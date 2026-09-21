#!/usr/bin/env python3
"""
Collect one local day's transcripts into a single document and send it.

Why this is separate from process.py
------------------------------------
process.py runs once per recording, on a runner that is destroyed at the end of
the job. It has no memory of the day and no way to acquire one. The archive it
writes to (archive_transcript() in process.py) is what gives the day a memory,
and this is the reader for it.

Why a day boundary needs stating
--------------------------------
Object keys are UTC, because that is what the writer had. The day the owner means
is local. So this converts: for local day D and offset T, the UTC window is
[D 00:00 - T, D 00:00 - T + 24h), which can span two UTC date prefixes. Both are
listed and each key's own timestamp is parsed and checked against the window, so
the boundary is applied here and nowhere else. Changing LOCAL_TZ later re-groups
everything correctly instead of splitting days at a stale line.

What it will not do
-------------------
Print a transcript. These logs are public. Only counts, durations and dates are
ever logged; the text goes to Telegram and nowhere else.

    python digest.py                    # yesterday, local time
    python digest.py --date 2026-09-21  # a specific local day
    python digest.py --date 2026-09-21 --dry-run   # summarise, send nothing
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

import requests

API_ROOT = "https://api.telegram.org"
KEY = re.compile(r"^transcripts/(\d{4}-\d{2}-\d{2})/(\d{2})(\d{2})(\d{2})-[^/]+\.txt$")

# A day that produced more than this is not a day, it is a bug - and Telegram
# would refuse it anyway. Truncate rather than fail, and say so in the document.
MAX_CHARS = 8_000_000


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def client():
    import boto3
    from botocore.config import Config

    account = env("R2_ACCOUNT_ID")
    bucket = env("R2_BUCKET")
    if not (account and bucket):
        sys.exit("R2_ACCOUNT_ID and R2_BUCKET are required")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "standard"},
                      connect_timeout=10, read_timeout=30),
    ), bucket


def window(day: dt.date, offset: int) -> tuple[dt.datetime, dt.datetime]:
    """The UTC half-open window that corresponds to one local calendar day."""
    start = dt.datetime.combine(day, dt.time(0, 0)) - dt.timedelta(hours=offset)
    return start, start + dt.timedelta(days=1)


def collect(s3, bucket: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
    """Every archived transcript whose own timestamp falls inside the window."""
    prefixes = set()
    cursor = start
    while cursor < end:
        prefixes.add(f"transcripts/{cursor:%Y-%m-%d}/")
        cursor += dt.timedelta(hours=12)

    found: list[dict] = []
    for prefix in sorted(prefixes):
        token = None
        while True:
            page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, **(
                {"ContinuationToken": token} if token else {}))
            for item in page.get("Contents", []):
                match = KEY.match(item["Key"])
                if not match:
                    continue
                stamp = dt.datetime(
                    int(match.group(1)[:4]), int(match.group(1)[5:7]), int(match.group(1)[8:10]),
                    int(match.group(2)), int(match.group(3)), int(match.group(4)),
                )
                if start <= stamp < end:
                    found.append({"key": item["Key"], "stamp": stamp,
                                  "size": item.get("Size", 0)})
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")

    found.sort(key=lambda entry: entry["stamp"])
    return found


def body_and_metrics(s3, bucket: str, key: str) -> tuple[str, dict]:
    """One object's text and its metrics, in a single request.

    get_object returns the user metadata alongside the body, so the day can be
    totalled without a separate HEAD for every recording.
    """
    response = s3.get_object(Bucket=bucket, Key=key)
    text = response["Body"].read().decode("utf-8", "replace")
    return text, {k.lower(): v for k, v in (response.get("Metadata") or {}).items()}


def number(metrics: dict, name: str) -> float:
    try:
        return float(metrics.get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def build(day: dt.date, offset: int, entries: list[dict]) -> tuple[str, dict]:
    totals = {"recorded": 0.0, "speech": 0.0, "voice": 0.0, "chars": 0}
    per_hour: dict[int, float] = {}
    blocks: list[str] = []

    for entry in entries:
        local = entry["stamp"] + dt.timedelta(hours=offset)
        metrics = entry["metrics"]
        recorded = number(metrics, "recorded_s")
        speech = number(metrics, "speech_s")
        voice = number(metrics, "voice_s")
        chars = int(number(metrics, "chars"))

        totals["recorded"] += recorded
        totals["speech"] += speech
        totals["voice"] += voice
        totals["chars"] += chars
        per_hour[local.hour] = per_hour.get(local.hour, 0.0) + speech

        blocks.append("=" * 72)
        blocks.append(
            f"{local:%H:%M}  recorded {recorded:.0f}s | speech {speech:.0f}s | "
            f"your voice {voice:.0f}s | {chars} chars"
        )
        blocks.append("=" * 72)
        blocks.append(entry["text"].rstrip())
        blocks.append("")

    header = [
        f"Day transcript - {day}  (local UTC{offset:+d})",
        "",
        f"recordings      : {len(entries)}",
        f"recorded        : {totals['recorded'] / 60:.1f} min",
        f"speech detected : {totals['speech'] / 60:.1f} min",
        f"your voice      : {totals['voice'] / 60:.1f} min",
        f"transcript      : {totals['chars']} chars",
        "",
    ]
    if per_hour:
        header.append("speech by hour (local)")
        peak = max(per_hour.values()) or 1.0
        for hour in sorted(per_hour):
            width = max(1, round(per_hour[hour] / peak * 30))
            header.append(f"  {hour:02d}:00  {per_hour[hour] / 60:6.1f}m  {'#' * width}")
        header.append("")

    return "\n".join(header + blocks), totals


def caption(day: dt.date, totals: dict, count: int) -> str:
    return (f"{day} | {count} recording(s) | "
            f"{totals['recorded'] / 60:.0f} min recorded | "
            f"{totals['speech'] / 60:.0f} min speech | "
            f"{totals['voice'] / 60:.0f} min your voice | "
            f"{totals['chars']} chars")


def send(token: str, chat_id: str, name: str, document: str, note: str) -> bool:
    try:
        response = requests.post(
            f"{API_ROOT}/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": note},
            files={"document": (name, document.encode("utf-8"), "text/plain")},
            timeout=120,
        )
    except Exception:
        print("sendDocument: network error")
        return False
    # The status alone is not proof Telegram accepted it, but the response body
    # can carry the file id, so only the code is reported.
    print(f"sendDocument: http {response.status_code}")
    return response.status_code == 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="local date, YYYY-MM-DD (default yesterday)")
    parser.add_argument("--tz", type=int, help="local UTC offset in hours (default $LOCAL_TZ or 0)")
    parser.add_argument("--dry-run", action="store_true", help="summarise, send nothing")
    args = parser.parse_args()

    offset = args.tz if args.tz is not None else int(env("LOCAL_TZ", "0") or 0)
    if args.date:
        day = dt.date.fromisoformat(args.date)
    else:
        # Yesterday, because a digest for a day that is still happening is a
        # partial answer that looks like a complete one.
        now_local = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=offset)
        day = now_local.date() - dt.timedelta(days=1)

    start, end = window(day, offset)
    print(f"day {day} (local UTC{offset:+d}) -> UTC {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M}")

    s3, bucket = client()
    entries = collect(s3, bucket, start, end)
    print(f"archived transcripts found: {len(entries)}")

    if not entries:
        # Silence would be indistinguishable from a broken digest, and the whole
        # point of this pipeline is that a gap is never silent.
        text = f"No recordings were archived for {day}."
        print(text)
        if not args.dry_run:
            token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("ALLOWED_CHAT_ID")
            if token and chat_id:
                requests.post(f"{API_ROOT}/bot{token}/sendMessage",
                              data={"chat_id": chat_id, "text": text}, timeout=30)
        return 0

    total_chars = 0
    for entry in entries:
        text, metrics = body_and_metrics(s3, bucket, entry["key"])
        entry["text"] = text
        entry["metrics"] = metrics
        total_chars += len(text)
        print(f"  {entry['stamp']:%H:%M:%S}Z  {len(text):>7} chars  "
              f"recorded {number(metrics, 'recorded_s'):.0f}s  "
              f"voice {number(metrics, 'voice_s'):.0f}s")

    document, totals = build(day, offset, entries)

    truncated = False
    if len(document) > MAX_CHARS:
        document = document[:MAX_CHARS] + "\n\n[truncated: the day exceeded the size cap]\n"
        truncated = True

    print(f"\ndocument: {len(document)} chars"
          f"{' (truncated)' if truncated else ''}, "
          f"recorded {totals['recorded'] / 60:.1f} min, "
          f"speech {totals['speech'] / 60:.1f} min, "
          f"your voice {totals['voice'] / 60:.1f} min")

    if args.dry_run:
        print("dry run: nothing sent")
        return 0

    token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("ALLOWED_CHAT_ID")
    if not (token and chat_id):
        sys.exit("TELEGRAM_BOT_TOKEN and ALLOWED_CHAT_ID are required to send")

    note = caption(day, totals, len(entries))
    if truncated:
        note += " | truncated"
    return 0 if send(token, chat_id, f"day-{day}.txt", document, note) else 1


if __name__ == "__main__":
    sys.exit(main())
