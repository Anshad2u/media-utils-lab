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
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import requests

import archive

API_ROOT = "https://api.telegram.org"
# "140512-35555043181.txt" - UTC time of day, then the run id, under a UTC date
# directory. archive.py decides this shape; this only has to read it.
NAME = re.compile(r"^(\d{2})(\d{2})(\d{2})-[^/]+\.txt$")

# A day that produced more than this is not a day, it is a bug - and Telegram
# would refuse it anyway. Truncate rather than fail, and say so in the document.
MAX_CHARS = 8_000_000


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def configured() -> bool:
    """Whether an archive exists to read.

    Checked before anything else so that an unconfigured repository is quiet
    rather than broken. The scheduled run happens every night whether or not
    anyone has set the secrets up yet, and a job that fails nightly and sends a
    failure notice for a feature nobody has switched on is worse than no feature:
    it trains the owner to ignore the one channel that is supposed to mean
    something. The archive in archive.py is inert the same way, so the two agree.
    """
    return bool(archive.settings()[0])


def window(day: dt.date, offset: int) -> tuple[dt.datetime, dt.datetime]:
    """The UTC half-open window that corresponds to one local calendar day."""
    start = dt.datetime.combine(day, dt.time(0, 0)) - dt.timedelta(hours=offset)
    return start, start + dt.timedelta(days=1)


def utc_days(start: dt.datetime, end: dt.datetime) -> list[str]:
    """Every UTC date directory the window can reach into.

    The window is local, the directories are UTC, so one local day can touch two
    of them. Stepping in twelve-hour increments covers a window of any length
    without arithmetic that has to be right about month ends.

    The last instant is `end` minus a second, not `end` itself: the window is
    half-open, so nothing is ever stored at `end`, and naming that directory
    would check out a folder that cannot contribute a file.
    """
    days = set()
    last = end - dt.timedelta(seconds=1)
    cursor = start
    while cursor <= last:
        days.add(f"{cursor:%Y-%m-%d}")
        cursor += dt.timedelta(hours=12)
    days.add(f"{last:%Y-%m-%d}")
    return sorted(days)


def collect(work: Path, start: dt.datetime, end: dt.datetime) -> list[dict]:
    """Every archived transcript whose own timestamp falls inside the window.

    The directory name is not trusted as the timestamp: a file's date prefix is
    UTC and the window is local, so the name is parsed and the result is checked
    against the window. That keeps the day boundary in one place, and it means
    changing LOCAL_TZ later re-groups everything correctly instead of splitting
    days at a stale line.
    """
    found: list[dict] = []
    for folder in sorted((work / archive.DIR).glob("*")):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.txt")):
            match = NAME.match(path.name)
            if not match:
                continue
            stamp = dt.datetime(
                int(folder.name[:4]), int(folder.name[5:7]), int(folder.name[8:10]),
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
            )
            if not (start <= stamp < end):
                continue
            found.append({
                "key": f"{archive.DIR}/{folder.name}/{path.name}",
                "stamp": stamp,
                "size": path.stat().st_size,
                "text": path.read_text(encoding="utf-8", errors="replace"),
                "metrics": metrics_for(path),
            })
    found.sort(key=lambda entry: entry["stamp"])
    return found


def metrics_for(path: Path) -> dict:
    """The sidecar written beside a transcript, or {} if it is missing.

    Missing is not fatal: the document still compiles, and the header simply
    totals less. A day that cannot be compiled because one sidecar was lost
    would be a worse outcome than a day with one recording's duration absent.

    The reason is not logged. The exception text carries the path, and the path
    carries the recording's own UTC timestamp - which is exactly the kind of
    thing that has no business in a world-readable log.
    """
    sidecar = path.with_suffix(".json")
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
    except Exception as error:
        # Through the sanitiser, not str(error). A requests failure embeds the
        # whole request URL, and this one carries the bot token - so the raw
        # message is the single most dangerous thing this file could print.
        print(f"sendDocument: {archive.describe(error)}")
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

    if not configured():
        # Deliberately silent on Telegram. See configured().
        print("the transcript archive is not configured, so there is nothing to compile")
        print("set ARCHIVE_REPO and ARCHIVE_SSH_KEY as repository secrets, and the")
        print("archive in process.py will start filling it")
        return 0

    start, end = window(day, offset)
    print(f"day {day} (local UTC{offset:+d}) -> UTC {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M}")

    repo, key = archive.settings()
    days = utc_days(start, end)
    with tempfile.TemporaryDirectory() as workspace:
        work = archive.checkout(print, Path(workspace), repo, key, days)
        if work is None:
            # Not the same thing as an empty day, and must not be reported as
            # one. Exiting non-zero is what makes the workflow's failure notice
            # fire, which is the only way this becomes visible.
            sys.exit("could not read the archive; the digest is not a zero-day report")

        entries = collect(work, start, end)
        print(f"archived transcripts found: {len(entries)}")

        if not entries:
            # Silence would be indistinguishable from a broken digest, and the
            # whole point of this pipeline is that a gap is never silent.
            text = f"No recordings were archived for {day}."
            print(text)
            if not args.dry_run:
                token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("ALLOWED_CHAT_ID")
                if token and chat_id:
                    requests.post(f"{API_ROOT}/bot{token}/sendMessage",
                                  data={"chat_id": chat_id, "text": text}, timeout=30)
            return 0

        for entry in entries:
            metrics = entry["metrics"]
            print(f"  {entry['stamp']:%H:%M:%S}Z  {len(entry['text']):>7} chars  "
                  f"recorded {number(metrics, 'recorded_s'):.0f}s  "
                  f"voice {number(metrics, 'voice_s'):.0f}s")

        document, totals = build(day, offset, entries)

    # Everything below works from the document in memory, so the working copy is
    # gone by this point. Deliberately: the clone holds a credential, and the
    # window in which that credential exists on disk should be as short as it
    # can be while still doing the job.
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
