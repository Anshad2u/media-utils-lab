#!/usr/bin/env python3
"""
Extract the speech belonging to one enrolled speaker from an inbound audio message.

Runs inside CI on a throwaway runner: the audio is downloaded to a temp directory,
processed, sent back, and deleted. Nothing is uploaded as a build artifact.

Design note: every log line here is world-readable, so nothing derived from the
message (ids, paths, urls, names) is ever formatted into a log record.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import requests

SAMPLE_RATE = 16000
API_ROOT = "https://api.telegram.org"

MAX_SPEECH_S = 10.0  # speech runs longer than this get split ...
WINDOW_S = 4.0  # ... into windows of roughly this length
MIN_WINDOW_S = 0.5  # windows shorter than this are padded up to it
SEGMENT_PAD_S = 0.15  # context added around a kept run
MERGE_GAP_S = 0.15  # kept runs closer than this are fused
JOIN_GAP_S = 0.15  # silence inserted between fused runs

# Transcription / language mix
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
CHUNK_S = 8.0  # short enough that a chunk rarely spans two languages
MIN_CHUNK_S = 1.0  # trailing fragments below this are dropped
MAX_CHUNKS = 80  # bounds the API calls one message can trigger (~10 min of speech)
CHUNK_PAUSE_S = 5.0  # two calls per chunk, paced under the free tier's request cap
TRANSCRIPT_LIMIT = 3500  # Telegram caps a message at 4096 characters

# A chunk is transcribed once per candidate language, and the more confident
# reading wins. This is content-aware rather than acoustic, which matters here:
# the acoustic identifier handles a clean studio sample at 0.95 confidence and
# still scatters a compressed voice note with an Indian accent across Tamil,
# Telugu, Malayalam, Urdu and Assamese. The transcriber is far better at accented
# speech, and the confidence tells us when neither language fits - which is what
# Beary looks like to it.
LANGUAGE_CANDIDATES = ("ar", "en")
LANGUAGE_FLOOR = -0.80  # below this, the reading is not credible
LANGUAGE_MARGIN = 0.15  # the winner must beat the runner-up by this much
SILENCE_CEILING = 0.60  # no_speech_prob above this means there was nothing to read
REPETITION_FLOOR = 0.40  # unique-word ratio below this means the model was looping

# Language identification runs locally, so it is free and can be as fine grained
# as we like. Whisper's own `language` field is deliberately NOT used for this:
# it is a byproduct of decoding rather than a real classifier, it returns a
# single label for a whole 28 s chunk, and it falls back to English on anything
# outside its language set - which is exactly how a clip that moved through
# English, Beary and Arabic came back as 100% English.
LID_WINDOW_S = 3.0
LID_HOP_S = 1.5
MIN_LID_S = 1.0
LANGUAGE_BUCKETS = {"ar": "Arabic", "en": "English"}
OTHER_BUCKET = "Beary / other"

# A Thai sample that ships with the language model. The self-test classifies it
# and expects Thai back: a known-answer check. Without one, a language report
# that is quietly nonsense looks exactly like a language report that works.
LID_KNOWN_ANSWER_URL = (
    "https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/resolve/main/udhr_th.wav"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("process")


def setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(f"missing configuration: {name}")
    return value


def payload() -> dict:
    """Read the dispatch payload from the event file, not from the environment.

    The runner echoes every env value into the public log, and this payload carries
    the file id, so it is deliberately kept out of the environment entirely.
    """
    path = os.environ.get("EVENT_PATH") or os.environ.get("GITHUB_EVENT_PATH") or ""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            event = json.load(handle)
    except Exception:
        return {}
    return (event or {}).get("client_payload") or {}


def describe(error: BaseException) -> str:
    """A log-safe summary of an exception.

    These logs are world-readable and exception text routinely embeds URLs - a
    requests error carries the whole request URL, and a Telegram URL carries the
    bot token - so URL-shaped text and long opaque strings are stripped first.
    """
    text = str(error).replace("\n", " ")
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b", "<token>", text)
    text = re.sub(r"\bgsk_[A-Za-z0-9]{20,}\b", "<key>", text)
    text = re.sub(r"\b[A-Za-z0-9_-]{40,}\b", "<opaque>", text)
    return f"{type(error).__name__}: {text.strip()[:300]}"


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #

def telegram_json(token: str, method: str, data: dict | None = None, timeout: int = 60):
    """Sole entry point for JSON bot calls; the URL carries the token, so it is never logged."""
    try:
        response = requests.post(f"{API_ROOT}/bot{token}/{method}", data=data, timeout=timeout)
    except Exception:
        log.warning("telegram %s: network error", method)
        return None
    if response.status_code != 200:
        log.warning("telegram %s: http %s", method, response.status_code)
        return None
    try:
        return response.json()
    except Exception:
        log.warning("telegram %s: unreadable response", method)
        return None


def telegram_text(token: str, chat_id: str, text: str) -> None:
    telegram_json(token, "sendMessage", {"chat_id": chat_id, "text": text})


def download(token: str, file_id: str, destination: Path) -> bool:
    meta = telegram_json(token, "getFile", {"file_id": file_id})
    if not meta or not meta.get("ok"):
        log.warning("could not resolve the file")
        return False

    remote = (meta.get("result") or {}).get("file_path")
    if not remote:
        log.warning("could not resolve the file")
        return False

    # Derived value: mask it even though we never intend to print it.
    print(f"::add-mask::{remote}", flush=True)

    try:
        with requests.get(
            f"{API_ROOT}/file/bot{token}/{remote}", stream=True, timeout=300
        ) as response:
            if response.status_code != 200:
                log.warning("download failed (status code %s)", response.status_code)
                return False
            with destination.open("wb") as handle:
                for chunk in response.iter_content(1 << 16):
                    handle.write(chunk)
    except Exception:
        log.warning("download failed (network error)")
        return False

    return destination.stat().st_size > 0


def send_document(
    token: str,
    chat_id: str,
    path: Path,
    caption: str,
    filename: str = "clip.m4a",
    mime: str = "audio/mp4",
) -> bool:
    try:
        with path.open("rb") as handle:
            response = requests.post(
                f"{API_ROOT}/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": (filename, handle, mime)},
                timeout=600,
            )
    except Exception:
        log.warning("upload failed (network error)")
        return False
    if response.status_code != 200:
        log.warning("upload failed (status code %s)", response.status_code)
        return False
    return True


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #

def ffmpeg(arguments: list[str]) -> bool:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments]
    # stderr is swallowed: ffmpeg echoes the input filename into its errors.
    return subprocess.run(command, capture_output=True).returncode == 0


def decode(source: Path, destination: Path) -> bool:
    return ffmpeg(
        ["-i", str(source), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(destination)]
    )


def encode(source: Path, destination: Path) -> bool:
    return ffmpeg(["-i", str(source), "-c:a", "aac", "-b:a", "80k", "-ac", "1", str(destination)])


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


# --------------------------------------------------------------------------- #
# Detection and matching
# --------------------------------------------------------------------------- #

def speech_runs(wav_path: Path) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    model = load_silero_vad()
    audio = read_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    stamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=300,
        min_silence_duration_ms=400,
        speech_pad_ms=0,
    )
    return [(stamp["start"] / SAMPLE_RATE, stamp["end"] / SAMPLE_RATE) for stamp in stamps]


def load_encoder():
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path("pretrained_models") / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )


def embed(encoder, samples: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(samples.astype(np.float32) / 32768.0).unsqueeze(0)
    with torch.no_grad():
        vector = encoder.encode_batch(tensor).squeeze().cpu().numpy()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


# --------------------------------------------------------------------------- #
# Transcription and language mix
# --------------------------------------------------------------------------- #
#
# Whisper handles Arabic and English well. It has no Beary at all, and neither
# does anything else released - the best available model (SraVaani-1.0) scores
# 77.8% WER on Bearybashe, which is unusable. So Beary is measured as a bucket
# rather than transcribed. That is enough for the question that matters here -
# how much Arabic is actually spoken - and it needs only the language label.

def transcribe_mode() -> str:
    """TRANSCRIBE = on | off."""
    value = (os.environ.get("TRANSCRIBE") or "off").strip().lower()
    return "on" if value in {"1", "true", "yes", "on"} else "off"


def groq_keys() -> list[str]:
    """Both configured keys, primary first, so one running dry falls through."""
    return [
        value
        for value in (
            (os.environ.get("GROQ_API_KEY") or "").strip(),
            (os.environ.get("GROQ_API_KEY_2") or "").strip(),
        )
        if value
    ]


def read_chunk(keys: list[str], samples: np.ndarray, language: str) -> tuple[str, float, float] | None:
    """Transcribe one chunk in a forced language.

    Returns (text, mean avg_logprob, no_speech_prob), or None if no key worked.

    `language` is an ISO-639-1 code and is always passed. Left out, the model
    picks for itself and may answer in English, which silently turns a
    transcription into a translation - that is why Arabic came back as English.
    """
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.astype(np.int16).tobytes())
    payload = buffer.getvalue()

    for key in keys:
        try:
            response = requests.post(
                GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("chunk.wav", payload, "audio/wav")},
                data={
                    "model": GROQ_MODEL,
                    "response_format": "verbose_json",
                    "language": language,
                },
                timeout=120,
            )
        except Exception as error:
            # describe() strips URLs and long opaque strings, so neither the
            # endpoint nor the key can reach the public log from here.
            log.warning("transcription request failed: %s", describe(error))
            continue

        if response.status_code == 200:
            try:
                body = response.json()
            except Exception:
                log.warning("transcription: unreadable response")
                continue
            segments = body.get("segments") or []
            scores = [
                float(segment["avg_logprob"])
                for segment in segments
                if segment.get("avg_logprob") is not None
            ]
            silence = max(
                (float(segment.get("no_speech_prob", 0.0)) for segment in segments),
                default=0.0,
            )
            mean = sum(scores) / len(scores) if scores else LANGUAGE_FLOOR
            return str(body.get("text") or "").strip(), mean, silence

        if response.status_code == 429:
            # Rate limited rather than rejected, so waiting is worth it.
            log.info("transcription rate limited, backing off")
            time.sleep(5.0)
            continue

        # 401/403 means the key is dead. Either way the next key is worth trying.
        log.warning("transcription http %s", response.status_code)

    return None


def repetitive(text: str) -> bool:
    """True if a reading looks like the model looping rather than listening.

    Forced to a language it cannot actually hear, Whisper does not return nothing
    - it returns a short phrase repeated until it fills the window. The user's own
    Beary came back as "I am recording. I am recording. I am recording.", which is
    the tell. Real speech does not collapse to a handful of distinct words.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < 12:
        return False
    return len(set(words)) / len(words) < REPETITION_FLOOR


def decide_chunk(keys: list[str], samples: np.ndarray) -> tuple[str, str | None, str]:
    """(bucket, iso code, text) for one chunk.

    Every candidate language gets a reading. A reading is discarded if the model
    found no speech, if it looped, or if it is not credible on its own. The
    survivor must then beat the runner-up by a clear margin - two readings that
    are nearly as good as each other mean neither language actually fits, which
    is exactly what Beary looks like to a model that has neither language.
    """
    readings: list[tuple[str, float, str]] = []
    for code in LANGUAGE_CANDIDATES:
        reading = read_chunk(keys, samples, code)
        if reading is not None:
            text, score, silence = reading
            if silence >= SILENCE_CEILING:
                log.info("chunk reading for %s was silent", code)
            elif repetitive(text):
                log.info("chunk reading for %s was looping", code)
            else:
                readings.append((code, score, text))
        time.sleep(CHUNK_PAUSE_S / len(LANGUAGE_CANDIDATES))

    if not readings:
        return OTHER_BUCKET, None, ""

    readings.sort(key=lambda item: -item[1])
    code, score, text = readings[0]
    runner_up = readings[1][1] if len(readings) > 1 else None
    # A language code and two floats carry no speech content, so this is log-safe.
    log.info(
        "chunk reading: %s at %.2f, next %s",
        code,
        score,
        f"{runner_up:.2f}" if runner_up is not None else "n/a",
    )

    if score < LANGUAGE_FLOOR:
        return OTHER_BUCKET, None, ""
    if runner_up is not None and score - runner_up < LANGUAGE_MARGIN:
        return OTHER_BUCKET, None, ""
    return LANGUAGE_BUCKETS[code], code, text


def load_lid():
    """The local language identifier. Same framework as the speaker encoder."""
    from speechbrain.inference.classifiers import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=str(Path("pretrained_models") / "lang-id-voxlingua107-ecapa"),
        run_opts={"device": "cpu"},
    )


def language_windows(lid, samples: np.ndarray) -> list[tuple[float, float, str, float]]:
    """Classify short overlapping windows -> [(start_s, end_s, iso, confidence)].

    Windows are deliberately short. A single 28 s window can hold two or three
    languages, and any classifier asked about a whole chunk has to answer with a
    single label - which is how a mixed clip collapses to one language. Short
    windows with a hop let the mix survive.
    """
    import torch

    span = int(LID_WINDOW_S * SAMPLE_RATE)
    hop = int(LID_HOP_S * SAMPLE_RATE)
    floor = int(MIN_LID_S * SAMPLE_RATE)

    found: list[tuple[float, float, str, float]] = []
    for start in range(0, max(1, len(samples) - span + hop), hop):
        piece = samples[start: start + span]
        if len(piece) < floor:
            continue
        waveform = torch.from_numpy(piece.astype(np.float32) / 32768.0)
        try:
            with torch.no_grad():
                _, score, _, label = lid.classify_batch(waveform)
        except Exception as error:
            log.warning("language window failed: %s", describe(error))
            continue
        # label reads like "ar: Arabic"; the ISO code is what we bucket on.
        code = str(label[0]).split(":")[0].strip().lower()
        found.append((start / SAMPLE_RATE, (start + len(piece)) / SAMPLE_RATE, code, float(score[0].exp())))
    return found


def raw_labels(windows: list[tuple[float, float, str, float]]) -> str:
    """Unfiltered label counts and mean confidence, for the log.

    The confidence is the important half. A model that is out of its depth does
    not fail loudly - it spreads its answers thinly across unrelated languages at
    low confidence, which looks like a result and is not one. `af 4@0.12` says
    "four windows, and the model was guessing". Language codes carry no speech
    content, so this is safe in a public log.
    """
    stats: dict[str, list[float]] = {}
    for _, _, code, confidence in windows:
        stats.setdefault(code, []).append(confidence)
    ordered = sorted(stats.items(), key=lambda item: -len(item[1]))
    return ", ".join(f"{code} {len(c):d}@{sum(c) / len(c):.2f}" for code, c in ordered) or "none"


def language_report(samples: np.ndarray) -> tuple[str, str]:
    """The mix line for the caption, and the transcript for a follow-up message.

    Language is decided by reading each chunk in every candidate language and
    keeping the most confident reading - not by an acoustic classifier. The
    acoustic identifier still runs, but only so the two can be compared in the
    log: on this speaker's audio it scatters across unrelated languages while the
    transcriber's confidence stays meaningful.

    Pulled out of process() so the self-test exercises the same code path a real
    message takes. Loading a model successfully proves very little about the
    wiring around it.
    """
    if transcribe_mode() != "on":
        return "", ""

    # Comparison only. Never used to decide anything.
    try:
        log.info("acoustic labels: %s", raw_labels(language_windows(load_lid(), samples)))
    except Exception as error:
        log.warning("acoustic labelling failed: %s", describe(error))

    keys = groq_keys()
    if not keys:
        log.info("language mix skipped: no key configured")
        return "", ""

    step = int(CHUNK_S * SAMPLE_RATE)
    floor = int(MIN_CHUNK_S * SAMPLE_RATE)
    pieces = [samples[index: index + step] for index in range(0, len(samples), step)]
    pieces = [piece for piece in pieces if len(piece) >= floor][:MAX_CHUNKS]
    if not pieces:
        log.info("language mix: nothing long enough to read")
        return "", ""

    seconds: dict[str, float] = {}
    lines: list[str] = []
    for piece in pieces:
        bucket, _, text = decide_chunk(keys, piece)
        seconds[bucket] = seconds.get(bucket, 0.0) + len(piece) / SAMPLE_RATE
        # Beary has no model that can read it, so it is labelled and left alone
        # rather than turned into confident nonsense - which also saves the call.
        lines.append(f"[{bucket}] {text}" if text else f"[{bucket}] (not transcribed)")

    total = sum(seconds.values())
    mix_line = " | " + " ".join(
        f"{name} {value / total * 100:.0f}%"
        for name, value in sorted(seconds.items(), key=lambda item: -item[1])
    )
    log.info("language mix over %.1f s in %d chunk(s):%s", total, len(pieces), mix_line)
    return mix_line, "\n".join(lines)


def windows(run: tuple[float, float], total_s: float) -> list[tuple[float, float]]:
    start, end = run
    span = end - start
    if span <= MAX_SPEECH_S:
        if span >= MIN_WINDOW_S:
            return [(start, end)]
        pad = (MIN_WINDOW_S - span) / 2
        return [(max(0.0, start - pad), min(end + pad, total_s))]

    count = max(1, math.ceil(span / WINDOW_S))
    step = span / count
    pieces = []
    for index in range(count):
        left = start + index * step
        right = min(end, left + step)
        if right - left >= MIN_WINDOW_S:
            pieces.append((left, right))
    return pieces


def fuse(runs: list[tuple[float, float]], total_s: float) -> list[tuple[float, float]]:
    expanded = sorted((max(0.0, s - SEGMENT_PAD_S), min(total_s, e + SEGMENT_PAD_S)) for s, e in runs)
    fused: list[tuple[float, float]] = []
    for start, end in expanded:
        if fused and start - fused[-1][1] <= MERGE_GAP_S:
            fused[-1] = (fused[-1][0], max(fused[-1][1], end))
        else:
            fused.append((start, end))
    return fused


def render(samples: np.ndarray, runs: list[tuple[float, float]], destination: Path) -> float:
    gap = np.zeros(int(JOIN_GAP_S * SAMPLE_RATE), dtype=np.int16)
    pieces = []
    for start, end in runs:
        left = int(start * SAMPLE_RATE)
        right = min(int(end * SAMPLE_RATE), len(samples))
        if right > left:
            pieces.append(samples[left:right])
    if not pieces:
        return 0.0

    joined = pieces[0]
    for piece in pieces[1:]:
        joined = np.concatenate([joined, gap, piece])

    write_wav(destination, joined)
    return len(joined) / SAMPLE_RATE


# --------------------------------------------------------------------------- #
# Enrolment
# --------------------------------------------------------------------------- #

def enroll(token: str, chat_id: str, file_id: str) -> int:
    """Bootstrap: turn one clean recording into the reference vector.

    Only reachable while VOICE_EMBEDDING is unset, and only from the allowed chat,
    so this stops working the moment enrolment succeeds. Computing the vector here
    rather than on a laptop guarantees it comes from the same model, the same
    library versions and the same 16 kHz mono preprocessing as every comparison
    it will later be measured against.
    """
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        source, decoded, payload = root / "ref.bin", root / "ref.wav", root / "embedding.json"

        if not download(token, file_id, source):
            telegram_text(token, chat_id, "Could not fetch that file.")
            return 1
        if not decode(source, decoded):
            telegram_text(token, chat_id, "Could not decode that file.")
            return 1

        samples = read_wav(decoded)
        seconds = len(samples) / SAMPLE_RATE
        log.info("enrolment clip length: %.1f s", seconds)
        if seconds < 3.0:
            telegram_text(token, chat_id, "Too short. Send 10-20 seconds of clean speech.")
            return 0

        vector = embed(load_encoder(), samples)
        payload.write_text(
            json.dumps([round(float(value), 6) for value in vector]), encoding="utf-8"
        )

        caption = (
            f"Reference embedding from {seconds:.1f}s of audio. "
            "Store the contents of this file as the VOICE_EMBEDDING secret, "
            "then delete this message."
        )
        if not send_document(token, chat_id, payload, caption, "embedding.json", "application/json"):
            telegram_text(token, chat_id, "Could not send the result.")
            return 1

        log.info("enrolment complete")
        return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def process(token: str, chat_id: str, file_id: str, reference: np.ndarray, threshold: float) -> int:
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        source, decoded, joined, output = root / "in.bin", root / "in.wav", root / "join.wav", root / "out.m4a"

        if not download(token, file_id, source):
            telegram_text(token, chat_id, "Could not fetch that file.")
            return 1
        if not decode(source, decoded):
            telegram_text(token, chat_id, "Could not decode that file.")
            return 1

        original = duration(decoded)
        runs = speech_runs(decoded)
        log.info("speech runs detected: %d", len(runs))
        if not runs:
            telegram_text(token, chat_id, "No matching speech found.")
            return 0

        samples = read_wav(decoded)
        total_s = len(samples) / SAMPLE_RATE
        encoder = load_encoder()

        kept: list[tuple[float, float]] = []
        for run in runs:
            for start, end in windows(run, total_s):
                chunk = samples[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
                if len(chunk) < int(MIN_WINDOW_S * SAMPLE_RATE):
                    continue
                if float(np.dot(embed(encoder, chunk), reference)) >= threshold:
                    kept.append((start, end))

        log.info("matching runs: %d", len(kept))
        if not kept:
            telegram_text(token, chat_id, "No matching speech found.")
            return 0

        merged = fuse(kept, total_s)
        kept_seconds = render(samples, merged, joined)
        if kept_seconds <= 0 or not encode(joined, output):
            telegram_text(token, chat_id, "Processing failed.")
            return 1

        # Language mix. Non-fatal by design: the recording is the deliverable, and
        # a spent API quota must not cost the user their audio.
        mix_line, transcript = "", ""
        if transcribe_mode() == "on":
            if kept_seconds > MAX_CHUNKS * CHUNK_S:
                log.info("language mix skipped: over the chunk budget")
            else:
                try:
                    mix_line, transcript = language_report(read_wav(joined))
                except Exception as error:
                    log.warning("language mix failed: %s", describe(error))

        caption = (
            f"original {original:.1f}s | kept {kept_seconds:.1f}s | "
            f"{len(merged)} segment(s) | threshold {threshold:.2f}{mix_line}"
        )
        if not send_document(token, chat_id, output, caption):
            telegram_text(token, chat_id, "Could not send the result.")
            return 1

        if transcript:
            telegram_text(token, chat_id, transcript[:TRANSCRIPT_LIMIT])

        log.info("finished")
        return 0


def main() -> int:
    token = setting("TELEGRAM_BOT_TOKEN")

    if (os.environ.get("SELFTEST") or "").strip().lower() in {"1", "true", "yes"}:
        # Environment check: prove the models load before trusting a real run.
        try:
            encoder = load_encoder()
            noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 4) * 1000
            vector = embed(encoder, noise.astype(np.int16))
            log.info("selftest ok, embedding dim %d", vector.shape[0])
        except Exception as error:
            log.warning("selftest failed: %s", describe(error))
            return 1

        if transcribe_mode() == "on":
            # Exercises the same helper a real message calls, end to end: the
            # local identifier, the bucketing, and the transcription round trip.
            # The input is noise so the labels mean nothing, but a broken code
            # path shows up here rather than on someone's recording.
            try:
                mix_line, transcript = language_report(noise.astype(np.int16))
                log.info(
                    "selftest ok, mix line %d char(s), transcript %d char(s)",
                    len(mix_line),
                    len(transcript),
                )
            except Exception as error:
                log.warning("selftest failed: %s", describe(error))
                return 1

            # Known-answer test for the identifier itself. A clean Thai sample
            # must come back as Thai. If it does not, either the model or the way
            # it is being called is wrong, and every language report is
            # meaningless - which is precisely the failure that is invisible,
            # because a confused model still returns confident-looking labels.
            try:
                with tempfile.TemporaryDirectory() as workspace:
                    root = Path(workspace)
                    raw_file, wav = root / "known.bin", root / "known.wav"
                    response = requests.get(LID_KNOWN_ANSWER_URL, timeout=120)
                    response.raise_for_status()
                    raw_file.write_bytes(response.content)
                    if not decode(raw_file, wav):
                        log.warning("selftest failed: could not decode the known-answer sample")
                        return 1
                    known = language_windows(load_lid(), read_wav(wav))
                    log.info("selftest lid known answer (expect th): %s", raw_labels(known))
                    if "th" not in {code for _, _, code, _ in known}:
                        log.warning("selftest failed: known-answer sample not identified as Thai")
                        return 1
            except Exception as error:
                log.warning("selftest known answer failed: %s", describe(error))
                return 1

        return 0

    allowed = setting("ALLOWED_CHAT_ID").strip()
    dispatch = payload()
    chat_id = str(dispatch.get("chat_id") or "").strip()
    file_id = str(dispatch.get("file_id") or "")
    threshold = float(os.environ.get("MATCH_THRESHOLD") or "0.45")

    # Re-checked here as well as in the relay: a dispatch can be replayed.
    if not chat_id or chat_id != allowed:
        log.info("ignoring unexpected chat")
        return 0
    if not file_id:
        log.info("no file reference in payload")
        return 0

    raw_reference = (os.environ.get("VOICE_EMBEDDING") or "").strip()

    if not raw_reference:
        # Nothing enrolled yet, so treat this message as the reference recording.
        try:
            return enroll(token, chat_id, file_id)
        except Exception as error:
            log.warning("enrolment failed: %s", describe(error))
            telegram_text(token, chat_id, "Enrolment failed.")
            return 1

    try:
        reference = np.asarray(json.loads(raw_reference), dtype=np.float32)
    except Exception as error:
        log.warning("reference embedding unusable: %s", describe(error))
        return 1

    try:
        return process(token, chat_id, file_id, reference, threshold)
    except Exception as error:
        log.warning("processing failed: %s", describe(error))
        telegram_text(token, chat_id, "Processing failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
