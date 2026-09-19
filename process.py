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
import subprocess
import sys
import tempfile
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("process")


def setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(f"missing configuration: {name}")
    return value


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

        caption = (
            f"original {original:.1f}s | kept {kept_seconds:.1f}s | "
            f"{len(merged)} segment(s) | threshold {threshold:.2f}"
        )
        if not send_document(token, chat_id, output, caption):
            telegram_text(token, chat_id, "Could not send the result.")
            return 1

        log.info("finished")
        return 0


def main() -> int:
    token = setting("TELEGRAM_BOT_TOKEN")
    allowed = setting("ALLOWED_CHAT_ID").strip()
    chat_id = (os.environ.get("CHAT_ID") or "").strip()
    file_id = os.environ.get("FILE_ID") or ""
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
        except Exception:
            log.warning("enrolment failed")
            telegram_text(token, chat_id, "Enrolment failed.")
            return 1

    try:
        reference = np.asarray(json.loads(raw_reference), dtype=np.float32)
    except Exception:
        log.warning("reference embedding unusable")
        return 1

    try:
        return process(token, chat_id, file_id, reference, threshold)
    except Exception:
        # Tracebacks and exception text can contain URLs, so they stop here.
        log.warning("processing failed")
        telegram_text(token, chat_id, "Processing failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
