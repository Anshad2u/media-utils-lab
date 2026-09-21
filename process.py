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
# whisper-large-v3, not the turbo variant. Turbo is a pruned model and is weaker
# on non-English audio, which is exactly where this pipeline needs accuracy. The
# free tier covers both, and nothing here is latency bound.
GROQ_MODEL = "whisper-large-v3"
CHUNK_S = 8.0  # short enough that a chunk rarely spans two languages
MIN_CHUNK_S = 1.0  # trailing fragments below this are dropped
# Bounds the API calls one recording can trigger, and it is set by quota, not by
# taste. Two Groq accounts give 2 x 2,000 = 4,000 audio requests/day. The target
# volume is ~300 h/month, which at a 10 minute cap is ~60 recordings/day. So the
# allowance per recording is 4,000 / 60 = 66 calls, and with two candidate
# languages that is 33 chunks. The cap is 28 rather than 33 deliberately: running
# at 96% of the allowance leaves nothing for retries or a burst, and the failure
# mode when the quota runs out is severe - see below. 28 uses ~84%.
# Raise this and the daily quota runs out mid-afternoon; the run then reports
# every chunk as "Beary / other" because every call failed, which looks exactly
# like a language-detection bug. The arithmetic above is the guard against that.
MAX_CHUNKS = 28
CHUNK_PAUSE_S = 6.0  # two calls per chunk, paced under the free tier's 20 rpm
# Telegram caps a message at 4096 characters, so this is the most that can be
# sent inline. It is a fallback now, not the delivery: the full transcript goes
# out as a .txt document, which has no such limit.
TRANSCRIPT_LIMIT = 3500
# The full transcript's filename, and the size past which it cannot be sent.
# Groq refuses an upload over its own limit with a 413, and a 10 minute file at
# the encoder's 80k is about 6 MB - so this is a guard against a longer
# accumulation, not against the normal case.
TRANSCRIPT_SUFFIX = "-transcript"
FULL_SUFFIX = "-full"
GROQ_MAX_BYTES = 24 * 1024 * 1024

# A chunk is transcribed once per candidate language, and the most confident
# reading wins. This is content-aware rather than acoustic, which matters here:
# the acoustic identifier handles a clean studio sample at 0.95 confidence and
# still scatters a compressed voice note with an Indian accent across Tamil,
# Telugu, Malayalam, Urdu and Assamese. The transcriber is far better at accented
# speech, and its confidence tells us when neither candidate fits - which is what
# Beary looks like to it.
#
# Beary is deliberately absent and must stay absent: the model has no Beary, so a
# "beary" candidate cannot be passed and could never win a reading. Beary falls
# through to OTHER_BUCKET by design, which is what we want - labelled and left
# alone rather than forced into Arabic or English.
#
# Hindi was tried here and has been removed. The reasoning for adding it was sound
# - an omitted language gets forced into one that is being measured, which corrupts
# the figure - but the evidence went the other way. On a recording containing no
# Hindi at all, BOTH Beary chunks were read as Hindi, one of them rendering "Beary"
# as Devanagari (बेरी). Beary is Dravidian and close enough in sound for the model to
# prefer it over admitting defeat, so Hindi did not absorb a stray 1% of speech; it
# stole the very language we are trying to isolate. Two candidates only.
LANGUAGE_CANDIDATES = ("ar", "en")

# A single gate, replacing the floor-plus-margin pair that was here before.
#
# The margin rule required the winner to beat the runner-up by 0.15 and was the
# largest single source of error. On a real recording it discarded `ar at -0.14` -
# the most confident reading in the entire file, and correct - purely because `en`
# scored -0.26. A right answer thrown away for being nearly right twice. The floor
# alone does what the margin was meant to do.
#
# -0.45 is measured, not guessed. From one recording with known ground truth:
#   real English   -0.17, -0.28, -0.58
#   real Arabic    -0.14
#   Beary          -0.50, -0.73   (read as Hindi, which is what exposed this)
# Real speech lands at -0.58 and above, Beary below. The ranges still overlap, so
# this is a starting point fitted to six chunks, not a settled value. Every
# candidate score is logged per chunk precisely so the next run can do better.
LANGUAGE_FLOOR = -0.45
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
# How much of its probability mass the identifier must put on a candidate before
# its vote counts for anything. The model has 107 classes, so a uniform guess is
# about 0.01, and a reading near that is the model declining to answer rather
# than answering. Observed real readings sit at 0.33-1.00 and observed
# non-answers at 0.00-0.09, so the floor sits in the gap. Without it, a chunk
# whose windows scored 0.00 against 0.00 was still being decided by whichever
# candidate won on the third decimal place - which quietly overrode a confident
# English reading and turned an English chunk into Arabic, with nothing in the
# log to say the vote had been a coin toss between two non-answers.
LID_MASS_FLOOR = 0.25
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


def routine_notice(token: str, chat_id: str, kind: str, text: str) -> None:
    """Report an outcome that is normal, not a fault.

    Whether to speak up depends on who is listening. A person who just sent a
    voice note and hears nothing back cannot tell "nothing matched" from "the
    pipeline is broken", so they are told. An automated upload has nobody waiting
    on it, and once the recorder only sends speech, a recording the voice match
    declines is an ordinary result rather than an exception - at roughly a hundred
    files an hour, reporting each one buries the messages that do need a person.

    Faults do not come through here. A file that could not be fetched, decoded or
    sent is a real problem and is reported for both kinds, because the alternative
    is a failure that nobody ever learns about.
    """
    if kind == "document":
        log.info("not notifying an automated upload: %s", text)
        return
    telegram_text(token, chat_id, text)


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


def level(path: Path) -> tuple[float, float]:
    """Peak and RMS of a decoded 16-bit mono wav, in dBFS.

    Two numbers and no content, so this is log-safe - and it is the only way to
    tell a silent recording apart from a merely quiet one from the outside. The
    two cases look identical downstream: the detector finds no runs and the
    pipeline reports "no matching speech", which reads like a matching fault
    when it may be a recording fault. A peak near the floor means there was
    never any speech to find; a healthy peak with no runs means the detector's
    own threshold is the thing to look at.
    """
    samples = read_wav(path).astype(np.float64) / 32768.0
    if samples.size == 0:
        return -math.inf, -math.inf
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return (
        20.0 * math.log10(peak) if peak > 0 else -math.inf,
        20.0 * math.log10(rms) if rms > 0 else -math.inf,
    )


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
    # Both of these were raised from 300/400 on 2026-09-21, after measuring what
    # the old values produced. Across 1,242 scored windows in 12 recordings, the
    # window length turned out to be the single strongest predictor of whether the
    # owner's voice was recognised at all:
    #
    #   under 1.0 s   429 windows   1.9% cleared the threshold
    #   1.0 - 2.0 s   335 windows   9.6%
    #   2.0 - 4.0 s   333 windows  17.7%
    #   4.0 - 8.0 s   125 windows  22.4%
    #   8.0 - 10  s    20 windows  10.0%
    #
    # So a third of all the work was going into windows that almost never matched.
    # The cause was here: min_speech_duration_ms=300 let the detector emit runs as
    # short as 0.3 s, and min_silence_duration_ms=400 cut runs apart at pauses that
    # a single speaker makes mid-sentence. Both then arrived at ECAPA as fragments
    # too short to carry a voiceprint, and windows() padded anything under 0.5 s up
    # to exactly 0.5 s, which is not enough audio to embed.
    #
    # Raising them fuses those fragments back into runs of a few seconds, which is
    # the band that matches best. It does not push towards long windows: the 8-10 s
    # band is the second worst, because a long run can span more than one speaker.
    # MAX_SPEECH_S and WINDOW_S already re-split anything long into ~4 s pieces, so
    # over-merging here is corrected downstream rather than left to hurt.
    #
    # 700 ms is chosen as just above a normal inter-word pause and just below the
    # gap between two people taking turns, so it joins a sentence without joining a
    # conversation. It is a guess with a measurement behind it, not a fitted value:
    # re-check the table above over the next recordings and move it if the 2-4 s
    # band does not grow.
    stamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=500,
        min_silence_duration_ms=700,
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
    """Both configured keys. They belong to two separate Groq accounts, and Groq
    meters per organisation, so each carries its own independent allowance.
    read_chunk rotates between them rather than always starting at the first."""
    return [
        value
        for value in (
            (os.environ.get("GROQ_API_KEY") or "").strip(),
            (os.environ.get("GROQ_API_KEY_2") or "").strip(),
        )
        if value
    ]


# Rotates which account a call starts from. See read_chunk for why.
_key_cursor = 0


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

    # Start from the next account each time. Always starting at the first key
    # spends one account's daily allowance and leaves the other idle until the
    # first is rate limited, which throws away half the usable quota. Rotating
    # spreads the load; the fall-through still applies when one is exhausted.
    global _key_cursor
    start = _key_cursor % len(keys)
    _key_cursor += 1
    for key in keys[start:] + keys[:start]:
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


def full_transcript(keys: list[str], audio: Path) -> str:
    """The whole recording, read in a single pass.

    The per-chunk pass above exists to *decide* a language and a percentage, and
    it is budgeted to MAX_CHUNKS for exactly that reason. It is not a transcript.
    It samples, so it can only ever report a fraction of what was said, and it was
    being handed the kept audio rather than the recording - which is why a ten
    minute file came back as six lines covering fifty seconds.

    This is the transcript. One call for the whole file, and deliberately no
    `language`: naming one forces every window into it, and speech the model cannot
    read is then rendered as confident boilerplate in that language instead of
    being reported as uncertain. Left to choose, it decides per window, which is
    the only thing that works on a recording that moves between languages.

    Returns "" on failure, and the caller reads that as "no transcript" rather than
    as an error. The recording is the deliverable; a spent quota or a bad response
    must never cost the user their audio.
    """
    try:
        size = audio.stat().st_size if audio.exists() else 0
    except Exception as error:
        log.warning("full transcript unreadable: %s", describe(error))
        return ""
    if size == 0:
        return ""
    if size > GROQ_MAX_BYTES:
        # Retrying on another key would send the same bytes and get the same 413.
        log.warning("full transcript skipped: %.1f MB is over the endpoint limit",
                    size / 1024 / 1024)
        return ""
    try:
        payload = audio.read_bytes()
    except Exception as error:
        log.warning("full transcript unreadable: %s", describe(error))
        return ""

    # Same rotation as read_chunk, for the same reason: always starting at the
    # first key spends one account's allowance while the other sits idle.
    global _key_cursor
    start = _key_cursor % len(keys)
    _key_cursor += 1
    for key in keys[start:] + keys[:start]:
        try:
            response = requests.post(
                GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("recording.m4a", payload, "audio/mp4")},
                data={"model": GROQ_MODEL, "response_format": "json"},
                timeout=600,
            )
        except Exception as error:
            log.warning("full transcript request failed: %s", describe(error))
            continue

        if response.status_code == 200:
            try:
                return str(response.json().get("text") or "").strip()
            except Exception:
                log.warning("full transcript: unreadable response")
                continue

        if response.status_code == 413:
            log.warning("full transcript refused: file too large")
            return ""

        if response.status_code == 429:
            log.info("full transcript rate limited, backing off")
            time.sleep(5.0)
            continue

        log.warning("full transcript http %s", response.status_code)

    return ""


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


def decide_chunk(keys: list[str], samples: np.ndarray, preferred: str | None = None) -> tuple[str, str | None, str]:
    """(bucket, iso code, text) for one chunk.

    Every candidate language gets a reading. A reading is discarded if the model
    found no speech or if it looped. If none survives, or none clears
    LANGUAGE_FLOOR, the chunk is left as "other", which is what Beary looks like
    to a model that has none of its languages.

    Among the readings that clear the floor, the acoustics choose - `preferred`,
    from the local identifier. The reading's own avg_logprob cannot make that
    choice, because it measures how fluent the text is and a *translation* is
    fluent. Forced into a language it cannot hear, Whisper renders the speech in
    that language instead of refusing, so the two readings are not two
    transcriptions of the same audio: one is a transcription and the other is a
    rendering, and the rendering can outscore it. On a chunk holding two
    languages the comparison then picks the wrong one with confidence - which is
    how a sentence spoken in English came back as Arabic. The floor still decides
    whether any candidate fits at all, so "other" is unaffected; falling back to
    the most confident reading when the acoustics name no survivor keeps this a
    tie-break rather than a takeover.
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
    # Log every reading, not just the winner. The thresholds are set from these
    # numbers, and a winner-plus-runner-up line cannot show what the other
    # candidates scored - which is exactly the gap that made the last calibration
    # a guess. A language code and a float carry no speech content, so this is
    # log-safe.
    log.info("chunk readings: %s", ", ".join(f"{code} {score:.2f}" for code, score, _ in readings))

    # A reading is a fit only if it clears the floor. The list is sorted by score,
    # so an empty fits list is exactly the old "the winner is below the floor"
    # case - nothing about the "other" bucket moves.
    fits = [item for item in readings if item[1] >= LANGUAGE_FLOOR]
    if not fits:
        return OTHER_BUCKET, None, ""
    code, _, text = next((item for item in fits if item[0] == preferred), fits[0])
    return LANGUAGE_BUCKETS[code], code, text


def load_lid():
    """The local language identifier. Same framework as the speaker encoder."""
    from speechbrain.inference.classifiers import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=str(Path("pretrained_models") / "lang-id-voxlingua107-ecapa"),
        run_opts={"device": "cpu"},
    )


def candidate_indices(lid) -> dict[str, list[int]]:
    """ISO code -> the class indices whose label carries that code.

    Read from the loaded model rather than hard-coded. An index guessed from a
    model card would be a silent mistake: the numbers would still look like
    scores, and every decision built on them would be quietly meaningless.
    """
    encoder = lid.hparams.label_encoder
    table = getattr(encoder, "ind2lab", None)
    if not table:
        table = {index: label for label, index in encoder.lab2ind.items()}
    indices: dict[str, list[int]] = {}
    for index, label in table.items():
        code = str(label).split(":")[0].strip().lower()
        if code in LANGUAGE_CANDIDATES:
            indices.setdefault(code, []).append(int(index))
    return indices


def language_windows(lid, samples: np.ndarray) -> list[tuple[float, float, str, float, dict[str, float]]]:
    """Classify short windows -> [(start_s, end_s, iso, confidence, candidate_probs)].

    Windows are deliberately short. A single 28 s window can hold two or three
    languages, and any classifier asked about a whole chunk has to answer with a
    single label - which is how a mixed clip collapses to one language. Short
    windows with a hop let the mix survive.

    The fifth field is the posterior mass each candidate language holds in that
    window. The identifier's top-1 label is unreliable on this speaker - asked
    about Beary it spreads across Kannada, Telugu, Nepali and Sinhala at low
    confidence - but the question asked of it here is far narrower, and a two-way
    choice between languages as unlike each other as Arabic and English is a much
    easier call than a 107-way argmax. Same pass over the audio, same model, one
    extra number per window.
    """
    import torch

    span = int(LID_WINDOW_S * SAMPLE_RATE)
    hop = int(LID_HOP_S * SAMPLE_RATE)
    floor = int(MIN_LID_S * SAMPLE_RATE)
    indices = candidate_indices(lid)
    absent = [code for code in LANGUAGE_CANDIDATES if code not in indices]
    if absent:
        # Without this the vote is silently empty, every chunk falls back to the
        # old behaviour, and the change looks like it simply did not work rather
        # than like a broken label lookup.
        log.warning("language identifier has no class for: %s", ", ".join(absent))

    found: list[tuple[float, float, str, float, dict[str, float]]] = []
    for start in range(0, max(1, len(samples) - span + hop), hop):
        piece = samples[start: start + span]
        if len(piece) < floor:
            continue
        waveform = torch.from_numpy(piece.astype(np.float32) / 32768.0)
        try:
            with torch.no_grad():
                posterior, score, _, label = lid.classify_batch(waveform)
        except Exception as error:
            log.warning("language window failed: %s", describe(error))
            continue
        # label reads like "ar: Arabic"; the ISO code is what we bucket on.
        code = str(label[0]).split(":")[0].strip().lower()
        row = posterior[0]
        # The classifier hands back log-probabilities - the confidence above is
        # already one of them exponentiated to get a value in [0, 1] - so they are
        # exponentiated before being summed. Summing the logs instead would give a
        # negative number that still sorts the right way most of the time, which
        # is the worst kind of wrong: it would look like it worked.
        probs = {name: float(row[at].exp().sum()) for name, at in indices.items()}
        found.append(
            (start / SAMPLE_RATE, (start + len(piece)) / SAMPLE_RATE, code, float(score[0].exp()), probs)
        )
    return found


def preferred_language(
    windows: list[tuple[float, float, str, float, dict[str, float]]],
) -> tuple[str | None, str]:
    """(favoured candidate, log-safe reading of the vote) for one chunk.

    Windows vote rather than one window deciding, because a single 3 s window
    inside a chunk can land on a pause and name the wrong language for the whole
    chunk. The reading is what makes a wrong call diagnosable: it shows whether
    the vote was lopsided or a coin toss, which a bare winner cannot.

    Returns no preference when the vote is a tie, or when neither candidate drew
    LID_MASS_FLOOR of the mass. Both are cases where the identifier has not
    actually named a language, and in both the caller falls back to the
    transcriber's own confidence.
    """
    if not windows:
        return None, "no reading"
    totals = {code: 0.0 for code in LANGUAGE_CANDIDATES}
    votes = {code: 0 for code in LANGUAGE_CANDIDATES}
    for _, _, _, _, probs in windows:
        for code in LANGUAGE_CANDIDATES:
            totals[code] += probs.get(code, 0.0)
        winner = max(LANGUAGE_CANDIDATES, key=lambda code: probs.get(code, 0.0))
        votes[winner] += 1
    reading = " ".join(
        f"{code} {votes[code]}/{len(windows)}@{totals[code] / len(windows):.2f}"
        for code in LANGUAGE_CANDIDATES
    )
    first, second = LANGUAGE_CANDIDATES
    if totals[first] == totals[second]:
        return None, f"{reading} (tie)"

    winner = max(LANGUAGE_CANDIDATES, key=lambda code: totals[code])
    # A vote nobody can win is not a vote. If the identifier put no mass on
    # either candidate, it is saying "not one of these" - which is what Beary
    # looks like - and the third decimal place must not then pick a language.
    # The fallback is the transcriber's own confidence, i.e. the old behaviour.
    if totals[winner] / len(windows) < LID_MASS_FLOOR:
        return None, f"{reading} (no candidate recognised)"
    return winner, reading


def raw_labels(windows: list[tuple[float, float, str, float, dict[str, float]]]) -> str:
    """Unfiltered label counts and mean confidence, for the log.

    The confidence is the important half. A model that is out of its depth does
    not fail loudly - it spreads its answers thinly across unrelated languages at
    low confidence, which looks like a result and is not one. `af 4@0.12` says
    "four windows, and the model was guessing". Language codes carry no speech
    content, so this is safe in a public log.
    """
    stats: dict[str, list[float]] = {}
    for _, _, code, confidence, _ in windows:
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

    # One pass over the audio, used twice: the whole-recording label counts are
    # logged for comparison, and the same windows are re-read per chunk to say
    # which candidate the acoustics favour there.
    marks: list[tuple[float, float, str, float, dict[str, float]]] = []
    try:
        marks = language_windows(load_lid(), samples)
        log.info("acoustic labels: %s", raw_labels(marks))
    except Exception as error:
        log.warning("acoustic labelling failed: %s", describe(error))

    keys = groq_keys()
    if not keys:
        log.info("language mix skipped: no key configured")
        return "", ""

    step = int(CHUNK_S * SAMPLE_RATE)
    floor = int(MIN_CHUNK_S * SAMPLE_RATE)
    # The offset travels with the piece so each chunk can claim the identifier
    # windows that fall inside it.
    pieces = [(index, samples[index: index + step]) for index in range(0, len(samples), step)]
    pieces = [(index, piece) for index, piece in pieces if len(piece) >= floor]
    if not pieces:
        log.info("language mix: nothing long enough to read")
        return "", ""

    # Even sampling. Truncating to the first MAX_CHUNKS chunks would describe only
    # the opening minutes of a long recording, and a note that runs past the
    # budget used to be skipped outright - which silently reported nothing for
    # the long notes. Sampling across the whole span keeps the same call budget
    # while making the figure an estimate of the entire recording.
    sampled = len(pieces) > MAX_CHUNKS
    if sampled:
        stride = len(pieces) / MAX_CHUNKS
        pieces = [pieces[int(index * stride)] for index in range(MAX_CHUNKS)]
        log.info("language mix sampled every %.1f chunk(s) across the recording", stride)

    seconds: dict[str, float] = {}
    lines: list[str] = []
    for offset, piece in pieces:
        left = offset / SAMPLE_RATE
        right = (offset + len(piece)) / SAMPLE_RATE
        # Only windows lying wholly inside this chunk vote on it, so a window
        # straddling the boundary cannot colour both neighbours.
        inside = [window for window in marks if window[0] >= left and window[1] <= right]
        preferred, vote = preferred_language(inside)
        log.info("chunk acoustics: %s", vote)
        bucket, _, text = decide_chunk(keys, piece, preferred)
        seconds[bucket] = seconds.get(bucket, 0.0) + len(piece) / SAMPLE_RATE
        # Beary has no model that can read it, so it is labelled and left alone
        # rather than turned into confident nonsense - which also saves the call.
        lines.append(f"[{bucket}] {text}" if text else f"[{bucket}] (not transcribed)")

    total = sum(seconds.values())
    # A sampled figure is an estimate over the whole recording, so it is marked
    # as approximate rather than passed off as a measured breakdown.
    mark = "~" if sampled else ""
    mix_line = " | " + " ".join(
        f"{name} {mark}{value / total * 100:.0f}%"
        for name, value in sorted(seconds.items(), key=lambda item: -item[1])
    )
    log.info("language mix over %.1f s in %d chunk(s)%s:%s",
             total, len(pieces), " (sampled)" if sampled else "", mix_line)
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

CLEAN_SUFFIX = "-clean"


def _safe_stem(file_name: str) -> str:
    """The uploader's name, reduced to something safe to interpolate.

    The name arrives from whoever uploaded the file, so it is treated as
    untrusted and is never logged. Only the base name survives, anything outside
    letters, digits, dot, dash and underscore becomes an underscore, and the
    length is capped - a name carrying a path separator or a control character
    would otherwise be interpolated into a multipart header. An empty result is
    the caller's cue to fall back, because Telegram rejects a document sent with
    an empty filename.
    """
    base = Path(str(file_name or "")).name
    stem = Path(base).stem if base else ""
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:60].strip("._-")


def cleaned_name(file_name: str) -> str:
    """The name to give the cleaned audio: the original, suffixed.

    The name arrives from whoever uploaded the file, so it is treated as
    untrusted and is never logged. Only the base name survives, anything outside
    letters, digits, dot, dash and underscore becomes an underscore, and the
    length is capped - a name carrying a path separator or a control character
    would otherwise be interpolated into a multipart header. A name that
    survives none of that falls back to a fixed one, because Telegram rejects a
    document sent with an empty filename.

    The suffix is what keeps a cleaned file distinguishable from the original
    once a day's uploads are all sitting in the same chat.
    """
    keep = _safe_stem(file_name)
    if not keep:
        return f"clip{CLEAN_SUFFIX}.m4a"
    extension = Path(Path(str(file_name or "")).name).suffix
    return f"{keep}{CLEAN_SUFFIX}{extension or '.m4a'}"


def transcript_name(file_name: str) -> str:
    """The full transcript's name, built from the same uploader-supplied stem.

    Sanitised identically to the cleaned audio so the two sort together in the
    chat, and with a fixed .txt because the body is text whatever the recording
    happened to be.
    """
    return f"{_safe_stem(file_name) or 'clip'}{TRANSCRIPT_SUFFIX}.txt"


def full_name(file_name: str) -> str:
    """The whole recording's name: every speaker, dead air removed.

    A fixed .m4a rather than the uploaded extension, because this file is always
    re-encoded here whatever arrived.
    """
    return f"{_safe_stem(file_name) or 'clip'}{FULL_SUFFIX}.m4a"


def minute_map(runs: list[tuple[float, float]],
               mine: list[tuple[float, float]],
               total_s: float) -> str:
    """A minute-by-minute picture of where speech was, and where it was yours.

    The transcript says what was said; this says when, and - by separating the
    seconds that matched the reference voice from the rest - whether the speech
    in a given minute was the person the recording was made for or somebody
    else. That distinction is invisible in the text, which is a single stream
    with no notion of who was talking.

    Only offsets and durations, so it sits alongside a transcript without
    adding anything the transcript does not already carry.

    Written for a plain-text reader rather than a terminal: fixed columns, no
    escape codes, and bars short enough not to wrap on a phone.
    """
    if not runs or total_s <= 0:
        return ""

    span = max(1, math.ceil(total_s / 60.0))
    speech = [0.0] * span
    voice = [0.0] * span

    for start, end in runs:
        index = min(span - 1, max(0, int(start // 60.0)))
        speech[index] += max(0.0, min(end, total_s) - start)
    for start, end in mine:
        index = min(span - 1, max(0, int(start // 60.0)))
        voice[index] += max(0.0, end - start)

    # The caller passes fused runs, which do not overlap, so this should never
    # bite. Clamp anyway: the two columns come from different lists, and a minute
    # claiming more of the user's voice than it claims speech is a contradiction
    # the reader cannot resolve - they cannot tell which column is lying. A
    # slightly conservative bar is better than a visibly impossible one.
    for index in range(span):
        voice[index] = min(voice[index], speech[index])

    peak = max(speech) or 1.0
    width = 20
    lines = [
        "where you spoke, minute by minute",
        "# is all speech found, + is the part that matched your voice",
        "",
        "minute   speech   your voice   all speech            your voice",
    ]
    for index in range(span):
        lines.append(
            f"{index:>4}-{index + 1:<3}"
            f"{speech[index]:>7.1f}s{voice[index]:>11.1f}s   "
            f"{'#' * max(0, round(speech[index] / peak * width)):<{width}}  "
            f"{'+' * max(0, round(voice[index] / peak * width))}"
        )
    return "\n".join(lines) + "\n"


def process(
    token: str,
    chat_id: str,
    file_id: str,
    reference: np.ndarray,
    threshold: float,
    kind: str = "voice",
    file_name: str = "",
) -> int:
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        source, decoded, joined, output = root / "in.bin", root / "in.wav", root / "join.wav", root / "out.m4a"
        # The whole recording with the dead air removed, and the transcript. This
        # file is both sent back and read for the transcript, so the text and the
        # audio describe the same thing.
        all_wav, all_out = root / "all.wav", root / "full.m4a"
        text_out = root / "transcript.txt"

        if not download(token, file_id, source):
            # The one cause worth naming. Telegram lets a bot *send* 50 MB but only
            # *download* 20 MB, so a recording above that size sits in the chat and
            # is permanently unreadable. The relay refuses those now, so this should
            # be rare - but "Could not fetch that file." told the owner nothing about
            # what to do, and on 2026-09-21 that is exactly what a 27.8 MB upload
            # produced, after a full runner had been spun up to fail.
            telegram_text(token, chat_id,
                          "Could not fetch that file from Telegram. If it was a large "
                          "recording, note that Telegram will not return files over "
                          "20 MB - record at a lower bitrate.")
            return 1
        if not decode(source, decoded):
            telegram_text(token, chat_id, "Could not decode that file.")
            return 1

        original = duration(decoded)
        peak_db, rms_db = level(decoded)
        # The upload's own size, which nothing else in the pipeline reports. Telegram
        # caps downloads at 20 MB while allowing 50 MB uploads, so this number is the
        # first thing worth knowing when a fetch fails - and without it the log cannot
        # separate "too big" from "wrong file id". A byte count carries no speech.
        log.info("upload: %.1f MB", source.stat().st_size / 1024 / 1024)
        log.info("audio: %.1f s, peak %.1f dBFS, rms %.1f dBFS", original, peak_db, rms_db)

        runs = speech_runs(decoded)
        log.info("speech runs detected: %d, totalling %.1f s",
                 len(runs), sum(end - start for start, end in runs))
        if not runs:
            # "No matching speech" reads like the matcher is at fault, and on an
            # empty recording it is not. Quoting the level back separates the two
            # cases for the person who made the recording, without quoting
            # anything they said.
            detail = "silent recording" if peak_db < -50 else f"peak {peak_db:.0f} dBFS"
            routine_notice(token, chat_id, kind,
                           f"No speech found in {original:.0f}s ({detail}).")
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
                score = float(np.dot(embed(encoder, chunk), reference))
                # A score, a duration and an offset carry no speech content, so this
                # is log-safe. Without it the match is the only stage in the pipeline
                # with no visibility, which is exactly why a recording that keeps
                # 2.3 s out of 17.6 s cannot be explained from the outside: the count
                # says something was dropped, never what it scored or how long the
                # window was - and window length is what makes a short run's
                # embedding unreliable in the first place.
                log.info(
                    "match at %.1fs, %.1fs long: %.3f %s",
                    start, end - start, score, "keep" if score >= threshold else "drop",
                )
                if score >= threshold:
                    kept.append((start, end))

        log.info("matching runs: %d", len(kept))

        # The whole recording with the dead air removed: every speaker, not just the
        # one that matched. This is what "include everything" means - the kept audio
        # answers "what did I say", and this answers "what was said". It is also what
        # the transcript is read from, so the text and the audio sent back describe
        # the same thing, and so the model is never handed long stretches of silence
        # - which is its own source of hallucination, separate from the
        # forced-language one.
        #
        # Built first, and independently of the match, because the recording is the
        # deliverable. A run that found 422 s of speech and matched none of it used
        # to return before this point and send nothing at all - and silently, since
        # an automated upload sends no notice for an ordinary result. That is the
        # one case "include everything" must not lose.
        whole_seconds = render(samples, fuse(runs, total_s), all_wav)
        whole_ok = whole_seconds > 0 and encode(all_wav, all_out)
        if not whole_ok:
            log.warning("full audio could not be built")

        # The voice-matched extract. Absent when nothing matched, which is a result
        # rather than a fault: the full recording still goes out below.
        merged: list[tuple[float, float]] = []
        kept_seconds = 0.0
        if kept:
            merged = fuse(kept, total_s)
            kept_seconds = render(samples, merged, joined)
            if kept_seconds <= 0 or not encode(joined, output):
                log.warning("kept audio could not be built; sending the full audio only")
                kept_seconds = 0.0
        else:
            log.info("no run matched the reference voice")

        if not kept_seconds and not whole_ok:
            if kept:
                # A genuine fault: runs matched, and the audio still could not be
                # built. Reported for both kinds, because the alternative is a
                # failure nobody ever learns about.
                telegram_text(token, chat_id, "Processing failed.")
                return 1
            routine_notice(token, chat_id, kind, "No matching speech found.")
            return 0

        # Language mix. Non-fatal by design: the recording is the deliverable, and
        # a spent API quota must not cost the user their audio. Read from the kept
        # audio when there is one, because the question this line answers is "what
        # languages did I speak"; when nothing matched there is no kept audio, so it
        # reads the full recording rather than reporting nothing.
        census = joined if kept_seconds > 0 else all_wav
        mix_line, transcript = "", ""
        if transcribe_mode() == "on" and census.exists():
            try:
                mix_line, transcript = language_report(read_wav(census))
            except Exception as error:
                log.warning("language mix failed: %s", describe(error))

        if kept_seconds > 0:
            caption = (
                f"original {original:.1f}s | kept {kept_seconds:.1f}s | "
                f"{len(merged)} segment(s) | threshold {threshold:.2f}{mix_line}"
            )
        else:
            # No "kept" figure: quoting 0.0 s beside a threshold reads as a failure,
            # and on a recording where nobody matched the reference it is not one.
            caption = (
                f"original {original:.1f}s | no voice match | "
                f"{whole_seconds:.1f}s of speech | threshold {threshold:.2f}{mix_line}"
            )

        if kind == "document":
            # An automated upload. The cleaned audio goes back as well as the
            # transcript: the runner is thrown away at the end of the job, so the
            # cleaned file would otherwise exist for a few minutes and then never
            # again, and Telegram is the only durable copy kept. Sent first and
            # without a caption, so the file stands on its own suffixed name and
            # the caption below reads as its summary.
            if kept_seconds > 0 and not send_document(
                token, chat_id, output, "", cleaned_name(file_name)
            ):
                telegram_text(token, chat_id, "Could not send the cleaned audio.")
            if whole_ok and not send_document(token, chat_id, all_out, "", full_name(file_name)):
                telegram_text(token, chat_id, "Could not send the full audio.")

            # The full transcript, over the whole recording rather than over the
            # kept audio. The sampled pass above is a language census, not a
            # transcript: it reads at most MAX_CHUNKS chunks, so on a ten minute
            # file it could only ever describe a fraction of what was said. This
            # is the part that answers "what did they actually say".
            whole = ""
            if transcribe_mode() == "on" and whole_ok:
                try:
                    whole = full_transcript(groq_keys(), all_out)
                except Exception as error:
                    log.warning("full transcript failed: %s", describe(error))
                # A character count carries no speech content, so this is log-safe.
                # Without it there is no way to tell a transcript that worked from
                # one that never ran - both leave the log identical, and the run at
                # 02:22 could not be checked either way for exactly that reason.
                log.info("full transcript: %d char(s) over %.1f s of speech",
                         len(whole), whole_seconds)

            if whole:
                # A document rather than a message: Telegram caps a message at
                # 4096 characters, which a ten minute transcript passes easily,
                # and the cap was silently cutting the tail off every one.
                #
                # The minute map leads, because it orients the reader before the
                # text: where the recording was loud, and which parts of it were
                # actually the voice this pipeline is looking for. Empty when the
                # recording had no runs, hence the guard rather than a stray blank.
                profile = minute_map(runs, merged, total_s)
                text_out.write_text(
                    f"{profile}\n{whole}\n\n" if profile else f"{whole}\n\n"
                    f"---\n"
                    f"sampled chunks, with the language each was read as. The full\n"
                    f"transcript above carries no per-line label.\n\n"
                    f"{transcript}\n",
                    encoding="utf-8",
                )
                if not send_document(token, chat_id, text_out, caption,
                                    transcript_name(file_name), "text/plain"):
                    telegram_text(token, chat_id, "Could not send the transcript.")
            else:
                # No full transcript - a spent quota, or the call failed. The
                # caption still goes out with whatever the sampled pass produced,
                # because a file that was processed must never be silent: silence
                # is indistinguishable from a file that was dropped.
                body = transcript[:TRANSCRIPT_LIMIT]
                telegram_text(token, chat_id, f"{caption}\n\n{body}" if body else caption)
        else:
            # A hand-sent note: someone is waiting on it, so the matched audio goes
            # back when there is one. When nothing matched, the full recording still
            # goes back, because the caption on it says "no voice match" - and an
            # empty reply is the one outcome that cannot be told apart from a broken
            # pipeline.
            if kept_seconds > 0:
                if not send_document(token, chat_id, output, caption):
                    telegram_text(token, chat_id, "Could not send the result.")
                    return 1
            elif whole_ok:
                if not send_document(token, chat_id, all_out, caption, full_name(file_name)):
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
                    if "th" not in {code for _, _, code, _, _ in known}:
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
    # "document" means an automated upload (the watch): transcript only, no audio
    # sent back. Anything else is a hand-sent note and gets the audio too.
    kind = str(dispatch.get("kind") or "voice").strip().lower()
    # Only ever used to name the outgoing file. It is supplied by whoever uploaded
    # the recording, so it is never logged - see cleaned_name().
    file_name = str(dispatch.get("file_name") or "")
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
        return process(token, chat_id, file_id, reference, threshold, kind, file_name)
    except Exception as error:
        log.warning("processing failed: %s", describe(error))
        telegram_text(token, chat_id, "Processing failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
