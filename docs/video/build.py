# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "soundfile"]
# ///
"""Build the Anometa explainer video from `narration.md` and `scenes.py`.

Steps, in order (each writes into `docs/video/out/`):

1. ``tts``: one WAV per narration block in ``audio/<scene>.wav``, padded with a
   short lead-in and tail and loudness-matched, plus ``audio/durations.json``.
2. ``music``: a quiet ambient pad synthesised with numpy (``music.wav``), about
   24 dB under the voice, with a fade in and out.
3. ``render``: ``manim -qh --frame_rate 30`` (1080p30) on ``scenes.py``; each segment lasts
   exactly as long as its WAV and carries it as its soundtrack. Real images come
   from ``data/ad2/`` (``uv run anometa download``) and ``assets/``.
4. ``mux``: video + narration + music + soft subtitles (``anometa_explainer.srt``,
   ``mov_text``, never burned in) into ``anometa_explainer.mp4``.

Voices (``--voice``):

- ``breeze-male`` (default), ``breeze-female``: Breeze TTS 2 voice design (text
  description only, fixed seed), run over SSH on a GPU host named by
  ``ANOMETA_TTS_HOST`` and ``ANOMETA_TTS_DIR``. Weights and
  outputs: BreezeBlue Research and Non-Commercial License.
- ``kokoro-af_heart``, ``kokoro-am_michael``: Kokoro-82M (Apache-2.0), local.

Usage (from the repository root; needs ffmpeg and the Homebrew ``manim``)::

    uv run docs/video/build.py                                 # every step
    uv run docs/video/build.py --voice kokoro-af_heart         # switch voice, rebuild
    uv run docs/video/build.py --steps render,mux              # reuse audio and music
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
AUDIO = OUT / "audio"
SR = 24_000
LEAD_S, TAIL_S = 0.35, 0.85
VOICE_DBFS = -20.0
MUSIC_REL_DB = -24.0

# Breeze voices run over SSH on a GPU host: ANOMETA_TTS_HOST (e.g. user@host) and
# ANOMETA_TTS_DIR (folder holding breeze-tts/ with its venv and the breeze-tts-2 weights).
HOST = os.environ.get("ANOMETA_TTS_HOST", "")
HOST_TTS = os.environ.get("ANOMETA_TTS_DIR", "")
BREEZE_STYLE = (
    "in their thirties with a neutral American accent, speaking at a steady, unhurried "
    "documentary pace with precise articulation."
)
VOICES = {
    "breeze-male": f"A calm, clear male narrator {BREEZE_STYLE}",
    "breeze-female": f"A calm, clear female narrator {BREEZE_STYLE}",
    "kokoro-af_heart": "af_heart",
    "kokoro-am_michael": "am_michael",
}
# TTS-only respellings; narration.md keeps the written form.
RESPELL: dict[str, str] = {}

KOKORO_SCRIPT = """
import json, sys
import numpy as np, soundfile as sf
from kokoro import KPipeline
job = json.load(sys.stdin)
pipe = KPipeline(lang_code="a")
for name, text in job["blocks"].items():
    audio = np.concatenate([a for _, _, a in pipe(text, voice=job["voice"], speed=1.0)])
    sf.write(f"{job['out']}/{name}.raw.wav", audio, 24000)
"""


def blocks() -> dict[str, str]:
    """Return the narration blocks of ``narration.md`` as ``{scene: text}``, in order."""
    md = (HERE / "narration.md").read_text()
    parts = re.split(r"^## (\w+)\n", md, flags=re.M)[1:]
    return {
        name: " ".join(body.split()) for name, body in zip(parts[::2], parts[1::2], strict=True)
    }


def run(cmd: list[str], **kw: object) -> None:
    """Run a command, echoing it, and fail loudly."""
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)  # type: ignore[arg-type]


def tts_breeze(texts: dict[str, str], instruction: str) -> None:
    """Synthesise each block on the 3090 host with Breeze TTS 2 voice design."""
    if not HOST or not HOST_TTS:
        sys.exit("Breeze voices need ANOMETA_TTS_HOST and ANOMETA_TTS_DIR; or use a kokoro voice")
    remote = f"{HOST_TTS}/video"
    run(["ssh", "-n", HOST, f"mkdir -p {remote}"])
    for name, text in texts.items():
        infer = (
            f"cd {HOST_TTS}/breeze-tts && .venv/bin/python infer.py ../breeze-tts-2 --seed 42 "
            f"--cfg-scale 4 --instruction {shlex.quote(instruction)} --text {shlex.quote(text)} "
            f"--output {remote}/{name}.wav > {remote}/{name}.log 2>&1"
        )
        run(["ssh", "-n", HOST, infer])
        run(["scp", "-q", f"{HOST}:{remote}/{name}.wav", str(AUDIO / f"{name}.raw.wav")])


def tts_kokoro(texts: dict[str, str], voice: str) -> None:
    """Synthesise each block locally with Kokoro-82M."""
    deps = ["kokoro>=0.9", "transformers>=4.45", "misaki[en]>=0.9", "soundfile"]
    cmd = [
        "uv",
        "run",
        "--python",
        "3.12",
        *(a for d in deps for a in ("--with", d)),
        "python",
        "-c",
        KOKORO_SCRIPT,
    ]
    job = json.dumps({"voice": voice, "out": str(AUDIO), "blocks": texts})
    run(cmd, input=job, text=True)


def tts(voice: str) -> None:
    """Write padded, loudness-matched ``<scene>.wav`` files and ``durations.json``."""
    AUDIO.mkdir(parents=True, exist_ok=True)
    texts = blocks()
    for a, b in RESPELL.items():
        texts = {n: t.replace(a, b) for n, t in texts.items()}
    if voice.startswith("breeze"):
        tts_breeze(texts, VOICES[voice])
    else:
        tts_kokoro(texts, VOICES[voice])
    durations = {}
    for name in texts:
        raw = AUDIO / f"{name}.raw.wav"
        audio, sr = sf.read(raw, dtype="float32")
        assert sr == SR, f"{raw}: expected {SR} Hz, got {sr}"
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        voiced = np.flatnonzero(np.abs(audio) >= 1e-3)
        audio = audio[voiced[0] : voiced[-1] + 1]
        rms = float(np.sqrt(np.mean(audio**2)))
        audio = audio * (10 ** (VOICE_DBFS / 20) / max(rms, 1e-9))
        audio = np.clip(audio, -0.98, 0.98)
        audio = np.concatenate([np.zeros(int(LEAD_S * SR)), audio, np.zeros(int(TAIL_S * SR))])
        sf.write(AUDIO / f"{name}.wav", audio, SR, subtype="PCM_16")
        raw.unlink()
        durations[name] = round(len(audio) / SR, 3)
    (AUDIO / "durations.json").write_text(
        json.dumps({"voice": voice, "durations": durations}, indent=1) + "\n"
    )
    print(json.dumps(durations, indent=1), f"total {sum(durations.values()):.1f} s")


def music() -> None:
    """Synthesise a slow four-chord ambient pad as long as the narration."""
    durations = json.loads((AUDIO / "durations.json").read_text())["durations"]
    total = sum(durations.values())
    sr = 48_000
    t = np.arange(int(total * sr)) / sr
    # A minor, F, C, G; root position plus fifth, an octave apart for warmth.
    chords = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 55, 60, 64], [55, 59, 62, 67]]
    chord_s = 8.0
    pad = np.zeros_like(t)
    for i, notes in enumerate(chords * int(np.ceil(total / chord_s / len(chords)) + 1)):
        start = i * chord_s
        if start >= total:
            break
        env = np.clip(1 - np.abs((t - start - chord_s / 2) / (chord_s * 0.75)), 0, 1) ** 1.5
        for midi in notes:
            f = 440.0 * 2 ** ((midi - 69 - 12) / 12)
            for detune in (-0.15, 0.15):
                ff = f + detune
                pad += env * (np.sin(2 * np.pi * ff * t) + 0.25 * np.sin(4 * np.pi * ff * t))
    pad *= 1 + 0.15 * np.sin(2 * np.pi * 0.07 * t)
    fade = np.minimum(1, np.minimum(t / 3.0, (total - t) / 4.0))
    pad *= np.clip(fade, 0, 1)
    rms = float(np.sqrt(np.mean(pad**2)))
    pad *= 10 ** ((VOICE_DBFS + MUSIC_REL_DB) / 20) / rms
    sf.write(OUT / "music.wav", pad.astype(np.float32), sr, subtype="PCM_16")


def srt() -> None:
    """Write captions: each block split into lines of at most 80 characters, timed by words."""
    durations = json.loads((AUDIO / "durations.json").read_text())["durations"]

    def stamp(s: float) -> str:
        ms = round(s * 1000)
        return (
            f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        )

    cues: list[tuple[float, float, str]] = []
    start = 0.0
    for name, text in blocks().items():
        speech = durations[name] - LEAD_S - TAIL_S
        chunks = [c for s in re.split(r"(?<=[.?!])\s+", text) for c in textwrap.wrap(s, 80)]
        words = [len(c.split()) for c in chunks]
        t = start + LEAD_S
        for chunk, n in zip(chunks, words, strict=True):
            d = speech * n / sum(words)
            cues.append((t, t + d, "\n".join(textwrap.wrap(chunk, 42))))
            t += d
        start += durations[name]
    body = "\n".join(
        f"{i}\n{stamp(a)} --> {stamp(b)}\n{c}\n" for i, (a, b, c) in enumerate(cues, 1)
    )
    (OUT / "anometa_explainer.srt").write_text(body)


def render() -> None:
    """Render ``scenes.py`` at 1080p30 with the Homebrew manim.

    Caching stays off: cached animations advance the scene clock by their nominal
    length instead of by written frames, which would let audio drift.
    """
    run(
        [
            "manim",
            "-qh",
            "--frame_rate",
            "30",
            "--disable_caching",
            "--media_dir",
            str(OUT / "media"),
            str(HERE / "scenes.py"),
            "Explainer",
        ]
    )


def mux() -> None:
    """Mix the music under the rendered narration and add the soft subtitle track."""
    srt()
    video = OUT / "media" / "videos" / "scenes" / "1080p30" / "Explainer.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-i",
            str(OUT / "music.wav"),
            "-i",
            str(OUT / "anometa_explainer.srt"),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-map",
            "2",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            "-movflags",
            "+faststart",
            str(OUT / "anometa_explainer.mp4"),
        ]
    )


def main() -> None:
    """Parse arguments and run the selected steps."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--voice", choices=VOICES, default="breeze-male")
    parser.add_argument("--steps", default="tts,music,render,mux")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    steps = {"tts": lambda: tts(args.voice), "music": music, "render": render, "mux": mux}
    for step in args.steps.split(","):
        if step not in steps:
            sys.exit(f"unknown step {step!r}; choose from {', '.join(steps)}")
        steps[step]()


if __name__ == "__main__":
    main()
