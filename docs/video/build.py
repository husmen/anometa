# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "soundfile"]
# ///
"""Build the Anometa explainer video from `narration.md` and `scenes.py`.

Steps, in order (each writes into `docs/video/out/`):

1. ``tts``: for each voice, one WAV per narration block in ``audio/<voice>/<scene>.wav``,
   padded with a short lead-in and tail and loudness-matched, plus ``durations.json``.
   A voice whose folder already holds every block is reused.
2. ``align``: each scene lasts as long as its longest voice; shorter clips get trailing
   silence. Writes the first voice's clips to ``audio/<scene>.wav`` (the render's
   soundtrack), ``audio/durations.json``, and one ``narration_<voice>.wav`` per voice.
3. ``music``: a quiet ambient pad synthesised with numpy (``music.wav``), about
   24 dB under the voice, with a fade in and out.
4. ``render``: ``manim -qh --frame_rate 30`` (1080p30) on ``scenes.py``; each segment lasts
   exactly as long as its aligned WAV. Real images come from ``data/ad2/``
   (``uv run anometa download``) and the report's ``docs/report/data/``.
5. ``mux``: video + one audio track per voice (narration + music) + one soft subtitle
   track per voice (``mov_text``, never burned in) into ``anometa_explainer.mp4``.
   The first voice is the default track; players such as VLC or QuickTime switch tracks.

Voices (``--voices``, comma-separated, first is the default track):

- ``kokoro-am_michael``, ``kokoro-af_heart``: Kokoro-82M (Apache-2.0), local.
  One fixed speaker embedding, so every block sounds the same.
- ``breeze-male``, ``breeze-female``: Breeze TTS 2 voice design (text description
  only, fixed seed; the timbre still varies between blocks). Weights and outputs:
  BreezeBlue Research and Non-Commercial License. Run over SSH on a GPU host.
- ``breeze`` (the published narration): ``breeze-direct``, with the scenes that say
  "Anometa" taken from ``breeze-clone`` (``MIXES``).
- ``breeze-clone``, ``breeze-direct``: Breeze TTS 2 anchored on one ``breeze-male`` clip
  (``ANCHOR``) with Voice Clone, or Voice Direction (clone plus a delivery instruction),
  so every block keeps the anchor's timbre. Needs the ``breeze-male`` clips first.

Remote voices need ``ANOMETA_TTS_HOST`` (e.g. user@host) and ``ANOMETA_TTS_DIR`` (folder
holding ``breeze-tts/`` with its venv and the ``breeze-tts-2`` weights).

Usage (from the repository root; needs ffmpeg and the Homebrew ``manim``)::

    uv run docs/video/build.py                                          # every step
    uv run docs/video/build.py --voices kokoro-am_michael,breeze-male   # two tracks
    uv run docs/video/build.py --steps align,music,render,mux           # reuse voices
"""

import argparse
import json
import os
import re
import shlex
import shutil
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
    "breeze-male": f"A calm, clear, engaging male narrator {BREEZE_STYLE}",
    "breeze-female": f"A calm, clear, engaging female narrator {BREEZE_STYLE}",
    # Anchored on one breeze-male clip (see ANCHOR): clone copies its voice as is; direct
    # also steers the delivery with this instruction.
    "breeze-clone": "",
    "breeze-direct": "Calm, clear documentary narration at a steady, unhurried pace with "
    "precise articulation.",
    "kokoro-af_heart": "af_heart",
    "kokoro-am_michael": "am_michael",
}
# Reference clip for the anchored Breeze voices: the first two sentences of the breeze-male
# "maps" clip (seconds within its padded WAV, cut in a pause; checked with Whisper).
ANCHOR = {"voice": "breeze-male", "scene": "maps", "span": (0.30, 12.36), "sentences": 2}
# Final narration: breeze-direct, except the scenes that say "Anometa", which come from
# breeze-clone (same anchor and timbre; preferred pronunciation of the name).
MIXES = {
    "breeze": {"default": "breeze-direct", "question": "breeze-clone", "takeaways": "breeze-clone"}
}
TRACK_TITLES = {
    "breeze-male": "Breeze TTS 2 (male)",
    "breeze-female": "Breeze TTS 2 (female)",
    "breeze-clone": "Breeze TTS 2 (anchored clone)",
    "breeze-direct": "Breeze TTS 2 (anchored, directed)",
    "breeze": "Breeze TTS 2",
    "kokoro-af_heart": "Kokoro (af_heart)",
    "kokoro-am_michael": "Kokoro (am_michael)",
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


def remote_ready() -> None:
    """Exit unless a remote TTS host is configured."""
    if not HOST or not HOST_TTS:
        sys.exit("remote voices need ANOMETA_TTS_HOST and ANOMETA_TTS_DIR; or use a kokoro voice")


def anchor() -> tuple[Path, str]:
    """Cut the reference clip for the anchored Breeze voices; return it and its transcript."""
    clip, _ = sf.read(AUDIO / ANCHOR["voice"] / f"{ANCHOR['scene']}.wav", dtype="float32")
    a, b = ANCHOR["span"]
    path = OUT / "anchor" / f"{ANCHOR['scene']}_anchor.wav"
    path.parent.mkdir(exist_ok=True)
    sf.write(path, clip[int(a * SR) : int(b * SR)], SR, subtype="PCM_16")
    sentences = re.split(r"(?<=[.?!])\s+", blocks()[ANCHOR["scene"]])
    return path, " ".join(sentences[: ANCHOR["sentences"]])


def tts_breeze(texts: dict[str, str], voice: str, out: Path) -> None:
    """Synthesise each block on the GPU host with Breeze TTS 2.

    ``breeze-male``/``breeze-female`` use voice design (instruction only). ``breeze-clone``
    uses Voice Clone on the anchor clip; ``breeze-direct`` adds the instruction (Voice
    Direction). Anchoring keeps one timbre across blocks.
    """
    remote_ready()
    remote = f"{HOST_TTS}/video/{voice}"
    run(["ssh", "-n", HOST, f"mkdir -p {remote}"])
    instruction = VOICES[voice]
    # Classifier-free guidance strengthens instructions; the clone template has none.
    args = f"--cfg-scale 4 --instruction {shlex.quote(instruction)} " if instruction else ""
    if voice in ("breeze-clone", "breeze-direct"):
        ref, ref_text = anchor()
        run(["scp", "-q", str(ref), f"{HOST}:{remote}/anchor.wav"])
        args += f"--ref-audio {remote}/anchor.wav --ref-text {shlex.quote(ref_text)} "
    for name, text in texts.items():
        infer = (
            f"cd {HOST_TTS}/breeze-tts && .venv/bin/python infer.py ../breeze-tts-2 --seed 42 "
            f"{args}--text {shlex.quote(text)} "
            f"--output {remote}/{name}.wav > {remote}/{name}.log 2>&1"
        )
        run(["ssh", "-n", HOST, infer])
        run(["scp", "-q", f"{HOST}:{remote}/{name}.wav", str(out / f"{name}.raw.wav")])


def tts_kokoro(texts: dict[str, str], voice: str, out: Path) -> None:
    """Synthesise each block locally with Kokoro-82M."""
    deps = ["kokoro>=0.9", "transformers>=4.45", "misaki[en]>=0.9", "soundfile"]
    cmd = [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.12",
        *(a for d in deps for a in ("--with", d)),
        "python",
        "-c",
        KOKORO_SCRIPT,
    ]
    job = json.dumps({"voice": voice, "out": str(out), "blocks": texts})
    run(cmd, input=job, text=True)


def mix(voice: str, texts: dict[str, str], out: Path) -> None:
    """Assemble a voice from other voices' clips, scene by scene (see ``MIXES``)."""
    sources = MIXES[voice]
    durations, done = {}, {}
    for name, text in texts.items():
        src = sources.get(name, sources["default"])
        tts(src)
        meta = json.loads((AUDIO / src / "durations.json").read_text())
        shutil.copyfile(AUDIO / src / f"{name}.wav", out / f"{name}.wav")
        durations[name], done[name] = meta["durations"][name], text
    (out / "durations.json").write_text(
        json.dumps({"voice": voice, "durations": durations, "texts": done}, indent=1) + "\n"
    )


def tts(voice: str) -> None:
    """Write padded, loudness-matched ``audio/<voice>/<scene>.wav`` and ``durations.json``.

    Only blocks that are missing or whose narration text changed are synthesised again.
    """
    out = AUDIO / voice
    out.mkdir(parents=True, exist_ok=True)
    texts = blocks()
    if voice in MIXES:
        mix(voice, texts, out)
        return
    meta_path = out / "durations.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    durations: dict[str, float] = meta.get("durations", {})
    done_texts: dict[str, str] = meta.get("texts", {})
    todo = {
        n: s for n, s in texts.items() if done_texts.get(n) != s or not (out / f"{n}.wav").exists()
    }
    if not todo:
        print(f"{voice}: reusing {out}")
        return
    spoken = dict(todo)
    for a, b in RESPELL.items():
        spoken = {n: s.replace(a, b) for n, s in spoken.items()}
    if voice.startswith("breeze"):
        tts_breeze(spoken, voice, out)
    else:
        tts_kokoro(spoken, VOICES[voice], out)
    for name in todo:
        raw = out / f"{name}.raw.wav"
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
        sf.write(out / f"{name}.wav", audio, SR, subtype="PCM_16")
        raw.unlink()
        durations[name] = round(len(audio) / SR, 3)
        done_texts[name] = texts[name]
    durations = {n: durations[n] for n in texts}
    meta_path.write_text(
        json.dumps({"voice": voice, "durations": durations, "texts": done_texts}, indent=1) + "\n"
    )
    print(voice, json.dumps(durations, indent=1), f"total {sum(durations.values()):.1f} s")


def align(voices: list[str]) -> None:
    """Give each scene its longest voice's length and pad every voice to it.

    ``audio/durations.json`` holds the aligned scene lengths and, per voice, the
    unpadded clip lengths (``speech``), which time the captions and the first
    voice's on-screen cues.
    """
    texts = blocks()
    per_voice = {
        v: json.loads((AUDIO / v / "durations.json").read_text())["durations"] for v in voices
    }
    clips = {
        v: {n: sf.read(AUDIO / v / f"{n}.wav", dtype="float32")[0] for n in texts} for v in voices
    }
    samples = {n: max(len(clips[v][n]) for v in voices) for n in texts}
    scene = {n: round(samples[n] / SR, 3) for n in texts}
    for v in voices:
        track = []
        for n in texts:
            clip = np.concatenate([clips[v][n], np.zeros(samples[n] - len(clips[v][n]))])
            track.append(clip)
            if v == voices[0]:
                sf.write(AUDIO / f"{n}.wav", clip, SR, subtype="PCM_16")
        sf.write(OUT / f"narration_{v}.wav", np.concatenate(track), SR, subtype="PCM_16")
    (AUDIO / "durations.json").write_text(
        json.dumps({"voices": voices, "durations": scene, "speech": per_voice}, indent=1) + "\n"
    )
    print(json.dumps(scene, indent=1), f"total {sum(scene.values()):.1f} s")


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


def srt(voice: str) -> Path:
    """Write one voice's captions: lines of at most 80 characters, timed by words."""
    meta = json.loads((AUDIO / "durations.json").read_text())
    durations, own = meta["durations"], meta["speech"][voice]

    def stamp(s: float) -> str:
        ms = round(s * 1000)
        return (
            f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        )

    cues: list[tuple[float, float, str]] = []
    start = 0.0
    for name, text in blocks().items():
        speech = own[name] - LEAD_S - TAIL_S
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
    path = OUT / f"anometa_explainer.{voice}.srt"
    path.write_text(body)
    return path


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


def mux(voices: list[str]) -> None:
    """Mix the music under each voice and add one audio and one subtitle track per voice."""
    subs = [srt(v) for v in voices]
    video = OUT / "media" / "videos" / "scenes" / "1080p30" / "Explainer.mp4"
    n = len(voices)
    inputs = ["-i", str(video), "-i", str(OUT / "music.wav")]
    for v in voices:
        inputs += ["-i", str(OUT / f"narration_{v}.wav")]
    for s in subs:
        inputs += ["-i", str(s)]
    music = "".join(f"[m{i}]" for i in range(n))
    graph = f"[1:a]asplit={n}{music};" if n > 1 else "[1:a]anull[m0];"
    graph += ";".join(
        f"[{2 + i}:a][m{i}]amix=inputs=2:duration=first:normalize=0[a{i}]" for i in range(n)
    )
    maps = ["-map", "0:v"]
    meta: list[str] = []
    for i, v in enumerate(voices):
        maps += ["-map", f"[a{i}]"]
        meta += [
            f"-metadata:s:a:{i}",
            f"title={TRACK_TITLES[v]}",
            f"-metadata:s:a:{i}",
            f"handler_name={TRACK_TITLES[v]}",
            f"-metadata:s:a:{i}",
            "language=eng",
            f"-disposition:a:{i}",
            "default" if i == 0 else "0",
        ]
    for i, v in enumerate(voices):
        maps += ["-map", f"{2 + n + i}"]
        meta += [
            f"-metadata:s:s:{i}",
            f"title=English ({TRACK_TITLES[v]} timing)",
            f"-metadata:s:s:{i}",
            f"handler_name=English ({TRACK_TITLES[v]} timing)",
            f"-metadata:s:s:{i}",
            "language=eng",
        ]
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *inputs,
            "-filter_complex",
            graph,
            *maps,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "mov_text",
            *meta,
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
    parser.add_argument("--voices", default="breeze")
    parser.add_argument("--steps", default="tts,align,music,render,mux")
    args = parser.parse_args()
    voices = args.voices.split(",")
    if unknown := [v for v in voices if v not in VOICES and v not in MIXES]:
        sys.exit(f"unknown voice(s) {unknown}; choose from {', '.join([*MIXES, *VOICES])}")
    OUT.mkdir(exist_ok=True)
    steps = {
        "tts": lambda: [tts(v) for v in voices],
        "align": lambda: align(voices),
        "music": music,
        "render": render,
        "mux": lambda: mux(voices),
    }
    for step in args.steps.split(","):
        if step not in steps:
            sys.exit(f"unknown step {step!r}; choose from {', '.join(steps)}")
        steps[step]()


if __name__ == "__main__":
    main()
