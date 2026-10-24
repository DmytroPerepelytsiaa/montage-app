from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dima_auto_editor.ffmpeg_tools import FFmpegBinaries, locate_binaries, run_capture, write_wav  # noqa: E402

SAMPLE_RATE = 48000


@pytest.fixture(scope="session")
def ffmpeg_bins() -> FFmpegBinaries:
    return locate_binaries()


def speech_like(seconds: float, sr: int = SAMPLE_RATE, amplitude: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    carrier = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t) + 0.1 * rng.standard_normal(t.size)
    syllables = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 4 * t))
    return (amplitude * carrier * syllables / 1.6).astype(np.float32)


def silence(seconds: float, sr: int = SAMPLE_RATE, noise_db: float = -70.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (10 ** (noise_db / 20) * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def build_pattern(pattern: list[tuple[str, float]], sr: int = SAMPLE_RATE) -> np.ndarray:
    parts = [speech_like(sec, sr, seed=i) if kind == "speech" else silence(sec, sr, seed=i) for i, (kind, sec) in enumerate(pattern)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


@pytest.fixture
def make_speech_like_wav():
    def _make(path: Path, pattern: list[tuple[str, float]], sr: int = SAMPLE_RATE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(path, build_pattern(pattern, sr), sr)
        return path

    return _make


@pytest.fixture
def make_tone_wav():
    def _make(path: Path, seconds: float = 0.5, freq: float = 880.0, amplitude: float = 0.8, sr: int = SAMPLE_RATE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(seconds * sr)) / sr
        write_wav(path, (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
        return path

    return _make


@pytest.fixture
def make_video(ffmpeg_bins):
    def _make(path: Path, seconds: float = 2.0, fps: float = 12, size: tuple[int, int] = (320, 568), source: str = "testsrc2") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        result = run_capture(
            ["-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i", f"{source}=s={width}x{height}:r={fps}:d={seconds}",
             "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", str(path)],
            binaries=ffmpeg_bins,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        return path

    return _make


@pytest.fixture
def make_image(ffmpeg_bins):
    def _make(path: Path, size: tuple[int, int] = (256, 448), color: str = "teal") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        result = run_capture(
            ["-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=s={width}x{height}:d=1",
             "-frames:v", "1", str(path)],
            binaries=ffmpeg_bins,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        return path

    return _make
