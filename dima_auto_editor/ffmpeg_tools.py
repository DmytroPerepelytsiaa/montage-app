from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .models import MediaInfo, MediaKind
from .naming import IMAGE_EXTENSIONS, AUDIO_EXTENSIONS

ProgressCallback = Callable[[float], None]
LogCallback = Callable[[str], None]

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FFmpegNotFound(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    def __init__(self, message: str, stderr_tail: str = ""):
        super().__init__(f"{message}\n{stderr_tail}".strip())
        self.stderr_tail = stderr_tail


class RenderCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class FFmpegBinaries:
    ffmpeg: str
    ffprobe: str | None


def _candidate_dirs() -> Iterable[Path]:
    env_dir = os.environ.get("DIMA_FFMPEG_DIR")
    if env_dir:
        yield Path(env_dir)
    yield _PROJECT_ROOT / "tools" / "ffmpeg" / "bin"
    yield _PROJECT_ROOT / "tools" / "ffmpeg"


def _find_in_dirs(name: str) -> str | None:
    exe = f"{name}.exe" if sys.platform == "win32" else name
    for directory in _candidate_dirs():
        candidate = directory / exe
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


@lru_cache(maxsize=8)
def locate_binaries(ffmpeg_override: str = "", ffprobe_override: str = "") -> FFmpegBinaries:
    ffmpeg = ffmpeg_override if ffmpeg_override and Path(ffmpeg_override).is_file() else _find_in_dirs("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if not ffmpeg:
        raise FFmpegNotFound(
            "FFmpeg was not found. Install it with 'winget install Gyan.FFmpeg' "
            "or run 'pip install imageio-ffmpeg', then restart the application."
        )
    ffprobe = ffprobe_override if ffprobe_override and Path(ffprobe_override).is_file() else None
    if not ffprobe:
        sibling = Path(ffmpeg).with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        ffprobe = str(sibling) if sibling.is_file() else _find_in_dirs("ffprobe")
    return FFmpegBinaries(ffmpeg=ffmpeg, ffprobe=ffprobe)


def run_ffmpeg(
    args: list[str],
    *,
    binaries: FFmpegBinaries | None = None,
    duration: float | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
) -> None:
    binaries = binaries or locate_binaries()
    command = [binaries.ffmpeg, "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", *args]
    if log:
        log("ffmpeg " + subprocess.list2cmdline(command[1:]))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr_tail: deque[str] = deque(maxlen=60)
    drain = threading.Thread(target=lambda: stderr_tail.extend(process.stderr), daemon=True)  # type: ignore[arg-type]
    drain.start()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if cancel is not None and cancel.is_set():
                process.kill()
                raise RenderCancelled("Rendering was cancelled by the user.")
            if progress and duration and line.startswith("out_time_us="):
                value = line.split("=", 1)[1].strip()
                if value.isdigit():
                    progress(min(1.0, int(value) / 1_000_000 / duration))
        process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        drain.join(timeout=5)
    if cancel is not None and cancel.is_set():
        raise RenderCancelled("Rendering was cancelled by the user.")
    if process.returncode != 0:
        tail = "".join(stderr_tail)
        if log:
            log(tail)
        raise FFmpegError(f"FFmpeg failed with exit code {process.returncode}.", tail)
    if progress:
        progress(1.0)


def run_capture(args: list[str], *, binaries: FFmpegBinaries | None = None, executable: str = "ffmpeg") -> subprocess.CompletedProcess[bytes]:
    binaries = binaries or locate_binaries()
    exe = binaries.ffprobe if executable == "ffprobe" else binaries.ffmpeg
    if not exe:
        raise FFmpegNotFound(f"{executable} is not available.")
    return subprocess.run([exe, *args], capture_output=True, stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)


def media_kind_for(path: str | Path) -> MediaKind:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return MediaKind.IMAGE
    if suffix in AUDIO_EXTENSIONS:
        return MediaKind.AUDIO
    return MediaKind.VIDEO


def probe(path: str | Path, binaries: FFmpegBinaries | None = None) -> MediaInfo:
    binaries = binaries or locate_binaries()
    kind = media_kind_for(path)
    if binaries.ffprobe:
        info = _probe_with_ffprobe(str(path), kind, binaries)
    else:
        info = _probe_with_ffmpeg(str(path), kind, binaries)
    if kind is MediaKind.IMAGE:
        info.duration = None
        info.fps = 0.0
        info.frame_count = 1
    elif info.frame_count <= 0 and info.duration and info.fps:
        info.frame_count = int(round(info.duration * info.fps))
    return info


def _parse_rate(rate: str) -> float:
    if not rate or rate in ("0/0", "N/A"):
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)


def _probe_with_ffprobe(path: str, kind: MediaKind, binaries: FFmpegBinaries) -> MediaInfo:
    result = run_capture(
        ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        binaries=binaries,
        executable="ffprobe",
    )
    if result.returncode != 0:
        raise FFmpegError(f"Cannot read media file: {path}", result.stderr.decode("utf-8", "replace"))
    data = json.loads(result.stdout.decode("utf-8", "replace") or "{}")
    info = MediaInfo(path=path, kind=kind)
    duration = data.get("format", {}).get("duration")
    info.duration = float(duration) if duration not in (None, "N/A") else None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not info.has_video:
            info.has_video = True
            info.width = int(stream.get("width") or 0)
            info.height = int(stream.get("height") or 0)
            info.fps = _parse_rate(stream.get("avg_frame_rate", "")) or _parse_rate(stream.get("r_frame_rate", ""))
            frames = stream.get("nb_frames")
            info.frame_count = int(frames) if frames and str(frames).isdigit() else 0
        elif stream.get("codec_type") == "audio" and not info.has_audio:
            info.has_audio = True
            info.sample_rate = int(stream.get("sample_rate") or 0)
            info.channels = int(stream.get("channels") or 0)
    return info


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #\S+.*?Video:.*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_TBR_RE = re.compile(r"(\d+(?:\.\d+)?k?)\s*tbr")
_AUDIO_RE = re.compile(r"Stream #\S+.*?Audio:.*?(\d+)\s*Hz,\s*([^,]+)")


def _probe_with_ffmpeg(path: str, kind: MediaKind, binaries: FFmpegBinaries) -> MediaInfo:
    result = run_capture(["-hide_banner", "-i", path], binaries=binaries)
    text = result.stderr.decode("utf-8", "replace")
    if "Invalid data found" in text or "No such file" in text or ("Stream #" not in text):
        raise FFmpegError(f"Cannot read media file: {path}", text[-2000:])
    info = MediaInfo(path=path, kind=kind)
    match = _DURATION_RE.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        info.duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    for line in text.splitlines():
        if "Video:" in line and not info.has_video:
            video = _VIDEO_RE.search(line)
            if video:
                info.has_video = True
                info.width, info.height = int(video.group(1)), int(video.group(2))
            fps = _FPS_RE.search(line)
            tbr = _TBR_RE.search(line)
            if fps:
                info.fps = float(fps.group(1))
            elif tbr and not tbr.group(1).endswith("k"):
                info.fps = float(tbr.group(1))
        elif "Audio:" in line and not info.has_audio:
            audio = _AUDIO_RE.search(line)
            info.has_audio = True
            if audio:
                info.sample_rate = int(audio.group(1))
                layout = audio.group(2).strip()
                info.channels = 1 if layout == "mono" else 2 if layout == "stereo" else int(re.sub(r"\D", "", layout) or 2)
    return info


def decode_audio(
    path: str | Path,
    sample_rate: int = 48000,
    channels: int = 1,
    binaries: FFmpegBinaries | None = None,
) -> np.ndarray:
    result = run_capture(
        ["-v", "error", "-i", str(path), "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
        binaries=binaries,
    )
    if result.returncode != 0:
        raise FFmpegError(f"Cannot decode audio: {path}", result.stderr.decode("utf-8", "replace")[-2000:])
    samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    import wave

    data = np.asarray(samples, dtype=np.float32)
    channels = 1 if data.ndim == 1 else data.shape[1]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


_ENCODER_ORDER = ("h264_nvenc", "h264_qsv", "h264_amf", "libx264")


@lru_cache(maxsize=4)
def working_encoders(binaries: FFmpegBinaries | None = None) -> tuple[str, ...]:
    binaries = binaries or locate_binaries()
    listed = run_capture(["-hide_banner", "-encoders"], binaries=binaries).stdout.decode("utf-8", "replace")
    usable: list[str] = []
    for encoder in _ENCODER_ORDER:
        if f" {encoder} " not in listed:
            continue
        test = run_capture(
            ["-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=60:d=0.1",
             "-pix_fmt", "yuv420p", "-c:v", encoder, "-f", "null", "-"],
            binaries=binaries,
        )
        if test.returncode == 0:
            usable.append(encoder)
    return tuple(usable)


def pick_encoder(preference: str = "auto", binaries: FFmpegBinaries | None = None) -> str:
    usable = working_encoders(binaries)
    if preference != "auto" and preference in usable:
        return preference
    return usable[0] if usable else "libx264"


def video_encoder_args(encoder: str, quality: int, preset: str) -> list[str]:
    if encoder == "h264_nvenc":
        nvenc_preset = preset if preset.startswith("p") and preset[1:].isdigit() else "p5"
        return ["-c:v", encoder, "-preset", nvenc_preset, "-rc", "vbr", "-cq", str(quality), "-b:v", "0", "-profile:v", "high"]
    if encoder == "h264_qsv":
        return ["-c:v", encoder, "-global_quality", str(quality), "-preset", "medium"]
    if encoder == "h264_amf":
        return ["-c:v", encoder, "-rc", "cqp", "-qp_i", str(quality), "-qp_p", str(quality), "-quality", "quality"]
    x264_preset = preset if not preset.startswith("p") or preset in ("placebo",) else "medium"
    return ["-c:v", "libx264", "-preset", x264_preset, "-crf", str(quality), "-profile:v", "high"]
