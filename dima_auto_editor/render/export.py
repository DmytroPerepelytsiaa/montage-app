from __future__ import annotations

import re
import threading
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..config import ProjectSettings
from ..ffmpeg_tools import (
    FFmpegBinaries,
    FFmpegError,
    LogCallback,
    ProgressCallback,
    locate_binaries,
    pick_encoder,
    probe,
    run_capture,
    run_ffmpeg,
    video_encoder_args,
)
from ..models import Issue, Severity

SOFTWARE_ENCODER = "libx264"
OUTPUT_SAMPLE_RATE = 48000
BT709_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
_FRAME_COUNT_RE = re.compile(r"frame=\s*(\d+)")
_SAFE_NAME_RE = re.compile(r"[^\w\-]+", re.UNICODE)


@dataclass
class ExportReport:
    output_path: str
    encoder: str
    frames: int
    expected_frames: int
    fps: float
    audio_sample_rate: int
    issues: list[Issue] = field(default_factory=list)


def binaries_for(settings: ProjectSettings) -> FFmpegBinaries:
    return locate_binaries(settings.export.ffmpeg_path, settings.export.ffprobe_path)


def preferred_encoder(settings: ProjectSettings) -> str:
    if settings.export.encoder == SOFTWARE_ENCODER:
        return SOFTWARE_ENCODER
    return pick_encoder(settings.export.encoder, binaries_for(settings))


def intermediate_video_args(encoder: str, preview: bool) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p1" if preview else "p4", "-rc", "constqp",
                "-qp", "24" if preview else "16", "-pix_fmt", "yuv420p", *BT709_TAGS]
    return ["-c:v", SOFTWARE_ENCODER, "-preset", "ultrafast" if preview else "veryfast",
            "-crf", "20" if preview else "10", "-pix_fmt", "yuv420p", *BT709_TAGS]


def encode_with_fallback(
    build_args: Callable[[str], list[str]],
    encoder: str,
    *,
    binaries: FFmpegBinaries,
    duration: float | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
) -> str:
    try:
        run_ffmpeg(build_args(encoder), binaries=binaries, duration=duration, progress=progress, cancel=cancel, log=log)
        return encoder
    except FFmpegError:
        if encoder == SOFTWARE_ENCODER:
            raise
        if log:
            log(f"Hardware encoder {encoder} failed; retrying with {SOFTWARE_ENCODER}.")
        run_ffmpeg(build_args(SOFTWARE_ENCODER), binaries=binaries, duration=duration, progress=progress, cancel=cancel, log=log)
        return SOFTWARE_ENCODER


def count_video_frames(path: str | Path, binaries: FFmpegBinaries | None = None) -> int:
    result = run_capture(["-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"], binaries=binaries)
    if result.returncode != 0:
        raise FFmpegError(f"Cannot count frames of {path}", result.stderr.decode("utf-8", "replace")[-2000:])
    matches = _FRAME_COUNT_RE.findall(result.stderr.decode("utf-8", "replace"))
    return int(matches[-1]) if matches else 0


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def safe_project_name(name: str, fallback: str = "video") -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    return cleaned or fallback


def unique_output_path(output_dir: str | Path, project_name: str, *, preview: bool = False,
                       now: datetime | None = None, suffix: str = ".mp4") -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = f"{safe_project_name(project_name)}{'_preview' if preview else ''}_{stamp}"
    directory = Path(output_dir)
    candidate = directory / f"{base}{suffix}"
    counter = 2
    while candidate.exists() or _partial_path(candidate).exists():
        candidate = directory / f"{base}_{counter}{suffix}"
        counter += 1
    return candidate


def is_within(path: str | Path, folder: str | Path) -> bool:
    if not str(folder).strip():
        return False
    resolved_path = Path(path).resolve()
    resolved_folder = Path(folder).resolve()
    return resolved_path == resolved_folder or resolved_folder in resolved_path.parents


def _partial_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.name + ".part")


def _final_encoder_args(encoder: str, settings: ProjectSettings, preview: bool) -> list[str]:
    quality = settings.export.quality
    preset = settings.export.preset
    if preview:
        quality = min(quality + 6, 40)
        preset = "p2" if encoder == "h264_nvenc" else "veryfast"
    return video_encoder_args(encoder, quality, preset)


def mux_final(
    video_path: str | Path,
    audio_wav: str | Path,
    out_path: str | Path,
    settings: ProjectSettings,
    *,
    preview: bool,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
) -> ExportReport:
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    fps = settings.video.fps
    binaries = binaries_for(settings)
    duration = wav_duration(audio_wav)
    expected_frames = round(duration * fps)
    partial = _partial_path(out_path)

    def build_args(encoder: str) -> list[str]:
        return [
            "-i", str(video_path), "-i", str(audio_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            *_final_encoder_args(encoder, settings, preview), "-pix_fmt", "yuv420p", *BT709_TAGS,
            "-r", str(fps), "-fps_mode", "cfr", "-frames:v", str(expected_frames),
            "-c:a", "aac", "-b:a", settings.export.audio_bitrate, "-ar", str(OUTPUT_SAMPLE_RATE),
            "-movflags", "+faststart", "-f", "mp4", str(partial),
        ]

    try:
        encoder = encode_with_fallback(build_args, preferred_encoder(settings), binaries=binaries, duration=duration,
                                       progress=progress, cancel=cancel, log=log)
        partial.rename(out_path)
    finally:
        partial.unlink(missing_ok=True)
    return verify_export(out_path, encoder, expected_frames, fps, binaries)


def verify_export(path: Path, encoder: str, expected_frames: int, fps: int, binaries: FFmpegBinaries) -> ExportReport:
    info = probe(path, binaries)
    frames = count_video_frames(path, binaries)
    report = ExportReport(str(path), encoder, frames, expected_frames, info.fps, info.sample_rate)
    if frames != expected_frames:
        report.issues.append(Issue(Severity.WARNING, "export_frame_mismatch",
                                   f"Exported video has {frames} frames, expected {expected_frames}.", str(path)))
    if abs(info.fps - fps) > 0.01:
        report.issues.append(Issue(Severity.WARNING, "export_fps_mismatch",
                                   f"Exported video runs at {info.fps:g} FPS instead of {fps}.", str(path)))
    if info.sample_rate != OUTPUT_SAMPLE_RATE:
        report.issues.append(Issue(Severity.WARNING, "export_audio_rate_mismatch",
                                   f"Exported audio sample rate is {info.sample_rate} Hz instead of {OUTPUT_SAMPLE_RATE} Hz.", str(path)))
    return report
