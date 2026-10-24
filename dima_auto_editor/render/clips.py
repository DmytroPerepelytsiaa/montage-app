from __future__ import annotations

import math
import shutil
import threading
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import MOTION_SMOOTH, ProjectSettings
from ..ffmpeg_tools import FFmpegBinaries, LogCallback, ProgressCallback, run_capture, run_ffmpeg
from ..models import Issue, MediaInfo, MediaKind, NumberedFile, Severity
from .export import binaries_for, encode_with_fallback, intermediate_video_args, preferred_encoder
from .rife import ENGINE_RIFE, FRAME_PATTERN, RifeError, interpolate_frames, rife_for

PIXEL_ART_MAX_WIDTH = 720
PROXY_SIZE = 48
HARD_CUT_MIN_DIFF = 30.0
HARD_CUT_MEDIAN_RATIO = 6.0
MAX_LOOP_BUFFER_FRAMES = 32767
MINTERPOLATE = "minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
SCALE_FLAGS = {"neighbor": "neighbor", "nearest": "neighbor", "point": "neighbor", "bilinear": "bilinear",
               "bicubic": "bicubic", "lanczos": "lanczos", "area": "area", "spline": "spline"}


@dataclass
class ClipReport:
    scene_path: str
    output_path: str
    frames: int
    method: str
    trimmed: bool = False
    extended: bool = False
    interpolated: bool = False
    source_start: float = 0.0
    speed_factor: float = 1.0
    encoder: str = ""
    warnings: list[Issue] = field(default_factory=list)


@dataclass
class _ClipPlan:
    input_args: list[str]
    graph: list[str]
    last_label: str
    method: str
    trimmed: bool = False
    extended: bool = False
    interpolated: bool = False
    source_start: float = 0.0
    speed_factor: float = 1.0
    rife_capable: bool = False
    rife_frame_limit: int | None = None
    warnings: list[Issue] = field(default_factory=list)


def output_size(settings: ProjectSettings, preview: bool) -> tuple[int, int]:
    scale = settings.export.preview_scale if preview else 1.0
    return _even(settings.video.width * scale), _even(settings.video.height * scale)


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def scale_flags(settings: ProjectSettings) -> str:
    return SCALE_FLAGS.get(settings.video.scale_filter.lower(), "bicubic")


def fit_filter(in_label: str, out_label: str, fit_mode: str, width: int, height: int, flags: str) -> str:
    color = "out_color_matrix=bt709:out_range=tv"
    if fit_mode == "contain_black":
        return (f"[{in_label}]scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags={flags}:{color},"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p[{out_label}]")
    if fit_mode == "contain_blur":
        small_w, small_h = _even(width / 8), _even(height / 8)
        return (f"[{in_label}]split[{out_label}_fg_src][{out_label}_bg_src];"
                f"[{out_label}_bg_src]scale={small_w}:{small_h}:force_original_aspect_ratio=increase:flags=area:{color},"
                f"crop={small_w}:{small_h},boxblur=8:2,scale={width}:{height}:flags=bilinear[{out_label}_bg];"
                f"[{out_label}_fg_src]scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags={flags}:{color}"
                f"[{out_label}_fg];"
                f"[{out_label}_bg][{out_label}_fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[{out_label}]")
    return (f"[{in_label}]scale={width}:{height}:force_original_aspect_ratio=increase:flags={flags}:{color},"
            f"crop={width}:{height},setsar=1,format=yuv420p[{out_label}]")


def render_scene_clip(
    scene: NumberedFile,
    info: MediaInfo,
    content_duration: float,
    out_path: str | Path,
    settings: ProjectSettings,
    *,
    preview: bool,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
) -> ClipReport:
    fps = settings.video.fps
    frame_count = max(1, round(content_duration * fps))
    binaries = binaries_for(settings)
    if scene.kind is MediaKind.IMAGE:
        plan = _plan_image(scene, frame_count, settings, preview)
    else:
        plan = _plan_video(scene, info, frame_count, settings, binaries, log)
        rife = rife_for(settings) if plan.interpolated and plan.rife_capable else None
        if rife is not None:
            try:
                encoder = _render_with_rife(plan, info, frame_count, Path(out_path), settings, preview, binaries, rife,
                                            progress, cancel, log)
                plan.method += "_rife"
                return _report(scene, out_path, frame_count, plan, encoder)
            except RifeError as error:
                if log:
                    log(f"RIFE failed for {Path(scene.path).name}: {error}")
                plan.warnings.append(Issue(
                    Severity.WARNING, "rife_failed",
                    f"{Path(scene.path).name}: RIFE frame generation failed ({str(error).splitlines()[0]}); "
                    "FFmpeg minterpolate was used instead.", scene.path))
        elif plan.interpolated and settings.video.interpolation_engine == ENGINE_RIFE:
            plan.warnings.append(Issue(
                Severity.WARNING, "rife_missing",
                f"{Path(scene.path).name}: RIFE was not found (tools\\rife or the path on the Video tab); "
                "FFmpeg minterpolate was used instead.", scene.path))
        _append_motion_and_fit(plan, info, settings, preview)
    graph = [*plan.graph, f"[{plan.last_label}]fps={fps},tpad=stop=-1:stop_mode=clone,format=yuv420p[out]"]

    def build_args(encoder: str) -> list[str]:
        return [*plan.input_args, "-filter_complex", ";".join(graph), "-map", "[out]", "-an",
                "-frames:v", str(frame_count), "-r", str(fps), *intermediate_video_args(encoder, preview), str(out_path)]

    encoder = encode_with_fallback(build_args, preferred_encoder(settings), binaries=binaries,
                                   duration=frame_count / fps, progress=progress, cancel=cancel, log=log)
    return _report(scene, out_path, frame_count, plan, encoder)


def _report(scene: NumberedFile, out_path: str | Path, frame_count: int, plan: _ClipPlan, encoder: str) -> ClipReport:
    return ClipReport(
        scene_path=scene.path, output_path=str(out_path), frames=frame_count, method=plan.method,
        trimmed=plan.trimmed, extended=plan.extended, interpolated=plan.interpolated,
        source_start=plan.source_start, speed_factor=plan.speed_factor, encoder=encoder, warnings=plan.warnings,
    )


def _source_is_small(info: MediaInfo, width: int, height: int) -> bool:
    return info.width * info.height <= width * height if info.width and info.height else True


def _stage_progress(progress: ProgressCallback | None, start: float, end: float) -> ProgressCallback | None:
    if progress is None:
        return None
    return lambda fraction: progress(start + (end - start) * fraction)


def _render_with_rife(plan: _ClipPlan, info: MediaInfo, frame_count: int, out_path: Path, settings: ProjectSettings,
                      preview: bool, binaries: FFmpegBinaries, rife: Path, progress: ProgressCallback | None,
                      cancel: threading.Event | None, log: LogCallback | None) -> str:
    fps = settings.video.fps
    width, height = output_size(settings, preview)
    fit = partial(fit_filter, fit_mode=settings.video.fit_mode, width=width, height=height, flags=scale_flags(settings))
    source_is_small = _source_is_small(info, width, height)
    work_dir = out_path.with_name(out_path.stem + "_rife")
    source_dir, generated_dir = work_dir / "source", work_dir / "generated"
    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        prepare = ("[0:v]setpts=PTS-STARTPTS,format=rgb24[frames]" if source_is_small
                   else "[0:v]setpts=PTS-STARTPTS[raw];" + fit("raw", "frames"))
        limit = ["-frames:v", str(plan.rife_frame_limit)] if plan.rife_frame_limit else []
        run_ffmpeg([*plan.input_args, "-filter_complex", prepare, "-map", "[frames]", "-fps_mode", "passthrough",
                    *limit, str(source_dir / FRAME_PATTERN)], binaries=binaries, cancel=cancel, log=log)
        if len(list(source_dir.glob("*.png"))) < 2:
            raise RifeError("the clip has fewer than two frames to interpolate between")
        interpolate_frames(rife, source_dir, generated_dir, frame_count, settings.video.rife_model,
                           progress=_stage_progress(progress, 0.1, 0.85), cancel=cancel, log=log)
        finish = ((fit("0:v", "fitted") if source_is_small else "[0:v]format=yuv420p[fitted]")
                  + f";[fitted]fps={fps},tpad=stop=-1:stop_mode=clone,format=yuv420p[out]")

        def build_args(encoder: str) -> list[str]:
            return ["-framerate", str(fps), "-i", str(generated_dir / FRAME_PATTERN), "-filter_complex", finish,
                    "-map", "[out]", "-an", "-frames:v", str(frame_count), "-r", str(fps),
                    *intermediate_video_args(encoder, preview), str(out_path)]

        return encode_with_fallback(build_args, preferred_encoder(settings), binaries=binaries, duration=frame_count / fps,
                                    progress=_stage_progress(progress, 0.85, 1.0), cancel=cancel, log=log)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _plan_image(scene: NumberedFile, frame_count: int, settings: ProjectSettings, preview: bool) -> _ClipPlan:
    video = settings.video
    width, height = output_size(settings, preview)
    flags = scale_flags(settings)
    input_args = ["-framerate", str(video.fps), "-i", scene.path]
    strength = max(0.0, video.image_motion_strength)
    if video.image_motion not in ("zoom_in", "zoom_out", "pan") or strength <= 0 or frame_count < 2:
        graph = [fit_filter("0:v", "fitted", video.fit_mode, width, height, flags),
                 f"[fitted]loop=loop=-1:size=1,setpts=N/({video.fps}*TB)[held]"]
        return _ClipPlan(input_args, graph, "held", "image_hold")
    zoompan = _zoompan_filter(video.image_motion, strength, frame_count, width, height, video.fps)
    graph = [fit_filter("0:v", "canvas", video.fit_mode, width * 2, height * 2, flags), f"[canvas]{zoompan}[fitted]"]
    return _ClipPlan(input_args, graph, "fitted", f"image_{video.image_motion}")


def _zoompan_filter(motion: str, strength: float, frame_count: int, width: int, height: int, fps: int) -> str:
    progress = f"on/{frame_count - 1}"
    centered_x, centered_y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    if motion == "zoom_in":
        zoom, x = f"1+{strength:.6f}*{progress}", centered_x
    elif motion == "zoom_out":
        zoom, x = f"1+{strength:.6f}*(1-{progress})", centered_x
    else:
        zoom, x = f"{1 + strength:.6f}", f"(iw-iw/zoom)*{progress}"
    return f"zoompan=z={zoom}:x={x}:y={centered_y}:d={frame_count}:s={width}x{height}:fps={fps}"


def _plan_video(scene: NumberedFile, info: MediaInfo, frame_count: int, settings: ProjectSettings,
                binaries: FFmpegBinaries, log: LogCallback | None) -> _ClipPlan:
    video = settings.video
    needed = frame_count / video.fps
    source_fps = info.fps if info.fps > 0 else video.source_fps_hint
    source_duration = info.duration or (info.frame_count / source_fps if info.frame_count else 0.0)
    if source_duration <= 0 or source_duration >= needed - 0.5 / video.fps:
        plan = _plan_trim(scene, needed, source_fps, source_duration, settings, binaries, log)
    else:
        plan = _plan_extend(scene, needed, source_fps, source_duration, settings)
    slowed = plan.speed_factor > 1.0001
    force_interpolation = plan.extended and video.extend_mode == "interpolate"
    wants_interpolation = (video.motion_mode == MOTION_SMOOTH or force_interpolation
                           or (slowed and video.interpolate_slowed_clips))
    needs_interpolation = source_fps < video.fps - 0.5 or slowed
    plan.interpolated = wants_interpolation and needs_interpolation
    if plan.interpolated:
        plan.warnings.extend(_interpolation_warnings(scene, info))
    return plan


def _plan_trim(scene: NumberedFile, needed: float, source_fps: float, source_duration: float,
               settings: ProjectSettings, binaries: FFmpegBinaries, log: LogCallback | None) -> _ClipPlan:
    mode = settings.video.trim_mode
    spare = max(0.0, source_duration - needed)
    start = 0.0
    if spare > 0 and mode == "center":
        start = spare / 2
    elif spare > 0 and mode == "smart":
        start = _smart_trim_start(scene.path, needed, source_fps, binaries, log)
    start = min(round(start * source_fps) / source_fps, spare)
    input_args = ["-ss", f"{start:.6f}"] if start > 0 else []
    input_args += ["-t", f"{needed + 2 / source_fps:.6f}", "-i", scene.path]
    trimmed = spare > 1 / settings.video.fps
    method = f"trim_{mode}" if trimmed else "as_is"
    return _ClipPlan(input_args, ["[0:v]setpts=PTS-STARTPTS[timed]"], "timed", method, trimmed=trimmed, source_start=start,
                     rife_capable=True, rife_frame_limit=round(needed * source_fps) + 1)


def _plan_extend(scene: NumberedFile, needed: float, source_fps: float, source_duration: float,
                 settings: ProjectSettings) -> _ClipPlan:
    video = settings.video
    mode = video.extend_mode
    regular_timing = f"setpts=N/({source_fps:.6f}*TB)"
    if mode == "loop":
        graph = [f"[0:v]{regular_timing}[timed]"]
        return _ClipPlan(["-stream_loop", "-1", "-i", scene.path], graph, "timed", "extend_loop", extended=True)
    if mode == "pingpong":
        graph = [
            "[0:v]setpts=PTS-STARTPTS,split[forward][backward_src]",
            "[backward_src]reverse[backward]",
            f"[forward][backward]concat=n=2:v=1:a=0,loop=loop=-1:size={MAX_LOOP_BUFFER_FRAMES},{regular_timing}[timed]",
        ]
        return _ClipPlan(["-i", scene.path], graph, "timed", "extend_pingpong", extended=True)
    speed = 1.0
    if mode in ("slow", "interpolate"):
        speed = max(1.0, needed / source_duration)
    method = f"extend_{mode}" if speed > 1.0 else "extend_freeze"
    graph = [f"[0:v]setpts=(PTS-STARTPTS)*{speed:.6f}[timed]"]
    return _ClipPlan(["-i", scene.path], graph, "timed", method, extended=True, speed_factor=speed, rife_capable=speed > 1.0)


def _append_motion_and_fit(plan: _ClipPlan, info: MediaInfo, settings: ProjectSettings, preview: bool) -> None:
    width, height = output_size(settings, preview)
    fit = partial(fit_filter, fit_mode=settings.video.fit_mode, width=width, height=height, flags=scale_flags(settings))
    if not plan.interpolated:
        plan.graph.append(fit(plan.last_label, "fitted"))
        plan.last_label = "fitted"
        return
    interpolate = MINTERPOLATE.format(fps=settings.video.fps)
    if _source_is_small(info, width, height):
        plan.graph.append(f"[{plan.last_label}]{interpolate}[interpolated]")
        plan.graph.append(fit("interpolated", "fitted"))
    else:
        plan.graph.append(fit(plan.last_label, "prefit"))
        plan.graph.append(f"[prefit]{interpolate}[fitted]")
    plan.last_label = "fitted"
    plan.method += "_interpolated"


def _interpolation_warnings(scene: NumberedFile, info: MediaInfo) -> list[Issue]:
    warnings: list[Issue] = []
    name = Path(scene.path).name
    if info.width and info.width < PIXEL_ART_MAX_WIDTH:
        warnings.append(Issue(
            Severity.WARNING, "interpolation_pixel_art",
            f"{name}: motion interpolation on low-resolution or pixel-art footage may smear pixel edges and create "
            f"warping artifacts. Consider the choppy 60 FPS mode for pixel art.", scene.path))
    return warnings


def _smart_trim_start(path: str, needed: float, source_fps: float, binaries: FFmpegBinaries, log: LogCallback | None) -> float:
    proxy = decode_gray_proxy(path, binaries)
    if proxy is None or len(proxy) < 2:
        if log:
            log(f"Smart trim analysis failed for {Path(path).name}; trimming from the start.")
        return 0.0
    window = min(len(proxy), max(1, math.ceil(needed * source_fps - 1e-6)))
    return choose_smart_window(proxy, window) / source_fps


def decode_gray_proxy(path: str, binaries: FFmpegBinaries | None = None, size: int = PROXY_SIZE) -> np.ndarray | None:
    result = run_capture(
        ["-v", "error", "-i", path, "-map", "0:v:0", "-vf", f"scale={size}:{size}:flags=area,format=gray",
         "-f", "rawvideo", "pipe:1"],
        binaries=binaries,
    )
    frame_bytes = size * size
    if result.returncode != 0 or len(result.stdout) < frame_bytes:
        return None
    usable = len(result.stdout) // frame_bytes * frame_bytes
    return np.frombuffer(result.stdout[:usable], dtype=np.uint8).reshape(-1, size, size)


def frame_motion(proxy: np.ndarray) -> np.ndarray:
    frames = proxy.astype(np.float32)
    motion = np.zeros(len(frames), dtype=np.float32)
    if len(frames) > 1:
        motion[1:] = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return motion


def detect_hard_cuts(motion: np.ndarray) -> np.ndarray:
    if len(motion) < 2:
        return np.zeros(len(motion), dtype=bool)
    threshold = max(HARD_CUT_MIN_DIFF, HARD_CUT_MEDIAN_RATIO * float(np.median(motion[1:])))
    is_cut = motion > threshold
    is_cut[0] = False
    return is_cut


def choose_smart_window(proxy: np.ndarray, window: int) -> int:
    total = len(proxy)
    if window >= total:
        return 0
    motion = frame_motion(proxy)
    is_cut = detect_hard_cuts(motion)
    calm_motion = np.where(is_cut, 0.0, motion)
    normalized = motion / (float(np.percentile(calm_motion, 95)) + 1e-6)
    cut_prefix = np.concatenate([[0], np.cumsum(is_cut)])
    starts = np.arange(total - window + 1)
    ends = starts + window - 1
    crossings = cut_prefix[ends + 1] - cut_prefix[starts + 1]
    following_is_cut = np.append(is_cut[1:], True)[ends]
    starts_naturally = (starts == 0) | is_cut[starts]
    score = (1000.0 * crossings + normalized[ends] + 0.2 * starts / max(1, total - window)
             - 0.5 * following_is_cut - 0.3 * starts_naturally)
    return int(np.argmin(score))
