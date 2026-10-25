from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import ProjectSettings
from ..ffmpeg_tools import FFmpegBinaries, FFmpegError, LogCallback, ProgressCallback, RenderCancelled
from ..models import TransitionEvent
from .export import SOFTWARE_ENCODER, binaries_for, intermediate_video_args
from .stylized_transitions import make_renderer

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
_TO_RGB = "scale=in_color_matrix=bt709:in_range=tv:out_range=full,format=rgb24"
_TO_YUV = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
RENDER_WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))


def _read_frames(binaries: FFmpegBinaries, path: Path, start_frame: int, count: int, width: int, height: int) -> Iterator[np.ndarray]:
    command = [binaries.ffmpeg, "-v", "error", "-i", str(path), "-vf",
               f"trim=start_frame={start_frame},setpts=PTS-STARTPTS,scale={width}:{height},{_TO_RGB}",
               "-fps_mode", "passthrough", "-frames:v", str(count), "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                               creationflags=_NO_WINDOW)
    frame_bytes = width * height * 3
    try:
        assert process.stdout is not None
        for _ in range(count):
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                raise FFmpegError(f"Could not read {count} frames from {path.name} starting at frame {start_frame}.")
            yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.kill()
        process.wait()


def render_transition_clip(
    event: TransitionEvent,
    outgoing_clip: Path,
    outgoing_frames: int,
    incoming_clip: Path,
    frame_count: int,
    size: tuple[int, int],
    out_path: Path,
    settings: ProjectSettings,
    *,
    preview: bool,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
) -> None:
    binaries = binaries_for(settings)
    fps = settings.video.fps
    width, height = size
    draw = make_renderer(event.xfade, height, width, frame_count, settings.transitions.seed + event.boundary_position)
    outgoing = _read_frames(binaries, outgoing_clip, outgoing_frames - frame_count, frame_count, width, height)
    incoming = _read_frames(binaries, incoming_clip, 0, frame_count, width, height)
    command = [binaries.ffmpeg, "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-vf", _TO_YUV, "-frames:v", str(frame_count),
               *intermediate_video_args(SOFTWARE_ENCODER, preview), str(out_path)]
    if log:
        log(f"Rendering transition '{event.name}' at {event.time:.3f} s ({frame_count} frames).")
    writer = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              creationflags=_NO_WINDOW)
    written = 0
    try:
        assert writer.stdin is not None
        with ThreadPoolExecutor(max_workers=RENDER_WORKERS, thread_name_prefix="transition") as pool:
            pending: deque[Future[np.ndarray]] = deque()

            def flush_one() -> None:
                nonlocal written
                writer.stdin.write(pending.popleft().result().tobytes())  # type: ignore[union-attr]
                written += 1
                if progress:
                    progress(written / frame_count)

            for index, (a, b) in enumerate(zip(outgoing, incoming)):
                if cancel is not None and cancel.is_set():
                    raise RenderCancelled("Rendering was cancelled by the user.")
                pending.append(pool.submit(draw, a, b, index))
                if len(pending) >= RENDER_WORKERS * 2:
                    flush_one()
            while pending:
                flush_one()
        writer.stdin.close()
        _, stderr = writer.communicate()
    except BaseException:
        writer.kill()
        writer.wait()
        raise
    if writer.returncode != 0 or written != frame_count:
        raise FFmpegError(f"Encoding transition '{event.name}' failed.", (stderr or b"").decode("utf-8", "replace")[-2000:])
