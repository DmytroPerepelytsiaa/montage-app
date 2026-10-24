from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from ..config import ProjectSettings
from ..ffmpeg_tools import LogCallback, ProgressCallback, RenderCancelled

ENGINE_AUTO = "auto"
ENGINE_RIFE = "rife"
ENGINE_MINTERPOLATE = "minterpolate"
RIFE_EXECUTABLE = "rife-ncnn-vulkan.exe" if sys.platform == "win32" else "rife-ncnn-vulkan"
FRAME_PATTERN = "%08d.png"
RIFE_THREADS = "2:4:4"
POLL_INTERVAL_S = 0.25

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
_GPU_LOCK = threading.Lock()


class RifeError(RuntimeError):
    pass


def bundled_rife_dir() -> Path:
    return _PROJECT_ROOT / "tools" / "rife"


def find_rife(settings: ProjectSettings) -> Path | None:
    configured = settings.video.rife_executable.strip()
    candidates = [Path(configured)] if configured else []
    candidates.append(bundled_rife_dir() / RIFE_EXECUTABLE)
    on_path = shutil.which("rife-ncnn-vulkan")
    if on_path:
        candidates.append(Path(on_path))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def rife_for(settings: ProjectSettings) -> Path | None:
    if settings.video.interpolation_engine == ENGINE_MINTERPOLATE:
        return None
    return find_rife(settings)


def model_dir(executable: Path, preferred: str) -> Path | None:
    exact = executable.parent / preferred
    if preferred and exact.is_dir():
        return exact
    v4_models = sorted(path for path in executable.parent.glob("rife-v4*") if path.is_dir())
    return v4_models[-1] if v4_models else None


def interpolate_frames(
    executable: Path,
    input_dir: Path,
    output_dir: Path,
    target_frames: int,
    model: str,
    *,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
) -> None:
    model_path = model_dir(executable, model)
    if model_path is None:
        raise RifeError(f"No RIFE v4 model folder was found next to {executable}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "-i", str(input_dir), "-o", str(output_dir), "-n", str(target_frames),
               "-m", str(model_path), "-f", FRAME_PATTERN, "-j", RIFE_THREADS]
    if log:
        log("rife " + subprocess.list2cmdline(command[1:]))
    with _GPU_LOCK:
        _run(command, executable.parent, output_dir, target_frames, progress, cancel)
    produced = len(list(output_dir.glob("*.png")))
    if produced != target_frames:
        raise RifeError(f"RIFE produced {produced} frames instead of {target_frames}.")


def _run(command: list[str], working_dir: Path, output_dir: Path, target_frames: int,
         progress: ProgressCallback | None, cancel: threading.Event | None) -> None:
    process = subprocess.Popen(command, cwd=working_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW, text=True, errors="replace")
    stderr_tail: deque[str] = deque(maxlen=40)
    drain = threading.Thread(target=lambda: stderr_tail.extend(process.stderr), daemon=True)  # type: ignore[arg-type]
    drain.start()
    try:
        while True:
            try:
                process.wait(timeout=POLL_INTERVAL_S)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    raise RenderCancelled("Rendering was cancelled by the user.")
                if progress:
                    progress(min(1.0, len(list(output_dir.glob("*.png"))) / target_frames))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        drain.join(timeout=5)
    if process.returncode != 0:
        raise RifeError(f"RIFE failed with exit code {process.returncode}.\n" + "".join(stderr_tail))
