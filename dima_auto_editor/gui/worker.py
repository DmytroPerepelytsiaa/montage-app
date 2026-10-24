from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Protocol

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from ..config import ProjectSettings
from ..ffmpeg_tools import RenderCancelled

ProgressCallback = Callable[[str, float], None]
LogCallback = Callable[[str], None]


class PipelineFactory(Protocol):
    def __call__(
        self,
        settings: ProjectSettings,
        progress_cb: ProgressCallback,
        log_cb: LogCallback,
        cancel_event: threading.Event,
    ) -> Any: ...


PipelineTask = Callable[[Any], Any]


def default_pipeline_factory(
    settings: ProjectSettings,
    progress_cb: ProgressCallback,
    log_cb: LogCallback,
    cancel_event: threading.Event,
) -> Any:
    from ..pipeline import BuildPipeline

    return BuildPipeline(settings, progress_cb, log_cb, cancel_event)


def _blocked_issues(error: BaseException) -> list | None:
    try:
        from ..pipeline import BuildBlocked
    except ImportError:
        return None
    return list(error.issues) if isinstance(error, BuildBlocked) else None


class PipelineWorker(QObject):
    progress = Signal(str, float)
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    blocked = Signal(object)
    cancelled = Signal()
    done = Signal()

    def __init__(
        self,
        settings: ProjectSettings,
        task: PipelineTask,
        cancel_event: threading.Event,
        pipeline_factory: PipelineFactory = default_pipeline_factory,
    ):
        super().__init__()
        self._settings = settings
        self._task = task
        self._cancel_event = cancel_event
        self._pipeline_factory = pipeline_factory

    @Slot()
    def run(self) -> None:
        try:
            pipeline = self._pipeline_factory(self._settings, self.progress.emit, self.log.emit, self._cancel_event)
            result = self._task(pipeline)
        except Exception as error:
            self._report_error(error)
        else:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit(result)
        finally:
            self.done.emit()

    def _report_error(self, error: Exception) -> None:
        if isinstance(error, RenderCancelled) or self._cancel_event.is_set():
            self.cancelled.emit()
            return
        issues = _blocked_issues(error)
        if issues is not None:
            self.blocked.emit(issues)
            return
        self.log.emit("".join(traceback.format_exception(error)).rstrip())
        self.failed.emit(str(error) or type(error).__name__)


class BackgroundTask(QObject):
    """Owns one worker thread; keeps Python references alive until the thread has fully stopped."""

    stopped = Signal()

    def __init__(self, worker: PipelineWorker, cancel_event: threading.Event, parent: QObject | None = None):
        super().__init__(parent)
        self.worker = worker
        self.cancel_event = cancel_event
        self._thread = QThread()
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.done.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._on_thread_finished)

    def start(self) -> None:
        self._thread.start()

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread.wait()
        self.stopped.emit()

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def wait(self, timeout_ms: int) -> bool:
        return self._thread.wait(timeout_ms)
