from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QFont, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..config import ProjectSettings, default_settings_path
from ..ffmpeg_tools import FFmpegNotFound, locate_binaries
from ..models import Issue, Severity, Timeline
from .eta import EtaEstimator, format_seconds
from .preview_panel import PreviewPanel
from .results_view import IssuesList, TimelineTable, issue_counts
from .settings_panel import JSON_FILTER, SettingsPanel
from .theme import BUILD_BUTTON_STYLE
from .worker import BackgroundTask, PipelineFactory, PipelineWorker, default_pipeline_factory

SETTINGS_SAVE_DELAY_MS = 800
SHUTDOWN_WAIT_MS = 15_000
PROGRESS_STEPS = 1000

FFMPEG_HELP = (
    "FFmpeg is required to read and render media, but it was not found.\n\n"
    "How to fix (any one is enough):\n"
    "  1. Run install.ps1 again: it installs the 'imageio-ffmpeg' package, which bundles a working FFmpeg.\n"
    "     Manually: .venv\\Scripts\\python -m pip install imageio-ffmpeg\n"
    "  2. Install FFmpeg system-wide: winget install Gyan.FFmpeg  (then restart the app)\n"
    "  3. Point the app to ffmpeg.exe on the Export tab (\"FFmpeg executable\")."
)



class MainWindow(QMainWindow):
    def __init__(self, settings_path: Path | None = None, pipeline_factory: PipelineFactory = default_pipeline_factory):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self._settings_path = Path(settings_path) if settings_path else default_settings_path()
        self._pipeline_factory = pipeline_factory
        self._task: BackgroundTask | None = None
        self._on_task_success: Callable[[Any], None] | None = None
        self._last_output_path = ""
        self._eta = EtaEstimator()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SETTINGS_SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._on_settings_settled)

        self.settings_panel = SettingsPanel()
        self.issues_list = IssuesList()
        self.timeline_table = TimelineTable()
        self.preview_panel = PreviewPanel()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20_000)
        self.log_view.setFont(QFont("Consolas", 9))

        self._build_layout()
        self._build_menu()
        self._connect_signals()
        self._restore_settings()
        self._set_busy(False)

    def _build_layout(self) -> None:
        self.results_tabs = QTabWidget()
        self.results_tabs.addTab(self.timeline_table, "Timeline")
        self.results_tabs.addTab(self.preview_panel, "Preview")
        self.messages_tabs = QTabWidget()
        self.messages_tabs.addTab(self.issues_list, "Issues")
        self.messages_tabs.addTab(self.log_view, "Log")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.results_tabs)
        right.addWidget(self.messages_tabs)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(self.settings_panel)
        main_split.addWidget(right)
        main_split.setStretchFactor(0, 2)
        main_split.setStretchFactor(1, 3)
        main_split.setSizes([620, 880])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(main_split, 1)
        layout.addLayout(self._build_status_row())
        layout.addLayout(self._build_action_row())
        self.setCentralWidget(central)

    def _build_status_row(self) -> QHBoxLayout:
        self.stage_label = QLabel("Ready.")
        self.stage_label.setMinimumWidth(260)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setValue(0)
        self.eta_label = QLabel("")
        self.eta_label.setMinimumWidth(170)
        row = QHBoxLayout()
        row.addWidget(self.stage_label)
        row.addWidget(self.progress_bar, 1)
        row.addWidget(self.eta_label)
        return row

    def _build_action_row(self) -> QHBoxLayout:
        self.scan_button = QPushButton("Scan inputs")
        self.plan_button = QPushButton("Plan timeline")
        self.cancel_button = QPushButton("Cancel")
        self.open_output_button = QPushButton("Open output folder")
        self.build_button = QPushButton("Build Video")
        self.build_button.setStyleSheet(BUILD_BUTTON_STYLE)
        self.build_button.setMinimumHeight(46)
        self.build_button.setToolTip("Render the final 1080×1920 60 FPS video with the processed narration.")
        row = QHBoxLayout()
        for button in (self.scan_button, self.plan_button):
            row.addWidget(button)
        row.addStretch(1)
        for button in (self.cancel_button, self.open_output_button, self.build_button):
            row.addWidget(button)
        return row

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._add_action(file_menu, "&Load project settings…", self.load_project_settings, QKeySequence.StandardKey.Open)
        self._add_action(file_menu, "&Save project settings as…", self.save_project_settings, QKeySequence.StandardKey.Save)
        self._add_action(file_menu, "Reset settings to defaults", self.reset_settings)
        file_menu.addSeparator()
        self._add_action(file_menu, "Open output folder", self.open_output_folder)
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", self.close, QKeySequence.StandardKey.Quit)
        tools_menu = self.menuBar().addMenu("&Tools")
        self._add_action(tools_menu, "Check FFmpeg", lambda: self.check_ffmpeg(report_success=True))
        help_menu = self.menuBar().addMenu("&Help")
        self._add_action(help_menu, "About", self._show_about)

    def _add_action(self, menu, text: str, handler: Callable[[], Any], shortcut: QKeySequence.StandardKey | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: handler())
        menu.addAction(action)
        return action

    def _connect_signals(self) -> None:
        self.settings_panel.changed.connect(self._save_timer.start)
        self.preview_panel.voice_preview_requested.connect(self.preview_voice)
        self.preview_panel.render_preview_requested.connect(self.preview_render)
        self.scan_button.clicked.connect(self.scan_inputs)
        self.plan_button.clicked.connect(self.plan_timeline)
        self.build_button.clicked.connect(self.build_video)
        self.cancel_button.clicked.connect(self.cancel_task)
        self.open_output_button.clicked.connect(self.open_output_folder)

    def current_settings(self) -> ProjectSettings:
        return self.settings_panel.collect_settings()

    def _restore_settings(self) -> None:
        settings = ProjectSettings()
        if self._settings_path.is_file():
            try:
                settings = ProjectSettings.load(self._settings_path)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                self.append_log(f"Could not read saved settings ({self._settings_path}): {error}. Using defaults.")
        self._apply(settings)

    def _apply(self, settings: ProjectSettings) -> None:
        rejected = self.settings_panel.apply_settings(settings)
        if rejected:
            self.append_log("Ignored invalid saved values (defaults kept): " + ", ".join(rejected))

    def save_settings(self) -> None:
        self._save_timer.stop()
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.current_settings().save(self._settings_path)
        except OSError as error:
            self.append_log(f"Could not save settings to {self._settings_path}: {error}")

    def _on_settings_settled(self) -> None:
        self.save_settings()

    def _apply_and_save(self, settings: ProjectSettings) -> None:
        self._apply(settings)
        self.save_settings()

    def load_project_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load project settings", "", JSON_FILTER)
        if not path:
            return
        try:
            settings = ProjectSettings.load(path)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self._show_error("Cannot load project settings", f"Could not read {path}:\n{error}")
            return
        self._apply_and_save(settings)
        self.append_log(f"Loaded project settings from {path}")

    def save_project_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save project settings", "project.json", JSON_FILTER)
        if not path:
            return
        try:
            self.current_settings().save(path)
        except OSError as error:
            self._show_error("Cannot save project settings", f"Could not write {path}:\n{error}")
            return
        self.append_log(f"Saved project settings to {path}")

    def reset_settings(self) -> None:
        answer = QMessageBox.question(self, "Reset settings", "Reset all settings (including folders) to defaults?")
        if answer == QMessageBox.StandardButton.Yes:
            self._apply_and_save(ProjectSettings())

    def check_ffmpeg(self, report_success: bool = False) -> bool:
        try:
            binaries = locate_binaries(self.current_settings().export.ffmpeg_path)
        except FFmpegNotFound:
            QMessageBox.warning(self, "FFmpeg not found", FFMPEG_HELP)
            return False
        if report_success:
            probe_note = binaries.ffprobe or "not found (not required; FFmpeg is used for probing)"
            QMessageBox.information(self, "FFmpeg", f"FFmpeg: {binaries.ffmpeg}\nffprobe: {probe_note}")
        return True

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} {__version__}\n\nBuilds vertical 1080×1920 60 FPS biography videos from numbered "
            "scenes and voice-overs: pause removal, even scene timing, frame generation, music and export.",
        )

    def scan_inputs(self) -> None:
        self._run_task("Scanning inputs", lambda pipeline: pipeline.scan(), self._show_scan, needs_output=False)

    def plan_timeline(self) -> None:
        self._run_task("Planning timeline", lambda pipeline: pipeline.plan(), self._show_timeline)

    def preview_voice(self) -> None:
        self._run_task("Processing voice-over", lambda pipeline: pipeline.preview_voice(), self._show_voice_preview)

    def preview_render(self) -> None:
        self._run_task("Rendering preview", lambda pipeline: pipeline.build(preview=True), self._show_render_preview)

    def build_video(self) -> None:
        self._run_task("Building video", lambda pipeline: pipeline.build(preview=False), self._show_build_result)

    def cancel_task(self) -> None:
        if self._task is not None and self._task.is_running():
            self._task.cancel()
            self.stage_label.setText("Cancelling…")
            self.append_log("Cancellation requested.")

    def is_busy(self) -> bool:
        return self._task is not None

    def _missing_inputs(self, settings: ProjectSettings, needs_output: bool) -> list[str]:
        required = [("Voice-over folder", settings.paths.voice_dir), ("Scenes folder", settings.paths.scenes_dir)]
        if needs_output:
            required.append(("Output folder", settings.paths.output_dir))
        return [label for label, value in required if not value]

    def _run_task(self, title: str, task: Callable[[Any], Any], on_success: Callable[[Any], None], needs_output: bool = True) -> None:
        if self.is_busy():
            return
        settings = self.current_settings()
        missing = self._missing_inputs(settings, needs_output)
        if missing:
            QMessageBox.warning(self, "Missing folders", "Please choose: " + ", ".join(missing) + " (Inputs tab).")
            self.settings_panel.tabs.setCurrentIndex(0)
            return
        self.save_settings()
        self.preview_panel.release_media()
        cancel_event = threading.Event()
        worker = PipelineWorker(settings, task, cancel_event, self._pipeline_factory)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self.append_log)
        worker.finished.connect(self._on_task_finished)
        worker.failed.connect(self._on_task_failed)
        worker.blocked.connect(self._on_task_blocked)
        worker.cancelled.connect(self._on_task_cancelled)
        self._task = BackgroundTask(worker, cancel_event, self)
        self._task.stopped.connect(self._on_task_stopped)
        self._on_task_success = on_success
        self._eta.reset()
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.stage_label.setText(f"{title}…")
        self.append_log(f"=== {title} ===")
        self._set_busy(True)
        self._task.start()

    def _set_busy(self, busy: bool) -> None:
        for button in (self.scan_button, self.plan_button, self.build_button):
            button.setEnabled(not busy)
        self.preview_panel.set_busy(busy)
        self.cancel_button.setEnabled(busy)

    def _on_progress(self, stage: str, fraction: float) -> None:
        self.stage_label.setText(stage)
        self.progress_bar.setValue(int(round(min(1.0, max(0.0, fraction)) * PROGRESS_STEPS)))
        remaining = self._eta.remaining(fraction)
        elapsed = format_seconds(self._eta.elapsed())
        self.eta_label.setText(f"Elapsed {elapsed}" + (f" · ETA {format_seconds(remaining)}" if remaining is not None else ""))

    def _on_task_finished(self, result: Any) -> None:
        self.progress_bar.setValue(PROGRESS_STEPS)
        self.stage_label.setText("Done.")
        callback, self._on_task_success = self._on_task_success, None
        if callback is not None:
            try:
                callback(result)
            except Exception as error:
                self._show_error("Unexpected result", f"The task finished but its result could not be shown:\n{error}")

    def _on_task_failed(self, message: str) -> None:
        self.stage_label.setText("Failed.")
        self.messages_tabs.setCurrentWidget(self.log_view)
        self._show_error("Error", message)

    def _on_task_blocked(self, issues: list[Issue]) -> None:
        self.stage_label.setText("Blocked by input errors.")
        self.show_issues(issues)
        errors = issue_counts(issues)[Severity.ERROR]
        QMessageBox.warning(
            self,
            "Cannot build yet",
            f"{errors} problem(s) must be fixed before rendering. See the Issues list for details.",
        )

    def _on_task_cancelled(self) -> None:
        self.stage_label.setText("Cancelled.")
        self.append_log("Task cancelled.")

    def _on_task_stopped(self) -> None:
        if self._task is not None:
            self._task.wait(SHUTDOWN_WAIT_MS)
            self._task.deleteLater()
        self._task = None
        self._on_task_success = None
        self.eta_label.setText(f"Total {format_seconds(self._eta.elapsed())}")
        self._set_busy(False)

    def show_issues(self, issues: list[Issue]) -> None:
        self.issues_list.set_issues(issues)
        counts = issue_counts(issues)
        self.messages_tabs.setTabText(
            0, f"Issues ({counts[Severity.ERROR]} errors, {counts[Severity.WARNING]} warnings, {counts[Severity.INFO]} info)"
        )
        if issues:
            self.messages_tabs.setCurrentWidget(self.issues_list)

    def _show_scan(self, scan: Any) -> None:
        self.show_issues(list(scan.issues))
        parts = [f"{len(scan.voices)} voice-over(s)", f"{len(scan.scenes)} scene(s)"]
        if scan.music:
            parts.append(f"music: {Path(scan.music).name}")
        summary = "Scan: " + ", ".join(parts) + "."
        self.append_log(summary)
        self.stage_label.setText(summary)

    def _show_timeline(self, timeline: Timeline) -> None:
        self.timeline_table.show_timeline(timeline)
        self.results_tabs.setCurrentWidget(self.timeline_table)
        self.show_issues(list(timeline.issues))
        self.append_log(
            f"Timeline: {len(timeline.entries)} scene(s), {len(timeline.transitions)} transition(s), "
            f"{timeline.duration:.2f} s at {timeline.fps} FPS."
        )

    def _show_voice_preview(self, wav_path: str) -> None:
        self.preview_panel.set_voice(str(wav_path))
        self.results_tabs.setCurrentWidget(self.preview_panel)
        self.preview_panel.play_voice()

    def _show_build_outcome(self, result: Any) -> None:
        self._last_output_path = str(result.output_path)
        self._show_timeline(result.timeline)
        self.show_issues(list(result.issues))
        self.append_log(f"Video: {result.output_path}\nManifest: {result.manifest_path}\nLog: {result.log_path}")
        for key, value in (result.stats or {}).items():
            self.append_log(f"  {key}: {value}")

    def _show_render_preview(self, result: Any) -> None:
        self._show_build_outcome(result)
        self.preview_panel.set_video(str(result.output_path))
        self.results_tabs.setCurrentWidget(self.preview_panel)
        self.preview_panel.toggle_video()

    def _show_build_result(self, result: Any) -> None:
        self._show_build_outcome(result)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Video ready")
        box.setText(f"The video was saved to:\n{result.output_path}")
        open_button = box.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_button:
            self.open_output_folder()

    def _output_folder(self) -> Path | None:
        if self._last_output_path:
            return Path(self._last_output_path).parent
        output_dir = self.current_settings().paths.output_dir
        return Path(output_dir) if output_dir else None

    def open_output_folder(self) -> None:
        folder = self._output_folder()
        if folder is None:
            QMessageBox.information(self, "Output folder", "Choose an output folder on the Inputs tab first.")
            return
        if not folder.is_dir():
            QMessageBox.information(self, "Output folder", f"The folder does not exist yet:\n{folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _show_error(self, title: str, message: str) -> None:
        self.append_log(f"ERROR: {message}")
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._task is not None and self._task.is_running():
            answer = QMessageBox.question(self, "Quit", "A task is still running. Cancel it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._task.cancel()
            self._task.wait(SHUTDOWN_WAIT_MS)
        self.preview_panel.release_media()
        self.save_settings()
        super().closeEvent(event)
