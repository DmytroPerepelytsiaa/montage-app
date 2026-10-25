from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("dima_auto_editor.transitions")
pytest.importorskip("dima_auto_editor.manifest")

from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from dima_auto_editor.config import MOTION_CHOPPY, MOTION_SMOOTH, ProjectSettings  # noqa: E402
from dima_auto_editor.ffmpeg_tools import RenderCancelled  # noqa: E402
from dima_auto_editor.gui.eta import EtaEstimator, format_seconds  # noqa: E402
from dima_auto_editor.gui.main_window import MainWindow  # noqa: E402
from dima_auto_editor.gui.settings_panel import SettingsPanel  # noqa: E402
from dima_auto_editor.gui.transition_editor import TransitionEditorDialog  # noqa: E402
from dima_auto_editor.models import (  # noqa: E402
    CutStyle,
    Issue,
    MediaKind,
    NumberedFile,
    Severity,
    Timeline,
    TimelineEntry,
    TransitionDef,
    TransitionEvent,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def non_blocking_dialogs(monkeypatch):
    shown: list[tuple[str, str]] = []

    def record(kind: str, answer=QMessageBox.StandardButton.Ok):
        def show(_parent, title, text, *args, **kwargs):
            shown.append((kind, f"{title}: {text}"))
            return answer

        return staticmethod(show)

    for kind in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, kind, record(kind))
    monkeypatch.setattr(QMessageBox, "question", record("question", QMessageBox.StandardButton.Yes))
    return shown


def wait_until(app: QApplication, predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for the GUI condition.")
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)


class FakePipeline:
    calls: list[str] = []
    worker_threads: list[threading.Thread] = []

    def __init__(self, settings, progress_cb, log_cb, cancel_event):
        self.settings = settings
        self.progress = progress_cb
        self.log = log_cb
        self.cancel_event = cancel_event

    def scan(self):
        FakePipeline.worker_threads.append(threading.current_thread())
        self.progress("Scanning inputs", 0.5)
        self.log("scanning fake inputs")
        return SimpleNamespace(
            voices=["01.wav"], scenes=["01.mp4", "02.png"], sfx=[], music="",
            issues=[
                Issue(Severity.WARNING, "orphan_scene", "Scene 02.png has no voice-over."),
                Issue(Severity.ERROR, "missing_scene", "Voice-over 03.wav has no scene."),
            ],
        )

    def plan(self):
        while not self.cancel_event.is_set():
            time.sleep(0.01)
        raise RenderCancelled("Rendering was cancelled by the user.")


def sample_timeline() -> Timeline:
    scenes = [NumberedFile(f"C:/scenes/0{i}.mp4", i, 0, f"0{i}", MediaKind.VIDEO) for i in (1, 2)]
    entries = [
        TimelineEntry(0, scenes[0], 0.0, 3.0, [1], end_cut=CutStyle.TRANSITION),
        TimelineEntry(1, scenes[1], 3.0, 5.5, [2]),
    ]
    transition = TransitionEvent(0, 3.0, 0.5, "Crossfade", "fade", "C:/sfx/whoosh.wav", -10.0, 2.75)
    return Timeline(entries=entries, transitions=[transition], duration=5.5, fps=60)


def customized_settings() -> ProjectSettings:
    settings = ProjectSettings()
    settings.paths.voice_dir = "C:/project/voice"
    settings.paths.scenes_dir = "C:/project/scenes"
    settings.paths.sfx_dir = "C:/project/sfx"
    settings.paths.music_path = "C:/project/music.mp3"
    settings.paths.output_dir = "C:/project/out"
    audio = settings.audio
    audio.silence_threshold_db = -42.5
    audio.min_silence_s = 0.65
    audio.target_silence_s = 0.25
    audio.fade_ms = 6.5
    audio.clip_edge_silence_s = 0.2
    audio.speech_pad_before_ms = 80.0
    audio.speech_pad_after_ms = 160.0
    audio.speech_aware = False
    audio.voice_gain_db = 1.5
    audio.sfx_volume_db = -12.5
    audio.music_volume_db = -20.0
    audio.music_ducking = False
    audio.limiter_ceiling_db = -1.5
    audio.highpass_hz = 100.0
    video = settings.video
    video.motion_mode = MOTION_SMOOTH
    video.source_fps_hint = 15.0
    video.fit_mode = "contain_blur"
    video.scale_filter = "lanczos"
    video.image_motion = "pan"
    video.image_motion_strength = 0.1
    video.extend_mode = "pingpong"
    video.trim_mode = "center"
    video.width = 720
    settings.transitions.library_path = "C:/project/library.json"
    settings.transitions.selection = "sequential"
    settings.transitions.seed = 77
    settings.transitions.duration_s = 0.6
    export = settings.export
    export.encoder = "libx264"
    export.quality = 20
    export.preset = "slow"
    export.audio_bitrate = "256k"
    export.preview_scale = 0.35
    export.preview_max_seconds = 30.0
    export.keep_temp = True
    export.ffmpeg_path = "C:/tools/ffmpeg.exe"
    return settings


@pytest.fixture
def window(qapp, tmp_path):
    FakePipeline.calls = []
    FakePipeline.worker_threads = []
    main_window = MainWindow(settings_path=tmp_path / "settings.json", pipeline_factory=FakePipeline)
    yield main_window
    main_window.close()
    main_window.deleteLater()
    qapp.processEvents()


def test_main_window_constructs_with_required_controls(window):
    assert window.windowTitle() == "Dima Auto Editor"
    assert window.build_button.text() == "Build Video"
    assert window.build_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert window.preview_panel.voice_button.text() == "Preview voice-over"
    assert window.preview_panel.render_button.text() == "Preview render"
    tab_titles = [window.settings_panel.tabs.tabText(i) for i in range(window.settings_panel.tabs.count())]
    assert tab_titles == ["Inputs", "Audio", "Video", "Transitions", "Export"]


def test_settings_round_trip_through_widgets(qapp):
    panel = SettingsPanel()
    settings = customized_settings()
    panel.apply_settings(settings)
    assert panel.collect_settings().to_dict() == settings.to_dict()


def test_widget_edits_reach_project_settings(qapp):
    panel = SettingsPanel()
    changes: list[bool] = []
    panel.changed.connect(lambda: changes.append(True))
    panel.voice_dir.setText("D:/voices")
    panel.motion_buttons[MOTION_SMOOTH].setChecked(True)
    collected = panel.collect_settings()
    assert collected.paths.voice_dir == "D:/voices"
    assert collected.video.motion_mode == MOTION_SMOOTH
    assert changes


def test_unknown_option_values_are_preserved(qapp):
    panel = SettingsPanel()
    settings = ProjectSettings()
    settings.video.fit_mode = "future_fit_mode"
    panel.apply_settings(settings)
    assert panel.collect_settings().video.fit_mode == "future_fit_mode"


def test_motion_warning_visible_only_for_smooth_with_neighbor(qapp):
    panel = SettingsPanel()
    settings = ProjectSettings()
    settings.video.motion_mode = MOTION_CHOPPY
    settings.video.scale_filter = "neighbor"
    panel.apply_settings(settings)
    assert panel.motion_warning_label.isHidden()

    panel.motion_buttons[MOTION_SMOOTH].setChecked(True)
    assert not panel.motion_warning_label.isHidden()

    panel.scale_filter.setCurrentIndex(panel.scale_filter.findData("lanczos"))
    assert panel.motion_warning_label.isHidden()

    panel.scale_filter.setCurrentIndex(panel.scale_filter.findData("neighbor"))
    panel.motion_buttons[MOTION_CHOPPY].setChecked(True)
    assert panel.motion_warning_label.isHidden()


@pytest.fixture
def sfx_dir(tmp_path, make_tone_wav) -> Path:
    folder = tmp_path / "sfx"
    make_tone_wav(folder / "whoosh_01.wav", seconds=0.1)
    make_tone_wav(folder / "impact.wav", seconds=0.1)
    return folder


def test_transition_editor_round_trips_library(qapp, sfx_dir, tmp_path):
    library = [
        TransitionDef("Crossfade", "fade", 0.5, [str(sfx_dir / "whoosh_01.wav")], -9.5, 0.1, True),
        TransitionDef("Legacy", "not_a_real_xfade", 0.8, [], -12.0, -0.2, False),
    ]
    dialog = TransitionEditorDialog(library, str(sfx_dir), "sequential", 42)
    assert dialog.library() == library
    assert (dialog.selection(), dialog.seed()) == ("sequential", 42)
    assert dialog.issues_list.count() >= 1

    target = tmp_path / "library.json"
    assert dialog.save_file(str(target))
    reloaded = TransitionEditorDialog([], str(sfx_dir))
    assert reloaded.load_file(str(target))
    assert reloaded.library() == library
    assert reloaded.library_path() == str(target)


def test_transition_editor_row_operations(qapp, sfx_dir):
    dialog = TransitionEditorDialog([TransitionDef("Crossfade", "fade")], str(sfx_dir))
    dialog.table.selectRow(0)
    dialog._duplicate_transition()
    dialog._add_transition()
    names = [transition.name for transition in dialog.library()]
    assert names == ["Crossfade", "Crossfade copy", "New transition"]
    assert dialog.table.rowCount() == 3

    dialog.table.item(0, 1).setText("Renamed")
    dialog.table.cellWidget(0, 3).setValue(1.25)
    assert dialog.library()[0].name == "Renamed"
    assert dialog.library()[0].duration_s == pytest.approx(1.25)

    dialog.table.selectRow(1)
    dialog._remove_transition()
    assert [transition.name for transition in dialog.library()] == ["Renamed", "New transition"]


def test_settings_persist_and_restore(qapp, tmp_path):
    path = tmp_path / "persist" / "settings.json"
    first = MainWindow(settings_path=path, pipeline_factory=FakePipeline)
    first.settings_panel.apply_settings(customized_settings())
    first.close()
    assert path.is_file()
    second = MainWindow(settings_path=path, pipeline_factory=FakePipeline)
    assert second.current_settings().to_dict() == customized_settings().to_dict()
    second.close()


def test_corrupt_saved_settings_fall_back_to_defaults(qapp, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    first = MainWindow(settings_path=broken, pipeline_factory=FakePipeline)
    assert first.current_settings().to_dict() == ProjectSettings().to_dict()
    assert "Could not read saved settings" in first.log_view.toPlainText()
    first.close()

    wrong_type = tmp_path / "wrong_type.json"
    wrong_type.write_text('{"audio": {"min_silence_s": "abc"}}', encoding="utf-8")
    second = MainWindow(settings_path=wrong_type, pipeline_factory=FakePipeline)
    assert second.current_settings().audio.min_silence_s == ProjectSettings().audio.min_silence_s
    assert "audio.min_silence_s" in second.log_view.toPlainText()
    second.close()


def test_scan_runs_in_worker_thread_and_lists_issues(qapp, window):
    window.settings_panel.apply_settings(customized_settings())
    window.scan_inputs()
    assert window.is_busy()
    assert window.cancel_button.isEnabled()
    wait_until(qapp, lambda: not window.is_busy())
    assert FakePipeline.worker_threads and FakePipeline.worker_threads[0] is not threading.main_thread()
    assert window.issues_list.count() == 2
    assert window.issues_list.item(0).text().startswith("[ERROR]")
    assert "scanning fake inputs" in window.log_view.toPlainText()
    assert window.build_button.isEnabled()


def test_cancel_stops_running_task(qapp, window):
    window.settings_panel.apply_settings(customized_settings())
    window.plan_timeline()
    assert window.is_busy()
    window.cancel_task()
    wait_until(qapp, lambda: not window.is_busy())
    assert window.stage_label.text() == "Cancelled."


def test_build_blocked_lists_issues(qapp, tmp_path, non_blocking_dialogs):
    pipeline_module = pytest.importorskip("dima_auto_editor.pipeline")
    blocking_issue = Issue(Severity.ERROR, "missing_scene", "Voice-over 04.wav has no scene.")

    class BlockedPipeline(FakePipeline):
        def build(self, preview: bool):
            error = pipeline_module.BuildBlocked.__new__(pipeline_module.BuildBlocked)
            Exception.__init__(error, "Build blocked")
            error.issues = [blocking_issue]
            raise error

    main_window = MainWindow(settings_path=tmp_path / "settings.json", pipeline_factory=BlockedPipeline)
    main_window.settings_panel.apply_settings(customized_settings())
    main_window.build_video()
    wait_until(qapp, lambda: not main_window.is_busy())
    assert main_window.issues_list.issues() == [blocking_issue]
    assert any(kind == "warning" for kind, _ in non_blocking_dialogs)
    main_window.close()


def test_missing_folders_prevent_task(window, non_blocking_dialogs):
    window.build_video()
    assert not window.is_busy()
    assert any("Voice-over folder" in text for _, text in non_blocking_dialogs)


def test_timeline_table_shows_summary_rows(window):
    window.timeline_table.show_timeline(sample_timeline())
    table = window.timeline_table
    assert table.rowCount() == 2
    assert table.item(0, 1).text() == "01.mp4"
    assert "Crossfade" in table.item(0, 5).text()
    assert "whoosh.wav" in table.item(0, 6).text()


def test_eta_estimator_uses_overall_fraction():
    now = [0.0]
    eta = EtaEstimator(clock=lambda: now[0])
    assert eta.remaining(0.0) is None
    now[0] = 10.0
    assert eta.remaining(0.25) == pytest.approx(30.0)
    eta.reset()
    assert eta.remaining(0.5) is None
    assert format_seconds(75) == "1:15"
    assert format_seconds(3725) == "1:02:05"
