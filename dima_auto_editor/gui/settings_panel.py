from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import MOTION_CHOPPY, MOTION_SMOOTH, ProjectSettings
from ..render.rife import ENGINE_AUTO, ENGINE_MINTERPOLATE, ENGINE_RIFE, find_rife
from ..render.stylized_transitions import STYLE_LABELS
from .theme import WARNING_LABEL_STYLE
from .widgets import ComboBox, DoubleSpinBox, PathMode, PathSelector, SpinBox

Options = Sequence[tuple[str, Any]]

MOTION_MODES: Options = (
    ("Choppy 60 FPS: keep original frames (frame repeat)", MOTION_CHOPPY),
    ("Smooth 60 FPS: generate frames for every clip", MOTION_SMOOTH),
)
INTERPOLATION_ENGINES: Options = (
    ("Auto: RIFE if installed, otherwise FFmpeg", ENGINE_AUTO),
    ("RIFE neural optical flow (GPU, best)", ENGINE_RIFE),
    ("FFmpeg minterpolate (CPU, slow)", ENGINE_MINTERPOLATE),
)
FIT_MODES: Options = (
    ("Cover (scale and crop to fill)", "cover"),
    ("Contain on blurred background", "contain_blur"),
    ("Contain on black bars", "contain_black"),
)
SCALE_FILTERS: Options = (
    ("Nearest neighbor (crisp pixel art)", "neighbor"),
    ("Bilinear", "bilinear"),
    ("Bicubic", "bicubic"),
    ("Lanczos (sharp, photo)", "lanczos"),
)
IMAGE_MOTIONS: Options = (
    ("Static hold", "none"),
    ("Slow zoom in", "zoom_in"),
    ("Slow zoom out", "zoom_out"),
    ("Slow pan", "pan"),
)
EXTEND_MODES: Options = (
    ("Slow down to fit", "slow"),
    ("Freeze last frame", "freeze"),
    ("Loop", "loop"),
    ("Ping-pong", "pingpong"),
    ("Slow down with interpolation", "interpolate"),
)
TRIM_MODES: Options = (
    ("Smart (calm cut point, no internal cuts)", "smart"),
    ("Keep the start", "start"),
    ("Keep the center", "center"),
)
ENCODERS: Options = (
    ("Auto (best available)", "auto"),
    ("CPU: libx264", "libx264"),
    ("NVIDIA: h264_nvenc", "h264_nvenc"),
    ("Intel: h264_qsv", "h264_qsv"),
    ("AMD: h264_amf", "h264_amf"),
)
PRESETS: Options = tuple(
    (name, name)
    for name in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow",
                 "p1", "p2", "p3", "p4", "p5", "p6", "p7")
)
AUDIO_BITRATES: Options = tuple((rate, rate) for rate in ("128k", "160k", "192k", "256k", "320k"))

AUDIO_FILTER = "Audio files (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma);;All files (*)"
JSON_FILTER = "JSON (*.json);;All files (*)"
EXECUTABLE_FILTER = "Executable (*.exe);;All files (*)"

SMOOTH_NEIGHBOR_WARNING = (
    "Warning: smooth interpolation with nearest-neighbor scaling may damage pixel-art edges "
    "(blended, wobbling pixels). Use the choppy mode for pixel art."
)


@dataclass(frozen=True)
class FieldBinding:
    section: str
    name: str
    read: Callable[[], Any]
    write: Callable[[Any], None]


class SettingsPanel(QWidget):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._bindings: list[FieldBinding] = []
        self._base = ProjectSettings()
        self._applying = False
        self.tabs = QTabWidget()
        self.tabs.addTab(self._scrollable(self._build_inputs_tab()), "Inputs")
        self.tabs.addTab(self._scrollable(self._build_audio_tab()), "Audio")
        self.tabs.addTab(self._scrollable(self._build_video_tab()), "Video")
        self.tabs.addTab(self._scrollable(self._build_transitions_tab()), "Transitions")
        self.tabs.addTab(self._scrollable(self._build_export_tab()), "Export")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.apply_settings(self._base)

    def apply_settings(self, settings: ProjectSettings) -> list[str]:
        """Shows the settings in the widgets; returns the names of fields whose stored values were invalid."""
        self._base = copy.deepcopy(settings)
        rejected: list[str] = []
        self._applying = True
        try:
            for binding in self._bindings:
                try:
                    binding.write(getattr(getattr(settings, binding.section), binding.name))
                except (TypeError, ValueError, AttributeError):
                    rejected.append(f"{binding.section}.{binding.name}")
        finally:
            self._applying = False
        self._refresh_motion_warning()
        self._refresh_rife_status()
        return rejected

    def collect_settings(self) -> ProjectSettings:
        result = copy.deepcopy(self._base)
        for binding in self._bindings:
            setattr(getattr(result, binding.section), binding.name, binding.read())
        return result

    def _emit_changed(self, *_: Any) -> None:
        if not self._applying:
            self.changed.emit()

    def _bind(self, section: str, name: str, read: Callable[[], Any], write: Callable[[Any], None]) -> None:
        self._bindings.append(FieldBinding(section, name, read, write))

    @staticmethod
    def _scrollable(content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(content)
        return area

    @staticmethod
    def _group(title: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return group, form

    @staticmethod
    def _page(*groups: QWidget) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        for group in groups:
            layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _path(self, section: str, name: str, mode: PathMode, file_filter: str = "All files (*)", placeholder: str = "") -> PathSelector:
        selector = PathSelector(mode, file_filter, placeholder)
        selector.changed.connect(self._emit_changed)
        self._bind(section, name, selector.text, selector.setText)
        return selector

    def _double(self, section: str, name: str, minimum: float, maximum: float, step: float, decimals: int,
                suffix: str = "", tooltip: str = "") -> DoubleSpinBox:
        spin = DoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(self._emit_changed)
        self._bind(section, name, spin.value, lambda value: spin.setValue(float(value)))
        return spin

    def _int(self, section: str, name: str, minimum: int, maximum: int, tooltip: str = "") -> SpinBox:
        spin = SpinBox()
        spin.setRange(minimum, maximum)
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(self._emit_changed)
        self._bind(section, name, spin.value, lambda value: spin.setValue(int(value)))
        return spin

    def _check(self, section: str, name: str, text: str, tooltip: str = "") -> QCheckBox:
        box = QCheckBox(text)
        box.setToolTip(tooltip)
        box.toggled.connect(self._emit_changed)
        self._bind(section, name, box.isChecked, lambda value: box.setChecked(bool(value)))
        return box

    def _combo(self, section: str, name: str, options: Options, tooltip: str = "") -> ComboBox:
        combo = ComboBox()
        combo.setToolTip(tooltip)
        for label, value in options:
            combo.addItem(label, value)

        def write(value: Any) -> None:
            index = combo.findData(value)
            if index < 0:
                combo.addItem(str(value), value)
                index = combo.count() - 1
            combo.setCurrentIndex(index)

        combo.currentIndexChanged.connect(self._emit_changed)
        self._bind(section, name, combo.currentData, write)
        return combo

    def _multi_check(self, section: str, name: str, options: Options) -> tuple[QWidget, dict[Any, QCheckBox]]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        boxes: dict[Any, QCheckBox] = {}
        for label, value in options:
            box = QCheckBox(label)
            box.toggled.connect(self._emit_changed)
            layout.addWidget(box)
            boxes[value] = box

        def read() -> list[Any]:
            return [value for value, box in boxes.items() if box.isChecked()]

        def write(values: Any) -> None:
            for value, box in boxes.items():
                box.setChecked(value in values)

        self._bind(section, name, read, write)
        return container, boxes

    def _radios(self, section: str, name: str, options: Options) -> tuple[QWidget, dict[Any, QRadioButton]]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        group = QButtonGroup(container)
        buttons: dict[Any, QRadioButton] = {}
        for label, value in options:
            button = QRadioButton(label)
            group.addButton(button)
            layout.addWidget(button)
            buttons[value] = button
            button.toggled.connect(self._emit_changed)
        fallback = options[0][1]

        def read() -> Any:
            return next((value for value, button in buttons.items() if button.isChecked()), fallback)

        def write(value: Any) -> None:
            buttons.get(value, buttons[fallback]).setChecked(True)

        self._bind(section, name, read, write)
        return container, buttons

    def _build_inputs_tab(self) -> QWidget:
        group, form = self._group("Input and output folders")
        self.voice_dir = self._path("paths", "voice_dir", PathMode.DIRECTORY, placeholder="Folder with 01.wav, 02.wav…")
        self.scenes_dir = self._path("paths", "scenes_dir", PathMode.DIRECTORY, placeholder="Folder with 01.mp4, 02.png, 03_1.mp4…")
        self.music_path = self._path("paths", "music_path", PathMode.FILE_OR_DIRECTORY, AUDIO_FILTER, "Optional background music (file or folder)")
        self.output_dir = self._path("paths", "output_dir", PathMode.DIRECTORY, placeholder="Where the video, manifest and logs are written")
        form.addRow("Voice-over folder", self.voice_dir)
        form.addRow("Scenes folder", self.scenes_dir)
        form.addRow("Background music", self.music_path)
        form.addRow("Output folder", self.output_dir)
        hint = QLabel(
            "Scenes and voice-overs are paired by number: 01.wav ↔ 01.mp4 / 01.png. "
            "Several scenes for one voice-over (03_1.mp4, 03_2.mp4…) share its time equally; each clip is "
            "slowed down or trimmed to fit. Scenes change with hard cuts. Voice-overs and scenes may be in "
            "the same folder. Source folders are never modified."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        return self._page(group)

    def _build_audio_tab(self) -> QWidget:
        silence, silence_form = self._group("Pause removal")
        silence_form.addRow("Silence threshold", self._double(
            "audio", "silence_threshold_db", -80, -10, 1, 1, " dB", "Levels below this count as silence."))
        silence_form.addRow("Minimum silence", self._double(
            "audio", "min_silence_s", 0.05, 5, 0.05, 2, " s", "Only pauses at least this long are shortened."))
        silence_form.addRow("Target pause", self._double(
            "audio", "target_silence_s", 0.0, 3, 0.05, 2, " s", "Length a long pause is shortened to."))
        silence_form.addRow("Fade at cuts", self._double(
            "audio", "fade_ms", 0, 50, 1, 1, " ms", "Short fade at each audio cut to avoid clicks."))
        silence_form.addRow("Clip edge silence", self._double(
            "audio", "clip_edge_silence_s", 0, 2, 0.05, 2, " s", "Silence kept at the start and end of each voice file."))
        silence_form.addRow("Pad before speech", self._double(
            "audio", "speech_pad_before_ms", 0, 500, 10, 0, " ms", "Protects soft word onsets."))
        silence_form.addRow("Pad after speech", self._double(
            "audio", "speech_pad_after_ms", 0, 1000, 10, 0, " ms", "Protects word endings and natural decay."))
        silence_form.addRow(self._check(
            "audio", "speech_aware", "Speech-aware detection (high-pass, hangover, protected words)",
            "Never cut inside detected speech; more robust against breaths and room noise."))

        mix, mix_form = self._group("Mix")
        mix_form.addRow("Voice volume", self._double("audio", "voice_gain_db", -24, 24, 0.5, 1, " dB"))
        mix_form.addRow("SFX volume", self._double(
            "audio", "sfx_volume_db", -60, 12, 0.5, 1, " dB", "Global SFX level; the narration always stays louder."))
        mix_form.addRow("Music volume", self._double("audio", "music_volume_db", -60, 0, 0.5, 1, " dB"))
        mix_form.addRow(self._check("audio", "music_ducking", "Duck music under speech"))
        mix_form.addRow("Limiter ceiling", self._double(
            "audio", "limiter_ceiling_db", -12, 0, 0.1, 1, " dB", "No sample in the final mix exceeds this level."))
        return self._page(silence, mix)

    def _build_video_tab(self) -> QWidget:
        motion, motion_form = self._group("Motion mode")
        radios, self.motion_buttons = self._radios("video", "motion_mode", MOTION_MODES)
        motion_form.addRow(radios)
        self.motion_warning_label = QLabel(SMOOTH_NEIGHBOR_WARNING)
        self.motion_warning_label.setWordWrap(True)
        self.motion_warning_label.setStyleSheet(WARNING_LABEL_STYLE)
        motion_form.addRow(self.motion_warning_label)
        motion_form.addRow("Source FPS hint", self._double(
            "video", "source_fps_hint", 1, 120, 1, 1, " fps", "Used when a clip does not report its frame rate."))

        framing, framing_form = self._group("Framing (1080×1920 vertical)")
        framing_form.addRow("Fit mode", self._combo("video", "fit_mode", FIT_MODES))
        self.scale_filter = self._combo("video", "scale_filter", SCALE_FILTERS)
        framing_form.addRow("Scaling filter", self.scale_filter)
        framing_form.addRow("Image motion", self._combo("video", "image_motion", IMAGE_MOTIONS))
        framing_form.addRow("Image motion strength", self._double("video", "image_motion_strength", 0, 0.5, 0.01, 2))

        timing, timing_form = self._group("Fitting clips to narration")
        timing_form.addRow("Clip too short", self._combo("video", "extend_mode", EXTEND_MODES))
        timing_form.addRow("Clip too long", self._combo("video", "trim_mode", TRIM_MODES))

        frames, frames_form = self._group("Frame generation (optical flow)")
        frames_form.addRow(self._check(
            "video", "interpolate_slowed_clips", "Generate in-between frames for slowed-down clips",
            "Slowed clips get new frames so they play as smooth 60 FPS instead of repeating frames."))
        frames_form.addRow("Engine", self._combo("video", "interpolation_engine", INTERPOLATION_ENGINES))
        self.rife_path = self._path("video", "rife_executable", PathMode.FILE, EXECUTABLE_FILTER,
                                    "Empty = tools\\rife\\rife-ncnn-vulkan.exe")
        frames_form.addRow("RIFE executable", self.rife_path)
        self.rife_status_label = QLabel("")
        self.rife_status_label.setWordWrap(True)
        frames_form.addRow(self.rife_status_label)

        for button in self.motion_buttons.values():
            button.toggled.connect(self._refresh_motion_warning)
        self.scale_filter.currentIndexChanged.connect(self._refresh_motion_warning)
        self.changed.connect(self._refresh_rife_status)
        return self._page(motion, framing, timing, frames)

    def _refresh_rife_status(self) -> None:
        rife = find_rife(self.collect_settings())
        self.rife_status_label.setText(f"RIFE found: {rife}" if rife else
                                       "RIFE not found: FFmpeg minterpolate (slower, lower quality) will be used. "
                                       "Run install.ps1 to download RIFE.")

    def _refresh_motion_warning(self, *_: Any) -> None:
        smooth = self.motion_buttons[MOTION_SMOOTH].isChecked()
        neighbor = self.scale_filter.currentData() == "neighbor"
        self.motion_warning_label.setHidden(not (smooth and neighbor))

    def _build_transitions_tab(self) -> QWidget:
        group, form = self._group("Transitions between voice-overs")
        self.transitions_enabled = self._check(
            "transitions", "enabled", "Add a transition where one voice-over ends and the next begins (02 → 03)",
            "Off: every scene change is a hard cut. Scenes inside one voice-over always change with a hard cut.")
        form.addRow(self.transitions_enabled)
        styles, self.transition_style_boxes = self._multi_check("transitions", "styles", tuple(
            (label, style) for style, label in STYLE_LABELS.items()))
        form.addRow("Use", styles)
        form.addRow("Duration", self._double("transitions", "duration_s", 0.2, 3.0, 0.05, 2, " s"))
        form.addRow("Sound effects", self._path(
            "paths", "sfx_dir", PathMode.DIRECTORY, placeholder="Optional: folder with transition sounds"))
        note = QLabel("Transitions are picked at random and the same one never plays twice in a row. "
                      "With a sound folder, a random sound plays at each transition (also never twice in a row).")
        note.setWordWrap(True)
        form.addRow(note)
        return self._page(group)

    def _build_export_tab(self) -> QWidget:
        video, video_form = self._group("Video encoding")
        video_form.addRow("Encoder", self._combo("export", "encoder", ENCODERS))
        video_form.addRow("Quality (CRF/CQ)", self._int("export", "quality", 0, 51, "Lower is better quality and bigger files."))
        preset = self._combo("export", "preset", PRESETS, "x264: ultrafast…veryslow; NVENC: p1…p7.")
        video_form.addRow("Preset", preset)

        audio, audio_form = self._group("Audio encoding")
        audio_form.addRow("AAC bitrate", self._combo("export", "audio_bitrate", AUDIO_BITRATES))

        other, other_form = self._group("Preview and files")
        other_form.addRow("Preview scale", self._double("export", "preview_scale", 0.1, 1, 0.05, 2, "×"))
        other_form.addRow("Preview length limit", self._double(
            "export", "preview_max_seconds", 0, 3600, 5, 0, " s", "0 renders the full length."))
        other_form.addRow(self._check("export", "keep_temp", "Keep temporary files (for debugging)"))
        other_form.addRow("FFmpeg executable", self._path(
            "export", "ffmpeg_path", PathMode.FILE, EXECUTABLE_FILTER, "Empty = auto-detect (bundled FFmpeg works)"))
        return self._page(video, audio, other)
