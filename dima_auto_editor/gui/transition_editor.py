from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import Severity, TransitionDef
from ..naming import AUDIO_EXTENSIONS, natural_sorted
from ..transitions import (
    SELECTION_MODES,
    XFADE_TRANSITIONS,
    default_library,
    load_library,
    resolve_sfx_path,
    save_library,
    validate_library,
)
from .media import AudioClipPlayer
from .results_view import IssuesList, issue_counts
from .widgets import ComboBox, DoubleSpinBox, SpinBox

JSON_FILTER = "Transition library (*.json);;All files (*)"
COLUMN_ENABLED, COLUMN_NAME, COLUMN_XFADE, COLUMN_DURATION, COLUMN_SFX, COLUMN_VOLUME, COLUMN_OFFSET, COLUMN_PLAY = range(8)
HEADERS = ("On", "Name", "Transition type", "Duration", "Sound effects", "SFX volume", "SFX offset", "")


def list_sfx_files(sfx_dir: str) -> list[str]:
    folder = Path(sfx_dir) if sfx_dir else None
    if folder is None or not folder.is_dir():
        return []
    return [str(path) for path in natural_sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)]


def describe_sfx(files: list[str]) -> str:
    if not files:
        return "No sound"
    names = ", ".join(Path(entry).name for entry in files[:2])
    more = f" +{len(files) - 2}" if len(files) > 2 else ""
    return f"{names}{more}"


class SfxPickerDialog(QDialog):
    def __init__(self, available: list[str], selected: list[str], player: AudioClipPlayer, gain_db: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Choose sound effects")
        self.resize(460, 480)
        self._player = player
        self._gain_db = gain_db
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._play_item)
        claimed: set[str] = set()
        for entry in selected:
            resolved = resolve_sfx_path(entry, available)
            if resolved is None:
                self._add_item(entry, f"{Path(entry).name} (missing)", checked=True)
            else:
                claimed.add(resolved)
        for path in available:
            self._add_item(path, Path(path).name, checked=path in claimed)

        hint = QLabel("Tick the sounds this transition may use (one is picked per transition). Double-click to listen.")
        hint.setWordWrap(True)
        if not available:
            hint.setText("The SFX folder is empty or not set. Choose it on the Inputs tab first.")
        play_button = QPushButton("Play selected")
        play_button.clicked.connect(lambda: self._play_item(self.list_widget.currentItem()))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.list_widget, 1)
        layout.addWidget(play_button)
        layout.addWidget(buttons)

    def _add_item(self, entry: str, label: str, checked: bool) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, entry)
        item.setToolTip(entry)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.list_widget.addItem(item)

    def _play_item(self, item: QListWidgetItem | None) -> None:
        if item is not None and Path(item.data(Qt.ItemDataRole.UserRole)).is_file():
            self._player.play(item.data(Qt.ItemDataRole.UserRole), self._gain_db)

    def selected_files(self) -> list[str]:
        items = (self.list_widget.item(row) for row in range(self.list_widget.count()))
        return [item.data(Qt.ItemDataRole.UserRole) for item in items if item.checkState() == Qt.CheckState.Checked]


class _SelectRowOnFocus(QObject):
    def __init__(self, table: QTableWidget):
        super().__init__(table)
        self._table = table

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.FocusIn and isinstance(watched, QWidget):
            row = self._table.indexAt(watched.pos()).row()
            if row >= 0:
                self._table.selectRow(row)
        return super().eventFilter(watched, event)


class TransitionEditorDialog(QDialog):
    def __init__(
        self,
        library: list[TransitionDef],
        sfx_dir: str,
        selection: str = "random",
        seed: int = 0,
        library_path: str = "",
        parent: QWidget | None = None,
        default_duration_s: float | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Transition & SFX library")
        self.resize(1080, 640)
        self._library = [copy.deepcopy(transition) for transition in library]
        self._default_duration_s = default_duration_s
        self._sfx_files = list_sfx_files(sfx_dir)
        self._library_path = library_path
        self._play_cursors: dict[int, int] = {}
        self._player: AudioClipPlayer | None = None
        self._populating = False

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMN_SFX, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self._row_focus = _SelectRowOnFocus(self.table)

        self.selection_combo = ComboBox()
        for mode in SELECTION_MODES:
            self.selection_combo.addItem(mode.capitalize(), mode)
        self.selection_combo.setCurrentIndex(max(0, self.selection_combo.findData(selection)))
        self.seed_spin = SpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(seed)

        self.issues_list = IssuesList()
        self.issues_list.setMaximumHeight(130)
        self.validation_label = QLabel()
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"SFX folder: {sfx_dir or '(not set)'} — {len(self._sfx_files)} sound file(s)"))
        layout.addWidget(self.table, 1)
        layout.addLayout(self._build_row_buttons())
        options = QHBoxLayout()
        options.addWidget(QLabel("Selection mode"))
        options.addWidget(self.selection_combo)
        options.addSpacing(16)
        options.addWidget(QLabel("Seed"))
        options.addWidget(self.seed_spin)
        options.addStretch(1)
        layout.addLayout(options)
        layout.addWidget(self.validation_label)
        layout.addWidget(self.issues_list)
        layout.addWidget(self.path_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate()

    def library(self) -> list[TransitionDef]:
        return [copy.deepcopy(transition) for transition in self._library]

    def selection(self) -> str:
        return self.selection_combo.currentData()

    def seed(self) -> int:
        return self.seed_spin.value()

    def library_path(self) -> str:
        return self._library_path

    def _build_row_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        for text, handler in (
            ("Add", self._add_transition),
            ("Duplicate", self._duplicate_transition),
            ("Remove", self._remove_transition),
            ("Move up", lambda: self._move_transition(-1)),
            ("Move down", lambda: self._move_transition(1)),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            row.addWidget(button)
        row.addStretch(1)
        for text, handler in (
            ("Reset to defaults", self._reset_to_defaults),
            ("Load JSON…", self._load_from_file),
            ("Save JSON…", self._save_to_file),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            row.addWidget(button)
        return row

    def _populate(self, select_row: int | None = None) -> None:
        self._populating = True
        try:
            self.table.setRowCount(0)
            self.table.setRowCount(len(self._library))
            for row, transition in enumerate(self._library):
                self._fill_row(row, transition)
        finally:
            self._populating = False
        if select_row is not None and 0 <= select_row < len(self._library):
            self.table.selectRow(select_row)
        self._refresh_validation()

    def _fill_row(self, row: int, transition: TransitionDef) -> None:
        enabled_item = QTableWidgetItem()
        enabled_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
        enabled_item.setCheckState(Qt.CheckState.Checked if transition.enabled else Qt.CheckState.Unchecked)
        self.table.setItem(row, COLUMN_ENABLED, enabled_item)
        self.table.setItem(row, COLUMN_NAME, QTableWidgetItem(transition.name))

        xfade_combo = ComboBox()
        for label, xfade in XFADE_TRANSITIONS.items():
            xfade_combo.addItem(f"{label} ({xfade})", xfade)
        if xfade_combo.findData(transition.xfade) < 0:
            xfade_combo.addItem(f"Unknown: {transition.xfade}", transition.xfade)
        xfade_combo.setCurrentIndex(xfade_combo.findData(transition.xfade))
        xfade_combo.currentIndexChanged.connect(lambda _: self._update(transition, xfade=xfade_combo.currentData()))
        self._place(row, COLUMN_XFADE, xfade_combo)

        duration = self._spin(transition.duration_s, 0.04, 5.0, 0.05, " s")
        duration.valueChanged.connect(lambda value: self._update(transition, duration_s=value))
        self._place(row, COLUMN_DURATION, duration)

        sfx_button = QPushButton(describe_sfx(transition.sfx_files))
        sfx_button.setToolTip("\n".join(transition.sfx_files) or "Click to choose sound effects")
        sfx_button.clicked.connect(lambda: self._choose_sfx(transition, sfx_button))
        self._place(row, COLUMN_SFX, sfx_button)

        volume = self._spin(transition.sfx_volume_db, -60.0, 12.0, 0.5, " dB")
        volume.valueChanged.connect(lambda value: self._update(transition, sfx_volume_db=value))
        self._place(row, COLUMN_VOLUME, volume)

        offset = self._spin(transition.sfx_offset_s, -3.0, 3.0, 0.05, " s")
        offset.setToolTip("Shift of the sound relative to the start of the transition.")
        offset.valueChanged.connect(lambda value: self._update(transition, sfx_offset_s=value))
        self._place(row, COLUMN_OFFSET, offset)

        play_button = QPushButton("Play")
        play_button.setToolTip("Play this transition's sound effects one after another.")
        play_button.clicked.connect(lambda: self._play_next_sfx(transition))
        self._place(row, COLUMN_PLAY, play_button)

    @staticmethod
    def _spin(value: float, minimum: float, maximum: float, step: float, suffix: str) -> DoubleSpinBox:
        spin = DoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(value)
        return spin

    def _place(self, row: int, column: int, widget: QWidget) -> None:
        widget.installEventFilter(self._row_focus)
        self.table.setCellWidget(row, column, widget)

    def _update(self, transition: TransitionDef, **changes) -> None:
        for field_name, value in changes.items():
            setattr(transition, field_name, value)
        self._refresh_validation()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or not 0 <= item.row() < len(self._library):
            return
        transition = self._library[item.row()]
        if item.column() == COLUMN_NAME:
            self._update(transition, name=item.text().strip())
        elif item.column() == COLUMN_ENABLED:
            self._update(transition, enabled=item.checkState() == Qt.CheckState.Checked)

    def _refresh_validation(self) -> None:
        issues = validate_library(self._library, self._sfx_files)
        self.issues_list.set_issues(issues)
        counts = issue_counts(issues)
        enabled = sum(transition.enabled for transition in self._library)
        status = f"{len(self._library)} transitions, {enabled} enabled."
        if counts[Severity.ERROR] or counts[Severity.WARNING]:
            status += f" {counts[Severity.ERROR]} error(s), {counts[Severity.WARNING]} warning(s)."
        else:
            status += " Library is valid."
        self.validation_label.setText(status)
        self.path_label.setText(f"Library file: {self._library_path or '(built-in defaults, not saved yet)'}")

    def _add_transition(self) -> None:
        new_transition = TransitionDef(name=self._unique_name("New transition"), sfx_files=self._sfx_files[:1])
        if self._default_duration_s:
            new_transition.duration_s = self._default_duration_s
        self._library.append(new_transition)
        self._populate(len(self._library) - 1)

    def _duplicate_transition(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._library):
            return
        original = self._library[row]
        clone = replace(original, name=self._unique_name(f"{original.name} copy"), sfx_files=list(original.sfx_files))
        self._library.insert(row + 1, clone)
        self._populate(row + 1)

    def _remove_transition(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._library):
            return
        del self._library[row]
        self._populate(min(row, len(self._library) - 1))

    def _move_transition(self, step: int) -> None:
        row = self.table.currentRow()
        target = row + step
        if not (0 <= row < len(self._library) and 0 <= target < len(self._library)):
            return
        self._library[row], self._library[target] = self._library[target], self._library[row]
        self._populate(target)

    def _unique_name(self, base: str) -> str:
        taken = {transition.name.casefold() for transition in self._library}
        candidate, counter = base, 2
        while candidate.casefold() in taken:
            candidate, counter = f"{base} {counter}", counter + 1
        return candidate

    def _reset_to_defaults(self) -> None:
        self._library = default_library(self._sfx_files, self._default_duration_s)
        self._populate(0)

    def _load_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load transition library", self._library_path, JSON_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> bool:
        try:
            self._library = load_library(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot load library", f"Could not read {path}:\n{error}")
            return False
        self._library_path = path
        self._populate(0)
        return True

    def _save_to_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save transition library", self._library_path or "transitions.json", JSON_FILTER)
        if path:
            self.save_file(path)

    def save_file(self, path: str) -> bool:
        try:
            save_library(path, self._library)
        except OSError as error:
            QMessageBox.warning(self, "Cannot save library", f"Could not write {path}:\n{error}")
            return False
        self._library_path = path
        self._refresh_validation()
        return True

    def _audio_player(self) -> AudioClipPlayer:
        if self._player is None:
            self._player = AudioClipPlayer(self)
        return self._player

    def _choose_sfx(self, transition: TransitionDef, button: QPushButton) -> None:
        picker = SfxPickerDialog(self._sfx_files, transition.sfx_files, self._audio_player(), transition.sfx_volume_db, self)
        if picker.exec() == QDialog.DialogCode.Accepted:
            self._update(transition, sfx_files=picker.selected_files())
            button.setText(describe_sfx(transition.sfx_files))
            button.setToolTip("\n".join(transition.sfx_files) or "Click to choose sound effects")

    def _play_next_sfx(self, transition: TransitionDef) -> None:
        playable = [path for path in (resolve_sfx_path(entry, self._sfx_files) for entry in transition.sfx_files) if path]
        if not playable:
            self.validation_label.setText(f"'{transition.name}' has no playable sound effect.")
            return
        cursor = self._play_cursors.get(id(transition), 0)
        self._play_cursors[id(transition)] = cursor + 1
        self._audio_player().play(playable[cursor % len(playable)], transition.sfx_volume_db)

    def done(self, result: int) -> None:
        if self._player is not None:
            self._player.stop()
        super().done(result)
