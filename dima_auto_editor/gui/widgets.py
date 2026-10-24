from __future__ import annotations

from enum import Enum
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)


class PathMode(str, Enum):
    DIRECTORY = "directory"
    FILE = "file"
    FILE_OR_DIRECTORY = "file_or_directory"
    SAVE_FILE = "save_file"


class PathSelector(QWidget):
    changed = Signal(str)

    def __init__(self, mode: PathMode, file_filter: str = "All files (*)", placeholder: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._mode = mode
        self._file_filter = file_filter
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.setClearButtonEnabled(True)
        self.line_edit.textChanged.connect(self.changed.emit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit, 1)
        if mode in (PathMode.FILE, PathMode.FILE_OR_DIRECTORY, PathMode.SAVE_FILE):
            layout.addWidget(self._button("File…", self._browse_file))
        if mode in (PathMode.DIRECTORY, PathMode.FILE_OR_DIRECTORY):
            layout.addWidget(self._button("Folder…", self._browse_directory))

    def _button(self, text: str, handler) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(handler)
        return button

    def text(self) -> str:
        return self.line_edit.text().strip()

    def setText(self, value: str) -> None:
        self.line_edit.setText(value or "")

    def _start_directory(self) -> str:
        current = Path(self.text()) if self.text() else None
        if current is None:
            return ""
        return str(current if current.is_dir() else current.parent)

    def _browse_directory(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose folder", self._start_directory())
        if chosen:
            self.setText(str(Path(chosen)))

    def _browse_file(self) -> None:
        if self._mode is PathMode.SAVE_FILE:
            chosen, _ = QFileDialog.getSaveFileName(self, "Choose file", self._start_directory(), self._file_filter)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "Choose file", self._start_directory(), self._file_filter)
        if chosen:
            self.setText(str(Path(chosen)))


class _FocusedWheelOnly:
    """Lets the surrounding scroll area scroll instead of silently changing an unfocused value."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SpinBox(_FocusedWheelOnly, QSpinBox):
    pass


class DoubleSpinBox(_FocusedWheelOnly, QDoubleSpinBox):
    pass


class ComboBox(_FocusedWheelOnly, QComboBox):
    pass
