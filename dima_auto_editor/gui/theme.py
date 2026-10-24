from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from ..models import Severity

SEVERITY_COLORS: dict[Severity, QColor] = {
    Severity.ERROR: QColor("#ff6b6b"),
    Severity.WARNING: QColor("#ffb84d"),
    Severity.INFO: QColor("#8fb8ff"),
}

WARNING_LABEL_STYLE = "color: #ffb84d; font-weight: 600;"
BUILD_BUTTON_STYLE = (
    "QPushButton { background-color: #2e7d32; color: white; font-size: 16px; font-weight: 700;"
    " padding: 10px 28px; border-radius: 6px; }"
    "QPushButton:hover { background-color: #388e3c; }"
    "QPushButton:disabled { background-color: #3a3a3a; color: #777; }"
)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    window = QColor(37, 37, 40)
    base = QColor(28, 28, 30)
    text = QColor(225, 225, 228)
    disabled_text = QColor(120, 120, 125)
    accent = QColor(64, 132, 214)
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, window)
    palette.setColor(QPalette.ColorRole.ToolTipBase, base)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, disabled_text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(48, 48, 52))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled_text)
    app.setPalette(palette)
