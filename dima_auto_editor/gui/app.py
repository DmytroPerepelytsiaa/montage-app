from __future__ import annotations

import sys
from typing import Sequence

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .. import APP_NAME, __version__
from .main_window import MainWindow
from .theme import apply_dark_theme


def run_app(argv: Sequence[str] | None = None, quit_after_s: float | None = None) -> int:
    app = QApplication.instance() or QApplication(list(argv if argv is not None else sys.argv))
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    if quit_after_s is None:
        QTimer.singleShot(0, window.check_ffmpeg)
    else:
        QTimer.singleShot(int(quit_after_s * 1000), window.close)
    return app.exec()
