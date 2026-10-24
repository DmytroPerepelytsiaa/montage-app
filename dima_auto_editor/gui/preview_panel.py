from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .eta import format_seconds
from .media import AudioClipPlayer


def open_with_system_player(path: str) -> None:
    if hasattr(os, "startfile"):
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class PreviewPanel(QWidget):
    voice_preview_requested = Signal()
    render_preview_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._voice_path = ""
        self._video_path = ""
        self._voice_player: AudioClipPlayer | None = None
        self._video_player: QMediaPlayer | None = None

        self.voice_button = QPushButton("Preview voice-over")
        self.voice_button.setToolTip("Process the narration (pause removal) and listen to it.")
        self.voice_button.clicked.connect(self.voice_preview_requested.emit)
        self.voice_play_button = QPushButton("Play")
        self.voice_play_button.clicked.connect(self.play_voice)
        self.voice_stop_button = QPushButton("Stop")
        self.voice_stop_button.clicked.connect(self.stop_voice)
        self.voice_status = QLabel("No processed narration yet.")
        self.voice_status.setWordWrap(True)

        voice_group = QGroupBox("Voice-over")
        voice_layout = QVBoxLayout(voice_group)
        voice_buttons = QHBoxLayout()
        for button in (self.voice_button, self.voice_play_button, self.voice_stop_button):
            voice_buttons.addWidget(button)
        voice_buttons.addStretch(1)
        voice_layout.addLayout(voice_buttons)
        voice_layout.addWidget(self.voice_status)

        self.render_button = QPushButton("Preview render")
        self.render_button.setToolTip("Render a fast low-resolution preview of the whole video.")
        self.render_button.clicked.connect(self.render_preview_requested.emit)
        self.video_play_button = QPushButton("Play / Pause")
        self.video_play_button.clicked.connect(self.toggle_video)
        self.video_stop_button = QPushButton("Stop")
        self.video_stop_button.clicked.connect(self.stop_video)
        self.open_external_button = QPushButton("Open in player")
        self.open_external_button.clicked.connect(self.open_video_externally)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(220)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_status = QLabel("No preview rendered yet.")
        self.video_status.setWordWrap(True)

        video_group = QGroupBox("Video preview")
        video_layout = QVBoxLayout(video_group)
        video_buttons = QHBoxLayout()
        for button in (self.render_button, self.video_play_button, self.video_stop_button, self.open_external_button):
            video_buttons.addWidget(button)
        video_buttons.addStretch(1)
        video_layout.addLayout(video_buttons)
        video_layout.addWidget(self.video_widget, 1)
        video_layout.addWidget(self.video_status)

        layout = QVBoxLayout(self)
        layout.addWidget(voice_group)
        layout.addWidget(video_group, 1)
        self._refresh_buttons()

    def set_busy(self, busy: bool) -> None:
        self.voice_button.setEnabled(not busy)
        self.render_button.setEnabled(not busy)

    def set_voice(self, path: str) -> None:
        self.stop_voice()
        self._voice_path = path
        self.voice_status.setText(f"Processed narration: {path}")
        self._refresh_buttons()

    def set_video(self, path: str) -> None:
        self.stop_video()
        self._video_path = path
        self.video_status.setText(f"Preview: {path}")
        self._refresh_buttons()

    def play_voice(self) -> None:
        if not self._voice_path or not Path(self._voice_path).is_file():
            return
        if self._voice_player is None:
            self._voice_player = AudioClipPlayer(self)
            self._voice_player.player.positionChanged.connect(self._show_voice_position)
        self.stop_video()
        self._voice_player.play(self._voice_path)

    def stop_voice(self) -> None:
        if self._voice_player is not None:
            self._voice_player.stop()

    def _show_voice_position(self, position_ms: int) -> None:
        duration_ms = self._voice_player.player.duration() if self._voice_player else 0
        self.voice_status.setText(
            f"{format_seconds(position_ms / 1000)} / {format_seconds(duration_ms / 1000)} — {Path(self._voice_path).name}"
        )

    def _ensure_video_player(self) -> QMediaPlayer:
        if self._video_player is None:
            self._video_player = QMediaPlayer(self)
            self._video_player.setAudioOutput(QAudioOutput(self._video_player))
            self._video_player.setVideoOutput(self.video_widget)
        return self._video_player

    def toggle_video(self) -> None:
        if not self._video_path:
            return
        player = self._ensure_video_player()
        if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            player.pause()
            return
        if player.source() != QUrl.fromLocalFile(self._video_path):
            player.setSource(QUrl.fromLocalFile(self._video_path))
        self.stop_voice()
        player.play()

    def stop_video(self) -> None:
        if self._video_player is not None:
            self._video_player.stop()

    def open_video_externally(self) -> None:
        if self._video_path and Path(self._video_path).is_file():
            self.stop_video()
            open_with_system_player(self._video_path)

    def release_media(self) -> None:
        """Stops playback and closes file handles so the pipeline may rewrite or clean up preview files."""
        if self._voice_player is not None:
            self._voice_player.release()
        if self._video_player is not None:
            self._video_player.stop()
            self._video_player.setSource(QUrl())

    def _refresh_buttons(self) -> None:
        has_voice = bool(self._voice_path)
        has_video = bool(self._video_path)
        self.voice_play_button.setEnabled(has_voice)
        self.voice_stop_button.setEnabled(has_voice)
        for button in (self.video_play_button, self.video_stop_button, self.open_external_button):
            button.setEnabled(has_video)
