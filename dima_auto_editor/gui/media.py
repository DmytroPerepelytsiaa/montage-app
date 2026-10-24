from __future__ import annotations

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


def db_to_volume(gain_db: float) -> float:
    return max(0.0, min(1.0, 10 ** (gain_db / 20)))


class AudioClipPlayer(QObject):
    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)

    def play(self, path: str, gain_db: float = 0.0) -> None:
        self.player.stop()
        self.audio_output.setVolume(db_to_volume(gain_db))
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

    def stop(self) -> None:
        self.player.stop()

    def release(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())

    def is_playing(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
