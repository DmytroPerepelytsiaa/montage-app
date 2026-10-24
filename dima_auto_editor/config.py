from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

MOTION_CHOPPY = "choppy"
MOTION_SMOOTH = "smooth"


@dataclass
class InputPaths:
    voice_dir: str = ""
    scenes_dir: str = ""
    sfx_dir: str = ""
    music_path: str = ""
    output_dir: str = ""


@dataclass
class AudioSettings:
    silence_threshold_db: float = -38.0
    min_silence_s: float = 0.5
    target_silence_s: float = 0.3
    fade_ms: float = 8.0
    clip_edge_silence_s: float = 0.15
    speech_pad_before_ms: float = 60.0
    speech_pad_after_ms: float = 140.0
    analysis_window_ms: float = 20.0
    highpass_hz: float = 80.0
    speech_aware: bool = True
    voice_gain_db: float = 0.0
    sfx_volume_db: float = -10.0
    music_volume_db: float = -24.0
    music_ducking: bool = True
    limiter_ceiling_db: float = -1.0
    sample_rate: int = 48000


@dataclass
class VideoSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 60
    motion_mode: str = MOTION_CHOPPY
    source_fps_hint: float = 12.0
    fit_mode: str = "cover"
    scale_filter: str = "neighbor"
    image_motion: str = "zoom_in"
    image_motion_strength: float = 0.06
    extend_mode: str = "slow"
    trim_mode: str = "smart"
    interpolate_slowed_clips: bool = True
    interpolation_engine: str = "auto"
    rife_executable: str = ""
    rife_model: str = "rife-v4.6"


@dataclass
class TransitionSettings:
    enabled: bool = False
    library_path: str = ""
    selection: str = "random"
    seed: int = 12345
    default_duration_s: float = 0.5


@dataclass
class ExportSettings:
    encoder: str = "auto"
    quality: int = 18
    preset: str = "medium"
    audio_bitrate: str = "192k"
    preview_scale: float = 0.5
    preview_max_seconds: float = 0.0
    keep_temp: bool = False
    ffmpeg_path: str = ""
    ffprobe_path: str = ""


@dataclass
class ProjectSettings:
    paths: InputPaths = field(default_factory=InputPaths)
    audio: AudioSettings = field(default_factory=AudioSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    transitions: TransitionSettings = field(default_factory=TransitionSettings)
    export: ExportSettings = field(default_factory=ExportSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectSettings":
        return _from_dict(cls, data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectSettings":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else None  # type: ignore[misc]
        if default is not None and is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        kwargs[f.name] = value
    return cls(**kwargs)


def default_settings_path() -> Path:
    return Path.home() / ".dima_auto_editor" / "settings.json"
