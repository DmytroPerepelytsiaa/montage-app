from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CutStyle(str, Enum):
    TRANSITION = "transition"
    HARD_CUT = "hard_cut"


@dataclass
class Issue:
    severity: Severity
    code: str
    message: str
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity.value, "code": self.code, "message": self.message, "path": self.path}


@dataclass
class MediaInfo:
    path: str
    kind: MediaKind
    duration: float | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    has_audio: bool = False
    has_video: bool = False
    sample_rate: int = 0
    channels: int = 0


@dataclass(frozen=True)
class NumberedFile:
    path: str
    index: int
    sub_index: int
    stem: str
    kind: MediaKind


@dataclass
class TransitionDef:
    name: str
    xfade: str = "fade"
    duration_s: float = 0.5
    sfx_files: list[str] = field(default_factory=list)
    sfx_volume_db: float = -10.0
    sfx_offset_s: float = 0.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionDef":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class KeptSegment:
    src_start: float
    src_end: float
    out_start: float

    @property
    def duration(self) -> float:
        return self.src_end - self.src_start

    @property
    def out_end(self) -> float:
        return self.out_start + self.duration


@dataclass
class ProcessedVoice:
    index: int
    source_path: str
    source_duration: float
    duration: float
    global_offset: float
    segments: list[KeptSegment] = field(default_factory=list)


@dataclass
class Narration:
    wav_path: str
    duration: float
    sample_rate: int
    voices: list[ProcessedVoice] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


@dataclass
class TimelineEntry:
    position: int
    scene: NumberedFile
    start: float
    end: float
    voice_indices: list[int] = field(default_factory=list)
    end_cut: CutStyle | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TransitionEvent:
    boundary_position: int
    time: float
    duration: float
    name: str
    xfade: str
    sfx_path: str = ""
    sfx_volume_db: float = 0.0
    sfx_start: float = 0.0

    @property
    def start(self) -> float:
        return self.time - self.duration / 2

    @property
    def end(self) -> float:
        return self.time + self.duration / 2


@dataclass
class Timeline:
    entries: list[TimelineEntry] = field(default_factory=list)
    transitions: list[TransitionEvent] = field(default_factory=list)
    duration: float = 0.0
    fps: int = 60
    issues: list[Issue] = field(default_factory=list)

    def transition_at(self, boundary_position: int) -> TransitionEvent | None:
        for event in self.transitions:
            if event.boundary_position == boundary_position:
                return event
        return None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
