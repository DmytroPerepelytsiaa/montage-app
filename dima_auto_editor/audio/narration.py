from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import ProjectSettings
from ..ffmpeg_tools import RenderCancelled, decode_audio, write_wav
from ..models import Issue, KeptSegment, Narration, NumberedFile, ProcessedVoice, Severity
from .silence import Interval, SilenceParams, process_samples

NARRATION_FILENAME = "narration.wav"
_EDGE_EPSILON = 1e-6

ProgressCallback = Callable[[float], None]


@dataclass
class _ProcessedClip:
    voice: NumberedFile
    source_duration: float
    samples: np.ndarray
    segments: list[KeptSegment]
    silences: list[Interval]

    @property
    def has_speech(self) -> bool:
        return not any(start <= _EDGE_EPSILON and end >= self.source_duration - _EDGE_EPSILON for start, end in self.silences)


def build_narration(voices: list[NumberedFile], settings: ProjectSettings, work_dir: Path,
                    progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> Narration:
    sample_rate = settings.audio.sample_rate
    fps = settings.video.fps
    params = SilenceParams.from_settings(settings.audio)
    clips = [_process_clip(voice, sample_rate, params, cancel, progress, position, len(voices))
             for position, voice in enumerate(voices)]

    total_samples = sum(len(clip.samples) for clip in clips)
    frame_aligned_samples = round(math.ceil(total_samples * fps / sample_rate - 1e-9) * sample_rate / fps)
    padding = max(0, frame_aligned_samples - total_samples)
    audio = np.concatenate([clip.samples for clip in clips] + [np.zeros(padding, dtype=np.float32)])
    duration = len(audio) / sample_rate

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / NARRATION_FILENAME
    write_wav(wav_path, audio, sample_rate)

    offsets = np.concatenate(([0], np.cumsum([len(clip.samples) for clip in clips]))) / sample_rate
    processed_voices = [
        ProcessedVoice(
            index=clip.voice.index,
            source_path=clip.voice.path,
            source_duration=clip.source_duration,
            duration=(len(clip.samples) + (padding if i == len(clips) - 1 else 0)) / sample_rate,
            global_offset=float(offsets[i]),
            segments=clip.segments,
        )
        for i, clip in enumerate(clips)
    ]
    return Narration(
        wav_path=str(wav_path),
        duration=duration,
        sample_rate=sample_rate,
        voices=processed_voices,
        issues=[_no_speech_issue(clip, params) for clip in clips if not clip.has_speech],
    )


def _process_clip(voice: NumberedFile, sample_rate: int, params: SilenceParams, cancel: threading.Event | None,
                  progress: ProgressCallback | None, position: int, count: int) -> _ProcessedClip:
    if cancel is not None and cancel.is_set():
        raise RenderCancelled("Voice-over processing was cancelled by the user.")
    source = decode_audio(voice.path, sample_rate, 1)
    samples, segments, silences = process_samples(source, sample_rate, params)
    if progress is not None:
        progress((position + 1) / count)
    return _ProcessedClip(voice, len(source) / sample_rate, samples, segments, silences)


def _no_speech_issue(clip: _ProcessedClip, params: SilenceParams) -> Issue:
    name = Path(clip.voice.path).name
    return Issue(Severity.WARNING, "no_speech_detected",
                 f"No speech above {params.threshold_db:g} dBFS was detected in voice-over '{name}'; "
                 f"it was kept untouched. Check the recording level or lower the silence threshold.",
                 clip.voice.path)
