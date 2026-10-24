from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectSettings
from .models import (
    CutStyle,
    Issue,
    Narration,
    NumberedFile,
    ProcessedVoice,
    Severity,
    Timeline,
    TimelineEntry,
    TransitionDef,
    TransitionEvent,
)
from .naming import natural_key
from .transitions import SELECTION_MODES, SELECTION_RANDOM, TransitionPicker

MAX_TRANSITION_FRACTION = 0.45


def format_time(seconds: float) -> str:
    minutes, millis = divmod(int(round(seconds * 1000)), 60_000)
    return f"{minutes:02d}:{millis / 1000:06.3f}"


def _scene_name(scene: NumberedFile) -> str:
    return Path(scene.path).name or scene.stem


def _voice_name(voice: ProcessedVoice) -> str:
    return Path(voice.source_path).name or f"voice {voice.index:02d}"


@dataclass
class _VoiceSpan:
    voice_index: int
    scenes: list[NumberedFile]
    start_frame: int
    end_frame: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def plan_timeline(narration: Narration, scenes: list[NumberedFile], settings: ProjectSettings,
                  library: list[TransitionDef]) -> Timeline:
    return _TimelinePlanner(narration, settings, library).plan(scenes)


class _TimelinePlanner:
    def __init__(self, narration: Narration, settings: ProjectSettings, library: list[TransitionDef]):
        self.settings = settings
        self.library = library
        self.fps = settings.video.fps
        self.total_frames = int(round(narration.duration * self.fps))
        self.voices = sorted(narration.voices, key=lambda voice: voice.global_offset)
        self.issues: list[Issue] = []

    def plan(self, scenes: list[NumberedFile]) -> Timeline:
        timeline = Timeline(duration=self._seconds(self.total_frames), fps=self.fps, issues=self.issues)
        if self.total_frames <= 0:
            self._issue(Severity.ERROR, "empty_narration", "The processed narration is empty; there is nothing to place scenes on.")
            return timeline
        spans = self._voice_spans(scenes)
        if not spans:
            self._issue(Severity.ERROR, "no_scenes", "No scenes could be matched to the narration.")
            return timeline
        timeline.entries = self._entries(spans)
        timeline.transitions = self._transitions(timeline.entries)
        return timeline

    def _seconds(self, frames: int) -> float:
        return frames / self.fps

    def _issue(self, severity: Severity, code: str, message: str, path: str = "") -> None:
        self.issues.append(Issue(severity, code, message, path))

    def _voice_spans(self, scenes: list[NumberedFile]) -> list[_VoiceSpan]:
        ordered = sorted(scenes, key=lambda scene: (scene.index, scene.sub_index, natural_key(scene.stem)))
        if not ordered:
            return []
        if len(self.voices) <= 1:
            voice_index = self.voices[0].index if self.voices else 0
            return [_VoiceSpan(voice_index, ordered, 0, self.total_frames)]

        scenes_by_index: dict[int, list[NumberedFile]] = defaultdict(list)
        for scene in ordered:
            scenes_by_index[scene.index].append(scene)
        voice_indices = {voice.index for voice in self.voices}
        for index, unused in scenes_by_index.items():
            if index not in voice_indices:
                for scene in unused:
                    self._issue(Severity.WARNING, "scene_unused",
                                f"Scene {_scene_name(scene)} has no voice-over numbered {index:02d} and is not used.", scene.path)

        starts = [0, *(int(round(voice.global_offset * self.fps)) for voice in self.voices[1:])]
        ends = [*starts[1:], self.total_frames]
        spans: list[_VoiceSpan] = []
        uncovered_start: int | None = None
        for voice, start, end in zip(self.voices, starts, ends):
            own_scenes = scenes_by_index.get(voice.index)
            if own_scenes:
                spans.append(_VoiceSpan(voice.index, own_scenes, start if uncovered_start is None else uncovered_start, end))
                uncovered_start = None
                continue
            if spans:
                spans[-1].end_frame = end
                held = spans[-1].scenes[-1]
            else:
                uncovered_start = start if uncovered_start is None else uncovered_start
                held = next((scenes_by_index[v.index][0] for v in self.voices if scenes_by_index.get(v.index)), None)
            if held is not None:
                self._issue(Severity.WARNING, "voice_without_scene",
                            f"Voice-over {_voice_name(voice)} has no matching scene; scene {_scene_name(held)} stays on screen during it.",
                            voice.source_path)
        return spans

    def _entries(self, spans: list[_VoiceSpan]) -> list[TimelineEntry]:
        entries: list[TimelineEntry] = []
        for span in spans:
            scenes = span.scenes
            if len(scenes) > span.frames:
                for scene in scenes[span.frames:]:
                    self._issue(Severity.WARNING, "voice_too_short",
                                f"Voice-over {span.voice_index:02d} is too short to show scene {_scene_name(scene)}.", scene.path)
                scenes = scenes[:span.frames]
            count = len(scenes)
            bounds = [span.start_frame + round(span.frames * part / count) for part in range(count + 1)]
            for part, scene in enumerate(scenes):
                entries.append(TimelineEntry(
                    position=len(entries),
                    scene=scene,
                    start=self._seconds(bounds[part]),
                    end=self._seconds(bounds[part + 1]),
                    voice_indices=[span.voice_index],
                ))
        for entry in entries[:-1]:
            entry.end_cut = CutStyle.HARD_CUT
        return entries

    def _picker(self) -> TransitionPicker:
        mode = self.settings.transitions.selection
        if mode not in SELECTION_MODES:
            self._issue(Severity.WARNING, "unknown_selection_mode", f"Unknown transition selection mode '{mode}'; using random selection.")
            mode = SELECTION_RANDOM
        return TransitionPicker(self.library, mode, self.settings.transitions.seed)

    def _transitions_wanted(self) -> bool:
        return self.settings.transitions.enabled and bool(self.settings.paths.sfx_dir.strip())

    def _transitions(self, entries: list[TimelineEntry]) -> list[TransitionEvent]:
        junctions = [position for position in range(len(entries) - 1)
                     if entries[position].voice_indices != entries[position + 1].voice_indices]
        if not junctions or not self._transitions_wanted():
            return []
        picker = self._picker()
        if not picker.has_choices:
            self._issue(Severity.INFO, "no_transitions", "No transitions are enabled, so every scene change is a hard cut.")
            return []
        events = []
        for position in junctions:
            event = self._transition_event(entries[position], entries[position + 1], picker.next(), picker)
            if event is not None:
                entries[position].end_cut = CutStyle.TRANSITION
                events.append(event)
        return events

    def _transition_event(self, outgoing: TimelineEntry, incoming: TimelineEntry, transition: TransitionDef,
                          picker: TransitionPicker) -> TransitionEvent | None:
        shortest_frames = min(round(outgoing.duration * self.fps), round(incoming.duration * self.fps))
        fitting_frames = 2 * math.floor(MAX_TRANSITION_FRACTION * shortest_frames)
        duration_frames = max(2, 2 * round(transition.duration_s * self.fps / 2))
        if fitting_frames < 2:
            outgoing.notes.append("scenes too short for a transition; hard cut used")
            return None
        if duration_frames > fitting_frames:
            outgoing.notes.append(f"transition shortened to {self._seconds(fitting_frames):.2f} s to fit the scenes")
            duration_frames = fitting_frames
        duration = self._seconds(duration_frames)
        sfx_path = picker.pick_sfx(transition)
        sfx_start = self._quantize(min(max(0.0, outgoing.end - duration / 2 + transition.sfx_offset_s), self._seconds(self.total_frames)))
        return TransitionEvent(
            boundary_position=outgoing.position,
            time=outgoing.end,
            duration=duration,
            name=transition.name,
            xfade=transition.xfade,
            sfx_path=sfx_path,
            sfx_volume_db=transition.sfx_volume_db if sfx_path else 0.0,
            sfx_start=sfx_start if sfx_path else 0.0,
        )

    def _quantize(self, seconds: float) -> float:
        return round(seconds * self.fps) / self.fps
