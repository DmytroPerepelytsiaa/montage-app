from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from . import APP_NAME, __version__
from .config import ProjectSettings
from .models import (
    CutStyle,
    Issue,
    MediaKind,
    Narration,
    NumberedFile,
    Severity,
    Timeline,
    TimelineEntry,
    TransitionDef,
    TransitionEvent,
    to_jsonable,
)
from .timeline import format_time

MANIFEST_FORMAT_VERSION = 1
SUMMARY_COLUMNS = ("#", "scene", "start", "end", "duration", "cut", "transition", "sfx", "notes")

def write_manifest(
    path: str | Path,
    *,
    settings: ProjectSettings,
    scan: Any,
    narration: Narration | None,
    timeline: Timeline,
    library: list[TransitionDef],
    outputs: dict,
    stats: dict,
    started_at: datetime | float | str | None,
    finished_at: datetime | float | str | None,
) -> None:
    scan_issues = list(getattr(scan, "issues", []) or [])
    document = {
        "app": APP_NAME,
        "version": __version__,
        "manifest_version": MANIFEST_FORMAT_VERSION,
        "started_at": _timestamp(started_at),
        "finished_at": _timestamp(finished_at),
        "settings": settings.to_dict(),
        "inputs": _inputs_to_dict(scan),
        "narration": _narration_to_dict(narration),
        "timeline": timeline_to_dict(timeline),
        "transitions": [_transition_choice(event) for event in timeline.transitions],
        "transition_library": [transition.to_dict() for transition in library],
        "issues": [issue.to_dict() for issue in [*scan_issues, *timeline.issues]],
        "outputs": to_jsonable(outputs),
        "stats": to_jsonable(stats),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, target)


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _timestamp(value: datetime | float | str | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value)
    return value.isoformat(timespec="seconds")


def _inputs_to_dict(scan: Any) -> dict:
    if scan is None:
        return {}
    return {
        "voices": [to_jsonable(item) for item in getattr(scan, "voices", [])],
        "scenes": [to_jsonable(item) for item in getattr(scan, "scenes", [])],
        "sfx": list(getattr(scan, "sfx", [])),
        "music": getattr(scan, "music", ""),
    }


def _narration_to_dict(narration: Narration | None) -> dict:
    if narration is None:
        return {}
    return {
        "wav_path": narration.wav_path,
        "duration": narration.duration,
        "sample_rate": narration.sample_rate,
        "voices": [
            {
                "index": voice.index,
                "source_path": voice.source_path,
                "source_duration": voice.source_duration,
                "duration": voice.duration,
                "global_offset": voice.global_offset,
                "kept_segments": [to_jsonable(segment) for segment in voice.segments],
            }
            for voice in narration.voices
        ],
    }


def _scene_from_dict(data: dict) -> NumberedFile:
    return NumberedFile(data["path"], int(data["index"]), int(data["sub_index"]), data["stem"], MediaKind(data["kind"]))


def _transition_choice(event: TransitionEvent) -> dict:
    return {
        "boundary_position": event.boundary_position,
        "name": event.name,
        "xfade": event.xfade,
        "time": event.time,
        "duration": event.duration,
        "start": event.start,
        "end": event.end,
        "sfx": event.sfx_path,
        "volume_db": event.sfx_volume_db,
        "sfx_start": event.sfx_start,
    }


def timeline_to_dict(timeline: Timeline) -> dict:
    return {
        "duration": timeline.duration,
        "fps": timeline.fps,
        "entries": [to_jsonable(entry) | {"duration": entry.duration} for entry in timeline.entries],
        "transitions": [to_jsonable(event) for event in timeline.transitions],
        "issues": [issue.to_dict() for issue in timeline.issues],
    }


def timeline_from_dict(data: dict) -> Timeline:
    entries = [
        TimelineEntry(
            position=int(item["position"]),
            scene=_scene_from_dict(item["scene"]),
            start=float(item["start"]),
            end=float(item["end"]),
            voice_indices=[int(index) for index in item.get("voice_indices", [])],
            end_cut=CutStyle(item["end_cut"]) if item.get("end_cut") else None,
            notes=list(item.get("notes", [])),
        )
        for item in data.get("entries", [])
    ]
    transitions = [TransitionEvent(**item) for item in data.get("transitions", [])]
    issues = [Issue(Severity(item["severity"]), item["code"], item["message"], item.get("path", "")) for item in data.get("issues", [])]
    return Timeline(entries=entries, transitions=transitions, duration=float(data["duration"]), fps=int(data["fps"]), issues=issues)


def _cut_label(entry: TimelineEntry) -> str:
    if entry.end_cut is None:
        return "end"
    return "transition" if entry.end_cut is CutStyle.TRANSITION else "hard cut"


def timeline_summary_rows(timeline: Timeline) -> list[dict]:
    rows = []
    for entry in timeline.entries:
        event = timeline.transition_at(entry.position)
        rows.append({
            "#": entry.position + 1,
            "scene": Path(entry.scene.path).name or entry.scene.stem,
            "start": format_time(entry.start),
            "end": format_time(entry.end),
            "duration": f"{entry.duration:.2f} s",
            "cut": _cut_label(entry),
            "transition": f"{event.name} ({event.xfade}, {event.duration:.2f} s)" if event else "",
            "sfx": f"{Path(event.sfx_path).name} ({event.sfx_volume_db:+.1f} dB @ {format_time(event.sfx_start)})" if event and event.sfx_path else "",
            "notes": "; ".join(entry.notes),
        })
    return rows


def format_summary_text(timeline: Timeline) -> str:
    rows = timeline_summary_rows(timeline)
    if not rows:
        return "Timeline is empty."
    widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in SUMMARY_COLUMNS}
    lines = [
        "  ".join(column.ljust(widths[column]) for column in SUMMARY_COLUMNS).rstrip(),
        "  ".join("-" * widths[column] for column in SUMMARY_COLUMNS),
        *("  ".join(str(row[column]).ljust(widths[column]) for column in SUMMARY_COLUMNS).rstrip() for row in rows),
        "",
        f"Scenes: {len(timeline.entries)}, transitions: {len(timeline.transitions)}, "
        f"hard cuts: {len(timeline.entries) - 1 - len(timeline.transitions)}, total: {format_time(timeline.duration)} at {timeline.fps} fps.",
    ]
    if timeline.issues:
        lines.append("Issues:")
        lines.extend(f"  [{issue.severity.value.upper()}] {issue.message}" for issue in timeline.issues)
    return "\n".join(lines)
