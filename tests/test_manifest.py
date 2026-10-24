from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from dima_auto_editor import APP_NAME, __version__
from dima_auto_editor.config import ProjectSettings
from dima_auto_editor.manifest import (
    SUMMARY_COLUMNS,
    format_summary_text,
    load_manifest,
    timeline_from_dict,
    timeline_summary_rows,
    timeline_to_dict,
    write_manifest,
)
from dima_auto_editor.models import (
    Issue,
    KeptSegment,
    MediaKind,
    Narration,
    NumberedFile,
    ProcessedVoice,
    Severity,
    TransitionDef,
)
from dima_auto_editor.timeline import plan_timeline

LIBRARY = [TransitionDef(name="Slide left", xfade="slideleft", duration_s=0.4, sfx_files=["C:/sfx/whoosh.wav"], sfx_volume_db=-8.0)]


def _scene(stem: str, index: int) -> NumberedFile:
    return NumberedFile(f"C:/scenes/{stem}.png", index, 0, stem, MediaKind.IMAGE)


def _voice_file(index: int) -> NumberedFile:
    return NumberedFile(f"C:/voice/{index:02d}.wav", index, 0, f"{index:02d}", MediaKind.AUDIO)


def _planned():
    voices = [
        ProcessedVoice(1, "C:/voice/01.wav", 7.5, 7.0, 0.0, [KeptSegment(0.0, 3.0, 0.0), KeptSegment(3.5, 7.5, 3.0)]),
        ProcessedVoice(2, "C:/voice/02.wav", 3.0, 3.0, 7.0, [KeptSegment(0.0, 3.0, 0.0)]),
        ProcessedVoice(3, "C:/voice/03.wav", 9.0, 9.0, 10.0, [KeptSegment(0.0, 9.0, 0.0)]),
    ]
    narration = Narration("C:/out/narration.wav", 19.0, 48000, voices)
    scenes = [_scene("01", 1), _scene("02", 2), _scene("03_1", 3), NumberedFile("C:/scenes/03_2.mp4", 3, 2, "03_2", MediaKind.VIDEO)]
    settings = ProjectSettings()
    settings.transitions.enabled = True
    settings.paths.sfx_dir = "C:/sfx"
    return settings, narration, scenes, plan_timeline(narration, scenes, settings, LIBRARY)


def test_manifest_round_trip(tmp_path):
    settings, narration, scenes, timeline = _planned()
    scan = SimpleNamespace(
        voices=[_voice_file(n) for n in (1, 2, 3)], scenes=scenes, sfx=["C:/sfx/whoosh.wav"], music="",
        issues=[Issue(Severity.WARNING, "numbering_gap", "Numbering gap after 03.")],
    )
    path = tmp_path / "out" / "manifest.json"
    write_manifest(path, settings=settings, scan=scan, narration=narration, timeline=timeline, library=LIBRARY,
                   outputs={"video": "C:/out/final.mp4"}, stats={"peak_db": -1.2},
                   started_at=datetime(2026, 1, 2, 3, 4, 5), finished_at=datetime(2026, 1, 2, 3, 5, 0))

    assert [item.name for item in path.parent.iterdir()] == ["manifest.json"]
    data = load_manifest(path)
    assert data["app"] == APP_NAME and data["version"] == __version__
    assert data["started_at"] == "2026-01-02T03:04:05" and data["finished_at"] == "2026-01-02T03:05:00"
    assert ProjectSettings.from_dict(data["settings"]) == settings
    assert [scene["stem"] for scene in data["inputs"]["scenes"]] == ["01", "02", "03_1", "03_2"]
    assert data["narration"]["voices"][0]["kept_segments"][1] == {"src_start": 3.5, "src_end": 7.5, "out_start": 3.0}
    assert data["outputs"] == {"video": "C:/out/final.mp4"} and data["stats"] == {"peak_db": -1.2}
    assert [TransitionDef.from_dict(item) for item in data["transition_library"]] == LIBRARY
    assert {issue["code"] for issue in data["issues"]} >= {"numbering_gap"}

    choices = data["transitions"]
    assert choices and set(choices[0]) >= {"name", "xfade", "duration", "time", "sfx", "volume_db", "sfx_start"}
    assert choices[0]["xfade"] == "slideleft" and choices[0]["sfx"] == "C:/sfx/whoosh.wav" and choices[0]["volume_db"] == -8.0

    restored = timeline_from_dict(data["timeline"])
    assert restored == timeline


def test_manifest_preserves_unicode_and_accepts_missing_parts(tmp_path):
    settings, _, _, timeline = _planned()
    settings.paths.voice_dir = "C:/Озвучка"
    path = tmp_path / "manifest.json"
    write_manifest(path, settings=settings, scan=None, narration=None, timeline=timeline, library=[],
                   outputs={}, stats={}, started_at=1_700_000_000.0, finished_at=None)
    assert "Озвучка" in path.read_text(encoding="utf-8")
    data = load_manifest(path)
    assert data["inputs"] == {} and data["finished_at"] is None
    assert data["started_at"].startswith("2023-11-")


def test_timeline_dict_is_json_serialisable():
    _, _, _, timeline = _planned()
    assert timeline_from_dict(json.loads(json.dumps(timeline_to_dict(timeline)))) == timeline


def test_summary_rows_and_text():
    _, _, _, timeline = _planned()
    rows = timeline_summary_rows(timeline)

    assert len(rows) == len(timeline.entries) == 4
    assert all(tuple(row) == SUMMARY_COLUMNS for row in rows)
    assert [row["#"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["scene"] == "01.png" and rows[0]["start"] == "00:00.000" and rows[0]["end"] == "00:07.000"
    assert rows[-1]["cut"] == "end"
    transition_rows = [row for row in rows if row["cut"] == "transition"]
    assert transition_rows and "slideleft" in transition_rows[0]["transition"] and "whoosh.wav" in transition_rows[0]["sfx"]
    assert any(row["cut"] == "hard cut" and row["transition"] == "" for row in rows)
    assert [row["cut"] for row in rows] == ["transition", "transition", "hard cut", "end"]
    assert rows[2]["end"] == "00:14.500"

    text = format_summary_text(timeline)
    assert "03_2.mp4" in text and "Scenes: 4" in text
