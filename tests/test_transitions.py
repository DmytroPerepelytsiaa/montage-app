from __future__ import annotations

import json
from collections import Counter

import pytest

from dima_auto_editor.ffmpeg_tools import run_capture
from dima_auto_editor.models import Severity, TransitionDef
from dima_auto_editor.transitions import (
    XFADE_NAMES,
    XFADE_TRANSITIONS,
    TransitionPicker,
    default_library,
    load_library,
    resolve_library_sfx,
    save_library,
    validate_library,
)

REQUIRED_XFADES = {
    "fade", "dissolve", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright", "slideup", "slidedown",
    "smoothleft", "smoothright", "circleopen", "circleclose", "pixelize", "radial", "zoomin", "fadeblack", "fadewhite",
    "hblur", "squeezeh", "squeezev", "diagtl", "diagbr",
}


def _library(*names: str, enabled: bool = True) -> list[TransitionDef]:
    return [TransitionDef(name=name, xfade="fade", enabled=enabled) for name in names]


def test_catalogue_contains_practical_types():
    assert REQUIRED_XFADES <= XFADE_NAMES
    assert len(set(XFADE_TRANSITIONS)) == len(XFADE_TRANSITIONS)


def test_catalogue_names_are_supported_by_bundled_ffmpeg(ffmpeg_bins):
    help_text = run_capture(["-hide_banner", "-h", "filter=xfade"], binaries=ffmpeg_bins).stdout.decode("utf-8", "replace")
    supported = {line.split()[0] for line in help_text.splitlines() if "transition" in line and line.startswith("     ")}
    assert XFADE_NAMES <= supported


def test_default_library_without_sfx_has_no_sound():
    library = default_library([])
    assert library and all(not transition.sfx_files for transition in library)
    assert all(transition.xfade in XFADE_NAMES for transition in library)
    assert any(transition.enabled for transition in library)
    assert validate_library(library, []) == []


def test_default_library_uses_configured_duration():
    assert {transition.duration_s for transition in default_library([], 0.75)} == {0.75}
    assert len({transition.duration_s for transition in default_library([])}) > 1


def test_default_library_maps_sfx_by_keywords():
    whoosh, impact, glitch = "C:/sfx/Whoosh_01.wav", "C:/sfx/big-impact.wav", "C:/sfx/glitch3.mp3"
    library = {transition.xfade: transition for transition in default_library([whoosh, impact, glitch])}
    for xfade in ("slideleft", "slideright", "wipeleft", "smoothleft"):
        assert library[xfade].sfx_files == [whoosh]
    for xfade in ("fadewhite", "zoomin"):
        assert library[xfade].sfx_files == [impact]
    for xfade in ("pixelize", "hblur"):
        assert library[xfade].sfx_files == [glitch]


def test_default_library_round_robins_unclassified_sfx():
    sounds = ["C:/sfx/a.wav", "C:/sfx/b.wav", "C:/sfx/white_noise.wav"]
    library = default_library(sounds)
    assigned = Counter(path for transition in library for path in transition.sfx_files)
    assert all(len(transition.sfx_files) == 1 for transition in library)
    assert set(assigned) == set(sounds)
    assert max(assigned.values()) - min(assigned.values()) <= 1


def test_library_save_load_round_trip(tmp_path):
    library = default_library(["C:/sfx/whoosh.wav"])
    library[0].sfx_volume_db = -6.5
    library[1].sfx_offset_s = -0.1
    library[2].enabled = False
    path = tmp_path / "nested" / "library.json"
    save_library(path, library)
    assert load_library(path) == library
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_load_library_accepts_plain_list_and_string_sfx(tmp_path):
    path = tmp_path / "library.json"
    path.write_text(json.dumps([{"name": "Pop", "xfade": "zoomin", "sfx_files": "pop.wav", "unknown_key": 1}]), encoding="utf-8")
    assert load_library(path) == [TransitionDef(name="Pop", xfade="zoomin", sfx_files=["pop.wav"])]


def test_validate_library_reports_problems(tmp_path):
    existing = tmp_path / "whoosh.wav"
    existing.write_bytes(b"")
    library = [
        TransitionDef(name="Bad type", xfade="capcut_magic"),
        TransitionDef(name="Missing sound", xfade="fade", sfx_files=["nope.wav"]),
        TransitionDef(name="By name", xfade="fade", sfx_files=["WHOOSH.wav"]),
        TransitionDef(name="Zero", xfade="fade", duration_s=0),
        TransitionDef(name="Long", xfade="fade", duration_s=4.0),
    ]
    issues = {(issue.code, issue.severity) for issue in validate_library(library, [str(existing)])}
    assert (("unknown_xfade", Severity.ERROR)) in issues
    assert ("missing_sfx", Severity.WARNING) in issues
    assert ("bad_transition_duration", Severity.ERROR) in issues
    assert ("bad_transition_duration", Severity.WARNING) in issues
    messages = [issue.message for issue in validate_library(library, [str(existing)]) if issue.code == "missing_sfx"]
    assert len(messages) == 1 and "nope.wav" in messages[0]


def test_validate_library_flags_all_disabled():
    codes = {issue.code for issue in validate_library(_library("A", enabled=False), [])}
    assert "no_transitions_enabled" in codes


def test_resolve_library_sfx_maps_names_and_drops_missing(tmp_path):
    whoosh = tmp_path / "whoosh.wav"
    whoosh.write_bytes(b"")
    library = [TransitionDef(name="Slide", xfade="slideleft", sfx_files=["whoosh.wav", "gone.wav"])]
    resolved = resolve_library_sfx(library, [str(whoosh)])
    assert resolved[0].sfx_files == [str(whoosh)]
    assert library[0].sfx_files == ["whoosh.wav", "gone.wav"]


def test_picker_is_deterministic_for_seed():
    library = _library("A", "B", "C", "D")
    picker_a, picker_b = TransitionPicker(library, "random", seed=7), TransitionPicker(library, "random", seed=7)
    sequence_a = [picker_a.next().name for _ in range(50)]
    assert sequence_a == [picker_b.next().name for _ in range(50)]
    other = TransitionPicker(library, "random", seed=8)
    assert sequence_a != [other.next().name for _ in range(50)]


@pytest.mark.parametrize("seed", range(20))
def test_random_picker_never_repeats_immediately(seed):
    picker = TransitionPicker(_library("A", "B", "C"), "random", seed=seed)
    names = [picker.next().name for _ in range(200)]
    assert all(left != right for left, right in zip(names, names[1:]))
    assert set(names) == {"A", "B", "C"}


def test_picker_with_single_choice_repeats():
    picker = TransitionPicker(_library("Only"), "random", seed=1)
    assert [picker.next().name for _ in range(3)] == ["Only"] * 3


def test_sequential_picker_cycles_enabled_entries_in_order():
    library = _library("A", "B", "C") + _library("Disabled", enabled=False)
    picker = TransitionPicker(library, "sequential")
    assert [picker.next().name for _ in range(7)] == ["A", "B", "C", "A", "B", "C", "A"]


def test_picker_uses_only_enabled_entries():
    library = _library("On") + _library("Off1", "Off2", enabled=False)
    picker = TransitionPicker(library, "random", seed=3)
    assert {picker.next().name for _ in range(20)} == {"On"}


def test_picker_without_choices_and_bad_mode():
    picker = TransitionPicker(_library("Off", enabled=False))
    assert not picker.has_choices
    with pytest.raises(LookupError):
        picker.next()
    with pytest.raises(ValueError):
        TransitionPicker(_library("A"), "shuffle")


def test_pick_sfx_modes():
    silent = TransitionDef(name="Silent")
    noisy = TransitionDef(name="Noisy", sfx_files=["a.wav", "b.wav", "c.wav"])
    sequential = TransitionPicker([noisy], "sequential")
    assert sequential.pick_sfx(silent) == ""
    assert [sequential.pick_sfx(noisy) for _ in range(4)] == ["a.wav", "b.wav", "c.wav", "a.wav"]
    random_a, random_b = TransitionPicker([noisy], seed=5), TransitionPicker([noisy], seed=5)
    picks = [random_a.pick_sfx(noisy) for _ in range(30)]
    assert picks == [random_b.pick_sfx(noisy) for _ in range(30)]
    assert all(left != right for left, right in zip(picks, picks[1:]))
