from __future__ import annotations

import pytest

from dima_auto_editor.config import ProjectSettings
from dima_auto_editor.models import CutStyle, MediaKind, Narration, NumberedFile, ProcessedVoice, Severity, Timeline, TransitionDef
from dima_auto_editor.naming import parse_numbered_stem
from dima_auto_editor.timeline import format_time, plan_timeline

FPS = 60
LIBRARY = [
    TransitionDef(name="Crossfade", xfade="fade", duration_s=0.5, sfx_files=["C:/sfx/soft.wav"], sfx_offset_s=0.1),
    TransitionDef(name="Slide left", xfade="slideleft", duration_s=0.4, sfx_files=["C:/sfx/whoosh.wav"]),
    TransitionDef(name="Zoom in", xfade="zoomin", duration_s=0.6),
]


def scene(stem: str) -> NumberedFile:
    index, sub_index = parse_numbered_stem(stem)
    return NumberedFile(f"C:/scenes/{stem}.mp4", index, sub_index, stem, MediaKind.VIDEO)


def narration_of(durations: dict[int, float]) -> Narration:
    voices, offset = [], 0.0
    for index, duration in durations.items():
        voices.append(ProcessedVoice(index, f"C:/voice/{index}.mp3", duration, duration, offset))
        offset += duration
    return Narration("narration.wav", round(offset * FPS) / FPS, 48000, voices)


def plan(durations: dict[int, float], stems: list[str], library=LIBRARY, sfx_dir: str = "C:/sfx",
         **transition_settings) -> Timeline:
    settings = ProjectSettings()
    settings.paths.sfx_dir = sfx_dir
    settings.transitions.enabled = True
    for key, value in transition_settings.items():
        setattr(settings.transitions, key, value)
    return plan_timeline(narration_of(durations), [scene(stem) for stem in stems], settings, library)


def codes(timeline: Timeline, severity: Severity | None = None) -> list[str]:
    return [issue.code for issue in timeline.issues if severity is None or issue.severity is severity]


def frames(seconds: float) -> int:
    return round(seconds * FPS)


USER_PROJECT = {2: 8.3, 3: 7.8, 4: 13.8, 5: 11.0, 6: 10.2}
USER_SCENES = ["2_1", "2_2", "3_1", "3_2", "3_3", "4_1", "4_2", "4_3", "4_4", "4_5",
               "5_1", "5_2", "5_3", "5_4", "6_1", "6_2", "6_3", "6_4"]


def test_every_scene_of_a_voice_over_is_shown_in_order():
    timeline = plan(USER_PROJECT, USER_SCENES)
    assert [entry.scene.stem for entry in timeline.entries] == USER_SCENES
    assert not codes(timeline, Severity.WARNING) and not codes(timeline, Severity.ERROR)


def test_scenes_split_their_voice_over_equally():
    timeline = plan(USER_PROJECT, USER_SCENES)
    voices = narration_of(USER_PROJECT).voices
    for voice in voices:
        own = [entry for entry in timeline.entries if entry.voice_indices == [voice.index]]
        lengths = [frames(entry.duration) for entry in own]
        assert max(lengths) - min(lengths) <= 1
        assert frames(own[0].start) == frames(voice.global_offset)
        assert frames(own[-1].end) == frames(voice.global_offset + voice.duration)


def test_timeline_is_contiguous_frame_aligned_and_matches_narration():
    timeline = plan(USER_PROJECT, USER_SCENES)
    assert timeline.entries[0].start == 0.0
    assert timeline.entries[-1].end == pytest.approx(timeline.duration)
    for current, following in zip(timeline.entries, timeline.entries[1:]):
        assert current.end == following.start
    for entry in timeline.entries:
        assert entry.start * FPS == pytest.approx(round(entry.start * FPS))


def test_transitions_only_between_voice_overs():
    timeline = plan(USER_PROJECT, USER_SCENES)
    junction_positions = [entry.position for entry, following in zip(timeline.entries, timeline.entries[1:])
                          if entry.voice_indices != following.voice_indices]
    assert [event.boundary_position for event in timeline.transitions] == junction_positions == [1, 4, 9, 13]
    for entry in timeline.entries[:-1]:
        expected = CutStyle.TRANSITION if entry.position in junction_positions else CutStyle.HARD_CUT
        assert entry.end_cut is expected
    assert timeline.entries[-1].end_cut is None


def test_transition_midpoint_is_the_voice_over_junction():
    timeline = plan(USER_PROJECT, USER_SCENES)
    voices = narration_of(USER_PROJECT).voices
    assert [frames(event.time) for event in timeline.transitions] == [frames(voice.global_offset) for voice in voices[1:]]


def test_default_settings_use_hard_cuts_only():
    timeline = plan_timeline(narration_of(USER_PROJECT), [scene(stem) for stem in USER_SCENES], ProjectSettings(), LIBRARY)
    assert timeline.transitions == []
    assert all(entry.end_cut is CutStyle.HARD_CUT for entry in timeline.entries[:-1])


def test_no_transition_folder_means_hard_cuts():
    timeline = plan(USER_PROJECT, USER_SCENES, sfx_dir="")
    assert timeline.transitions == []
    assert all(entry.end_cut is CutStyle.HARD_CUT for entry in timeline.entries[:-1])


def test_random_transitions_never_repeat_back_to_back():
    durations = {index: 4.0 for index in range(1, 30)}
    timeline = plan(durations, [str(index) for index in durations], selection="random", seed=3)
    names = [event.name for event in timeline.transitions]
    sounds = [event.sfx_path for event in timeline.transitions]
    assert len(names) == 28
    assert all(first != second for first, second in zip(names, names[1:]))
    assert all(not first or first != second for first, second in zip(sounds, sounds[1:]))


def test_short_voice_overs_still_get_transitions():
    timeline = plan({1: 2.5, 2: 2.5, 3: 2.5}, ["1", "2", "3"])
    assert len(timeline.transitions) == 2


def test_transition_is_shortened_to_fit_short_scenes():
    timeline = plan({1: 4.0, 2: 0.8}, ["1_1", "1_2", "1_3", "2"], library=[TransitionDef("Zoom", "zoomin", duration_s=1.0)])
    event = timeline.transitions[0]
    assert frames(event.duration) <= 2 * int(0.45 * frames(0.8))
    assert frames(event.duration) % 2 == 0
    assert any("shortened" in note for note in timeline.entries[event.boundary_position].notes)


def test_sfx_starts_at_transition_start_plus_offset():
    timeline = plan({1: 5.0, 2: 5.0}, ["1", "2"], selection="sequential")
    event = timeline.transitions[0]
    assert event.name == "Crossfade" and event.sfx_path == "C:/sfx/soft.wav"
    assert event.sfx_start == pytest.approx(round((event.time - event.duration / 2 + 0.1) * FPS) / FPS)


def test_transition_without_sound_has_no_sfx():
    timeline = plan({1: 5.0, 2: 5.0}, ["1", "2"], library=[TransitionDef("Zoom", "zoomin")])
    assert timeline.transitions[0].sfx_path == "" and timeline.transitions[0].sfx_start == 0.0


def test_sequential_selection_cycles_through_library():
    timeline = plan({1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0}, ["1", "2", "3", "4"], selection="sequential")
    assert [event.name for event in timeline.transitions] == ["Crossfade", "Slide left", "Zoom in"]


def test_random_selection_is_reproducible_with_seed():
    first = plan(USER_PROJECT, USER_SCENES, selection="random", seed=7)
    second = plan(USER_PROJECT, USER_SCENES, selection="random", seed=7)
    assert [event.name for event in first.transitions] == [event.name for event in second.transitions]


def test_no_enabled_transitions_means_hard_cuts():
    library = [TransitionDef("Off", "fade", enabled=False)]
    timeline = plan({1: 4.0, 2: 4.0}, ["1", "2"], library=library)
    assert timeline.transitions == []
    assert timeline.entries[0].end_cut is CutStyle.HARD_CUT
    assert "no_transitions" in codes(timeline, Severity.INFO)


def test_single_narration_splits_all_scenes_equally_without_transitions():
    timeline = plan({1: 12.0}, ["1", "2", "3", "4"])
    assert [frames(entry.duration) for entry in timeline.entries] == [180, 180, 180, 180]
    assert timeline.transitions == []


def test_voice_without_scene_keeps_previous_scene_on_screen():
    timeline = plan({1: 3.0, 2: 3.0, 3: 3.0}, ["1", "3"])
    assert [entry.scene.stem for entry in timeline.entries] == ["1", "3"]
    assert frames(timeline.entries[0].end) == frames(6.0)
    assert "voice_without_scene" in codes(timeline, Severity.WARNING)


def test_leading_voice_without_scene_is_covered_by_first_scene():
    timeline = plan({1: 3.0, 2: 3.0}, ["2"])
    assert timeline.entries[0].start == 0.0 and frames(timeline.entries[0].end) == frames(6.0)
    assert "voice_without_scene" in codes(timeline, Severity.WARNING)


def test_scene_without_voice_is_reported_and_unused():
    timeline = plan({1: 3.0, 2: 3.0}, ["1", "2", "7"])
    assert [entry.scene.stem for entry in timeline.entries] == ["1", "2"]
    assert "scene_unused" in codes(timeline, Severity.WARNING)


def test_empty_narration_is_an_error():
    timeline = plan_timeline(Narration("n.wav", 0.0, 48000, []), [scene("1")], ProjectSettings(), LIBRARY)
    assert "empty_narration" in codes(timeline, Severity.ERROR)


def test_format_time():
    assert format_time(83.25) == "01:23.250"
