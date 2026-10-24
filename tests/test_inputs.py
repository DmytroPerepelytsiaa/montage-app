from pathlib import Path

import pytest

from dima_auto_editor.config import InputPaths
from dima_auto_editor.inputs import scan_inputs
from dima_auto_editor.models import MediaKind, Severity


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"\0")


def _codes(scan, severity=None):
    return [issue.code for issue in scan.issues if severity is None or issue.severity is severity]


def _issue(scan, code):
    matches = [issue for issue in scan.issues if issue.code == code]
    assert matches, f"expected issue {code!r}, got {_codes(scan)}"
    return matches[0]


@pytest.fixture
def folders(tmp_path):
    voice_dir, scene_dir = tmp_path / "voice", tmp_path / "scenes"
    voice_dir.mkdir()
    scene_dir.mkdir()
    return voice_dir, scene_dir


def _scan(voice_dir, scene_dir, **extra):
    return scan_inputs(InputPaths(voice_dir=str(voice_dir), scenes_dir=str(scene_dir), **extra), probe_media=False)


def test_clean_project_pairs_voices_with_scenes_in_natural_order(folders):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "10.wav", "2.wav", "01.wav")
    _touch(scene_dir, "10.mp4", "02_2.png", "02_1.jpg", "01.png")
    scan = _scan(voice_dir, scene_dir)

    assert [Path(v.path).name for v in scan.voices] == ["01.wav", "2.wav", "10.wav"]
    assert [Path(s.path).name for s in scan.scenes] == ["01.png", "02_1.jpg", "02_2.png", "10.mp4"]
    assert [Path(s.path).name for s in scan.pairs[2]] == ["02_1.jpg", "02_2.png"]
    assert scan.scenes[0].kind is MediaKind.IMAGE and scan.scenes[-1].kind is MediaKind.VIDEO
    assert not scan.has_errors()
    assert "numbering_gap" in _codes(scan, Severity.WARNING)


def test_voice_without_scene_is_an_error_naming_the_file(folders):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "01.wav", "02.wav")
    _touch(scene_dir, "01.png")
    scan = _scan(voice_dir, scene_dir)
    issue = _issue(scan, "missing_scene")
    assert issue.severity is Severity.ERROR and "02.wav" in issue.message
    assert scan.has_errors()


def test_scene_without_voice_is_a_warning(folders):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "01.wav", "02.wav")
    _touch(scene_dir, "01.png", "02.png", "03.mp4")
    issue = _issue(_scan(voice_dir, scene_dir), "orphan_scene")
    assert issue.severity is Severity.WARNING and "03.mp4" in issue.message


def test_unsupported_and_unnumbered_files_are_reported_not_skipped(folders):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "01.wav", "notes.txt", "intro.wav")
    _touch(scene_dir, "01.png", "02.psd")
    scan = _scan(voice_dir, scene_dir)
    unsupported = [i for i in scan.issues if i.code == "unsupported_file"]
    assert {Path(i.path).name for i in unsupported} == {"notes.txt", "02.psd"}
    assert all(i.severity is Severity.WARNING for i in unsupported)
    unnumbered = _issue(scan, "unnumbered_file")
    assert "intro.wav" in unnumbered.message and unnumbered.severity is Severity.WARNING


def test_duplicate_index_is_an_error(folders):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "01.wav", "1.mp3")
    _touch(scene_dir, "01.png", "01.mp4")
    scan = _scan(voice_dir, scene_dir)
    duplicates = [i for i in scan.issues if i.code == "duplicate_index"]
    assert len(duplicates) == 2 and all(i.severity is Severity.ERROR for i in duplicates)
    assert any("01.png" in i.message and "01.mp4" in i.message for i in duplicates)


def test_numbering_gap_lists_missing_numbers(folders):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "01.wav", "02.wav", "05.wav")
    _touch(scene_dir, "01.png", "02.png", "05.png")
    gaps = [i for i in _scan(voice_dir, scene_dir).issues if i.code == "numbering_gap"]
    assert gaps and all("03, 04" in i.message for i in gaps)


def test_nonexistent_and_unset_folders_are_errors(tmp_path):
    scan = scan_inputs(InputPaths(voice_dir=str(tmp_path / "missing"), scenes_dir=""), probe_media=False)
    assert "folder_not_found" in _codes(scan, Severity.ERROR)
    assert "folder_not_set" in _codes(scan, Severity.ERROR)
    assert "missing" in _issue(scan, "folder_not_found").message


def test_single_voice_with_many_scenes_is_continuous_narration(folders):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "01.wav")
    _touch(scene_dir, "01.png", "02.mp4", "03.png")
    scan = _scan(voice_dir, scene_dir)
    assert _issue(scan, "continuous_narration").severity is Severity.INFO
    assert "orphan_scene" not in _codes(scan)
    assert len(scan.pairs[1]) == 3
    assert not scan.has_errors()


def test_shared_folder_classifies_by_extension(tmp_path):
    folder = tmp_path / "all"
    _touch(folder, "01.wav", "01.png", "02.wav", "02.mp4")
    scan = scan_inputs(InputPaths(voice_dir=str(folder), scenes_dir=str(folder)), probe_media=False)
    assert len(scan.voices) == 2 and len(scan.scenes) == 2
    assert "unsupported_file" not in _codes(scan)


def test_music_folder_picks_first_audio(folders, tmp_path):
    voice_dir, scene_dir = folders
    _touch(voice_dir, "01.wav")
    _touch(scene_dir, "01.png")
    music_dir = tmp_path / "music"
    _touch(music_dir, "b.mp3", "a.wav")
    scan = _scan(voice_dir, scene_dir, music_path=str(music_dir))
    assert Path(scan.music).name == "a.wav"


def test_voice_overs_and_scenes_in_one_folder(tmp_path):
    shared = tmp_path / "gluing"
    _touch(shared, "2.mp3", "2_1.mp4", "2_2.mp4", "3.mp3", "3_1.mp4", "3_2.mp4", "3_3.mp4")
    scan = _scan(shared, shared)
    assert [Path(voice.path).name for voice in scan.voices] == ["2.mp3", "3.mp3"]
    assert [Path(scene.path).name for scene in scan.scenes] == ["2_1.mp4", "2_2.mp4", "3_1.mp4", "3_2.mp4", "3_3.mp4"]
    assert not scan.has_errors()
    assert not any(issue.code == "unsupported_file" for issue in scan.issues)


def test_probing_reports_unreadable_media(folders, make_tone_wav, make_image):
    voice_dir, scene_dir = folders
    make_tone_wav(voice_dir / "01.wav", seconds=0.3)
    make_tone_wav(voice_dir / "02.wav", seconds=0.3)
    make_image(scene_dir / "01.png")
    (scene_dir / "02.png").write_bytes(b"not an image")
    scan = scan_inputs(InputPaths(voice_dir=str(voice_dir), scenes_dir=str(scene_dir)))
    issue = _issue(scan, "unreadable_media")
    assert issue.severity is Severity.ERROR and "02.png" in issue.message
    assert [Path(i.path).name for i in scan.issues if i.severity is Severity.ERROR] == ["02.png"]
    assert scan.media[str(voice_dir / "01.wav")].duration == pytest.approx(0.3, abs=0.02)
