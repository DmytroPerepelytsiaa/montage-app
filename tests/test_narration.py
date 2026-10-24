import threading

import pytest

from dima_auto_editor.audio.narration import build_narration
from dima_auto_editor.config import ProjectSettings
from dima_auto_editor.ffmpeg_tools import RenderCancelled, decode_audio
from dima_auto_editor.models import MediaKind, NumberedFile, Severity

SAMPLE_RATE = 48000
FPS = 60
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS


def _voice(path, index):
    return NumberedFile(str(path), index, 0, path.stem, MediaKind.AUDIO)


@pytest.fixture
def two_voices(tmp_path, make_speech_like_wav):
    first = make_speech_like_wav(tmp_path / "src" / "01.wav", [("silence", 0.4), ("speech", 1.0), ("silence", 1.5), ("speech", 0.7), ("silence", 0.5)])
    second = make_speech_like_wav(tmp_path / "src" / "02.wav", [("silence", 0.3), ("speech", 0.9), ("silence", 0.6)])
    return [_voice(first, 1), _voice(second, 2)]


def test_narration_is_frame_aligned_and_equals_sum_of_clips(tmp_path, two_voices):
    narration = build_narration(two_voices, ProjectSettings(), tmp_path / "work")
    samples = decode_audio(narration.wav_path, SAMPLE_RATE, 1)

    assert len(samples) % SAMPLES_PER_FRAME == 0
    assert narration.duration == pytest.approx(len(samples) / SAMPLE_RATE, abs=1e-9)
    assert sum(v.duration for v in narration.voices) == pytest.approx(narration.duration, abs=1e-9)
    assert narration.voices[0].global_offset == 0.0
    assert narration.voices[1].global_offset == pytest.approx(narration.voices[0].duration, abs=1e-9)
    assert narration.voices[0].duration < narration.voices[0].source_duration
    assert (tmp_path / "work" / "narration.wav").is_file()
    assert not narration.issues


def test_progress_and_cancel(tmp_path, two_voices):
    reported = []
    build_narration(two_voices, ProjectSettings(), tmp_path / "a", progress=reported.append)
    assert reported == [0.5, 1.0]

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RenderCancelled):
        build_narration(two_voices, ProjectSettings(), tmp_path / "b", cancel=cancel)


def test_silent_voice_is_reported_and_kept(tmp_path, make_speech_like_wav):
    path = make_speech_like_wav(tmp_path / "src" / "01.wav", [("silence", 1.0)])
    narration = build_narration([_voice(path, 1)], ProjectSettings(), tmp_path / "work")
    assert narration.duration == pytest.approx(1.0, abs=1 / FPS)
    assert narration.issues[0].code == "no_speech_detected" and narration.issues[0].severity is Severity.WARNING
    assert "01.wav" in narration.issues[0].message


def test_source_files_are_not_modified(tmp_path, two_voices):
    before = [open(v.path, "rb").read() for v in two_voices]
    build_narration(two_voices, ProjectSettings(), tmp_path / "work")
    assert [open(v.path, "rb").read() for v in two_voices] == before
