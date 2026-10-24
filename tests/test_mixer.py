import time

import numpy as np
import pytest

from conftest import SAMPLE_RATE, build_pattern
from dima_auto_editor.audio.mixer import SfxCue, mix_final_audio
from dima_auto_editor.config import AudioSettings
from dima_auto_editor.ffmpeg_tools import decode_audio, write_wav

CEILING_DB = -1.0
QUANTIZATION_STEP = 1 / 32767


def _rms(signal):
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def _noise_burst(path, seconds, amplitude=1.0):
    rng = np.random.default_rng(5)
    write_wav(path, np.clip(amplitude * rng.standard_normal(int(seconds * SAMPLE_RATE)), -1, 1).astype(np.float32), SAMPLE_RATE)
    return path


@pytest.fixture
def narration_wav(tmp_path):
    path = tmp_path / "narration.wav"
    write_wav(path, build_pattern([("speech", 2.0), ("silence", 1.0), ("speech", 2.0)]) * 2.5, SAMPLE_RATE)
    return path


def _mix(tmp_path, narration_wav, cues, music="", duration=5.0, **audio):
    out = tmp_path / "mix.wav"
    stats = mix_final_audio(str(narration_wav), duration, cues, str(music), AudioSettings(**audio), out)
    return decode_audio(out, SAMPLE_RATE, 2), stats


def test_output_has_exact_length_and_never_exceeds_ceiling(tmp_path, narration_wav):
    loud = _noise_burst(tmp_path / "boom.wav", 0.5)
    cues = [SfxCue(str(loud), 0.5, 6.0), SfxCue(str(loud), 2.2, 6.0), SfxCue(str(loud), 4.8, 6.0)]
    mixed, stats = _mix(tmp_path, narration_wav, cues, duration=4.9871)

    assert mixed.shape == (round(4.9871 * SAMPLE_RATE), 2)
    ceiling = 10 ** (CEILING_DB / 20)
    assert np.abs(mixed).max() <= ceiling + QUANTIZATION_STEP
    assert stats["peak_db"] <= CEILING_DB + 1e-6
    assert stats["sfx_cues_placed"] == 3


def test_sfx_is_placed_at_the_exact_sample(tmp_path):
    silent = tmp_path / "silent.wav"
    write_wav(silent, np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
    click = tmp_path / "click.wav"
    write_wav(click, np.full(200, 0.5, dtype=np.float32), SAMPLE_RATE)
    start = 1.23457
    mixed, _ = _mix(tmp_path, silent, [SfxCue(str(click), start, 0.0)], duration=2.0)
    assert int(np.flatnonzero(np.abs(mixed[:, 0]) > 0.01)[0]) == round(start * SAMPLE_RATE)


def test_voice_stays_louder_than_sfx_during_speech(tmp_path, narration_wav):
    loud = _noise_burst(tmp_path / "whoosh.wav", 1.5, amplitude=0.9)
    voice_only, _ = _mix(tmp_path, narration_wav, [])
    mixed, stats = _mix(tmp_path, narration_wav, [SfxCue(str(loud), 0.25, 0.0)])

    region = slice(int(0.3 * SAMPLE_RATE), int(1.7 * SAMPLE_RATE))
    sfx_part = mixed[region] - voice_only[region]
    assert _rms(voice_only[region]) > 1.9 * _rms(sfx_part)
    assert stats["sfx_cap_reduction_db"] > 0
    assert stats["voice_rms_db"] > -30


def test_sfx_tail_after_end_is_cut_and_late_cue_is_reported(tmp_path, narration_wav):
    loud = _noise_burst(tmp_path / "long.wav", 2.0, amplitude=0.2)
    mixed, stats = _mix(tmp_path, narration_wav, [SfxCue(str(loud), 4.0, 0.0), SfxCue(str(loud), 9.0, 0.0)])
    assert len(mixed) == 5 * SAMPLE_RATE
    assert stats["sfx_cues_placed"] == 1
    assert any("long.wav" in warning for warning in stats["warnings"])


def test_music_is_looped_and_ducked_under_speech(tmp_path, narration_wav, make_tone_wav):
    music = make_tone_wav(tmp_path / "music.wav", seconds=1.3, freq=220.0, amplitude=0.5)
    silent = tmp_path / "silent.wav"
    write_wav(silent, np.zeros(5 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    music_only, _ = _mix(tmp_path, silent, [], music=music)
    with_voice, _ = _mix(tmp_path, narration_wav, [], music=music)
    without_ducking, _ = _mix(tmp_path, narration_wav, [], music=music, music_ducking=False)

    speech = slice(int(0.5 * SAMPLE_RATE), int(1.8 * SAMPLE_RATE))
    pause = slice(int(2.55 * SAMPLE_RATE), int(2.95 * SAMPLE_RATE))
    music_under_speech = with_voice[speech] - (without_ducking[speech] - music_only[speech])
    assert _rms(music_under_speech) < _rms(music_only[speech]) * 10 ** (-6 / 20)
    assert _rms(with_voice[pause]) > _rms(music_only[pause]) * 10 ** (-3 / 20)
    assert _rms(music_only[int(3.5 * SAMPLE_RATE):int(3.6 * SAMPLE_RATE)]) > 0


def test_three_minute_mix_is_fast(tmp_path, make_tone_wav):
    rng = np.random.default_rng(0)
    narration = tmp_path / "long_narration.wav"
    write_wav(narration, (0.3 * rng.standard_normal(180 * SAMPLE_RATE)).astype(np.float32), SAMPLE_RATE)
    sfx = _noise_burst(tmp_path / "hit.wav", 0.5)
    music = make_tone_wav(tmp_path / "bed.wav", seconds=20.0, freq=110.0)
    cues = [SfxCue(str(sfx), t, -3.0) for t in range(5, 180, 6)]

    started = time.perf_counter()
    mixed, _ = _mix(tmp_path, narration, cues, music=music, duration=180.0)
    assert time.perf_counter() - started < 5.0
    assert np.abs(mixed).max() <= 10 ** (CEILING_DB / 20) + QUANTIZATION_STEP
