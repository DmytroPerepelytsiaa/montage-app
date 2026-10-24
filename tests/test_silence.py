import time

import numpy as np
import pytest

from conftest import SAMPLE_RATE, build_pattern, silence, speech_like
from dima_auto_editor.audio.silence import (
    SilenceParams,
    apply_segments,
    detect_silences,
    frame_levels_db,
    map_time,
    plan_kept_segments,
    process_samples,
    process_voice,
)
from dima_auto_editor.config import AudioSettings
from dima_auto_editor.ffmpeg_tools import decode_audio
from dima_auto_editor.models import KeptSegment

FRAME_S = 1 / 60
PARAMS = SilenceParams.from_settings(AudioSettings())


def _process(pattern, protected=(), params=PARAMS):
    return process_samples(build_pattern(pattern), SAMPLE_RATE, params, protected)


def _kept_gaps(segments):
    return [(a.src_end, b.src_start) for a, b in zip(segments, segments[1:])]


def test_params_follow_audio_settings():
    params = SilenceParams.from_settings(AudioSettings(silence_threshold_db=-40, speech_pad_after_ms=200))
    assert params.threshold_db == -40 and params.pad_after_s == pytest.approx(0.2)
    assert params.min_silence_s == 0.5 and params.target_silence_s == 0.3


def test_frame_levels_distinguish_speech_from_silence():
    levels = frame_levels_db(build_pattern([("speech", 0.4), ("silence", 0.4)]), SAMPLE_RATE, 0.02, 80)
    assert levels.size == 40
    assert levels[:20].min() > -30 and levels[21:].max() < -60


def test_long_pause_is_reduced_to_target():
    audio, segments, _ = _process([("speech", 1.0), ("silence", 2.0), ("speech", 1.0)])
    assert len(audio) / SAMPLE_RATE == pytest.approx(2.3, abs=0.02)
    pause_out = map_time(segments, 3.0) - map_time(segments, 1.0)
    assert pause_out == pytest.approx(0.3, abs=0.02)
    assert map_time(segments, 1.0) == pytest.approx(1.0, abs=1e-6)


def test_short_pause_is_untouched():
    pattern = [("speech", 1.0), ("silence", 0.3), ("speech", 1.0)]
    audio, segments, _ = _process(pattern)
    assert len(segments) == 1
    assert len(audio) == len(build_pattern(pattern))


def test_speech_duration_is_preserved():
    pattern = [("speech", 0.8), ("silence", 1.2), ("speech", 1.3), ("silence", 0.9), ("speech", 0.6)]
    audio, segments, _ = _process(pattern)
    expected = 0.8 + 1.3 + 0.6 + 2 * PARAMS.retained_pause_s
    assert abs(len(audio) / SAMPLE_RATE - expected) <= FRAME_S
    for speech_start, speech_end in [(0.0, 0.8), (2.0, 3.3), (4.2, 4.8)]:
        assert map_time(segments, speech_end) - map_time(segments, speech_start) == pytest.approx(speech_end - speech_start, abs=1e-6)


def test_leading_and_trailing_silence_trimmed_to_edge():
    audio, segments, _ = _process([("silence", 1.0), ("speech", 1.0), ("silence", 1.5)])
    assert map_time(segments, 1.0) == pytest.approx(PARAMS.leading_edge_s, abs=1e-3)
    assert len(audio) / SAMPLE_RATE == pytest.approx(PARAMS.leading_edge_s + 1.0 + PARAMS.trailing_edge_s, abs=1e-3)


def test_protected_span_prevents_cut():
    pattern = [("speech", 1.0), ("silence", 2.0), ("speech", 1.0)]
    audio, segments, _ = _process(pattern, protected=[(1.5, 1.7)])
    assert len(segments) == 1 and len(audio) == len(build_pattern(pattern))


def test_soft_fricative_is_not_treated_as_silence():
    rng = np.random.default_rng(7)
    spectrum = np.fft.rfft(rng.standard_normal(int(0.7 * SAMPLE_RATE)))
    frequencies = np.fft.rfftfreq(int(0.7 * SAMPLE_RATE), 1 / SAMPLE_RATE)
    spectrum[(frequencies < 4000) | (frequencies > 9000)] = 0
    hiss = np.fft.irfft(spectrum, n=int(0.7 * SAMPLE_RATE))
    fricative = (hiss / np.sqrt(np.mean(hiss**2)) * 10 ** (-44 / 20)).astype(np.float32)
    samples = np.concatenate([silence(0.3), speech_like(0.5), fricative, speech_like(0.5, seed=3), silence(0.3)])

    aware = process_samples(samples, SAMPLE_RATE, PARAMS)[0]
    energy_only = process_samples(samples, SAMPLE_RATE, SilenceParams.from_settings(AudioSettings(speech_aware=False)))[0]
    content = 0.5 + 0.7 + 0.5
    assert len(aware) / SAMPLE_RATE == pytest.approx(content + PARAMS.leading_edge_s + PARAMS.trailing_edge_s, abs=0.03)
    assert len(energy_only) < len(aware)


def test_hangover_bridges_short_dips_inside_speech():
    pattern = [("speech", 0.5), ("silence", 0.04), ("speech", 0.5)]
    silences = detect_silences(build_pattern(pattern), SAMPLE_RATE, PARAMS)
    assert silences == []


def test_cut_points_have_no_clicks():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    segments = [KeptSegment(0.0, 0.30013, 0.0), KeptSegment(0.61237, 1.0, 0.30013)]
    output = apply_segments(tone, SAMPLE_RATE, segments, fade_ms=8.0)
    cut = round(0.30013 * SAMPLE_RATE)
    natural_step = np.abs(np.diff(tone)).max()
    assert np.abs(np.diff(output[cut - 400:cut + 400])).max() <= natural_step * 1.01
    assert abs(output[cut - 1]) < 0.01 and abs(output[cut]) < 0.01


def test_fades_never_exceed_half_a_segment():
    tone = np.ones(SAMPLE_RATE, dtype=np.float32)
    output = apply_segments(tone, SAMPLE_RATE, [KeptSegment(0.0, 0.004, 0.0)], fade_ms=8.0)
    assert len(output) == 192 and output.max() > 0.99


def test_map_time_is_monotonic_and_maps_removed_regions_to_cut_point():
    segments = plan_kept_segments(4.0, [(1.0, 3.0)], PARAMS)
    assert len(segments) == 2
    cut_point = segments[0].out_end
    assert map_time(segments, 1.5) == pytest.approx(cut_point)
    assert map_time(segments, 2.5) == pytest.approx(cut_point)
    times = np.linspace(-0.5, 4.5, 2001)
    mapped = np.array([map_time(segments, t) for t in times])
    assert np.all(np.diff(mapped) >= 0)
    assert mapped[-1] == pytest.approx(segments[-1].out_end)


def test_entirely_silent_file_is_kept_untouched():
    samples = silence(1.0)
    audio, segments, silences = process_samples(samples, SAMPLE_RATE, PARAMS)
    assert silences == [(0.0, 1.0)] and len(audio) == len(samples)


def test_process_voice_reads_file_without_modifying_it(tmp_path, make_speech_like_wav):
    path = make_speech_like_wav(tmp_path / "01.wav", [("silence", 0.5), ("speech", 0.6), ("silence", 1.0), ("speech", 0.6)])
    before = path.read_bytes()
    audio, segments, silences = process_voice(path, AudioSettings())
    assert path.read_bytes() == before
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) < len(decode_audio(path, SAMPLE_RATE, 1))
    assert len(silences) == 2


def test_three_minute_voice_is_processed_quickly():
    samples = build_pattern([("speech", 3.0), ("silence", 1.0)] * 45)
    started = time.perf_counter()
    audio, segments, _ = process_samples(samples, SAMPLE_RATE, PARAMS)
    assert time.perf_counter() - started < 4.0
    assert len(segments) == 45
