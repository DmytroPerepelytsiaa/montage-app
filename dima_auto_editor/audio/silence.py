from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import AudioSettings
from ..ffmpeg_tools import decode_audio
from ..models import KeptSegment

Interval = tuple[float, float]

HIGH_BAND_HZ = 2000.0
NOISE_FLOOR_PERCENTILE = 10.0
_POWER_FLOOR = 1e-12
_TIME_EPSILON = 1e-6
_WRAPAROUND_GUARD_S = 0.1


@dataclass
class SilenceParams:
    threshold_db: float = -38.0
    min_silence_s: float = 0.5
    target_silence_s: float = 0.3
    fade_ms: float = 8.0
    edge_silence_s: float = 0.15
    pad_before_s: float = 0.06
    pad_after_s: float = 0.14
    window_s: float = 0.02
    highpass_hz: float = 80.0
    speech_aware: bool = True
    hangover_s: float = 0.06
    high_band_sensitivity_db: float = 10.0
    noise_margin_db: float = 6.0

    @classmethod
    def from_settings(cls, audio: AudioSettings) -> SilenceParams:
        return cls(
            threshold_db=audio.silence_threshold_db,
            min_silence_s=audio.min_silence_s,
            target_silence_s=audio.target_silence_s,
            fade_ms=audio.fade_ms,
            edge_silence_s=audio.clip_edge_silence_s,
            pad_before_s=audio.speech_pad_before_ms / 1000.0,
            pad_after_s=audio.speech_pad_after_ms / 1000.0,
            window_s=audio.analysis_window_ms / 1000.0,
            highpass_hz=audio.highpass_hz,
            speech_aware=audio.speech_aware,
        )

    @property
    def retained_pause_s(self) -> float:
        return max(self.target_silence_s, self.pad_before_s + self.pad_after_s)

    @property
    def leading_edge_s(self) -> float:
        return max(self.edge_silence_s, self.pad_before_s)

    @property
    def trailing_edge_s(self) -> float:
        return max(self.edge_silence_s, self.pad_after_s)

    @property
    def hangover_windows(self) -> int:
        return max(2, round(self.hangover_s / self.window_s))


def window_length(sample_rate: int, window_s: float) -> int:
    return max(1, round(window_s * sample_rate))


def window_power(signal: np.ndarray, window: int) -> np.ndarray:
    squared = np.square(np.asarray(signal, dtype=np.float64))
    if squared.ndim > 1:
        squared = squared.mean(axis=1)
    if squared.size == 0:
        return np.zeros(0)
    starts = np.arange(0, squared.size, window)
    counts = np.diff(np.append(starts, squared.size))
    return np.add.reduceat(squared, starts) / counts


def power_to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, _POWER_FLOOR))


def frame_levels_db(samples: np.ndarray, sr: int, window_s: float, highpass_hz: float) -> np.ndarray:
    (filtered,) = _highpassed(samples, sr, [highpass_hz])
    return power_to_db(window_power(filtered, window_length(sr, window_s)))


def speech_levels_db(samples: np.ndarray, sr: int, params: SilenceParams) -> np.ndarray:
    if not params.speech_aware:
        return frame_levels_db(samples, sr, params.window_s, params.highpass_hz)
    window = window_length(sr, params.window_s)
    high_band_cutoff = min(max(HIGH_BAND_HZ, params.highpass_hz), sr / 4)
    broadband, high_band = _highpassed(samples, sr, [params.highpass_hz, high_band_cutoff])
    broadband_db = power_to_db(window_power(broadband, window))
    high_band_db = power_to_db(window_power(high_band, window))
    if high_band_db.size == 0:
        return broadband_db
    noise_floor_db = float(np.percentile(high_band_db, NOISE_FLOOR_PERCENTILE))
    high_band_threshold = max(params.threshold_db - params.high_band_sensitivity_db, noise_floor_db + params.noise_margin_db)
    return np.maximum(broadband_db, high_band_db - high_band_threshold + params.threshold_db)


def _highpassed(samples: np.ndarray, sr: int, cutoffs_hz: Sequence[float]) -> list[np.ndarray]:
    signal = np.asarray(samples, dtype=np.float32)
    if signal.size == 0:
        return [signal for _ in cutoffs_hz]
    padded_size = _fast_fft_size(signal.size + round(sr * _WRAPAROUND_GUARD_S))
    spectrum = np.fft.rfft(signal, n=padded_size)
    frequencies = np.fft.rfftfreq(padded_size, 1.0 / sr)
    return [np.fft.irfft(spectrum * _highpass_response(frequencies, cutoff), n=padded_size)[:signal.size] for cutoff in cutoffs_hz]


def _fast_fft_size(minimum: int) -> int:
    best = 1 << max(0, (minimum - 1).bit_length())
    power_of_five = 1
    while power_of_five < best:
        power_of_three = power_of_five
        while power_of_three < best:
            candidate = power_of_three << max(0, (-(-minimum // power_of_three) - 1).bit_length())
            best = min(best, candidate)
            power_of_three *= 3
        power_of_five *= 5
    return best


def _highpass_response(frequencies: np.ndarray, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return np.ones_like(frequencies, dtype=np.float32)
    ramp = np.clip((frequencies - cutoff_hz / 2) / (cutoff_hz / 2), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _bridge_short_gaps(speech: np.ndarray, max_gap_windows: int) -> np.ndarray:
    bridged = speech.copy()
    for start, end in _runs(~speech):
        is_internal = start > 0 and end < speech.size
        if is_internal and end - start <= max_gap_windows:
            bridged[start:end] = True
    return bridged


def detect_silences(samples: np.ndarray, sr: int, params: SilenceParams) -> list[Interval]:
    if len(samples) == 0:
        return []
    window = window_length(sr, params.window_s)
    duration = len(samples) / sr
    speech = speech_levels_db(samples, sr, params) > params.threshold_db
    if params.speech_aware:
        speech = _bridge_short_gaps(speech, params.hangover_windows)
    silences: list[Interval] = []
    for start, end in _runs(~speech):
        start_s, end_s = start * window / sr, min(end * window / sr, duration)
        touches_edge = start == 0 or end == speech.size
        if touches_edge or end_s - start_s >= params.min_silence_s - _TIME_EPSILON:
            silences.append((start_s, end_s))
    return silences


def _overlaps_any(start: float, end: float, protected: Sequence[Interval]) -> bool:
    return any(span_start < end and span_end > start for span_start, span_end in protected)


def _removable_part(start: float, end: float, duration: float, params: SilenceParams) -> Interval | None:
    is_leading = start <= _TIME_EPSILON
    is_trailing = end >= duration - _TIME_EPSILON
    if is_leading and is_trailing:
        return None
    if is_leading:
        return 0.0, end - params.leading_edge_s
    if is_trailing:
        return start + params.trailing_edge_s, duration
    if end - start < params.min_silence_s - _TIME_EPSILON:
        return None
    return start + params.pad_after_s, end - (params.retained_pause_s - params.pad_after_s)


def plan_kept_segments(duration: float, silences: Sequence[Interval], params: SilenceParams,
                       protected: Sequence[Interval] = ()) -> list[KeptSegment]:
    removed: list[Interval] = []
    for start, end in sorted(silences):
        if _overlaps_any(start, end, protected):
            continue
        cut = _removable_part(start, end, duration, params)
        if cut is not None and cut[1] - cut[0] > _TIME_EPSILON:
            removed.append(cut)
    return _complement(duration, removed)


def _complement(duration: float, removed: Sequence[Interval]) -> list[KeptSegment]:
    segments: list[KeptSegment] = []
    cursor = out_cursor = 0.0
    for cut_start, cut_end in sorted(removed) + [(duration, duration)]:
        if cut_start - cursor > _TIME_EPSILON:
            segments.append(KeptSegment(cursor, cut_start, out_cursor))
            out_cursor += cut_start - cursor
        cursor = max(cursor, cut_end)
    return segments


def snap_to_samples(segments: Sequence[KeptSegment], sr: int) -> list[KeptSegment]:
    snapped: list[KeptSegment] = []
    out_samples = 0
    for segment in segments:
        start, end = round(segment.src_start * sr), round(segment.src_end * sr)
        if end > start:
            snapped.append(KeptSegment(start / sr, end / sr, out_samples / sr))
            out_samples += end - start
    return snapped


def _fade_curve(length: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(np.pi * (np.arange(length) + 0.5) / length)).astype(np.float32)


def apply_segments(samples: np.ndarray, sr: int, segments: Sequence[KeptSegment], fade_ms: float) -> np.ndarray:
    max_fade = round(fade_ms * sr / 1000.0)
    pieces: list[np.ndarray] = []
    for segment in segments:
        start = max(0, round(segment.src_start * sr))
        end = min(len(samples), round(segment.src_end * sr))
        piece = np.array(samples[start:end], dtype=np.float32)
        fade = min(max_fade, len(piece) // 2)
        if fade > 0:
            ramp = _fade_curve(fade).reshape((fade,) + (1,) * (piece.ndim - 1))
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
        pieces.append(piece)
    if not pieces:
        return np.zeros((0,) + np.shape(samples)[1:], dtype=np.float32)
    return np.concatenate(pieces)


def process_samples(samples: np.ndarray, sr: int, params: SilenceParams,
                    protected: Sequence[Interval] = ()) -> tuple[np.ndarray, list[KeptSegment], list[Interval]]:
    silences = detect_silences(samples, sr, params)
    planned = plan_kept_segments(len(samples) / sr, silences, params, protected)
    segments = snap_to_samples(planned, sr)
    return apply_segments(samples, sr, segments, params.fade_ms), segments, silences


def process_voice(path: str | Path, settings: AudioSettings,
                  protected: Sequence[Interval] = ()) -> tuple[np.ndarray, list[KeptSegment], list[Interval]]:
    samples = decode_audio(path, settings.sample_rate, 1)
    return process_samples(samples, settings.sample_rate, SilenceParams.from_settings(settings), protected)


def map_time(segments: Sequence[KeptSegment], src_time: float) -> float:
    if not segments:
        return 0.0
    position = bisect.bisect_right([segment.src_start for segment in segments], src_time) - 1
    if position < 0:
        return segments[0].out_start
    segment = segments[position]
    return segment.out_start + min(src_time - segment.src_start, segment.duration)
