from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import AudioSettings
from ..ffmpeg_tools import FFmpegError, decode_audio, write_wav
from .silence import power_to_db, window_power

CHANNELS = 2
CONTROL_BLOCK_S = 0.01
LIMITER_BLOCK_SAMPLES = 32
DUCK_DEPTH_DB = 8.0
DUCK_ATTACK_S = 0.08
DUCK_RELEASE_S = 0.4
DUCK_HOLD_S = 0.25
SFX_HEADROOM_DB = 6.0
SFX_CAP_ATTACK_DB_PER_BLOCK = 6.0
SFX_CAP_RELEASE_DB_PER_S = 30.0
LIMITER_ATTACK_DB_PER_MS = 2.0
LIMITER_RELEASE_DB_PER_S = 60.0
TAIL_FADE_S = 0.005
LOOP_SEAM_FADE_S = 0.01
MUSIC_FADE_IN_S = 0.3
MUSIC_FADE_OUT_S = 1.5
_SILENCE_DB = -120.0
_DEFAULT_SFX_VOLUME_DB = AudioSettings().sfx_volume_db


@dataclass
class SfxCue:
    path: str
    start: float
    volume_db: float


def db_to_gain(db: float | np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def amplitude_db(peak: float) -> float:
    return round(20.0 * float(np.log10(peak)), 2) if peak > 0 else _SILENCE_DB


def mix_final_audio(narration_wav: str, duration: float, cues: list[SfxCue], music_path: str,
                    settings: AudioSettings, out_path: Path) -> dict:
    sample_rate = settings.sample_rate
    length = max(0, round(duration * sample_rate))
    control_block = max(1, round(CONTROL_BLOCK_S * sample_rate))
    warnings: list[str] = []

    voice = _fit_length(decode_audio(narration_wav, sample_rate, 1), length) * np.float32(db_to_gain(settings.voice_gain_db))
    voice_active, speech_level_db = _voice_activity(voice, control_block, settings.silence_threshold_db)

    sfx_master_db = settings.sfx_volume_db - _DEFAULT_SFX_VOLUME_DB
    sfx_bus, placed_cues = _render_sfx_bus(cues, length, sample_rate, sfx_master_db, warnings)
    sfx_reduction_db = 0.0
    if speech_level_db is not None:
        sfx_bus, sfx_reduction_db = _cap_level(sfx_bus, speech_level_db - SFX_HEADROOM_DB, control_block, sample_rate)

    music = _render_music(music_path, length, sample_rate, settings, voice_active, control_block, warnings)

    mix = voice[:, None] + sfx_bus + music
    output, limiter_reduction_db = _limit_peaks(mix, settings.limiter_ceiling_db, sample_rate)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, output, sample_rate)

    return {
        "sample_rate": sample_rate,
        "length_samples": length,
        "duration": length / sample_rate,
        "peak_db": amplitude_db(_peak(output)),
        "ceiling_db": settings.limiter_ceiling_db,
        "voice_rms_db": round(speech_level_db, 2) if speech_level_db is not None else _SILENCE_DB,
        "sfx_peak_db": amplitude_db(_peak(sfx_bus)),
        "sfx_cap_reduction_db": round(sfx_reduction_db, 2),
        "music_rms_db": _rms_db(music),
        "limiter_reduction_db": round(limiter_reduction_db, 2),
        "sfx_cues_placed": placed_cues,
        "sfx_cues_total": len(cues),
        "warnings": warnings,
    }


def _rms_db(signal: np.ndarray) -> float:
    if signal.size == 0:
        return _SILENCE_DB
    return round(float(power_to_db(np.mean(np.square(signal, dtype=np.float64)))), 2)


def _peak(signal: np.ndarray) -> float:
    return float(np.max(np.abs(signal))) if signal.size else 0.0


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    fitted = np.zeros((length,) + samples.shape[1:], dtype=np.float32)
    take = min(length, len(samples))
    fitted[:take] = samples[:take]
    return fitted


def _voice_activity(voice: np.ndarray, block: int, threshold_db: float) -> tuple[np.ndarray, float | None]:
    power = window_power(voice, block)
    active = power_to_db(power) > threshold_db
    if not active.any():
        return active, None
    return active, float(power_to_db(power[active].mean()))


def rate_limited_gain_db(target_db: np.ndarray, attack_db_per_block: float, release_db_per_block: float) -> np.ndarray:
    positions = np.arange(target_db.size, dtype=np.float64)
    released = np.minimum.accumulate(target_db - positions * release_db_per_block) + positions * release_db_per_block
    anticipated = np.minimum.accumulate((target_db + positions * attack_db_per_block)[::-1])[::-1] - positions * attack_db_per_block
    return np.minimum(released, anticipated)


def _block_gain_to_samples(gain_db: np.ndarray, block: int, length: int) -> np.ndarray:
    block_gain = db_to_gain(gain_db).astype(np.float32)
    falling_delta = np.minimum(0.0, np.diff(block_gain, append=block_gain[-1]))
    position_in_block = np.arange(block, dtype=np.float32) / block
    return (block_gain[:, None] + falling_delta[:, None] * position_in_block[None, :]).ravel()[:length]


def _render_sfx_bus(cues: Sequence[SfxCue], length: int, sample_rate: int, master_db: float,
                    warnings: list[str]) -> tuple[np.ndarray, int]:
    bus = np.zeros((length, CHANNELS), dtype=np.float32)
    decoded: dict[str, np.ndarray | None] = {}
    placed = 0
    for cue in cues:
        name = Path(cue.path).name
        start = round(max(0.0, cue.start) * sample_rate)
        if start >= length:
            warnings.append(f"Sound effect '{name}' at {cue.start:.2f} s starts after the end of the video and was not placed.")
            continue
        if cue.path not in decoded:
            decoded[cue.path] = _decode_or_warn(cue.path, sample_rate, warnings)
        source = decoded[cue.path]
        if source is None or len(source) == 0:
            continue
        take = min(len(source), length - start)
        chunk = source[:take] * np.float32(db_to_gain(cue.volume_db + master_db))
        if take < len(source):
            _fade_out_tail(chunk, round(TAIL_FADE_S * sample_rate))
        bus[start:start + take] += chunk
        placed += 1
    return bus, placed


def _decode_or_warn(path: str, sample_rate: int, warnings: list[str]) -> np.ndarray | None:
    try:
        samples = decode_audio(path, sample_rate, CHANNELS)
    except (FFmpegError, OSError) as exc:
        warnings.append(f"Sound effect '{Path(path).name}' could not be decoded and was skipped: {str(exc).splitlines()[0]}")
        return None
    if len(samples) == 0:
        warnings.append(f"Sound effect '{Path(path).name}' is empty and was skipped.")
    return samples


def _fade_out_tail(chunk: np.ndarray, fade_length: int) -> None:
    fade_length = min(fade_length, len(chunk))
    if fade_length > 0:
        chunk[-fade_length:] *= np.linspace(1.0, 0.0, fade_length, dtype=np.float32)[:, None]


def _fade_edges(signal: np.ndarray, fade_in: int, fade_out: int) -> None:
    fade_in, fade_out = min(fade_in, len(signal) // 2), min(fade_out, len(signal) // 2)
    if fade_in > 0:
        signal[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    if fade_out > 0:
        signal[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]


def _cap_level(bus: np.ndarray, cap_db: float, block: int, sample_rate: int) -> tuple[np.ndarray, float]:
    if not bus.any():
        return bus, 0.0
    level_db = power_to_db(window_power(bus, block))
    target_db = np.minimum(0.0, cap_db - level_db)
    if target_db.min() >= 0.0:
        return bus, 0.0
    release_per_block = SFX_CAP_RELEASE_DB_PER_S * block / sample_rate
    gain_db = rate_limited_gain_db(target_db, SFX_CAP_ATTACK_DB_PER_BLOCK, release_per_block)
    return bus * _block_gain_to_samples(gain_db, block, len(bus))[:, None], float(-gain_db.min())


def _render_music(music_path: str, length: int, sample_rate: int, settings: AudioSettings,
                  voice_active: np.ndarray, block: int, warnings: list[str]) -> np.ndarray:
    silence = np.zeros((length, CHANNELS), dtype=np.float32)
    if not music_path or length == 0:
        return silence
    try:
        source = decode_audio(music_path, sample_rate, CHANNELS)
    except (FFmpegError, OSError) as exc:
        warnings.append(f"Background music '{Path(music_path).name}' could not be decoded and was left out: {str(exc).splitlines()[0]}")
        return silence
    if len(source) == 0:
        warnings.append(f"Background music '{Path(music_path).name}' is empty and was left out.")
        return silence

    if len(source) < length:
        seam = round(LOOP_SEAM_FADE_S * sample_rate)
        _fade_edges(source, seam, seam)
        source = np.tile(source, (-(-length // len(source)), 1))
    music = source[:length] * np.float32(db_to_gain(settings.music_volume_db))
    _fade_edges(music, round(MUSIC_FADE_IN_S * sample_rate), round(MUSIC_FADE_OUT_S * sample_rate))
    if settings.music_ducking and voice_active.any():
        music *= _ducking_gain(voice_active, block, length, sample_rate)[:, None]
    return music


def _ducking_gain(voice_active: np.ndarray, block: int, length: int, sample_rate: int) -> np.ndarray:
    hold_blocks = max(1, round(DUCK_HOLD_S * sample_rate / block))
    held = np.convolve(voice_active.astype(np.float32), np.ones(hold_blocks, dtype=np.float32))[:voice_active.size] > 0
    target_db = np.where(held, -DUCK_DEPTH_DB, 0.0)
    blocks_per_second = sample_rate / block
    gain_db = rate_limited_gain_db(target_db, DUCK_DEPTH_DB / (DUCK_ATTACK_S * blocks_per_second),
                                   DUCK_DEPTH_DB / (DUCK_RELEASE_S * blocks_per_second))
    return _block_gain_to_samples(gain_db, block, length)


def _limit_peaks(mix: np.ndarray, ceiling_db: float, sample_rate: int) -> tuple[np.ndarray, float]:
    ceiling = float(db_to_gain(ceiling_db))
    if mix.size == 0:
        return mix.astype(np.float32), 0.0
    block = LIMITER_BLOCK_SAMPLES
    sample_peaks = np.abs(mix).max(axis=1)
    block_peaks = np.maximum.reduceat(sample_peaks, np.arange(0, len(mix), block))
    target_db = np.minimum(0.0, ceiling_db - 20.0 * np.log10(np.maximum(block_peaks, 1e-9)))
    if target_db.min() >= 0.0:
        return np.clip(mix, -ceiling, ceiling).astype(np.float32), 0.0
    block_ms = block * 1000.0 / sample_rate
    gain_db = rate_limited_gain_db(target_db, LIMITER_ATTACK_DB_PER_MS * block_ms, LIMITER_RELEASE_DB_PER_S * block_ms / 1000.0)
    limited = mix * _block_gain_to_samples(gain_db, block, len(mix))[:, None]
    np.clip(limited, -ceiling, ceiling, out=limited)
    return limited.astype(np.float32), float(-gain_db.min())
