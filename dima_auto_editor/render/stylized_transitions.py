from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

STRETCH_LEFT = "stretch_left"
STRETCH_SLIDE = "stretch_slide"
MULTI_TEARS = "multi_tears"
HEATED_TEARS = "heated_tears"
STYLE_LABELS = {
    STRETCH_LEFT: "Stretch Left",
    STRETCH_SLIDE: "Stretch Slide",
    MULTI_TEARS: "Multi Tears",
    HEATED_TEARS: "Heated Tears",
}
STYLIZED_TRANSITIONS = frozenset(STYLE_LABELS)

PAPER_COLOR = np.array([243, 240, 232], dtype=np.float32)
EMBER_CORE = np.array([255, 236, 170], dtype=np.float32)
EMBER_GLOW = np.array([255, 110, 20], dtype=np.float32)
REFERENCE_HEIGHT = 1920.0
TEAR_BANDS = 5


def is_stylized(name: str) -> bool:
    return name in STYLIZED_TRANSITIONS


def smoothstep(value: np.ndarray | float) -> np.ndarray | float:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ease_in(value: float, power: float = 2.2) -> float:
    return float(np.clip(value, 0.0, 1.0)) ** power


def torn_edge(length: int, rng: np.random.Generator, amplitude: float, wavelength: float) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)
    edge = np.zeros(length, dtype=np.float32)
    for octave in range(5):
        frequency = (2 ** octave) / wavelength
        edge += np.sin(2 * np.pi * frequency * positions + rng.uniform(0, 2 * np.pi)) * amplitude / (1.8 ** octave)
    fibres = rng.normal(0.0, amplitude * 0.12, length).astype(np.float32)
    return edge + np.convolve(fibres, np.ones(3, dtype=np.float32) / 3, mode="same")


def box_blur(frame: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius < 1:
        return frame
    size = 2 * radius + 1
    padded = np.pad(frame, [(radius + 1, radius) if dim == axis else (0, 0) for dim in range(frame.ndim)], mode="edge")
    summed = np.cumsum(padded, axis=axis, dtype=np.float32)
    upper = np.take(summed, np.arange(size, summed.shape[axis]), axis=axis)
    lower = np.take(summed, np.arange(0, summed.shape[axis] - size), axis=axis)
    return (upper - lower) / size


@dataclass
class _Context:
    height: int
    width: int
    frames: int
    rng: np.random.Generator

    @property
    def scale(self) -> float:
        return self.height / REFERENCE_HEIGHT


Renderer = Callable[[_Context], Callable[[np.ndarray, np.ndarray, float], np.ndarray]]
FrameRenderer = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def _strip_whip(ctx: _Context, axis: int) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    length = ctx.width if axis == 1 else ctx.height
    coordinates = np.arange(length, dtype=np.float32)
    centre = length / 2

    def offset(progress: float) -> float:
        return float(smoothstep(progress)) * length

    def frame(a: np.ndarray, b: np.ndarray, progress: float) -> np.ndarray:
        stretch = 1.0 + 2.2 * np.sin(np.pi * progress) ** 2
        source = offset(progress) + centre + (coordinates - centre) / stretch
        from_a = source < length
        index_a = np.clip(source, 0, length - 1).astype(np.int32)
        index_b = np.clip(source - length, 0, length - 1).astype(np.int32)
        if axis == 1:
            mixed = np.where(from_a[None, :, None], a[:, index_a], b[:, index_b])
        else:
            mixed = np.where(from_a[:, None, None], a[index_a], b[index_b])
        step = abs(offset(min(1.0, progress + 0.5 / ctx.frames)) - offset(max(0.0, progress - 0.5 / ctx.frames)))
        return box_blur(mixed.astype(np.float32), int(min(length * 0.12, step * 0.9)), axis)

    return frame


def _stretch_left(ctx: _Context):
    return _strip_whip(ctx, axis=1)


def _stretch_slide(ctx: _Context):
    return _strip_whip(ctx, axis=0)


def _settle_zoom(image: np.ndarray, progress: float, rows: np.ndarray, cols: np.ndarray, start_zoom: float = 1.08) -> np.ndarray:
    zoom = 1.0 + (start_zoom - 1.0) * (1.0 - float(smoothstep(progress)))
    height, width = image.shape[:2]
    row_index = np.clip((rows - height / 2) / zoom + height / 2, 0, height - 1).astype(np.int32)
    col_index = np.clip((cols - width / 2) / zoom + width / 2, 0, width - 1).astype(np.int32)
    return image[row_index][:, col_index].astype(np.float32)


def _multi_tears(ctx: _Context):
    height, width, scale = ctx.height, ctx.width, ctx.scale
    rows = np.arange(height, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)
    boundaries = [height * band / TEAR_BANDS + torn_edge(width, ctx.rng, 22 * scale, width * 0.6)
                  for band in range(1, TEAR_BANDS)]
    band_of = np.zeros((height, width), dtype=np.int8)
    paper = np.zeros((height, width), dtype=bool)
    for boundary in boundaries:
        below = rows[:, None] >= boundary[None, :]
        band_of += below
        rim = np.abs(rows[:, None] - boundary[None, :]) < (4 + 5 * ctx.rng.random(width).astype(np.float32))[None, :] * scale
        paper |= rim
    directions = [(-1) ** band for band in range(TEAR_BANDS)]
    delays = ctx.rng.uniform(0.0, 0.3, TEAR_BANDS)
    shadow_offset = max(2, int(10 * scale))

    def frame(a: np.ndarray, b: np.ndarray, progress: float) -> np.ndarray:
        out = _settle_zoom(b, progress, rows, cols)
        covered = np.zeros((height, width), dtype=bool)
        layers = []
        for band in range(TEAR_BANDS):
            travel = ease_in((progress - delays[band]) / (1.0 - delays[band])) * width * 1.15
            source = cols - directions[band] * travel
            valid = (source >= 0) & (source < width)
            index = np.clip(source, 0, width - 1).astype(np.int32)
            mask = (band_of[:, index] == band) & valid[None, :]
            layers.append((mask, index))
            covered |= mask
        shadow = np.roll(np.roll(covered, shadow_offset, axis=0), shadow_offset, axis=1) & ~covered
        out[shadow] *= 0.55
        for mask, index in layers:
            torn = a[:, index].astype(np.float32)
            torn[paper[:, index]] = PAPER_COLOR
            out[mask] = torn[mask]
        return out

    return frame


def _heated_tears(ctx: _Context):
    height, width, scale = ctx.height, ctx.width, ctx.scale
    rows = np.arange(height, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)
    edge = width / 2 + torn_edge(height, ctx.rng, 38 * scale, height * 0.45)
    flicker_seed = ctx.rng.uniform(0, 2 * np.pi, 3)
    reach = width / 2 + 60 * scale + float(np.abs(edge - width / 2).max())

    def frame(a: np.ndarray, b: np.ndarray, progress: float) -> np.ndarray:
        heat = float(smoothstep(progress / 0.25))
        spread = ease_in(progress, 2.0) * reach
        left_edge = (edge - spread)[:, None]
        right_edge = (edge + spread)[:, None]
        in_left = cols[None, :] < left_edge
        in_right = cols[None, :] > right_edge
        out = _settle_zoom(b, progress, rows, cols)
        shifted_left = a[:, np.clip(cols + spread, 0, width - 1).astype(np.int32)].astype(np.float32)
        shifted_right = a[:, np.clip(cols - spread, 0, width - 1).astype(np.int32)].astype(np.float32)
        out = np.where(in_left[..., None], shifted_left, out)
        out = np.where(in_right[..., None], shifted_right, out)

        distance = np.where(in_left, left_edge - cols[None, :], np.where(in_right, cols[None, :] - right_edge, 0.0))
        gap = np.minimum(np.abs(cols[None, :] - left_edge), np.abs(right_edge - cols[None, :]))
        flicker = 0.75 + 0.25 * np.sin(rows * 0.045 / scale + flicker_seed[0] + progress * 25)[:, None]
        on_paper = in_left | in_right
        char = np.where(on_paper, np.exp(-distance / (34 * scale)) * 0.65 * heat, 0.0)[..., None]
        core = np.where(on_paper, np.exp(-distance / (5 * scale)), 0.0)[..., None] * heat * flicker[..., None]
        halo = np.exp(-np.where(on_paper, distance, gap) / (26 * scale))[..., None] * heat * flicker[..., None]
        out = out * (1.0 - char) + EMBER_GLOW * halo * 0.85 + EMBER_CORE * core
        return out

    return frame


RENDERERS: dict[str, Renderer] = {
    STRETCH_LEFT: _stretch_left,
    STRETCH_SLIDE: _stretch_slide,
    MULTI_TEARS: _multi_tears,
    HEATED_TEARS: _heated_tears,
}


def make_renderer(style: str, height: int, width: int, frame_count: int, seed: int) -> FrameRenderer:
    draw = RENDERERS[style](_Context(height, width, frame_count, np.random.default_rng(seed)))

    def render(outgoing: np.ndarray, incoming: np.ndarray, index: int) -> np.ndarray:
        return np.clip(draw(outgoing, incoming, (index + 0.5) / frame_count), 0, 255).astype(np.uint8)

    return render
