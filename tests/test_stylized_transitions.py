from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dima_auto_editor.config import ProjectSettings
from dima_auto_editor.models import TransitionEvent
from dima_auto_editor.render.compose import clip_frame_counts, compose_video
from dima_auto_editor.render.export import count_video_frames
from dima_auto_editor.render.stylized_transitions import STYLIZED_TRANSITIONS, box_blur, make_renderer
from dima_auto_editor.transitions import stylized_library, validate_library

from test_compose import decode_rgb_means, make_color_clips, make_timeline

HEIGHT, WIDTH, FRAMES = 192, 108, 54
OUTGOING = np.full((HEIGHT, WIDTH, 3), (200, 30, 30), dtype=np.uint8)
INCOMING = np.full((HEIGHT, WIDTH, 3), (30, 30, 200), dtype=np.uint8)


def share_like(frame: np.ndarray, reference: np.ndarray) -> float:
    other = INCOMING if reference is OUTGOING else OUTGOING
    to_reference = np.abs(frame.astype(int) - reference).sum(axis=2)
    to_other = np.abs(frame.astype(int) - other).sum(axis=2)
    return float((to_reference < to_other).mean())


@pytest.mark.parametrize("style", sorted(STYLIZED_TRANSITIONS))
def test_transition_starts_on_outgoing_and_ends_on_incoming(style):
    draw = make_renderer(style, HEIGHT, WIDTH, FRAMES, seed=5)
    first, middle, last = draw(OUTGOING, INCOMING, 0), draw(OUTGOING, INCOMING, FRAMES // 2), draw(OUTGOING, INCOMING, FRAMES - 1)
    assert first.shape == (HEIGHT, WIDTH, 3) and first.dtype == np.uint8
    assert share_like(first, OUTGOING) > 0.8
    assert share_like(last, INCOMING) > 0.8
    assert 0.0 < share_like(middle, INCOMING) < 1.0 or style.startswith("stretch")


@pytest.mark.parametrize("style", sorted(STYLIZED_TRANSITIONS))
def test_transition_is_deterministic_for_a_seed(style):
    first = make_renderer(style, HEIGHT, WIDTH, FRAMES, seed=11)(OUTGOING, INCOMING, 20)
    second = make_renderer(style, HEIGHT, WIDTH, FRAMES, seed=11)(OUTGOING, INCOMING, 20)
    assert np.array_equal(first, second)


def test_box_blur_keeps_shape_and_constant_images():
    image = np.full((10, 20, 3), 77.0, dtype=np.float32)
    blurred = box_blur(image, 4, axis=1)
    assert blurred.shape == image.shape
    assert np.allclose(blurred, 77.0)


def test_stylized_library_is_valid_and_carries_every_sound():
    library = stylized_library(["multi_tears", "stretch_left", "unknown"], ["C:/sfx/a.wav", "C:/sfx/b.wav"], 0.9)
    assert [transition.xfade for transition in library] == ["multi_tears", "stretch_left"]
    assert all(transition.duration_s == 0.9 and transition.sfx_files == ["C:/sfx/a.wav", "C:/sfx/b.wav"] for transition in library)
    assert not [issue for issue in validate_library(library, ["C:/sfx/a.wav", "C:/sfx/b.wav"]) if issue.code == "unknown_xfade"]


def test_compose_with_stylized_transition_keeps_exact_length(tmp_path, ffmpeg_bins):
    timeline = make_timeline([0, 120, 240], {0: 54})
    timeline.transitions = [TransitionEvent(0, 2.0, 54 / 60, "Multi Tears", "multi_tears")]
    settings = ProjectSettings()
    settings.export.encoder = "libx264"
    settings.video.width, settings.video.height = 108, 192
    clips = make_color_clips(tmp_path, timeline, ffmpeg_bins)
    report = compose_video(clips, timeline, settings, tmp_path / "out.mp4", preview=False)
    assert report.frames == report.expected_frames == 240 == count_video_frames(tmp_path / "out.mp4", ffmpeg_bins)
    means = decode_rgb_means(tmp_path / "out.mp4", ffmpeg_bins)
    assert means[30, 0] > 200 and means[210, 2] > 200
    mid = means[120]
    assert mid[0] > 40 and mid[2] > 40
    assert clip_frame_counts(timeline) == [147, 147]
