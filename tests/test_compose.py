from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dima_auto_editor.config import ProjectSettings
from dima_auto_editor.ffmpeg_tools import probe, run_capture
from dima_auto_editor.models import MediaKind, NumberedFile, Timeline, TimelineEntry, TransitionEvent
from dima_auto_editor.render import compose
from dima_auto_editor.render.compose import clip_frame_counts, compose_video, plan_chunks
from dima_auto_editor.render.export import count_video_frames

FPS = 60
COLORS = ["red", "blue", "lime", "yellow", "white", "black"]
RGB = {"red": (255, 0, 0), "blue": (0, 0, 255), "lime": (0, 255, 0), "yellow": (255, 255, 0)}


def scene(index: int) -> NumberedFile:
    return NumberedFile(path=f"{index:02d}.png", index=index, sub_index=0, stem=f"{index:02d}", kind=MediaKind.IMAGE)


def make_timeline(boundary_frames: list[int], transition_frames: dict[int, int]) -> Timeline:
    entries = [TimelineEntry(position=i, scene=scene(i + 1), start=start / FPS, end=end / FPS)
               for i, (start, end) in enumerate(zip(boundary_frames, boundary_frames[1:]))]
    transitions = [TransitionEvent(boundary_position=position, time=entries[position].end, duration=frames / FPS,
                                   name="Fade", xfade="fade")
                   for position, frames in transition_frames.items()]
    return Timeline(entries=entries, transitions=transitions, duration=boundary_frames[-1] / FPS, fps=FPS)


def make_color_clips(tmp_path: Path, timeline: Timeline, ffmpeg_bins) -> list[Path]:
    paths = []
    for position, frames in enumerate(clip_frame_counts(timeline)):
        path = tmp_path / f"clip_{position:03d}.mp4"
        color = COLORS[position % len(COLORS)]
        result = run_capture(["-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=108x192:r={FPS}",
                              "-frames:v", str(frames), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
                              str(path)], binaries=ffmpeg_bins)
        assert result.returncode == 0
        paths.append(path)
    return paths


def decode_rgb_means(path: Path, ffmpeg_bins) -> np.ndarray:
    result = run_capture(["-v", "error", "-i", str(path), "-vf", "scale=4:4,format=rgb24", "-f", "rawvideo", "pipe:1"],
                         binaries=ffmpeg_bins)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 16, 3).mean(axis=1)


@pytest.fixture
def settings() -> ProjectSettings:
    settings = ProjectSettings()
    settings.export.encoder = "libx264"
    return settings


def test_clip_frame_counts_include_half_transitions():
    timeline = make_timeline([0, 120, 200, 330], {0: 30})
    assert clip_frame_counts(timeline) == [135, 95, 130]


def test_compose_mixed_transitions_and_cuts_has_exact_length(tmp_path, ffmpeg_bins, settings):
    timeline = make_timeline([0, 100, 190, 260, 377], {0: 30, 2: 20})
    clips = make_color_clips(tmp_path, timeline, ffmpeg_bins)
    report = compose_video(clips, timeline, settings, tmp_path / "composed.mp4", preview=True)
    assert report.frames == report.expected_frames == 377 == count_video_frames(tmp_path / "composed.mp4", ffmpeg_bins)
    assert report.issues == []
    assert probe(tmp_path / "composed.mp4").fps == 60

    colors = decode_rgb_means(tmp_path / "composed.mp4", ffmpeg_bins)
    assert np.allclose(colors[84], RGB["red"], atol=12)
    assert np.allclose(colors[116], RGB["blue"], atol=12)
    midpoint = colors[100]
    assert 60 < midpoint[0] < 200 and 60 < midpoint[2] < 200
    assert np.allclose(colors[189], RGB["blue"], atol=12)
    assert np.allclose(colors[190], RGB["lime"], atol=12)


def test_compose_single_clip(tmp_path, ffmpeg_bins, settings):
    timeline = make_timeline([0, 45], {})
    clips = make_color_clips(tmp_path, timeline, ffmpeg_bins)
    report = compose_video(clips, timeline, settings, tmp_path / "composed.mp4", preview=True)
    assert report.frames == 45


def test_chunks_split_only_at_hard_cuts():
    timeline = make_timeline(list(range(0, 11 * 40, 40)), {0: 10, 1: 10, 3: 10, 5: 10, 6: 10, 7: 10})
    chunks = plan_chunks(timeline, max_clips=4)
    assert [(chunk.first, chunk.last) for chunk in chunks] == [(0, 2), (3, 4), (5, 8), (9, 9)]
    for chunk in chunks[:-1]:
        assert timeline.transition_at(chunk.last) is None


def test_chunked_compose_matches_duration(tmp_path, ffmpeg_bins, settings, monkeypatch):
    monkeypatch.setattr(compose, "MAX_CLIPS_PER_CHUNK", 2)
    timeline = make_timeline([0, 40, 80, 130, 170, 211], {0: 10, 2: 10})
    clips = make_color_clips(tmp_path, timeline, ffmpeg_bins)
    report = compose_video(clips, timeline, settings, tmp_path / "composed.mp4", preview=True)
    assert report.chunk_count == 3
    assert report.frames == 211
    assert sorted(p.name for p in tmp_path.iterdir() if "chunk" in p.name) == []


def test_compose_rejects_clip_count_mismatch(tmp_path, settings):
    with pytest.raises(ValueError):
        compose_video([], make_timeline([0, 30], {}), settings, tmp_path / "x.mp4", preview=True)
