from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dima_auto_editor.config import MOTION_CHOPPY, MOTION_SMOOTH, ProjectSettings
from dima_auto_editor.ffmpeg_tools import probe, run_capture
from dima_auto_editor.models import MediaKind, NumberedFile
from dima_auto_editor.render.clips import choose_smart_window, output_size, render_scene_clip
from dima_auto_editor.render.export import count_video_frames
from dima_auto_editor.render.rife import ENGINE_MINTERPOLATE, ENGINE_RIFE, find_rife


def make_settings(**video_overrides) -> ProjectSettings:
    settings = ProjectSettings()
    settings.export.encoder = "libx264"
    for key, value in video_overrides.items():
        setattr(settings.video, key, value)
    return settings


def scene_for(path: Path, kind: MediaKind) -> NumberedFile:
    return NumberedFile(path=str(path), index=1, sub_index=0, stem=path.stem, kind=kind)


def render(path: Path, kind: MediaKind, seconds: float, out: Path, settings: ProjectSettings, preview: bool = True):
    report = render_scene_clip(scene_for(path, kind), probe(path), seconds, out, settings, preview=preview)
    return report, probe(out)


def consecutive_frame_differences(path: Path, ffmpeg_bins, count: int) -> np.ndarray:
    result = run_capture(["-v", "error", "-i", str(path), "-vf", "scale=64:112:flags=area,format=gray",
                          "-frames:v", str(count), "-f", "rawvideo", "pipe:1"], binaries=ffmpeg_bins)
    assert result.returncode == 0
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 112, 64).astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


def test_image_clip_has_exact_frames_and_output_size(tmp_path, make_image, ffmpeg_bins):
    image = make_image(tmp_path / "src" / "01.png")
    settings = make_settings(image_motion="zoom_in")
    report, info = render(image, MediaKind.IMAGE, 1.2583, tmp_path / "clip.mp4", settings)
    assert report.frames == 75 == count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins)
    assert (info.width, info.height) == output_size(settings, preview=True) == (540, 960)
    assert info.fps == 60
    assert report.method == "image_zoom_in"


def test_static_image_full_resolution(tmp_path, make_image, ffmpeg_bins):
    image = make_image(tmp_path / "src" / "02.png", size=(300, 200))
    report, info = render(image, MediaKind.IMAGE, 0.5, tmp_path / "clip.mp4",
                          make_settings(image_motion="none", fit_mode="contain_blur"), preview=False)
    assert (info.width, info.height) == (1080, 1920)
    assert count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins) == 30
    assert report.method == "image_hold"


@pytest.mark.parametrize("extend_mode", ["slow", "freeze", "loop", "pingpong"])
def test_short_video_is_extended_to_exact_frames(tmp_path, make_video, ffmpeg_bins, extend_mode):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=1.0, fps=12, size=(160, 284))
    report, info = render(video, MediaKind.VIDEO, 2.5, tmp_path / "clip.mp4",
                          make_settings(extend_mode=extend_mode, fit_mode="contain_black"))
    assert report.extended and not report.trimmed
    assert count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins) == 150
    assert info.fps == 60
    if extend_mode == "slow":
        assert report.speed_factor == pytest.approx(2.5, rel=0.05)
        assert report.method.startswith("extend_slow") and report.interpolated


def test_long_video_is_trimmed_to_exact_frames(tmp_path, make_video, ffmpeg_bins):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=3.0, fps=24, size=(160, 284))
    report, _ = render(video, MediaKind.VIDEO, 1.0, tmp_path / "clip.mp4", make_settings(trim_mode="smart"))
    assert report.trimmed and not report.extended
    assert report.method == "trim_smart"
    assert 0.0 <= report.source_start <= 2.0
    assert count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins) == 60


def test_choppy_mode_repeats_each_12fps_frame_five_times(tmp_path, make_video, ffmpeg_bins):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=1.0, fps=12, size=(160, 284))
    report, _ = render(video, MediaKind.VIDEO, 1.0, tmp_path / "clip.mp4",
                       make_settings(motion_mode=MOTION_CHOPPY, scale_filter="neighbor"))
    assert not report.interpolated and report.warnings == []
    differences = consecutive_frame_differences(tmp_path / "clip.mp4", ffmpeg_bins, 30)
    source_frame_changes = np.arange(4, 29, 5)
    repeated = np.setdiff1d(np.arange(29), source_frame_changes)
    assert differences[repeated].max() < 0.5
    assert differences[source_frame_changes].min() > 2.0


def test_smooth_mode_interpolates_and_warns_for_pixel_art(tmp_path, make_video, ffmpeg_bins):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=0.5, fps=12, size=(96, 160))
    report, _ = render(video, MediaKind.VIDEO, 0.5, tmp_path / "clip.mp4",
                       make_settings(motion_mode=MOTION_SMOOTH, scale_filter="neighbor"))
    assert report.interpolated
    assert [issue.code for issue in report.warnings] == ["interpolation_pixel_art"]
    assert count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins) == 30
    differences = consecutive_frame_differences(tmp_path / "clip.mp4", ffmpeg_bins, 10)
    assert (differences[:4] > 0.5).sum() >= 3


def test_smooth_mode_skips_interpolation_for_60fps_source(tmp_path, make_video, ffmpeg_bins):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=1.0, fps=60, size=(160, 284))
    report, _ = render(video, MediaKind.VIDEO, 0.5, tmp_path / "clip.mp4",
                       make_settings(motion_mode=MOTION_SMOOTH, scale_filter="lanczos"))
    assert not report.interpolated and report.warnings == []
    assert count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins) == 30


def test_smart_window_avoids_hard_cut_and_prefers_calm_ending():
    rng = np.random.default_rng(0)
    shot_a = np.repeat(rng.integers(0, 60, (1, 8, 8)), 20, axis=0).astype(np.uint8)
    shot_a = (shot_a + np.arange(20)[:, None, None] * 2).astype(np.uint8)
    shot_b = np.repeat(rng.integers(180, 255, (1, 8, 8)), 20, axis=0).astype(np.uint8)
    proxy = np.concatenate([shot_a, shot_b])
    start = choose_smart_window(proxy, 15)
    assert start >= 20


def test_smart_window_returns_zero_when_window_covers_everything():
    assert choose_smart_window(np.zeros((10, 8, 8), dtype=np.uint8), 10) == 0


def duplicate_ratio(path: Path, ffmpeg_bins, count: int) -> float:
    return float((consecutive_frame_differences(path, ffmpeg_bins, count) < 0.05).mean())


def test_slowed_clip_gets_generated_frames_in_choppy_mode(tmp_path, make_video, ffmpeg_bins):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=1.0, fps=24, size=(160, 284))
    settings = make_settings(motion_mode=MOTION_CHOPPY, interpolation_engine=ENGINE_MINTERPOLATE)
    report, _ = render(video, MediaKind.VIDEO, 2.0, tmp_path / "clip.mp4", settings)
    assert report.interpolated and report.method == "extend_slow_interpolated"
    assert count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins) == 120
    assert duplicate_ratio(tmp_path / "clip.mp4", ffmpeg_bins, 120) < 0.2


def test_slowed_clip_repeats_frames_when_generation_is_off(tmp_path, make_video, ffmpeg_bins):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=1.0, fps=24, size=(160, 284))
    settings = make_settings(motion_mode=MOTION_CHOPPY, interpolate_slowed_clips=False)
    report, _ = render(video, MediaKind.VIDEO, 2.0, tmp_path / "clip.mp4", settings)
    assert not report.interpolated and report.method == "extend_slow"
    assert duplicate_ratio(tmp_path / "clip.mp4", ffmpeg_bins, 120) > 0.5


@pytest.mark.skipif(find_rife(ProjectSettings()) is None, reason="RIFE is not installed in tools/rife")
def test_rife_generates_exact_smooth_frames(tmp_path, make_video, ffmpeg_bins):
    video = make_video(tmp_path / "src" / "01.mp4", seconds=1.0, fps=24, size=(160, 284))
    settings = make_settings(motion_mode=MOTION_CHOPPY, interpolation_engine=ENGINE_RIFE)
    report, info = render(video, MediaKind.VIDEO, 2.0, tmp_path / "clips" / "clip.mp4", settings)
    assert report.method == "extend_slow_rife"
    assert not {"rife_failed", "rife_missing"} & {warning.code for warning in report.warnings}
    assert count_video_frames(tmp_path / "clips" / "clip.mp4", ffmpeg_bins) == 120 and info.fps == 60
    assert duplicate_ratio(tmp_path / "clips" / "clip.mp4", ffmpeg_bins, 120) < 0.2
    assert [path.name for path in (tmp_path / "clips").iterdir()] == ["clip.mp4"]


def test_failing_rife_falls_back_to_ffmpeg(tmp_path, make_video, ffmpeg_bins):
    fake_dir = tmp_path / "fake_rife"
    (fake_dir / "rife-v4.6").mkdir(parents=True)
    fake_rife = fake_dir / "rife-ncnn-vulkan.bat"
    fake_rife.write_text("@exit /b 3\n", encoding="ascii")
    video = make_video(tmp_path / "src" / "01.mp4", seconds=1.0, fps=24, size=(160, 284))
    settings = make_settings(interpolation_engine=ENGINE_RIFE, rife_executable=str(fake_rife))
    report, _ = render(video, MediaKind.VIDEO, 2.0, tmp_path / "clip.mp4", settings)
    assert report.method == "extend_slow_interpolated"
    assert "rife_failed" in [warning.code for warning in report.warnings]
    assert count_video_frames(tmp_path / "clip.mp4", ffmpeg_bins) == 120
