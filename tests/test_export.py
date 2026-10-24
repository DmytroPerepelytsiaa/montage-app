from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from dima_auto_editor.config import ProjectSettings
from dima_auto_editor.ffmpeg_tools import FFmpegError, probe, run_capture
from dima_auto_editor.render import export
from dima_auto_editor.render.export import count_video_frames, is_within, mux_final, unique_output_path

FPS = 60


def make_video_frames(path: Path, frames: int, ffmpeg_bins) -> Path:
    result = run_capture(["-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=108x192:r={FPS}", "-frames:v", str(frames),
                          "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)], binaries=ffmpeg_bins)
    assert result.returncode == 0
    return path


@pytest.fixture
def settings() -> ProjectSettings:
    settings = ProjectSettings()
    settings.export.encoder = "libx264"
    return settings


def test_mux_final_produces_60fps_h264_with_48k_aac(tmp_path, ffmpeg_bins, make_tone_wav, settings):
    frames = 97
    video = make_video_frames(tmp_path / "composed.mp4", frames, ffmpeg_bins)
    audio = make_tone_wav(tmp_path / "mix.wav", seconds=frames / FPS, sr=44100)
    out = tmp_path / "out" / "final.mp4"
    out.parent.mkdir()
    report = mux_final(video, audio, out, settings, preview=False)
    info = probe(out)
    assert report.issues == []
    assert report.frames == report.expected_frames == frames == count_video_frames(out, ffmpeg_bins)
    assert info.fps == 60
    assert info.sample_rate == 48000
    assert info.has_audio
    assert not out.with_name(out.name + ".part").exists()


def test_mux_final_reports_frame_mismatch(tmp_path, ffmpeg_bins, make_tone_wav, settings):
    video = make_video_frames(tmp_path / "composed.mp4", 30, ffmpeg_bins)
    audio = make_tone_wav(tmp_path / "mix.wav", seconds=1.0)
    report = mux_final(video, audio, tmp_path / "final.mp4", settings, preview=True)
    assert [issue.code for issue in report.issues] == ["export_frame_mismatch"]


def test_mux_final_never_overwrites(tmp_path, ffmpeg_bins, make_tone_wav, settings):
    existing = tmp_path / "final.mp4"
    existing.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        mux_final(tmp_path / "v.mp4", tmp_path / "a.wav", existing, settings, preview=True)
    assert existing.read_bytes() == b"keep me"


def test_hardware_encoder_failure_falls_back_to_libx264(tmp_path, ffmpeg_bins, make_tone_wav, settings, monkeypatch):
    video = make_video_frames(tmp_path / "composed.mp4", 30, ffmpeg_bins)
    audio = make_tone_wav(tmp_path / "mix.wav", seconds=0.5)
    monkeypatch.setattr(export, "preferred_encoder", lambda _settings: "h264_broken_hw")
    monkeypatch.setattr(export, "video_encoder_args",
                        lambda encoder, quality, preset: ["-c:v", "libx264", "-bogus_option_xyz", "1"]
                        if encoder == "h264_broken_hw" else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", str(quality)])
    report = mux_final(video, audio, tmp_path / "final.mp4", settings, preview=True)
    assert report.encoder == "libx264"
    assert report.frames == 30


def test_libx264_failure_is_not_retried(tmp_path, settings):
    with pytest.raises(FFmpegError):
        export.encode_with_fallback(lambda encoder: ["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "x.mp4")],
                                    "libx264", binaries=export.binaries_for(settings))


def test_unique_output_path_is_timestamped_and_never_collides(tmp_path):
    moment = datetime(2026, 3, 1, 14, 5, 9)
    first = unique_output_path(tmp_path, "Einstein Bio", now=moment)
    assert first.name == "Einstein_Bio_20260301_140509.mp4"
    first.write_bytes(b"")
    second = unique_output_path(tmp_path, "Einstein Bio", now=moment)
    assert second.name == "Einstein_Bio_20260301_140509_2.mp4"
    assert unique_output_path(tmp_path, "Einstein Bio", preview=True, now=moment).name == "Einstein_Bio_preview_20260301_140509.mp4"
    assert unique_output_path(tmp_path, "???", now=moment).name == "video_20260301_140509.mp4"


def test_is_within(tmp_path):
    assert is_within(tmp_path / "scenes" / "out", tmp_path / "scenes")
    assert is_within(tmp_path / "scenes", tmp_path / "scenes")
    assert not is_within(tmp_path / "output", tmp_path / "scenes")
    assert not is_within(tmp_path / "output", "")
