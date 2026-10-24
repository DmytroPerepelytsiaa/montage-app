from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("dima_auto_editor.inputs")
pytest.importorskip("dima_auto_editor.audio.narration")
pytest.importorskip("dima_auto_editor.audio.mixer")
pytest.importorskip("dima_auto_editor.timeline")
pytest.importorskip("dima_auto_editor.transitions")
pytest.importorskip("dima_auto_editor.manifest")

from dima_auto_editor.config import ProjectSettings  # noqa: E402
from dima_auto_editor.ffmpeg_tools import RenderCancelled, probe  # noqa: E402
from dima_auto_editor.manifest import load_manifest  # noqa: E402
from dima_auto_editor.models import MediaKind, NumberedFile, Timeline, TimelineEntry, TransitionEvent  # noqa: E402
from dima_auto_editor.pipeline import BUILD_STAGES, BuildBlocked, BuildPipeline, limit_timeline  # noqa: E402
from dima_auto_editor.render.export import count_video_frames  # noqa: E402

VOICE_PATTERN = [("silence", 0.1), ("speech", 0.9), ("silence", 0.6), ("speech", 0.7), ("silence", 0.1)]


@pytest.fixture
def project(tmp_path, make_speech_like_wav, make_image, make_video, make_tone_wav) -> ProjectSettings:
    source = tmp_path / "Einstein"
    for index in (1, 2, 3):
        make_speech_like_wav(source / "voice" / f"{index:02d}.wav", VOICE_PATTERN)
    make_image(source / "scenes" / "01.png")
    make_video(source / "scenes" / "02.mp4", seconds=0.8, fps=12, size=(160, 284))
    make_video(source / "scenes" / "03.mp4", seconds=4.0, fps=24, size=(160, 284))
    make_tone_wav(source / "sfx" / "whoosh.wav", seconds=0.3, freq=600)
    make_tone_wav(source / "sfx" / "swish.wav", seconds=0.2, freq=900)

    settings = ProjectSettings()
    settings.paths.voice_dir = str(source / "voice")
    settings.paths.scenes_dir = str(source / "scenes")
    settings.paths.sfx_dir = str(source / "sfx")
    settings.paths.output_dir = str(tmp_path / "output")
    settings.export.encoder = "libx264"
    return settings


def source_snapshot(settings: ProjectSettings) -> dict[str, bytes]:
    root = Path(settings.paths.scenes_dir).parent
    return {str(path): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_preview_build_end_to_end(project, ffmpeg_bins):
    before = source_snapshot(project)
    progress: list[tuple[str, float]] = []
    messages: list[str] = []
    result = BuildPipeline(project, lambda stage, fraction: progress.append((stage, fraction)), messages.append).build(preview=True)

    output = Path(result.output_path)
    assert output.is_file() and output.parent == Path(project.paths.output_dir).resolve()
    assert output.name.startswith("Einstein_preview_")
    info = probe(output, ffmpeg_bins)
    assert info.fps == 60
    assert (info.width, info.height) == (540, 960)
    assert info.sample_rate == 48000
    manifest = load_manifest(result.manifest_path)
    narration_duration = manifest["narration"]["duration"]
    frames = count_video_frames(output, ffmpeg_bins)
    assert frames == round(result.timeline.duration * 60)
    assert abs(frames / 60 - narration_duration) <= 1 / 60 + 1e-9
    assert len(result.timeline.entries) == 3
    assert result.timeline.transitions == []

    assert Path(result.manifest_path).name == output.stem + ".manifest.json"
    assert Path(result.log_path).is_file() and Path(result.log_path).parent.name == "logs"
    assert "Rendered" in Path(result.log_path).read_text(encoding="utf-8")
    assert not list(Path(project.paths.output_dir).glob(".dima_tmp_*"))
    assert source_snapshot(project) == before

    stages = [stage for stage, _ in progress]
    assert [stage for stage in BUILD_STAGES if stage in stages] == list(BUILD_STAGES)
    fractions = [fraction for _, fraction in progress]
    assert all(0.0 <= fraction <= 1.0 for fraction in fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert messages


def test_build_blocked_when_scene_missing(project):
    Path(project.paths.scenes_dir, "03.mp4").unlink()
    with pytest.raises(BuildBlocked) as blocked:
        BuildPipeline(project).build(preview=True)
    assert any(issue.code == "missing_scene" for issue in blocked.value.issues)
    assert not list(Path(project.paths.output_dir).glob(".dima_tmp_*"))


def test_output_inside_source_folder_is_refused(project):
    project.paths.output_dir = str(Path(project.paths.scenes_dir) / "renders")
    with pytest.raises(BuildBlocked) as blocked:
        BuildPipeline(project).build(preview=True)
    assert blocked.value.issues[0].code == "output_in_source"
    assert not Path(project.paths.output_dir).exists()


def cancel_during_rendering(cancel: threading.Event):
    def report(stage: str, _fraction: float) -> None:
        if stage == "Rendering scenes":
            cancel.set()

    return report


def test_cancelled_build_cleans_temp_dir(project):
    cancel = threading.Event()
    with pytest.raises(RenderCancelled):
        BuildPipeline(project, cancel_during_rendering(cancel), None, cancel).build(preview=True)
    output_dir = Path(project.paths.output_dir)
    assert not list(output_dir.glob(".dima_tmp_*"))
    assert not list(output_dir.glob("*.mp4"))
    assert list((output_dir / "logs").glob("build_*.log"))


def test_keep_temp_preserves_work_files(project):
    project.export.keep_temp = True
    cancel = threading.Event()
    with pytest.raises(RenderCancelled):
        BuildPipeline(project, cancel_during_rendering(cancel), None, cancel).build(preview=True)
    temp_dirs = list(Path(project.paths.output_dir).glob(".dima_tmp_*"))
    assert len(temp_dirs) == 1 and (temp_dirs[0] / "narration.wav").is_file()


def test_plan_and_preview_voice_do_not_render(project):
    pipeline = BuildPipeline(project)
    timeline = pipeline.plan()
    assert [entry.scene.index for entry in timeline.entries] == [1, 2, 3]
    assert timeline.entries[-1].end == pytest.approx(timeline.duration)
    wav = Path(pipeline.preview_voice())
    assert wav.is_file() and wav.suffix == ".wav"
    assert not Path(project.paths.output_dir).exists()


def test_limit_timeline_truncates_and_drops_unfit_transition():
    scene = NumberedFile("01.png", 1, 0, "01", MediaKind.IMAGE)
    timeline = Timeline(
        entries=[TimelineEntry(0, scene, 0.0, 2.0), TimelineEntry(1, scene, 2.0, 4.0), TimelineEntry(2, scene, 4.0, 6.0)],
        transitions=[TransitionEvent(0, 2.0, 0.5, "Fade", "fade"), TransitionEvent(1, 4.0, 0.5, "Fade", "fade")],
        duration=6.0,
    )
    limited = limit_timeline(timeline, 4.1)
    assert limited.duration == pytest.approx(4.1)
    assert [entry.end for entry in limited.entries] == [2.0, 4.0, pytest.approx(4.1)]
    assert [event.boundary_position for event in limited.transitions] == [0]
    assert limit_timeline(timeline, 0) is timeline
