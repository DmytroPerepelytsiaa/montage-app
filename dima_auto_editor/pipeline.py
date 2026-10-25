from __future__ import annotations

import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, TextIO

from .audio.mixer import SfxCue, mix_final_audio
from .audio.narration import build_narration
from .config import ProjectSettings
from .ffmpeg_tools import RenderCancelled, probe
from .inputs import InputScan, scan_inputs
from .manifest import write_manifest
from .models import Issue, Narration, Severity, Timeline, TimelineEntry, TransitionDef
from .render.clips import ClipReport, output_size, render_scene_clip
from .render.compose import ComposeReport, clip_content_durations, clip_frame_counts, compose_video, transition_frames
from .render.export import ExportReport, is_within, mux_final, safe_project_name, unique_output_path
from .timeline import plan_timeline
from .transitions import load_library, resolve_library_sfx, stylized_library, validate_library

StageProgressCallback = Callable[[str, float], None]
LogCallback = Callable[[str], None]

STAGE_SCAN = "Scanning inputs"
STAGE_VOICE = "Processing voice-over"
STAGE_PLAN = "Planning timeline"
STAGE_RENDER = "Rendering scenes"
STAGE_COMPOSE = "Composing transitions"
STAGE_MIX = "Mixing audio"
STAGE_ENCODE = "Encoding"
STAGE_VERIFY = "Verifying"
STAGE_WEIGHTS: dict[str, float] = {
    STAGE_SCAN: 0.03, STAGE_VOICE: 0.10, STAGE_PLAN: 0.02, STAGE_RENDER: 0.45,
    STAGE_COMPOSE: 0.18, STAGE_MIX: 0.04, STAGE_ENCODE: 0.15, STAGE_VERIFY: 0.03,
}
BUILD_STAGES = tuple(STAGE_WEIGHTS)
PLAN_STAGES = (STAGE_SCAN, STAGE_VOICE, STAGE_PLAN)
VOICE_PREVIEW_STAGES = (STAGE_SCAN, STAGE_VOICE)
MAX_RENDER_WORKERS = 3
TEMP_DIR_PREFIX = ".dima_tmp_"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


class BuildBlocked(RuntimeError):
    def __init__(self, issues: list[Issue]):
        self.issues = list(issues)
        details = "\n".join(f"- {issue.message}" for issue in self.issues)
        super().__init__(f"The build cannot start until these problems are fixed:\n{details}")


@dataclass
class BuildResult:
    output_path: str
    manifest_path: str
    log_path: str
    timeline: Timeline
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class _Prepared:
    scan: InputScan
    narration: Narration
    library: list[TransitionDef]
    timeline: Timeline


class _ProgressTracker:
    def __init__(self, callback: StageProgressCallback | None, stages: tuple[str, ...]):
        self._callback = callback
        total = sum(STAGE_WEIGHTS[stage] for stage in stages)
        self._offsets: dict[str, float] = {}
        self._weights: dict[str, float] = {}
        running = 0.0
        for stage in stages:
            self._offsets[stage] = running / total
            self._weights[stage] = STAGE_WEIGHTS[stage] / total
            running += STAGE_WEIGHTS[stage]
        self._lock = threading.Lock()

    def report(self, stage: str, fraction: float) -> None:
        if self._callback is None:
            return
        overall = self._offsets[stage] + self._weights[stage] * min(1.0, max(0.0, fraction))
        with self._lock:
            self._callback(stage, overall)

    def for_stage(self, stage: str) -> Callable[[float], None]:
        self.report(stage, 0.0)
        return lambda fraction: self.report(stage, fraction)


class _EitherEvent:
    def __init__(self, *events: threading.Event):
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


def limit_timeline(timeline: Timeline, max_seconds: float) -> Timeline:
    fps = timeline.fps
    limit_frames = round(max_seconds * fps)
    if limit_frames <= 0 or limit_frames >= round(timeline.duration * fps):
        return timeline
    limit = limit_frames / fps
    entries = [replace(entry) for entry in timeline.entries if round(entry.start * fps) < limit_frames]
    entries[-1] = replace(entries[-1], end=limit, end_cut=None)
    last_position = len(entries) - 1
    last_span_frames = limit_frames - round(entries[-1].start * fps)
    transitions = [
        event for event in timeline.transitions
        if event.boundary_position < last_position
        and not (event.boundary_position == last_position - 1 and transition_frames(event, fps) // 2 > last_span_frames)
    ]
    return replace(timeline, entries=entries, transitions=transitions, duration=limit, issues=list(timeline.issues))


class BuildPipeline:
    def __init__(self, settings: ProjectSettings, progress_cb: StageProgressCallback | None = None,
                 log_cb: LogCallback | None = None, cancel_event: threading.Event | None = None):
        self.settings = settings
        self._progress_cb = progress_cb
        self._log_cb = log_cb
        self._cancel = cancel_event or threading.Event()
        self._log_lock = threading.Lock()
        self._log_file: TextIO | None = None

    def scan(self) -> InputScan:
        tracker = _ProgressTracker(self._progress_cb, (STAGE_SCAN,))
        return self._scan(tracker)

    def preview_voice(self) -> str:
        tracker = _ProgressTracker(self._progress_cb, VOICE_PREVIEW_STAGES)
        scan = self._scan(tracker, require_valid=True)
        preview_root = Path(tempfile.gettempdir()) / "dima_auto_editor" / "voice_preview"
        _remove_stale_previews(preview_root)
        work_dir = preview_root / datetime.now().strftime(TIMESTAMP_FORMAT + "_%f")
        narration = self._process_voice(scan, work_dir, tracker)
        self._log(f"Voice preview ready: {narration.wav_path} ({narration.duration:.2f} s)")
        return narration.wav_path

    def plan(self) -> Timeline:
        tracker = _ProgressTracker(self._progress_cb, PLAN_STAGES)
        with tempfile.TemporaryDirectory(prefix="dima_plan_") as work_dir:
            return self._prepare(Path(work_dir), tracker).timeline

    def build(self, preview: bool = False) -> BuildResult:
        self._check_cancel()
        started_at = datetime.now()
        stamp = started_at.strftime(TIMESTAMP_FORMAT)
        output_dir = self._validated_output_dir()
        log_path = _unique_path(output_dir / "logs" / f"build_{stamp}.log")
        with self._logging_to(log_path):
            temp_dir = self._create_temp_dir(output_dir, stamp)
            try:
                return self._run_build(preview, output_dir, temp_dir, log_path, started_at)
            except RenderCancelled:
                self._log("Build cancelled.")
                raise
            except BuildBlocked as blocked:
                self._log(str(blocked))
                raise
            except Exception:
                self._log_detail(traceback.format_exc())
                self._log("Build failed.")
                raise
            finally:
                self._dispose_temp_dir(temp_dir, output_dir)

    def _run_build(self, preview: bool, output_dir: Path, temp_dir: Path, log_path: Path, started_at: datetime) -> BuildResult:
        mode = "preview" if preview else "final"
        width, height = output_size(self.settings, preview)
        self._log(f"Starting {mode} build: {width}x{height} @ {self.settings.video.fps} FPS, "
                  f"motion mode '{self.settings.video.motion_mode}'. Temporary files: {temp_dir}")
        tracker = _ProgressTracker(self._progress_cb, BUILD_STAGES)
        prepared = self._prepare(temp_dir, tracker)
        timeline = prepared.timeline
        if preview and self.settings.export.preview_max_seconds > 0:
            timeline = limit_timeline(timeline, self.settings.export.preview_max_seconds)
            self._log(f"Preview limited to the first {timeline.duration:.2f} s.")

        clip_reports = self._render_scenes(timeline, prepared.scan, temp_dir / "clips", preview, tracker)
        for report in clip_reports:
            timeline.issues.extend(report.warnings)

        self._check_cancel()
        compose_report = compose_video(
            [report.output_path for report in clip_reports], timeline, self.settings, temp_dir / "composed.mp4",
            preview=preview, progress=tracker.for_stage(STAGE_COMPOSE), cancel=self._cancel, log=self._log_detail,
        )
        timeline.issues.extend(compose_report.issues)
        self._log(f"Composed {compose_report.frames} frames in {compose_report.chunk_count} chunk(s).")

        mix_stats = self._mix_audio(prepared, timeline, temp_dir / "mix.wav", tracker)

        self._check_cancel()
        output_path = unique_output_path(output_dir, self._project_name(), preview=preview, now=started_at)
        export_report = mux_final(
            compose_report.output_path, temp_dir / "mix.wav", output_path, self.settings, preview=preview,
            progress=tracker.for_stage(STAGE_ENCODE), cancel=self._cancel, log=self._log_detail,
        )
        self._log(f"Encoded {output_path.name} with {export_report.encoder}.")

        tracker.report(STAGE_VERIFY, 0.0)
        timeline.issues.extend(export_report.issues)
        timeline.issues.extend(self._duration_issues(prepared.narration, timeline, export_report.frames, preview))
        issues = [*prepared.scan.issues, *timeline.issues]
        self._log_issues(issues)
        manifest_path = output_path.with_name(output_path.stem + ".manifest.json")
        stats = self._stats(preview, timeline, clip_reports, compose_report, export_report, mix_stats, started_at)
        write_manifest(
            manifest_path, settings=self.settings, scan=prepared.scan, narration=prepared.narration, timeline=timeline,
            library=prepared.library, outputs={"video": str(output_path), "log": str(log_path), "preview": preview},
            stats=stats, started_at=started_at, finished_at=datetime.now(),
        )
        tracker.report(STAGE_VERIFY, 1.0)
        self._log(f"Done in {stats['elapsed_s']:.1f} s: {output_path}")
        return BuildResult(str(output_path), str(manifest_path), str(log_path), timeline, issues, stats)

    def _scan(self, tracker: _ProgressTracker, require_valid: bool = False) -> InputScan:
        self._check_cancel()
        tracker.report(STAGE_SCAN, 0.0)
        scan = scan_inputs(self.settings.paths)
        tracker.report(STAGE_SCAN, 1.0)
        self._log(f"Found {len(scan.voices)} voice-over(s), {len(scan.scenes)} scene(s), {len(scan.sfx)} sound effect(s).")
        if require_valid and scan.has_errors():
            raise BuildBlocked([issue for issue in scan.issues if issue.severity is Severity.ERROR])
        return scan

    def _process_voice(self, scan: InputScan, work_dir: Path, tracker: _ProgressTracker) -> Narration:
        self._check_cancel()
        narration = build_narration(scan.voices, self.settings, work_dir,
                                    progress=tracker.for_stage(STAGE_VOICE), cancel=self._cancel)
        tracker.report(STAGE_VOICE, 1.0)
        return narration

    def _prepare(self, work_dir: Path, tracker: _ProgressTracker) -> _Prepared:
        scan = self._scan(tracker, require_valid=True)
        library, library_issues = self._load_library(scan)
        narration = self._process_voice(scan, work_dir, tracker)
        self._log(f"Processed narration: {narration.duration:.2f} s.")
        self._check_cancel()
        tracker.report(STAGE_PLAN, 0.0)
        timeline = plan_timeline(narration, scan.scenes, self.settings, library)
        timeline.issues = [*narration.issues, *library_issues, *timeline.issues]
        tracker.report(STAGE_PLAN, 1.0)
        self._log(f"Planned {len(timeline.entries)} scene(s) with {len(timeline.transitions)} transition(s), "
                  f"{timeline.duration:.2f} s total.")
        return _Prepared(scan, narration, library, timeline)

    def _load_library(self, scan: InputScan) -> tuple[list[TransitionDef], list[Issue]]:
        library_path = self.settings.transitions.library_path.strip()
        issues: list[Issue] = []
        library: list[TransitionDef] | None = None
        if library_path and Path(library_path).is_file():
            try:
                library = load_library(library_path)
            except (OSError, ValueError) as error:
                issues.append(Issue(Severity.WARNING, "library_unreadable",
                                    f"Transition library could not be read ({error}); the default library is used.", library_path))
        elif library_path:
            issues.append(Issue(Severity.WARNING, "library_missing",
                                f"Transition library {library_path} was not found; the default library is used.", library_path))
        if library is None:
            transitions = self.settings.transitions
            library = stylized_library(transitions.styles, scan.sfx, transitions.duration_s)
        issues.extend(validate_library(library, scan.sfx))
        errors = [issue for issue in issues if issue.severity is Severity.ERROR]
        if errors:
            raise BuildBlocked(errors)
        return resolve_library_sfx(library, scan.sfx), issues

    def _render_scenes(self, timeline: Timeline, scan: InputScan, clips_dir: Path, preview: bool,
                       tracker: _ProgressTracker) -> list[ClipReport]:
        self._check_cancel()
        clips_dir.mkdir(parents=True, exist_ok=True)
        durations = clip_content_durations(timeline)
        weights = clip_frame_counts(timeline)
        total_weight = max(1, sum(weights))
        fractions = [0.0] * len(weights)
        fraction_lock = threading.Lock()
        stage_progress = tracker.for_stage(STAGE_RENDER)
        abort = threading.Event()
        stop = _EitherEvent(self._cancel, abort)

        def clip_progress(position: int) -> Callable[[float], None]:
            def update(fraction: float) -> None:
                with fraction_lock:
                    fractions[position] = fraction
                    done = sum(weight * value for weight, value in zip(weights, fractions))
                stage_progress(done / total_weight)
            return update

        def render(position: int, entry: TimelineEntry) -> ClipReport:
            if stop.is_set():
                raise RenderCancelled("Rendering was cancelled by the user.")
            info = scan.media.get(entry.scene.path) or probe(entry.scene.path)
            return render_scene_clip(
                entry.scene, info, durations[position], clips_dir / f"clip_{position:03d}.mp4", self.settings,
                preview=preview, progress=clip_progress(position), cancel=stop, log=self._log_detail,  # type: ignore[arg-type]
            )

        reports: list[ClipReport | None] = [None] * len(timeline.entries)
        pool = ThreadPoolExecutor(max_workers=max(1, min(MAX_RENDER_WORKERS, len(timeline.entries))),
                                  thread_name_prefix="clip-render")
        try:
            futures: dict[Future[ClipReport], int] = {
                pool.submit(render, position, entry): position for position, entry in enumerate(timeline.entries)
            }
            for future in as_completed(futures):
                report = future.result()
                reports[futures[future]] = report
                self._log(f"Rendered {Path(report.scene_path).name}: {report.frames} frames ({report.method}, {report.encoder}).")
        except BaseException:
            abort.set()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return [report for report in reports if report is not None]

    def _mix_audio(self, prepared: _Prepared, timeline: Timeline, out_path: Path, tracker: _ProgressTracker) -> dict:
        self._check_cancel()
        tracker.report(STAGE_MIX, 0.0)
        cues = [SfxCue(event.sfx_path, event.sfx_start, event.sfx_volume_db) for event in timeline.transitions if event.sfx_path]
        stats = mix_final_audio(prepared.narration.wav_path, timeline.duration, cues, prepared.scan.music,
                                self.settings.audio, out_path)
        for warning in stats.get("warnings", []):
            timeline.issues.append(Issue(Severity.WARNING, "audio_mix", str(warning)))
            self._log(f"WARNING: {warning}")
        tracker.report(STAGE_MIX, 1.0)
        self._log(f"Mixed audio with {len(cues)} sound effect(s); peak {stats.get('peak_db')} dBFS.")
        return stats

    def _duration_issues(self, narration: Narration, timeline: Timeline, exported_frames: int, preview: bool) -> list[Issue]:
        fps = timeline.fps
        issues: list[Issue] = []
        expected_frames = round(timeline.duration * fps)
        limited = preview and self.settings.export.preview_max_seconds > 0
        if not limited and round(narration.duration * fps) != expected_frames:
            issues.append(Issue(Severity.WARNING, "duration_mismatch",
                                f"Timeline length {timeline.duration:.3f} s differs from narration length {narration.duration:.3f} s."))
        if exported_frames != expected_frames:
            issues.append(Issue(Severity.WARNING, "sync_risk",
                                f"Video has {exported_frames} frames but the audio needs {expected_frames}; check A/V sync."))
        return issues

    def _stats(self, preview: bool, timeline: Timeline, clip_reports: list[ClipReport], compose_report: ComposeReport, export_report: ExportReport,
               mix_stats: dict, started_at: datetime) -> dict:
        width, height = output_size(self.settings, preview)
        return {
            "preview": preview,
            "resolution": f"{width}x{height}",
            "fps": self.settings.video.fps,
            "duration_s": round(timeline.duration, 3),
            "frames": export_report.frames,
            "scene_count": len(timeline.entries),
            "transition_count": len(timeline.transitions),
            "hard_cut_count": max(0, len(timeline.entries) - 1 - len(timeline.transitions)),
            "encoder": export_report.encoder,
            "clips": [{key: value for key, value in asdict(report).items() if key != "warnings"} for report in clip_reports],
            "compose": {"frames": compose_report.frames, "chunks": compose_report.chunk_count, "encoder": compose_report.encoder},
            "audio": mix_stats,
            "elapsed_s": round((datetime.now() - started_at).total_seconds(), 2),
        }

    def _validated_output_dir(self) -> Path:
        paths = self.settings.paths
        if not paths.output_dir.strip():
            raise BuildBlocked([Issue(Severity.ERROR, "output_dir_missing", "Choose an output folder before building.")])
        output_dir = Path(paths.output_dir).expanduser().resolve()
        source_folders = [paths.voice_dir, paths.scenes_dir, paths.sfx_dir]
        source_files: list[str] = []
        if paths.music_path and Path(paths.music_path).is_dir():
            source_folders.append(paths.music_path)
        else:
            source_files.append(paths.music_path)
        inside_source = any(folder.strip() and is_within(output_dir, folder) for folder in source_folders)
        beside_source_file = any(file.strip() and output_dir == Path(file).resolve().parent for file in source_files)
        if inside_source or beside_source_file:
            raise BuildBlocked([Issue(Severity.ERROR, "output_in_source",
                                      f"The output folder {output_dir} is inside a source folder. Choose a separate output "
                                      "folder so source files are never touched.", str(output_dir))])
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _create_temp_dir(self, output_dir: Path, stamp: str) -> Path:
        temp_dir = _unique_path(output_dir / f"{TEMP_DIR_PREFIX}{stamp}")
        temp_dir.mkdir(parents=False, exist_ok=False)
        return temp_dir

    def _dispose_temp_dir(self, temp_dir: Path, output_dir: Path) -> None:
        if self.settings.export.keep_temp:
            self._log(f"Keeping temporary files in {temp_dir}")
            return
        if temp_dir.parent != output_dir or not temp_dir.name.startswith(TEMP_DIR_PREFIX):
            self._log(f"Refusing to delete unexpected folder {temp_dir}")
            return
        for attempt in range(3):
            try:
                shutil.rmtree(temp_dir)
                return
            except FileNotFoundError:
                return
            except OSError as error:
                if attempt == 2:
                    self._log(f"Could not remove temporary folder {temp_dir}: {error}")
                else:
                    time.sleep(0.3)

    def _project_name(self) -> str:
        scenes_dir = self.settings.paths.scenes_dir.strip()
        parent_name = Path(scenes_dir).resolve().parent.name if scenes_dir else ""
        return safe_project_name(parent_name)

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise RenderCancelled("Rendering was cancelled by the user.")

    @contextmanager
    def _logging_to(self, log_path: Path) -> Iterator[None]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            self._log_file = handle
            try:
                yield
            finally:
                with self._log_lock:
                    self._log_file = None

    def _log_issues(self, issues: list[Issue]) -> None:
        for issue in issues:
            message = f"{issue.severity.value.upper()} [{issue.code}]: {issue.message}"
            if issue.severity is Severity.INFO:
                self._log_detail(message)
            else:
                self._log(message)

    def _log(self, message: str) -> None:
        self._write_log_file(message)
        if self._log_cb:
            self._log_cb(message)

    def _log_detail(self, message: str) -> None:
        self._write_log_file(message)

    def _write_log_file(self, message: str) -> None:
        with self._log_lock:
            if self._log_file is not None:
                self._log_file.write(f"[{datetime.now():%H:%M:%S}] {message.rstrip()}\n")
                self._log_file.flush()


def _unique_path(path: Path) -> Path:
    candidate = path
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate


def _remove_stale_previews(preview_root: Path) -> None:
    if not preview_root.is_dir():
        return
    for stale in preview_root.iterdir():
        shutil.rmtree(stale, ignore_errors=True)
