from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ProjectSettings
from ..ffmpeg_tools import LogCallback, ProgressCallback, run_ffmpeg
from ..models import Issue, Severity, Timeline, TransitionEvent
from .export import binaries_for, count_video_frames, encode_with_fallback, intermediate_video_args, preferred_encoder

MAX_CLIPS_PER_CHUNK = 40


@dataclass
class ComposeReport:
    output_path: str
    frames: int
    expected_frames: int
    encoder: str
    chunk_count: int
    issues: list[Issue] = field(default_factory=list)


@dataclass
class _Chunk:
    first: int
    last: int


def to_frames(seconds: float, fps: int) -> int:
    return round(seconds * fps)


def transition_frames(event: TransitionEvent | None, fps: int) -> int:
    if event is None:
        return 0
    return 2 * round(event.duration * fps / 2)


def clip_frame_counts(timeline: Timeline) -> list[int]:
    fps = timeline.fps
    counts: list[int] = []
    for position, entry in enumerate(timeline.entries):
        left_half = transition_frames(timeline.transition_at(position - 1), fps) // 2 if position > 0 else 0
        right_half = transition_frames(timeline.transition_at(position), fps) // 2
        counts.append(to_frames(entry.end, fps) - to_frames(entry.start, fps) + left_half + right_half)
    return counts


def clip_content_durations(timeline: Timeline) -> list[float]:
    return [frames / timeline.fps for frames in clip_frame_counts(timeline)]


def plan_chunks(timeline: Timeline, max_clips: int = MAX_CLIPS_PER_CHUNK) -> list[_Chunk]:
    count = len(timeline.entries)
    hard_cut_after = [timeline.transition_at(position) is None for position in range(count - 1)] + [True]
    chunks: list[_Chunk] = []
    first = 0
    last_safe_split: int | None = None
    for position in range(count):
        if position - first + 1 > max_clips and last_safe_split is not None:
            chunks.append(_Chunk(first, last_safe_split))
            first = last_safe_split + 1
            last_safe_split = None
        if hard_cut_after[position]:
            last_safe_split = position
    if first < count:
        chunks.append(_Chunk(first, count - 1))
    return chunks


def build_chunk_graph(timeline: Timeline, chunk: _Chunk, clip_frames: list[int]) -> tuple[str, int]:
    fps = timeline.fps
    statements: list[str] = []
    segment_labels: list[str] = []
    accumulated_label = ""
    accumulated_frames = 0
    chunk_frames = 0
    for local_index, position in enumerate(range(chunk.first, chunk.last + 1)):
        label = f"v{local_index}"
        statements.append(f"[{local_index}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={fps},format=yuv420p,setsar=1[{label}]")
        incoming = timeline.transition_at(position - 1) if position > chunk.first else None
        if incoming is None:
            if accumulated_label:
                segment_labels.append(accumulated_label)
            accumulated_label, accumulated_frames = label, clip_frames[position]
        else:
            overlap = transition_frames(incoming, fps)
            merged = f"x{local_index}"
            statements.append(
                f"[{accumulated_label}][{label}]xfade=transition={incoming.xfade}:duration={overlap / fps:.6f}"
                f":offset={(accumulated_frames - overlap) / fps:.6f}[{merged}]"
            )
            accumulated_label = merged
            accumulated_frames += clip_frames[position] - overlap
        entry = timeline.entries[position]
        chunk_frames += to_frames(entry.end, fps) - to_frames(entry.start, fps)
    segment_labels.append(accumulated_label)
    finishing = f"fps={fps},tpad=stop=-1:stop_mode=clone,format=yuv420p[out]"
    if len(segment_labels) == 1:
        statements.append(f"[{segment_labels[0]}]{finishing}")
    else:
        joined = "".join(f"[{label}]" for label in segment_labels)
        statements.append(f"{joined}concat=n={len(segment_labels)}:v=1:a=0,{finishing}")
    return ";".join(statements), chunk_frames


def compose_video(
    clip_paths: list[str | Path],
    timeline: Timeline,
    settings: ProjectSettings,
    out_path: str | Path,
    *,
    preview: bool,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    log: LogCallback | None = None,
) -> ComposeReport:
    if len(clip_paths) != len(timeline.entries):
        raise ValueError(f"Expected {len(timeline.entries)} clips, got {len(clip_paths)}.")
    if not clip_paths:
        raise ValueError("Cannot compose an empty timeline.")
    out_path = Path(out_path)
    fps = timeline.fps
    binaries = binaries_for(settings)
    encoder = preferred_encoder(settings)
    clip_frames = clip_frame_counts(timeline)
    expected_frames = to_frames(timeline.duration, fps)
    chunks = plan_chunks(timeline, MAX_CLIPS_PER_CHUNK)
    chunk_paths = [out_path] if len(chunks) == 1 else [
        out_path.with_name(f"{out_path.stem}_chunk{number:03d}{out_path.suffix}") for number in range(len(chunks))
    ]
    done_frames = 0
    for chunk, chunk_path in zip(chunks, chunk_paths):
        graph, chunk_frame_count = build_chunk_graph(timeline, chunk, clip_frames)
        inputs = [arg for path in clip_paths[chunk.first:chunk.last + 1] for arg in ("-i", str(path))]

        def build_args(chosen: str) -> list[str]:
            return [*inputs, "-filter_complex", graph, "-map", "[out]", "-an", "-frames:v", str(chunk_frame_count),
                    "-r", str(fps), *intermediate_video_args(chosen, preview), str(chunk_path)]

        encoder = encode_with_fallback(
            build_args, encoder, binaries=binaries, duration=chunk_frame_count / fps, cancel=cancel, log=log,
            progress=_scaled_progress(progress, done_frames, chunk_frame_count, expected_frames),
        )
        done_frames += chunk_frame_count
    if len(chunks) > 1:
        _concat_chunks(chunk_paths, out_path, settings, cancel, log)
    frames = count_video_frames(out_path, binaries)
    report = ComposeReport(str(out_path), frames, expected_frames, encoder, len(chunks))
    if frames != expected_frames:
        report.issues.append(Issue(Severity.WARNING, "compose_frame_mismatch",
                                   f"Composed video has {frames} frames, expected {expected_frames}.", str(out_path)))
    return report


def _scaled_progress(progress: ProgressCallback | None, done: int, span: int, total: int) -> ProgressCallback | None:
    if progress is None:
        return None
    return lambda fraction: progress((done + fraction * span) / max(1, total))


def _concat_chunks(chunk_paths: list[Path], out_path: Path, settings: ProjectSettings,
                   cancel: threading.Event | None, log: LogCallback | None) -> None:
    list_path = out_path.with_name(f"{out_path.stem}_chunks.txt")
    lines = ["file '" + path.resolve().as_posix().replace("'", "'\\''") + "'" for path in chunk_paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)],
               binaries=binaries_for(settings), cancel=cancel, log=log)
    for path in chunk_paths:
        path.unlink(missing_ok=True)
    list_path.unlink(missing_ok=True)
