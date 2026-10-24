from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable

from .config import InputPaths
from .ffmpeg_tools import FFmpegBinaries, FFmpegError, FFmpegNotFound, locate_binaries, media_kind_for, probe
from .models import Issue, MediaInfo, MediaKind, NumberedFile, Severity
from .naming import AUDIO_EXTENSIONS, SCENE_EXTENSIONS, natural_key, natural_sorted, parse_numbered_stem

OS_METADATA_FILES = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})
_PROBE_WORKERS = 8


@dataclass
class InputScan:
    voices: list[NumberedFile] = field(default_factory=list)
    scenes: list[NumberedFile] = field(default_factory=list)
    sfx: list[str] = field(default_factory=list)
    music: str = ""
    issues: list[Issue] = field(default_factory=list)
    pairs: dict[int, list[NumberedFile]] = field(default_factory=dict)
    media: dict[str, MediaInfo] = field(default_factory=dict)

    def has_errors(self) -> bool:
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def is_continuous_narration(self) -> bool:
        return len(self.voices) == 1 and len(self.scenes) > 1


@dataclass(frozen=True)
class _FolderRole:
    key: str
    folder_label: str
    item_label: str
    extensions: frozenset[str]


_VOICE_ROLE = _FolderRole("voice", "voice-over folder", "voice-over", AUDIO_EXTENSIONS)
_SCENE_ROLE = _FolderRole("scene", "scenes folder", "scene", SCENE_EXTENSIONS)
_SFX_ROLE = _FolderRole("sfx", "sound-effects folder", "sound effect", AUDIO_EXTENSIONS)
_MUSIC_ROLE = _FolderRole("music", "music folder", "music track", AUDIO_EXTENSIONS)


class _IssueLog:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def error(self, code: str, message: str, path: str | Path = "") -> None:
        self.issues.append(Issue(Severity.ERROR, code, message, str(path)))

    def warning(self, code: str, message: str, path: str | Path = "") -> None:
        self.issues.append(Issue(Severity.WARNING, code, message, str(path)))

    def info(self, code: str, message: str, path: str | Path = "") -> None:
        self.issues.append(Issue(Severity.INFO, code, message, str(path)))


def scan_inputs(paths: InputPaths, probe_media: bool = True) -> InputScan:
    log = _IssueLog()
    shares_folder = _same_folder(paths.voice_dir, paths.scenes_dir)
    voice_files = _supported_files(paths.voice_dir, _VOICE_ROLE, log, required=True,
                                   handled_elsewhere=SCENE_EXTENSIONS if shares_folder else frozenset())
    scene_files = _supported_files(paths.scenes_dir, _SCENE_ROLE, log, required=True,
                                   handled_elsewhere=AUDIO_EXTENSIONS if shares_folder else frozenset())

    voices = _unique_numbered(_numbered(voice_files, _VOICE_ROLE, log), lambda f: f.index, _VOICE_ROLE, log)
    scenes = _unique_numbered(_numbered(scene_files, _SCENE_ROLE, log), lambda f: (f.index, f.sub_index), _SCENE_ROLE, log)
    _report_numbering_gaps([f.index for f in voices], _VOICE_ROLE, log)
    _report_numbering_gaps([f.index for f in scenes], _SCENE_ROLE, log)
    _report_empty_role(voices, paths.voice_dir, _VOICE_ROLE, log)
    _report_empty_role(scenes, paths.scenes_dir, _SCENE_ROLE, log)

    scan = InputScan(
        voices=voices,
        scenes=scenes,
        sfx=_sfx_files(paths.sfx_dir, log),
        music=_music_file(paths.music_path, log),
    )
    scan.pairs = _pair_voices_with_scenes(voices, scenes, log)
    if probe_media:
        _probe_inputs(scan, log)
    scan.issues = log.issues
    return scan


def _same_folder(first: str, second: str) -> bool:
    if not first or not second:
        return False
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _list_folder(folder: str, role: _FolderRole, log: _IssueLog, required: bool) -> list[Path]:
    if not folder:
        if required:
            log.error("folder_not_set", f"No {role.folder_label} is selected.")
        return []
    path = Path(os.path.abspath(folder))
    if not path.exists():
        log.error("folder_not_found", f"The {role.folder_label} does not exist: {path}", path)
        return []
    if not path.is_dir():
        log.error("not_a_folder", f"The {role.folder_label} is not a folder: {path}", path)
        return []
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        log.error("folder_unreadable", f"The {role.folder_label} cannot be read: {path} ({exc})", path)
        return []
    return natural_sorted(entries)


def _supported_files(folder: str, role: _FolderRole, log: _IssueLog, required: bool,
                     handled_elsewhere: frozenset[str] = frozenset()) -> list[Path]:
    accepted: list[Path] = []
    for entry in _list_folder(folder, role, log, required):
        if entry.is_dir():
            log.info("subfolder_ignored", f"Subfolder '{entry.name}' inside the {role.folder_label} is not scanned.", entry)
            continue
        if entry.name.casefold() in OS_METADATA_FILES:
            continue
        suffix = entry.suffix.lower()
        if suffix in role.extensions:
            accepted.append(entry)
        elif suffix not in handled_elsewhere:
            supported = ", ".join(sorted(role.extensions))
            log.warning("unsupported_file",
                        f"'{entry.name}' in the {role.folder_label} is not a supported {role.item_label} format "
                        f"({supported}); it will not be used.", entry)
    return accepted


def _numbered(files: list[Path], role: _FolderRole, log: _IssueLog) -> list[NumberedFile]:
    numbered: list[NumberedFile] = []
    for path in files:
        parsed = parse_numbered_stem(path.stem)
        if parsed is None:
            log.warning("unnumbered_file",
                        f"'{path.name}' in the {role.folder_label} has no number in its name "
                        f"(expected names like 01{path.suffix}, 02{path.suffix}); it will not be used.", path)
            continue
        index, sub_index = parsed
        numbered.append(NumberedFile(str(path), index, sub_index, path.stem, media_kind_for(path)))
    return sorted(numbered, key=lambda f: (f.index, f.sub_index, natural_key(Path(f.path).name)))


def _unique_numbered(files: list[NumberedFile], key: Callable[[NumberedFile], Hashable],
                     role: _FolderRole, log: _IssueLog) -> list[NumberedFile]:
    groups: dict[Hashable, list[NumberedFile]] = defaultdict(list)
    for numbered_file in files:
        groups[key(numbered_file)].append(numbered_file)
    unique: list[NumberedFile] = []
    for group in groups.values():
        unique.append(group[0])
        if len(group) > 1:
            names = ", ".join(Path(f.path).name for f in group)
            log.error("duplicate_index",
                      f"Several {role.item_label} files share the number {_label(group[0])}: {names}. "
                      f"Rename or remove the duplicates so each number is unique.", group[1].path)
    return unique


def _label(numbered_file: NumberedFile) -> str:
    base = f"{numbered_file.index:02d}"
    return f"{base}_{numbered_file.sub_index}" if numbered_file.sub_index else base


def _report_numbering_gaps(indices: list[int], role: _FolderRole, log: _IssueLog) -> None:
    present = sorted(set(indices))
    if not present:
        return
    first_expected = 0 if present[0] == 0 else 1
    missing = sorted(set(range(first_expected, present[-1] + 1)) - set(present))
    if missing:
        listed = ", ".join(f"{index:02d}" for index in missing)
        log.warning("numbering_gap", f"The {role.item_label} numbering has gaps: number(s) {listed} are missing.")


def _report_empty_role(files: list[NumberedFile], folder: str, role: _FolderRole, log: _IssueLog) -> None:
    if not files and folder and Path(folder).is_dir():
        log.error(f"no_{role.key}s",
                  f"No numbered {role.item_label} files were found in {os.path.abspath(folder)}.", os.path.abspath(folder))


def _pair_voices_with_scenes(voices: list[NumberedFile], scenes: list[NumberedFile], log: _IssueLog) -> dict[int, list[NumberedFile]]:
    if len(voices) == 1 and len(scenes) > 1:
        voice = voices[0]
        log.info("continuous_narration",
                 f"Single continuous narration: '{Path(voice.path).name}' will be split across {len(scenes)} scenes.",
                 voice.path)
        return {voice.index: list(scenes)}

    scenes_by_index: dict[int, list[NumberedFile]] = defaultdict(list)
    for scene in scenes:
        scenes_by_index[scene.index].append(scene)
    voice_indices = {voice.index for voice in voices}

    pairs: dict[int, list[NumberedFile]] = {}
    for voice in voices:
        matching = sorted(scenes_by_index.get(voice.index, []), key=lambda f: f.sub_index)
        pairs[voice.index] = matching
        if not matching:
            log.error("missing_scene",
                      f"Voice-over '{Path(voice.path).name}' has no matching scene "
                      f"(expected a scene file numbered {voice.index:02d}, e.g. {voice.index:02d}.mp4 or {voice.index:02d}.png).",
                      voice.path)
    for scene in scenes:
        if scene.index not in voice_indices:
            log.warning("orphan_scene",
                        f"Scene '{Path(scene.path).name}' has no matching voice-over (number {scene.index:02d}); "
                        f"it has no narration to cover and will not appear in the video.", scene.path)
    return pairs


def _sfx_files(folder: str, log: _IssueLog) -> list[str]:
    if not folder:
        return []
    return [str(path) for path in _supported_files(folder, _SFX_ROLE, log, required=False)]


def _music_file(music_path: str, log: _IssueLog) -> str:
    if not music_path:
        return ""
    path = Path(os.path.abspath(music_path))
    if path.is_file():
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            log.error("unsupported_file", f"The music file '{path.name}' is not a supported audio format.", path)
            return ""
        return str(path)
    if not path.exists():
        log.error("file_not_found", f"The music file or folder does not exist: {path}", path)
        return ""
    tracks = _supported_files(str(path), _MUSIC_ROLE, log, required=False)
    if not tracks:
        log.warning("no_music_found", f"The music folder contains no supported audio files: {path}", path)
        return ""
    if len(tracks) > 1:
        log.info("music_selected",
                 f"Using '{tracks[0].name}' as background music (first of {len(tracks)} audio files in the music folder).",
                 tracks[0])
    return str(tracks[0])


def _probe_inputs(scan: InputScan, log: _IssueLog) -> None:
    try:
        binaries = locate_binaries()
    except FFmpegNotFound as exc:
        log.error("ffmpeg_missing", str(exc))
        return
    targets = [f.path for f in scan.voices] + [f.path for f in scan.scenes] + scan.sfx + ([scan.music] if scan.music else [])
    with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
        results = dict(zip(targets, pool.map(lambda target: _try_probe(target, binaries), targets)))

    for path, outcome in results.items():
        if isinstance(outcome, MediaInfo):
            scan.media[path] = outcome
    for voice in scan.voices:
        _validate_media(results[voice.path], voice.path, "Voice-over", log)
    for scene in scan.scenes:
        _validate_media(results[scene.path], scene.path, "Scene", log)
    if scan.music:
        _validate_media(results[scan.music], scan.music, "Music", log)

    readable_sfx: list[str] = []
    for sfx_path in scan.sfx:
        problem = _media_problem(results[sfx_path])
        if problem:
            code, description = problem
            log.warning(code, f"Sound effect '{Path(sfx_path).name}' {description}; it will not be used.", sfx_path)
        else:
            readable_sfx.append(sfx_path)
    scan.sfx = readable_sfx


def _try_probe(path: str, binaries: FFmpegBinaries) -> MediaInfo | str:
    try:
        return probe(path, binaries)
    except (FFmpegError, OSError) as exc:
        return str(exc).splitlines()[0]


def _validate_media(outcome: MediaInfo | str, path: str, label: str, log: _IssueLog) -> None:
    problem = _media_problem(outcome)
    if problem is not None:
        code, description = problem
        log.error(code, f"{label} '{Path(path).name}' {description}.", path)


def _media_problem(outcome: MediaInfo | str) -> tuple[str, str] | None:
    if isinstance(outcome, str):
        return "unreadable_media", f"cannot be read ({outcome})"
    if outcome.kind is MediaKind.AUDIO:
        if not outcome.has_audio:
            return "unreadable_media", "contains no audio stream"
        if not outcome.duration or outcome.duration <= 0:
            return "empty_media", "is empty (zero-length audio)"
        return None
    if not outcome.has_video:
        return "unreadable_media", "contains no picture stream"
    if outcome.kind is MediaKind.VIDEO and outcome.duration is not None and outcome.duration <= 0:
        return "empty_media", "is empty (zero-length video)"
    return None
