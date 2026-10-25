from __future__ import annotations

import json
import math
import random
import re
from dataclasses import replace
from itertools import cycle
from pathlib import Path
from typing import Iterable

from .models import Issue, Severity, TransitionDef
from .render.stylized_transitions import STYLE_LABELS, STYLIZED_TRANSITIONS

SELECTION_RANDOM = "random"
SELECTION_SEQUENTIAL = "sequential"
SELECTION_MODES = (SELECTION_RANDOM, SELECTION_SEQUENTIAL)

LIBRARY_FORMAT_VERSION = 1
MAX_TRANSITION_DURATION_S = 3.0

XFADE_TRANSITIONS: dict[str, str] = {
    "Crossfade": "fade",
    "Fast crossfade": "fadefast",
    "Slow crossfade": "fadeslow",
    "Dissolve": "dissolve",
    "Fade through black": "fadeblack",
    "Fade through white": "fadewhite",
    "Fade through grey": "fadegrays",
    "Distance": "distance",
    "Wipe left": "wipeleft",
    "Wipe right": "wiperight",
    "Wipe up": "wipeup",
    "Wipe down": "wipedown",
    "Wipe top-left": "wipetl",
    "Wipe top-right": "wipetr",
    "Wipe bottom-left": "wipebl",
    "Wipe bottom-right": "wipebr",
    "Slide left": "slideleft",
    "Slide right": "slideright",
    "Slide up": "slideup",
    "Slide down": "slidedown",
    "Smooth left": "smoothleft",
    "Smooth right": "smoothright",
    "Smooth up": "smoothup",
    "Smooth down": "smoothdown",
    "Cover left": "coverleft",
    "Cover right": "coverright",
    "Cover up": "coverup",
    "Cover down": "coverdown",
    "Reveal left": "revealleft",
    "Reveal right": "revealright",
    "Reveal up": "revealup",
    "Reveal down": "revealdown",
    "Circle open": "circleopen",
    "Circle close": "circleclose",
    "Circle crop": "circlecrop",
    "Rectangle crop": "rectcrop",
    "Vertical open": "vertopen",
    "Vertical close": "vertclose",
    "Horizontal open": "horzopen",
    "Horizontal close": "horzclose",
    "Diagonal top-left": "diagtl",
    "Diagonal top-right": "diagtr",
    "Diagonal bottom-left": "diagbl",
    "Diagonal bottom-right": "diagbr",
    "Slices left": "hlslice",
    "Slices right": "hrslice",
    "Slices up": "vuslice",
    "Slices down": "vdslice",
    "Wind left": "hlwind",
    "Wind right": "hrwind",
    "Wind up": "vuwind",
    "Wind down": "vdwind",
    "Pixelize": "pixelize",
    "Radial": "radial",
    "Zoom in": "zoomin",
    "Horizontal blur": "hblur",
    "Squeeze horizontal": "squeezeh",
    "Squeeze vertical": "squeezev",
}
XFADE_NAMES = frozenset(XFADE_TRANSITIONS.values())

_SFX_KEYWORDS: dict[str, tuple[str, ...]] = {
    "whoosh": ("whoosh", "woosh", "swoosh", "swish", "swoop", "swipe", "sweep"),
    "impact": ("impact", "hit", "boom", "punch", "thud", "slam", "bang"),
    "glitch": ("glitch", "digital", "static", "bitcrush", "distort"),
}
_MOVEMENT_PREFIXES = ("wipe", "slide", "smooth", "cover", "reveal")


def _sfx_category_for_xfade(xfade: str) -> str | None:
    if xfade.startswith(_MOVEMENT_PREFIXES):
        return "whoosh"
    if xfade in ("fadewhite", "zoomin"):
        return "impact"
    if xfade in ("pixelize", "hblur"):
        return "glitch"
    return None


_STARTER_SET: tuple[tuple[str, float, bool], ...] = (
    ("Crossfade", 0.5, True),
    ("Dissolve", 0.5, True),
    ("Fade through black", 0.6, True),
    ("Fade through white", 0.4, True),
    ("Slide left", 0.4, True),
    ("Slide right", 0.4, True),
    ("Slide up", 0.4, True),
    ("Slide down", 0.4, True),
    ("Wipe left", 0.4, True),
    ("Wipe right", 0.4, True),
    ("Wipe up", 0.4, False),
    ("Wipe down", 0.4, False),
    ("Smooth left", 0.5, True),
    ("Smooth right", 0.5, True),
    ("Circle open", 0.5, True),
    ("Circle close", 0.5, False),
    ("Zoom in", 0.4, True),
    ("Pixelize", 0.4, True),
    ("Horizontal blur", 0.4, True),
    ("Radial", 0.5, False),
    ("Squeeze horizontal", 0.4, False),
    ("Squeeze vertical", 0.4, False),
    ("Diagonal top-left", 0.5, False),
    ("Diagonal bottom-right", 0.5, False),
)


def _name_tokens(path: str) -> list[str]:
    return re.findall(r"[a-z]+", Path(path).stem.casefold())


def sfx_category(path: str) -> str | None:
    tokens = _name_tokens(path)
    for category, keywords in _SFX_KEYWORDS.items():
        if any(token.startswith(keyword) for token in tokens for keyword in keywords):
            return category
    return None


def default_library(sfx_files: list[str], duration_s: float | None = None) -> list[TransitionDef]:
    by_category: dict[str, list[str]] = {}
    for path in sfx_files:
        category = sfx_category(path)
        if category:
            by_category.setdefault(category, []).append(path)
    round_robin = cycle(sfx_files) if sfx_files else None

    library: list[TransitionDef] = []
    for label, duration, enabled in _STARTER_SET:
        xfade = XFADE_TRANSITIONS[label]
        matched = by_category.get(_sfx_category_for_xfade(xfade) or "", [])
        if matched:
            sfx = list(matched)
        elif round_robin is not None:
            sfx = [next(round_robin)]
        else:
            sfx = []
        library.append(TransitionDef(name=label, xfade=xfade, duration_s=duration_s or duration, sfx_files=sfx, enabled=enabled))
    return library


def stylized_library(styles: Iterable[str], sfx_files: list[str], duration_s: float) -> list[TransitionDef]:
    return [TransitionDef(name=STYLE_LABELS[style], xfade=style, duration_s=duration_s, sfx_files=list(sfx_files))
            for style in styles if style in STYLE_LABELS]


def save_library(path: str | Path, library: list[TransitionDef]) -> None:
    payload = {"version": LIBRARY_FORMAT_VERSION, "transitions": [transition.to_dict() for transition in library]}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_library(path: str | Path) -> list[TransitionDef]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_items = data.get("transitions", []) if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        raise ValueError(f"Transition library {path} must contain a list of transitions.")
    return [_transition_from_raw(item) for item in raw_items if isinstance(item, dict)]


def _transition_from_raw(item: dict) -> TransitionDef:
    transition = TransitionDef.from_dict(item)
    sfx = transition.sfx_files
    transition.sfx_files = [sfx] if isinstance(sfx, str) else [str(entry) for entry in sfx or [] if entry]
    transition.duration_s = float(transition.duration_s)
    transition.sfx_volume_db = float(transition.sfx_volume_db)
    transition.sfx_offset_s = float(transition.sfx_offset_s)
    transition.enabled = bool(transition.enabled)
    return transition


def resolve_sfx_path(entry: str, sfx_dir_files: Iterable[str]) -> str | None:
    if Path(entry).is_file():
        return str(Path(entry))
    wanted = Path(entry).name.casefold()
    for candidate in sfx_dir_files:
        if Path(candidate).name.casefold() == wanted:
            return str(candidate)
    return None


def resolve_library_sfx(library: list[TransitionDef], sfx_dir_files: Iterable[str]) -> list[TransitionDef]:
    available = list(sfx_dir_files)
    resolved_library = []
    for transition in library:
        resolved = [path for path in (resolve_sfx_path(entry, available) for entry in transition.sfx_files) if path]
        resolved_library.append(replace(transition, sfx_files=resolved))
    return resolved_library


def validate_library(library: list[TransitionDef], sfx_dir_files: Iterable[str]) -> list[Issue]:
    available = list(sfx_dir_files)
    issues: list[Issue] = []
    seen_names: set[str] = set()
    for transition in library:
        label = transition.name or "(unnamed)"
        if not transition.name.strip():
            issues.append(Issue(Severity.WARNING, "unnamed_transition", f"A transition of type '{transition.xfade}' has no name."))
        elif transition.name.casefold() in seen_names:
            issues.append(Issue(Severity.WARNING, "duplicate_transition_name", f"Transition name '{label}' is used more than once."))
        seen_names.add(transition.name.casefold())
        if transition.xfade not in XFADE_NAMES | STYLIZED_TRANSITIONS:
            issues.append(Issue(Severity.ERROR, "unknown_xfade", f"Transition '{label}' uses unknown FFmpeg xfade type '{transition.xfade}'."))
        issues.extend(_duration_issues(transition, label))
        for entry in transition.sfx_files:
            if resolve_sfx_path(entry, available) is None:
                issues.append(Issue(Severity.WARNING, "missing_sfx", f"Sound effect '{Path(entry).name}' of transition '{label}' was not found; it will be skipped.", entry))
    if not any(transition.enabled for transition in library):
        issues.append(Issue(Severity.INFO, "no_transitions_enabled", "No transitions are enabled; every scene change will be a hard cut."))
    return issues


def _duration_issues(transition: TransitionDef, label: str) -> list[Issue]:
    duration = transition.duration_s
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return [Issue(Severity.ERROR, "bad_transition_duration", f"Transition '{label}' has an invalid duration ({duration}); it must be greater than 0 s.")]
    if duration > MAX_TRANSITION_DURATION_S:
        return [Issue(Severity.WARNING, "bad_transition_duration",
                      f"Transition '{label}' lasts {duration:.2f} s (more than {MAX_TRANSITION_DURATION_S:.0f} s); it will be shortened to fit the scenes.")]
    return []


class TransitionPicker:
    def __init__(self, library: list[TransitionDef], mode: str = SELECTION_RANDOM, seed: int = 0):
        if mode not in SELECTION_MODES:
            raise ValueError(f"Unknown transition selection mode '{mode}'. Use one of: {', '.join(SELECTION_MODES)}.")
        self._choices = [transition for transition in library if transition.enabled]
        self._mode = mode
        self._rng = random.Random(seed)
        self._sequence_cursor = 0
        self._last_choice: int | None = None
        self._sfx_cursors: dict[str, int] = {}
        self._last_sfx: str | None = None

    @property
    def available(self) -> list[TransitionDef]:
        return list(self._choices)

    @property
    def has_choices(self) -> bool:
        return bool(self._choices)

    def next(self) -> TransitionDef:
        if not self._choices:
            raise LookupError("No enabled transitions in the library.")
        if self._mode == SELECTION_SEQUENTIAL:
            index = self._sequence_cursor % len(self._choices)
            self._sequence_cursor += 1
        else:
            index = self._draw_avoiding(len(self._choices), self._last_choice)
        self._last_choice = index
        return self._choices[index]

    def pick_sfx(self, transition: TransitionDef) -> str:
        files = [entry for entry in transition.sfx_files if entry]
        if not files:
            return ""
        if self._mode == SELECTION_SEQUENTIAL:
            cursor = self._sfx_cursors.get(transition.name, 0)
            self._sfx_cursors[transition.name] = cursor + 1
            chosen = files[cursor % len(files)]
        else:
            last = files.index(self._last_sfx) if self._last_sfx in files else None
            chosen = files[self._draw_avoiding(len(files), last)]
        self._last_sfx = chosen
        return chosen

    def _draw_avoiding(self, count: int, previous: int | None) -> int:
        if count == 1 or previous is None:
            return self._rng.randrange(count)
        index = self._rng.randrange(count - 1)
        return index + 1 if index >= previous else index
