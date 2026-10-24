from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"})
SCENE_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

_CHUNK_RE = re.compile(r"(\d+)")
_NUMBERED_STEM_RE = re.compile(r"^\D*?(\d+)(?:[ _.\-]*(?:([a-z])|[_.\-](\d+)))?(?:[ _.\-].*)?$", re.IGNORECASE)


def natural_key(name: str | Path) -> tuple:
    text = Path(name).name if isinstance(name, Path) else str(name)
    return tuple((0, int(chunk), "") if chunk.isdigit() else (1, 0, chunk.casefold()) for chunk in _CHUNK_RE.split(text) if chunk != "")


def natural_sorted(items: Iterable[str | Path]) -> list:
    return sorted(items, key=natural_key)


def parse_numbered_stem(stem: str) -> tuple[int, int] | None:
    match = _NUMBERED_STEM_RE.match(stem.strip())
    if not match:
        return None
    index = int(match.group(1))
    letter, number = match.group(2), match.group(3)
    if letter:
        sub_index = ord(letter.lower()) - ord("a") + 1
    elif number:
        sub_index = int(number)
    else:
        sub_index = 0
    return index, sub_index
