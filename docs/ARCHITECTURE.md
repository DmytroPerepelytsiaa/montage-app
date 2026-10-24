# Dima Auto Editor — Architecture & Module Contracts

Python 3.13, PySide6, numpy, FFmpeg (subprocess). Package: `dima_auto_editor`.
Venv: `.venv` (run tools with `.venv/Scripts/python`). Tests: `.venv/Scripts/python -m pytest -q`.

## Already written (shared foundation — extend carefully, never break signatures)

- `config.py` — `ProjectSettings` (nested dataclasses: `paths`, `audio`, `timeline`, `video`,
  `transitions`, `export`, `alignment`) with `to_dict/from_dict/save/load`. All tunables live here.
- `models.py` — all shared dataclasses/enums: `MediaKind`, `Severity`, `Issue`, `MediaInfo`,
  `NumberedFile`, `TransitionDef`, `KeptSegment`, `Word`, `Sentence`, `BoundaryCandidate`,
  `BoundaryKind`, `ProcessedVoice`, `Narration`, `SceneAssignment`, `TimelineEntry`, `CutStyle`,
  `TransitionEvent`, `Timeline`, `to_jsonable()`.
- `naming.py` — `natural_key`, `natural_sorted`, `parse_numbered_stem(stem) -> (index, sub_index) | None`,
  extension sets `IMAGE_EXTENSIONS`, `VIDEO_EXTENSIONS`, `AUDIO_EXTENSIONS`, `SCENE_EXTENSIONS`.
  `03` -> (3,0); `03_2`/`03-2` -> (3,2); `03a` -> (3,1).
- `ffmpeg_tools.py` — `locate_binaries()`, `run_ffmpeg(args, duration=, progress=, cancel=, log=)`
  (adds `-y -progress pipe:1`, raises `FFmpegError` / `RenderCancelled`), `run_capture`, `probe(path) -> MediaInfo`
  (ffprobe or `ffmpeg -i` fallback; the bundled imageio-ffmpeg has NO ffprobe), `decode_audio(path, sr, channels) -> float32 ndarray`
  (mono 1-D, multi-channel shape (n, ch)), `write_wav(path, samples, sr)` (16-bit PCM), `working_encoders()`,
  `pick_encoder(pref)`, `video_encoder_args(encoder, quality, preset)`.

## Core time conventions (critical for sync)

- Output timeline time `t` in seconds, starting at 0 = first sample of processed narration.
- All scene boundaries are quantized to the frame grid: `round(t * fps) / fps` (fps = 60).
- Total video duration == narration duration quantized to frames (narration WAV is padded/trimmed to exactly
  `n_frames / fps` seconds = `n_frames * 800` samples at 48 kHz).
- Each `TimelineEntry` covers `[start, end)`; entries are contiguous: `entries[i].end == entries[i+1].start`,
  `entries[0].start == 0`, `entries[-1].end == timeline.duration`.
- A boundary between entry `i` and `i+1` has `position = i` (boundary index). It is either a hard cut or a
  `TransitionEvent` with midpoint `time == entries[i].end`, duration `D` (a multiple of 1/fps, even number of frames).
- Rendering with xfade: rendered clip i has length `span_i + D_left/2 + D_right/2` where D is the transition
  duration at that side (0 for hard cuts or video ends). The xfade offset for a transition at boundary time T
  is `T - D/2` in output time. Hence output length == timeline duration, and transition midpoints == boundary times.
- Clip i's *content* starts `D_left/2` before its span start (it shows during the incoming fade).

## Modules to implement

### `inputs.py` (agent A)
```python
@dataclass
class InputScan:
    voices: list[NumberedFile]          # natural-sorted, kind AUDIO
    scenes: list[NumberedFile]          # natural-sorted by (index, sub_index), kind IMAGE/VIDEO
    sfx: list[str]                      # audio files in sfx_dir (natural-sorted, absolute)
    music: str                          # "" if none; music_path may be a file or a folder (first audio file)
    script: str                         # "" if none
    issues: list[Issue]
    pairs: dict[int, list[NumberedFile]]  # voice index -> scenes with same index (sorted by sub_index)
    def has_errors(self) -> bool

def scan_inputs(paths: InputPaths, probe_media: bool = True) -> InputScan
```
Rules: never silently skip. Unsupported extension -> WARNING `unsupported_file`. Non-numbered name -> WARNING
`unnumbered_file`. Duplicate (index, sub_index) -> ERROR `duplicate_index`. Voice with no scene -> ERROR
`missing_scene`; scene with no voice -> WARNING `orphan_scene` (unless single continuous narration mode:
exactly one voice file and several scenes -> INFO `continuous_narration`). Gaps in numbering -> WARNING `numbering_gap`.
Missing/nonexistent folder -> ERROR. Unreadable media (probe fails) -> ERROR `unreadable_media`.
Zero-length audio -> ERROR. Messages are human-readable English, include filenames.

### `audio/silence.py` (agent A)
```python
@dataclass
class SilenceParams:  # built from AudioSettings via SilenceParams.from_settings(audio)
    threshold_db, min_silence_s, target_silence_s, fade_ms, edge_silence_s,
    pad_before_s, pad_after_s, window_s, highpass_hz, speech_aware

def frame_levels_db(samples: np.ndarray, sr: int, window_s: float, highpass_hz: float) -> np.ndarray
def detect_silences(samples, sr, params) -> list[tuple[float, float]]   # runs >= min_silence_s, in seconds
def plan_kept_segments(duration: float, silences, params, protected: list[tuple[float,float]] = ()) -> list[KeptSegment]
def apply_segments(samples, sr, segments, fade_ms) -> np.ndarray        # concat with equal-power-ish short fades
def process_voice(path, settings: AudioSettings, protected=()) -> tuple[np.ndarray, list[KeptSegment], list[tuple[float,float]]]
def map_time(segments: list[KeptSegment], src_time: float) -> float     # original -> processed time
```
Semantics: a silence run `[s, e]` with `e - s >= min_silence_s` is shortened so that the retained gap is
`target_silence_s` (keep `pad_after` worth right after speech end and the rest just before next onset; never keep
less than `pad_before_s` before speech or `pad_after_s` after speech — those protect consonants/decays; if
pads exceed target, retained = pad_before+pad_after). Leading/trailing silence of each file trimmed to
`edge_silence_s`. Speech-aware mode: smooth level with hangover (≥ 2 windows), high-pass first, and a silence
run is only accepted if no `protected` interval (word spans from alignment) overlaps it. Silence shorter than
`min_silence_s` is kept intact. Fades at every cut ≤ `fade_ms`, never longer than half a segment. Original files
are never modified. Output sample rate 48 kHz mono float32.

### `audio/narration.py` (agent A)
```python
def build_narration(voices: list[NumberedFile], settings: ProjectSettings, work_dir: Path,
                    progress=None, cancel=None, word_spans: dict[int, list[Word]] | None = None) -> Narration
```
Processes each voice file, concatenates (each already has edge silence so clip-to-clip gap ≈ 2*edge),
pads the end so the total is an exact number of video frames, writes `work_dir/narration.wav` (48 kHz mono 16-bit),
fills `ProcessedVoice.global_offset/duration/segments`, and produces `Narration.candidates`:
- one `BoundaryCandidate(kind=CLIP, strength=1.0)` at each clip junction (time = middle of the junction gap),
- `kind=PAUSE` candidates at the centre of each retained internal pause (strength 0.4–0.8 scaled by original pause
  length) — pauses approximate sentence ends when no alignment is present,
- if `word_spans`/sentences present: `kind=SENTENCE` (strength 0.9) at sentence ends mapped through `map_time`,
  `WORD_GAP` (0.2) between words.
Candidate times are in output time and quantized to frames. `gap_start/gap_end` = retained silence extent.

### `audio/mixer.py` (agent A)
```python
@dataclass
class SfxCue: path: str; start: float; volume_db: float
def mix_final_audio(narration_wav: str, duration: float, cues: list[SfxCue], music_path: str,
                    settings: AudioSettings, out_path: Path) -> dict   # returns stats {peak_db, voice_rms_db, sfx_peak_db, ...}
```
Stereo 48 kHz. Voice at `voice_gain_db`. SFX placed sample-accurately at `start` (clamped to ≥0; tails cut at
duration). Music looped/trimmed to duration at `music_volume_db`, ducked under speech when `music_ducking`
(envelope-based gain, ~−8 dB extra while voice active, smooth attack/release). SFX bus capped so its short-term
level stays ≥ 6 dB below the voice's speech level (narration stays louder). Final look-ahead peak limiter to
`limiter_ceiling_db`, guaranteeing no sample exceeds the ceiling (no clipping). Output exact length
`round(duration*48000)` samples.

### `transitions.py` (agent B)
```python
XFADE_TRANSITIONS: dict[str, str]  # display label -> ffmpeg xfade name (fade, dissolve, wipeleft, wiperight,
                                   # wipeup, wipedown, slideleft, slideright, slideup, slidedown, circleopen,
                                   # circleclose, pixelize, radial, smoothleft, smoothright, zoomin, fadeblack,
                                   # fadewhite, distance, hblur, squeezeh, squeezev, diagtl ...)
def default_library(sfx_files: list[str]) -> list[TransitionDef]
def load_library(path) -> list[TransitionDef]; def save_library(path, library) -> None
def validate_library(library, sfx_dir_files) -> list[Issue]     # unknown xfade, missing sfx file, bad duration
class TransitionPicker:  # deterministic given seed
    def __init__(self, library: list[TransitionDef], mode: str = "random" | "sequential", seed: int = 0)
    def next(self) -> TransitionDef      # random mode avoids repeating the same transition twice in a row if >1 available
    def pick_sfx(self, transition) -> str
```

### `timeline.py` (agent B)
```python
def plan_timeline(narration: Narration, scenes: list[NumberedFile], settings: ProjectSettings,
                  library: list[TransitionDef], assignments: list[SceneAssignment] | None = None) -> Timeline
```
Steps:
1. Assign scenes to narration spans. Direct pairing (default): scenes with index N cover voice N. Several scenes
   for one voice (03_1, 03_2) split that voice's span at the best internal candidates (prefer SENTENCE > PAUSE >
   WORD_GAP; choose by DP minimising duration penalty relative to [min,max] and the ideal equal split, plus a
   weak-boundary penalty). Continuous narration (one voice, many scenes): partition all candidates into K groups via
   the same DP. If `assignments` given, use them (stage 2).
2. If a span exceeds `max_scene_s` and no extra visual exists -> WARNING `scene_too_long` (never invent visuals).
   Below `min_scene_s` -> WARNING `scene_too_short` (unless pairing forces it; try to merge only when the scene map
   allows). Boundary not at SENTENCE/CLIP -> INFO `boundary_not_sentence` with the time.
3. Choose full transitions: iterate boundaries in time order; a boundary gets a transition only if it is
   ≥ `min_transition_interval_s` after the previous transition midpoint (and ≥ that from timeline start? no —
   only between transitions); prefer CLIP/SENTENCE boundaries; also require the transition duration window to fit
   inside both adjacent spans (D/2 ≤ 0.45*span). Otherwise hard cut. DP or greedy-with-lookahead to maximise
   count of high-strength transitions while respecting spacing — the spacing constraint must ALWAYS hold.
4. Duration D quantized to even frames; SFX start = midpoint - D/2 + transition.sfx_offset_s (clamped ≥ 0), volume
   = transition.sfx_volume_db. SFX should land in the retained silence (gap) — record INFO when it starts inside speech.
5. All times frame-quantized; contiguous entries; `timeline.duration == narration.duration`.

### `manifest.py` (agent B)
```python
def write_manifest(path, *, settings, scan, narration, timeline, library, outputs: dict, stats: dict, started_at, finished_at) -> None
def load_manifest(path) -> dict
def timeline_summary_rows(timeline) -> list[dict]   # for GUI table: #, scene file, start, end, duration, cut, transition, sfx, notes
def format_summary_text(timeline) -> str
```
JSON (UTF-8, ensure_ascii=False) with version, app name, settings, inputs, kept segments, timeline, transitions, issues.

### `render/` (agent C)
- `render/clips.py`: `render_scene_clip(scene: NumberedFile, info: MediaInfo, content_duration: float, out_path,
  settings: ProjectSettings, *, preview: bool, progress, cancel, log) -> ClipReport`
  Produces an intermediate video with exactly `round(content_duration*fps)` frames at 60 fps CFR, 1080x1920 (or
  preview scale), yuv420p, no audio, high-quality intra-friendly encoding (libx264 crf 10 `-preset veryfast` or
  NVENC with constqp ~16; preview uses lower quality).
  - Fit: `cover` (scale+crop), `contain_blur` (blurred bg), `contain_black`. Scaling flags from `video.scale_filter`
    (`neighbor` keeps pixel art crisp).
  - Images: hold, optionally `zoom_in`/`zoom_out`/`pan` by `image_motion_strength` (smooth, no jitter: upscale
    first then animate crop/zoompan).
  - Video longer than needed: trim. `trim_mode`: `start`, `center`, `smart` (analyse a small grayscale proxy of
    frame differences via numpy; choose the window whose end lands on a low-motion frame and which doesn't cross a
    hard internal scene cut). Video shorter: `extend_mode` `slow` (setpts up to `max_slowdown`, then freeze rest),
    `freeze`, `loop`, `pingpong`, `interpolate` (slow + motion interpolation).
  - Motion modes: `choppy` -> keep source frames and repeat them to reach 60 fps (`fps=60` duplicates frames; no
    blending). `smooth` -> interpolate to 60 fps (`minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1`)
    only when needed (source fps < 60 or slowdown applied); never for clips already at 60 fps with right timing.
  - `ClipReport`: frames, trimmed/extended flags, method, warnings (Issue list) e.g. pixel-art + interpolation warning.
- `render/compose.py`: `compose_video(clip_paths, timeline, settings, out_path, *, preview, progress, cancel, log)`
  — one filter graph chaining `xfade=transition=<name>:duration=D:offset=T-D/2` for transitions and `concat` for hard
  cuts, `settb`, `fps=60`, `format=yuv420p`. For >40 clips render in chunks split at hard cuts then concat demuxer.
- `render/export.py`: `mux_final(video_path, audio_wav, out_path, settings, *, preview, ...)` -> H.264 (encoder via
  `pick_encoder`) + AAC 48 kHz `-b:a audio_bitrate`, `-r 60 -fps_mode cfr`, `-movflags +faststart`, `-shortest` never
  used to cut audio — instead lengths are made identical upstream. Verify output duration/fps via probe.
  Unique output filename: `<output_dir>/<project>_YYYYmmdd_HHMMSS.mp4` (never overwrite; never write into source dirs).
- `pipeline.py` (agent C): `class BuildPipeline` orchestrating stages with `progress(stage: str, fraction: float)`,
  `cancel` Event, log file in output dir (`logs/build_<ts>.log`), temp dir `output_dir/.dima_tmp_<ts>` cleaned up
  safely (only that dir, only if created by us, kept when `export.keep_temp`). Methods: `scan()`, `preview_voice()` ->
  wav path, `plan()` -> Timeline (no rendering), `build(preview: bool)` -> `BuildResult(output_path, manifest_path,
  log_path, timeline, issues, stats)`. Stages: Scanning inputs → Processing voice-over → Planning timeline →
  Rendering scenes → Composing transitions → Mixing audio → Encoding → Verifying. Refuses to render when scan has
  ERROR issues (raises `BuildBlocked` with the issues).

### `gui/` (agent D)
`gui/main_window.py`, `gui/transition_editor.py`, `gui/worker.py` (QThread / QObject worker wrapping BuildPipeline,
signals for progress/log/finished/failed), `gui/settings_panel.py`. `dima_auto_editor/__main__.py` launches the app.
Folder selectors, all settings, transition library editor (table: name, xfade type dropdown, duration, sfx files
multi-select, volume dB, offset, enabled; add/remove/duplicate; preview SFX sound), motion mode radio buttons
(choppy pixel-art 60 FPS / smooth interpolated 60 FPS + pixel-art warning), progress bar + stage label, log pane,
issues list (colour by severity), timeline summary table, "Scan", "Preview voice" (QtMultimedia player),
"Preview render", big "Build Video", "Cancel", "Open output folder". Settings persisted to
`~/.dima_auto_editor/settings.json`. Long work never blocks the UI thread.

## Coding rules
- No comments unless logic is genuinely non-obvious; intention-revealing names; SOLID/DRY/KISS.
- English UI strings. Raise/return `Issue`s — never silently drop inputs.
- Tests live in `tests/`, use pytest, generate synthetic media with numpy + ffmpeg (lavfi) into `tmp_path`;
  keep each test fast (< 10 s). Shared fixtures in `tests/conftest.py` (`ffmpeg_bins`, `make_tone_wav`,
  `make_speech_like_wav(path, pattern)`, `make_video(path, seconds, fps, size)`, `make_image(path, size)`).
