# Dima Auto Editor

Desktop app for Windows that automatically edits short vertical (1080×1920, 60 FPS) biography videos
from numbered scenes (`01.mp4`, `02.png`, …) and numbered voice-overs (`01.wav`, `02.wav`, …):
it removes long pauses from the narration, times every scene to its voice-over, adds transitions with
sound effects at sensible moments, mixes optional background music and exports a ready-to-upload MP4.

---

## Быстрый старт (по-русски)

1. Установите Python 3.11 или новее: https://www.python.org/downloads/ (отметьте **Add python.exe to PATH**)
   или в PowerShell: `winget install Python.Python.3.12`.
2. Скопируйте (или клонируйте) папку проекта, например в `E:\Projects\montage-app`.
3. Откройте PowerShell в этой папке и выполните **один раз**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File install.ps1
   ```
4. Запускайте программу двойным щелчком по **`run.bat`**.
5. В программе укажите папки: озвучка (`01.wav`, `02.wav`…), сцены (`01.mp4`, `02.png`, `03_1.mp4`, `03_2.png`…),
   звуки переходов (SFX), при желании музыку и папку для результата.
6. Нажмите **Scan inputs** → проверьте список проблем (Issues) → **Plan timeline** → **Build Video**.

FFmpeg отдельно ставить не нужно — используется встроенный (imageio-ffmpeg). Установщик также скачивает **RIFE** —
нейросеть, которая дорисовывает промежуточные кадры (optical flow) на видеокарте: замедленные клипы идут плавными
60 FPS. Исходные файлы никогда не изменяются, результат, манифест и логи сохраняются только в выбранную папку вывода.

---

## Installation (Windows)

1. **Install Python 3.11+** (3.10 is the minimum, 3.12/3.13 recommended)
   - from https://www.python.org/downloads/ — tick *Add python.exe to PATH* in the installer, or
   - in PowerShell: `winget install Python.Python.3.12`
2. **Get the project folder** — clone it with git or copy the folder, e.g. to `E:\Projects\montage-app`.
3. **Install dependencies** — open PowerShell in the project folder and run:
   ```powershell
   powershell -ExecutionPolicy Bypass -File install.ps1
   ```
   The script checks the Python version, creates `.venv`, installs `requirements.txt`
   (PySide6, numpy, imageio-ffmpeg), checks FFmpeg and downloads RIFE into `tools\rife` (neural frame generation
   on the GPU via Vulkan; about 430 MB download, 18 MB kept). Options:
   - `-Dev` — also installs `requirements-dev.txt` (pytest) for running the tests;
   - `-SkipRife` — do not download RIFE (frame generation then uses the slower FFmpeg `minterpolate`).
4. **Launch** — double-click `run.bat` (or `powershell -ExecutionPolicy Bypass -File run.ps1`,
   or `.venv\Scripts\python -m dima_auto_editor` from a terminal to see console output).

### FFmpeg

The app ships with FFmpeg through the `imageio-ffmpeg` package, so it works out of the box (that build has no
`ffprobe`; the app reads media information with `ffmpeg -i` instead). To use your own FFmpeg (for example a
newer build with more GPU encoders), either:

- install it system-wide: `winget install Gyan.FFmpeg`, then restart the app;
- put `ffmpeg.exe` into `tools\ffmpeg\bin\` inside the project folder;
- set the `DIMA_FFMPEG_DIR` environment variable to the folder containing `ffmpeg.exe`; or
- choose the executable on the **Export** tab (*FFmpeg executable*).

If no FFmpeg can be found the app shows a dialog explaining these options. *Tools → Check FFmpeg* shows which
binary is used.

---

## Input folder conventions

| Folder | Contents |
| --- | --- |
| Voice-over folder | `01.wav`, `02.wav`, … (also `.mp3 .m4a .aac .flac .ogg .opus .wma`) |
| Scenes folder | `01.mp4`, `02.png`, … (videos `.mp4 .mov .mkv .webm .avi .m4v .gif`, images `.png .jpg .jpeg .webp .bmp .tif .tiff`) |
| Transition SFX folder | Any audio files: `whoosh_01.wav`, `impact.mp3`, … |
| Background music (optional) | One audio file, or a folder (its first audio file is used) |
| Output folder | Where the video, manifest and logs are written. Must not be a source folder. |

- **Pairing by number**: scene `N` is shown while voice-over `N` is spoken. Names may contain text after the
  number (`01 intro.wav`, `02_childhood.mp4`).
- **Several scenes for one voice-over**: `03_1.mp4`, `03_2.mp4`, `03_3.png` (or `03-1`, `03a`/`03b`). All of them are
  shown, splitting the voice-over's time **equally**; each clip is slowed down or trimmed to fit its share.
- **Transitions with sound effects play only between voice-overs** (`02 → 03`); scenes inside one voice-over change
  with a hard cut.
- Voice-overs and scenes may live in the **same folder** (`2.mp3`, `2_1.mp4`, `2_2.mp4`, `3.mp3`, …).
- **Single continuous narration**: exactly one voice file (`01.wav`) with many scenes — the scenes split it equally.
- **Nothing is skipped silently**: unnumbered files, unsupported formats, duplicates, gaps, voice-overs without
  scenes or scenes without voice-overs are all reported in the *Issues* list (errors block the build).
- Source media is **never modified**, and nothing is written into source folders.

---

## Using the app

Left side — settings in tabs (all values are saved automatically to `~/.dima_auto_editor/settings.json` and restored
on the next start; *File → Save/Load project settings* stores a complete project as JSON).
Right side — *Timeline* summary, *Preview*, *Issues* (coloured by severity) and *Log*.
Bottom — stage, progress bar with elapsed time and ETA, and the action buttons.

| Button | What it does |
| --- | --- |
| **Scan inputs** | Lists the files, pairs scenes with voice-overs and reports problems. Fast. |
| **Plan timeline** | Processes the narration and plans scene times and transitions without rendering. |
| **Preview voice-over** (Preview tab) | Processes the narration (pause removal) and plays it. |
| **Preview render** (Preview tab) | Renders a fast low-resolution preview and plays it (or *Open in player*). |
| **Build Video** | Renders the final video. |
| **Cancel** | Stops the running task safely (temporary files are cleaned up). |
| **Open output folder** | Opens the folder with the results. |

All long operations run in a background thread — the window stays responsive.

### Settings

**Audio**
- *Silence threshold (dB)* — levels below this count as silence (−38 dB default; raise toward −30 for noisy recordings).
- *Minimum silence (s)* — only pauses at least this long are shortened; shorter pauses stay untouched.
- *Target pause (s)* — the length a long pause is shortened to.
- *Fade at cuts (ms)* — tiny fades at every audio cut to avoid clicks.
- *Clip edge silence*, *Pad before/after speech* — protect word onsets, endings and natural decays.
- *Speech-aware detection* — high-pass filtering and hangover smoothing so breaths, consonants and word tails are never cut.
- *Voice / SFX / Music volume*, *Duck music under speech*, *Limiter ceiling* — the narration always stays louder
  than the effects; the final mix never clips.

**Video**
- *Motion mode* (see below), *Fit mode* (cover / contain on blurred background / contain on black),
  *Scaling filter* (nearest neighbor keeps pixel art crisp), *Image motion* (static, slow zoom in/out, pan) and strength.
- *Clip too short*: slow down to fit exactly (default), freeze the last frame, loop, ping-pong, or slow down with
  motion interpolation (adds in-between frames).
- *Clip too long*: smart trim (chooses a calm cut point and avoids internal scene cuts), keep start, keep center.

**Transitions** — library file, random (seeded) or sequential selection, seed, default duration and the
*Transition & SFX library* editor.

**Export** — encoder (*Auto* picks the first working one of NVENC, QSV, AMF, libx264), quality (CRF/CQ, lower =
better), preset, AAC bitrate, preview scale/length, *Keep temporary files* (for debugging).

### Motion modes and frame generation

- **Slowed-down clips always get generated in-between frames** (option *Generate in-between frames for slowed-down
  clips*, on by default), so a clip stretched to fill its share of the voice-over plays as smooth 60 FPS instead of
  repeating frames.
- **Choppy 60 FPS (keep original frames)** — clips at normal speed keep their own frames, repeated to reach 60 FPS.
- **Smooth 60 FPS (generate frames for every clip)** — every clip below 60 FPS gets generated frames.
- **Engine**: *Auto* uses **RIFE** (neural optical flow on the GPU, `toolsifeife-ncnn-vulkan.exe`, the engine
  behind Flowframes) when installed, otherwise FFmpeg `minterpolate` (CPU, roughly 8× slower and with more artifacts).
  If RIFE fails (for example no Vulkan driver), the clip is rendered with `minterpolate` and a warning is shown.
- Frame generation can smear very low-resolution pixel art; the app warns for sources narrower than 720 px.

### Transition & SFX library

*Transitions → Edit transition & SFX library…* (or *Tools* menu) opens a table where each row is a transition:
enabled, name, FFmpeg xfade type, duration, sound effects (multi-select from the SFX folder), SFX volume (dB) and
SFX offset (s, relative to the transition start). Rows can be added, duplicated, removed and reordered; *Play*
auditions the sounds. Problems (unknown type, missing sound file, bad duration) are listed live at the bottom.
*Reset to defaults* builds a starter library that matches sounds by name (whoosh → slides/wipes, impact → flashes,
glitch → pixelize). On OK the library is saved (by default to `~/.dima_auto_editor/transitions.json`).

Library JSON format:

```json
{
  "version": 1,
  "transitions": [
    {
      "name": "Slide left",
      "xfade": "slideleft",
      "duration_s": 0.4,
      "sfx_files": ["E:/sfx/whoosh_01.wav", "whoosh_02.wav"],
      "sfx_volume_db": -10.0,
      "sfx_offset_s": 0.0,
      "enabled": true
    }
  ]
}
```

`sfx_files` may contain absolute paths or plain file names (resolved inside the SFX folder). One sound is picked
per transition (randomly with the seed, or in order).

---

## Outputs

Everything goes into the chosen output folder:

- `<project>_YYYYmmdd_HHMMSS.mp4` — the video (H.264, 1080×1920, 60 FPS CFR, AAC 48 kHz). Existing files are never overwritten.
- a JSON **manifest** in the output folder (its path is shown in the Log after each build) — settings, inputs, kept audio segments, every scene's start/end,
  transitions with their sound effects, and all issues. Useful to reproduce or audit an edit.
- `logs/build_<timestamp>.log` — the detailed build log including FFmpeg commands.
- `.dima_tmp_<timestamp>/` — temporary files, removed after the build unless *Keep temporary files* is on.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `run.bat` says `.venv` was not found | Run `powershell -ExecutionPolicy Bypass -File install.ps1` first. |
| "running scripts is disabled on this system" | Use the full command with `-ExecutionPolicy Bypass` shown above. |
| `py` / `python` not found | Install Python (see above) and open a **new** terminal window. |
| "FFmpeg not found" dialog | Re-run `install.ps1`, or `winget install Gyan.FFmpeg`, or set the path on the Export tab. |
| Build blocked with errors | Open the *Issues* list: fix the named files (missing scene for a voice-over, duplicate numbers, unreadable media) and *Scan* again. |
| Words cut off / pauses too long | Lower *Silence threshold* (e.g. −42 dB) or increase *Pad after speech*; increase *Target pause* for a calmer pace. |
| SFX too loud / too quiet | Adjust the global *SFX volume* or per-transition *SFX volume* in the library editor. |
| GPU encoder fails | Choose *Auto* or *CPU: libx264* on the Export tab. |
| App closes without a message | Start it from a terminal with `.venv\Scripts\python -m dima_auto_editor` to see the error, and check `logs/` in the output folder. |

---

## Running the tests

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -Dev
.venv\Scripts\python -m pytest -q
```

The tests create small synthetic media with FFmpeg in temporary folders; GUI tests run headless
(`QT_QPA_PLATFORM=offscreen`).
