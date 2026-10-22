# Demo video recorder

These scripts record the demo video (`storyboard.md`) as repeatable takes and turn them into one MP4 with burned-in
captions. This is a separate tool, not part of the Python package or the Docker image.

- `record.mjs` records one clean `.webm` per chapter, plus an event log with every caption, highlight box and click,
  timed against that take's video. Each take starts from a fresh page load. It waits on what the page shows, never on
  fixed sleeps alone, and stops with an error if a caption's text is not on screen.
- `burn.mjs` reads the logs and cuts the waits. It burns in the captions, boxes and click rings, joins the chapters
  and writes the subtitles and one frame per event, so every take can be checked by eye.

## Prerequisites

- Node 20 or newer, and Playwright 1.63.0 with its Chromium. Install them in this folder:
  ```bash
  npm install --prefer-offline
  npx playwright install chromium   # only if ~/.cache/ms-playwright has no chromium-1243 yet
  ```
- ffmpeg with libass and libx264 (the `subtitles` filter). On the development machine it is the Windows build from
  winget (Gyan.FFmpeg 9.0.2), called from WSL by its path:
  ```bash
  FFMPEG=/mnt/c/Users/aboel/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe
  ```
  `burn.mjs` gives ffmpeg only relative paths, and runs it with the output folder as its working directory. That way
  a Windows `ffmpeg.exe` works without path conversion, as long as the output folder is on a Windows drive (`/mnt/c`).
  A Linux ffmpeg on `PATH` also works.
- Fonts: the UI asks for Segoe UI and Consolas. Under WSL, `record.mjs` points its own Chromium at the Windows copies
  through `fonts.conf`. Nothing is installed; its font cache goes to `runs/video/fontcache`.
- The replay chapters use the final evaluation run `runs/dryrun-tf/c3-test` and its seed set `runs/seeds/test`. Both
  are in the main checkout and are not committed.

## Record and burn

From the repository root (Windows Python from WSL, as in `CLAUDE.md`), start the UI. It listens on the WSL gateway
address, the default route in `ip route`, so that Chromium in WSL can reach it:

```bash
PY=/mnt/c/Users/aboel/AppData/Local/Programs/Python/Python310/python.exe
export WSLENV=PYTHONUTF8:PYTHONPATH:MANIPULAPY_QUIET PYTHONUTF8=1 MANIPULAPY_QUIET=1
PYTHONPATH='src;.' $PY -m vplan ui --host 172.18.80.1 --port 7861 \
  --replay 'C:\Users\aboel\SW_Projects\Nvidia_hakthon\runs\dryrun-tf\c3-test' \
  --seed-dir 'C:\Users\aboel\SW_Projects\Nvidia_hakthon\runs\seeds\test' \
  --record-dir runs/video/ui-recordings
```

Then, in `scripts/video/`:

```bash
node record.mjs                     # every chapter but cosmos, about 4 minutes -> runs/video/<chapter>.webm and .events.json
node burn.mjs --ffmpeg "$FFMPEG"    # about 40 s -> runs/video/vplan-demo.mp4
```

Stop the UI afterwards. `burn.mjs` does not need it.

### The optional Cosmos chapter (`--cosmos`)

This chapter shows NVIDIA Cosmos3-Edge as a learned, advisory visual check on one live baseline run. It needs a local
Cosmos server and the UI started with `--cosmos-url`. Every other UI chapter needs the UI started without it, so record
this chapter on its own.

1. Check that the GPU is free (`nvidia-smi`), then serve Cosmos3-Edge with vLLM 0.31.0 from its own venv. This is the
   command the shadow study used; it takes about 15 of 16 GB:
   ```bash
   VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/cosmos/bin/vllm serve nvidia/Cosmos3-Edge \
     --revision 344d602b128d1bbdacb43b08d0a3626f46343e29 --host 127.0.0.1 --port 8000 \
     --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.85 \
     --limit-mm-per-prompt '{"image":3,"video":0}' \
     --mm-processor-kwargs '{"do_resize":true,"min_pixels":4096,"max_pixels":409600}'
   curl http://127.0.0.1:8000/v1/models   # wait until this answers (about a minute)
   ```
2. Start the UI as above, but with `--cosmos-url http://127.0.0.1:8000/v1` instead of `--replay` and `--seed-dir`. The
   Windows interpreter reaches the WSL server through localhost forwarding.
3. Record and burn:
   ```bash
   node record.mjs --cosmos            # only the cosmos chapter, about 30 s
   node burn.mjs --ffmpeg "$FFMPEG"    # joins it after the refusal chapter
   ```
4. Stop the UI and the vLLM server.

Cosmos's verdicts vary from run to run, even at temperature 0. The take reads the verdict from the page (Pass, Fail at a
named frame, or Unsure) and uses the caption written for that outcome. It stops with an error if the banner does not
say the gate is on, or if the gate reports no verdict. The live take stops if the banner says Cosmos is on.
`node burn.mjs --no-cosmos` builds the video without this chapter, and so does a burn with no recorded cosmos take.

Outputs, all in `runs/video/` (gitignored):

| File | What it is |
|---|---|
| `vplan-demo.mp4` | The video: H.264, yuv420p, 1920×1080, 30 fps |
| `captions.srt` | The captions on the video's timeline, for the YouTube upload |
| `vplan-demo.ass` | Every overlay on the video's timeline (captions, boxes, click rings) |
| `chapters/<chapter>.mp4` | One chapter with its overlays |
| `frames/` | One PNG per event and `index.tsv` (time, chapter, type, text). Check these by eye before using the video |

## Re-record one chapter

A take that drifted is recorded again, never patched. Examples: a caption that does not match the screen, a panel
that was not loaded yet, a highlight in the wrong place.

```bash
node record.mjs --list                              # title, live, repair, refusal, cosmos (optional), results, closing
node record.mjs --chapter repair                    # only this take; the others stay as they are
node burn.mjs --ffmpeg "$FFMPEG" --chapter repair   # re-burns repair, re-joins all chapters, re-extracts every frame
```

`record.mjs` records a take again by itself, up to 3 times, if the browser started sending frames late (the video is
then shorter than its log). A take that cannot find what a caption names stops with an error and writes nothing.

Options:
- `record.mjs --url` uses another UI address (default `http://172.18.80.1:7861/?__theme=light`).
- `record.mjs --headed` shows the browser.
- `record.mjs --cosmos` records the optional cosmos chapter alone (see above); `burn.mjs --no-cosmos` leaves it out.
- `--out DIR` (both scripts) uses another output folder.

## How it works

- **Video clock:** Playwright's recorder counts time from its own creation, inside `newPage`. Event times count from
  the middle of that call, and `burn.mjs` corrects the small rest from the video's `creation_time`. A barely visible
  heartbeat pixel in the cards keeps frames coming while a card is still. Without it, the video would start at the
  first change, and every event would be early.
- **Execution animation:** the UI shows the execution as a looping GIF. Chromium starts it at its first paint and
  keeps its clock running while it is scrolled away. Before scrolling to it, a take waits (cut from the video) until
  the GIF is about to start over, so the Execute and Judge captions cover one whole run.
- **Cuts:** `cut` events mark waits. `burn.mjs` keeps the first 0.6 s and the last 0.4 s of each, so the video still
  shows the page working.
