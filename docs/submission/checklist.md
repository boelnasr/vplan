# Submission checklist (W4.5)

Everything the submission needs is on GitHub, so it can be done from any laptop with a browser. No Python, GPU or keys
are needed for these steps.

- **Deadline:** 2026-10-30, 10:00 AM PT, which is 18:00 in Germany (CET; Europe leaves summer time on Oct 25, the US on
  Nov 1). The plan is to submit on Oct 29.
- **Repo:** https://github.com/boelnasr/vplan (public, AGPL-3.0, `main`).
- **Release with the files:** https://github.com/boelnasr/vplan/releases/tag/submission-data
  - `vplan-demo.mp4`: the demo video, 2:57, 1080p, silent with burned-in captions (D83).
  - `captions.srt`: the same captions as a subtitle file, for the upload.
  - `vplan-runs-lightning-test-d70.zip`: the final evaluation's attempt records (D70) and the `test` seed set.
  - `vplan-runs-ultra-test-d82.zip`: the Nemotron 3 Ultra runs on `test` (D82) and the same seed set.
  - `vplan-runs-a1-test-d85.zip`: config 3 rerun on `test` with the merged findings, Lightning and Ultra (D85), and the
    same seed set.

## Steps

1. **Upload the video. Done (Oct 23):** https://youtu.be/y_Ig1zOWKPY, public. The rules require the video to be
   "publicly visible on YouTube", so not unlisted. Add `captions.srt` from the release as English subtitles, and
   `youtube-thumbnail.png` (1920×1080) as the custom thumbnail.
2. **Put the link in the README. Done (Oct 23):** "Demo video" links to the YouTube URL.
3. **Fill in Devpost.**
   - Project name: **vplan: LLMs plan, physics verifies**.
   - Elevator pitch (179 of 200 characters): "NVIDIA Nemotron plans robot pick-and-place; physics checks every plan
     before anything moves and sends failures back for repair. 65% success vs 43% unverified, 0 unsafe executions."
   - Track: **Physical AI**.
   - Story sections (Inspiration, What it does, How we built it, Challenges, Accomplishments, What we learned, What's
     next): paste from [`description.md`](description.md). Its "How we use NVIDIA models" section answers the NVIDIA
     model question.
   - Feedback on the Nebius and NVIDIA tools: paste from [`feedback.md`](feedback.md).
   - Repo URL and video URL as above.
   - Thumbnail: [`thumbnail.png`](thumbnail.png) (3000×2000, 3:2).
   - Built with (25 tags): Python, PyBullet, ManipulaPy, NumPy, Pydantic, Gradio, Matplotlib, Pillow, pytest,
     OpenAI Python SDK, Nebius Token Factory, NVIDIA Nemotron, Nemotron 3.5 Lightning, Nemotron 3 Ultra, NVIDIA Cosmos,
     Cosmos3-Edge, vLLM, CUDA, Docker, Playwright, Node.js, FFmpeg, URDF, HTML, GitHub.
   - Image gallery: the eight images in [`media/`](media/README.md), in order, with the captions listed there.
4. **Check before submitting** (the [hackathon rules](https://nebiusglobalaihackathon.devpost.com/rules)):
   - the repo is public and has `LICENSE` and a README with setup instructions;
   - the video is under 3 minutes (2:57) and over 1 minute shows the modules in action (2:03);
   - Nemotron runs on Nebius Token Factory (D70, D82, D85), and at least one NVIDIA open model is used;
   - the description says how the NVIDIA models are used, and the feedback is filled in.

## Replaying the evaluation (optional)

The video's replay chapters use attempts `3:test:1` and `3:test:3` of the D70 run. To open them on another machine,
unzip `vplan-runs-lightning-test-d70.zip` at the repo root (it creates `runs/dryrun-tf/` and `runs/seeds/test/`), set
up Python as in the README, and run:

```bash
python -m vplan ui --replay runs/dryrun-tf/c3-test --seed-dir runs/seeds/test
```

`vplan report build --run runs/dryrun-tf/c1-test --run runs/dryrun-tf/c3-test --run runs/dryrun-tf/c7-test --run
runs/dryrun-tf/c5-test --output <dir> --seed-dir runs/seeds/test` rebuilds the D70 report. Each run's `workers/`
scratch folder is left out of the zips; replay and the report do not read it.

## Not on GitHub (on the development PC only, not needed to submit)

- The `dev` A/B runs (D75–D82), the Cosmos shadow-study renders (D73) and the video takes, under `runs/wt-*/`.
- The local Cosmos3-Edge server (vLLM) for the optional live Cosmos check.
- `.env` and `.env.tokenfactory`, which hold the API keys and are never committed.
