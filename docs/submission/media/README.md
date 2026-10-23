# Devpost gallery images

Upload these to **Project Media → Image gallery** in this order. Each is 2400×1600 (3:2) PNG, under 1.3 MB, so it meets
Devpost's limits (JPG, PNG or GIF, 5 MB, up to 15 images). The UI images are frames of the demo video's raw takes; the
diagram and the report are screenshots of `docs/diagrams/system-overview.html` and `docs/results/report.html`.

| File | Caption |
|---|---|
| `01-cover.png` | vplan: NVIDIA Nemotron plans, deterministic checks verify every plan before anything moves. |
| `02-architecture.png` | How it works: Nemotron on Nebius Token Factory proposes, 8 kinematic and collision checks decide, an independent Judge scores. |
| `03-repair.png` | A recorded Nemotron attempt: two plans rejected with measured findings, the third passes all 8 checks and executes. |
| `04-execute-judge.png` | Only an approved plan runs in PyBullet physics; the independent Judge confirms the result. |
| `05-refusal.png` | Nemotron refuses a target out of reach and says why; the Judge accepts the refusal. |
| `06-cosmos.png` | Optional NVIDIA Cosmos3-Edge check on rendered key frames: advisory only, here a false rejection of a valid plan. |
| `07-results.png` | Results on 300 held-out tasks: Lightning (the headline), Ultra and the richer findings, with 0 unsafe verified runs. |
| `08-report.png` | The evaluation report: every number with its 95% interval, in one self-contained HTML file per run. |

The replay header in `05-refusal.png` showed the run folder as an absolute local path; the image shows it relative to the
repo instead.
