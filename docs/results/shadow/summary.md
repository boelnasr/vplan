# Cosmos shadow study of c3-test: a learned visual check, not verification

A learned visual check, not verification. Cosmos never influenced the run, so this measures what a visual gate would have caught or falsely rejected, never incremental success (spec §11, D8 item 4). Truth is the state predicate of each kinematic key frame, posed from the plan's trajectory, not the executed outcome the Judge scored.

- Status: complete. Model `nvidia/Cosmos3-Edge`, thinking on; config 3 on the test seed set `834d19318aab`; 215 attempts.
- Questions: 1337 recorded, 1337 answered, 0 endpoint errors; injected frames skipped: none. Executed attempts never asked about: without key frames none; instruction names no single object none.
- Natural negatives: 19 executed rounds whose key frames fail a predicate, 19 of them rounds whose plan grasped another object than the instructed one.
- Latency per request: n 1695, median 1.30 s, p90 2.72 s, max 7.11 s. Per question (with re-asks): median 1.62 s, p90 3.72 s.

## Per stage

For predicate-true frames a fail is a false rejection and a pass a correct acceptance. For negatives a fail is a true rejection (a catch) and a pass a catch the gate missed. In the live gate a final uncertain also stops execution (visual_uncertain). A misaligned grasp on a wrong-target round is the misaligned grasp of a round whose plan grasped another object than the instructed one.

| Frames | Stage | N | pass | fail | uncertain | fail rate | pass rate | uncertain rate | errors | latency median / p90 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| predicate true (natural) | pregrasp | 196 | 188 | 3 | 5 | 1.5% | 95.9% | 2.6% | 0 | 1.14 / 2.09 |
| predicate true (natural) | grasp | 196 | 175 | 7 | 14 | 3.6% | 89.3% | 7.1% | 0 | 1.33 / 2.10 |
| predicate true (natural) | prerelease | 196 | 127 | 53 | 16 | 27.0% | 64.8% | 8.2% | 0 | 1.81 / 3.51 |
| predicate true (natural) | all | 588 | 490 | 63 | 35 | 10.7% | 83.3% | 6.0% | 0 | 1.32 / 2.76 |
| natural negatives | pregrasp | 19 | 11 | 0 | 8 | 0.0% | 57.9% | 42.1% | 0 | 1.02 / 1.53 |
| natural negatives | grasp | 19 | 10 | 6 | 3 | 31.6% | 52.6% | 15.8% | 0 | 1.64 / 2.71 |
| natural negatives | prerelease | 19 | 0 | 16 | 3 | 84.2% | 0.0% | 15.8% | 0 | 0.93 / 1.33 |
| natural negatives | all | 57 | 21 | 22 | 14 | 38.6% | 36.8% | 24.6% | 0 | 1.09 / 2.31 |
| injected negatives, all | pregrasp | 159 | 129 | 1 | 29 | 0.6% | 81.1% | 18.2% | 0 | 1.19 / 2.68 |
| injected negatives, all | grasp | 374 | 282 | 55 | 37 | 14.7% | 75.4% | 9.9% | 0 | 1.55 / 2.94 |
| injected negatives, all | prerelease | 159 | 14 | 92 | 53 | 57.9% | 8.8% | 33.3% | 0 | 1.00 / 2.03 |
| injected negatives, all | all | 692 | 425 | 148 | 119 | 21.4% | 61.4% | 17.2% | 0 | 1.30 / 2.72 |
| injected: wrong object | pregrasp | 159 | 129 | 1 | 29 | 0.6% | 81.1% | 18.2% | 0 | 1.19 / 2.68 |
| injected: wrong object | grasp | 159 | 127 | 20 | 12 | 12.6% | 79.9% | 7.5% | 0 | 1.63 / 3.16 |
| injected: wrong object | prerelease | 159 | 14 | 92 | 53 | 57.9% | 8.8% | 33.3% | 0 | 1.00 / 2.03 |
| injected: wrong object | all | 477 | 270 | 113 | 94 | 23.7% | 56.6% | 19.7% | 0 | 1.22 / 2.72 |
| injected: misaligned grasp | grasp | 196 | 150 | 25 | 21 | 12.8% | 76.5% | 10.7% | 0 | 1.45 / 2.49 |
| injected: misaligned grasp | all | 196 | 150 | 25 | 21 | 12.8% | 76.5% | 10.7% | 0 | 1.45 / 2.49 |
| injected: misaligned grasp on a wrong-target round | grasp | 19 | 5 | 10 | 4 | 52.6% | 26.3% | 21.1% | 0 | 2.32 / 3.27 |
| injected: misaligned grasp on a wrong-target round | all | 19 | 5 | 10 | 4 | 52.6% | 26.3% | 21.1% | 0 | 2.32 / 3.27 |

## Balanced accuracy

The D34 pilot's measure (pass for predicate-true frames, fail for negatives; a final uncertain counts as wrong), with the predicate-true and negative N in brackets. The D34 pilot scored 30 frames (15 predicate-true, 15 injected negatives: wrong object, offset grasp and outside the bin) with a lenient parser: 0.70 with thinking on. Its D36 note puts that at about 0.60 under the strict parser used here.

| Negatives | pregrasp | grasp | prerelease | all |
|---|---|---|---|---|
| natural | 0.48 (196 / 19) | 0.60 (196 / 19) | 0.75 (196 / 19) | 0.61 (588 / 57) |
| injected | 0.48 (196 / 159) | 0.52 (196 / 374) | 0.61 (196 / 159) | 0.52 (588 / 692) |

## The live gate replayed on the natural frames

The gate asks the stages in order and stops at the first that does not pass. A round's frames are predicate true when all three stage predicates hold.

| Round's frames | N | pass (executes) | stopped by fail | stopped by uncertain | stopped at pregrasp / grasp / prerelease |
|---|---|---|---|---|---|
| predicate true | 196 | 111 | 53 | 32 | 8 / 19 / 58 |
| predicate false | 19 | 0 | 8 | 11 | 8 / 3 / 8 |
