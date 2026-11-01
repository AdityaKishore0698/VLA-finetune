# Module 17 — Interview Kit

**Files covered:** none new. This module packages Modules 1–16 for speaking. Every claim points back to a module (`M4 §4.5`) and its model answer (`M4 A5` = Module 4, Appendix A, answer 5).

**Where we are:** The manual so far is for understanding. This module is for *delivery*: what to say in 30 seconds, 2 minutes and 5 minutes; how to answer the 45 questions you're most likely to get; how to tell the behavioural stories; which phrases to avoid; what to ask back; and every number and term on one card.

**The one-line thesis:** The project's strongest story isn't "I fine-tuned a VLA". It's **"I built an evaluation harness honest enough to tell me my VLA tied a nearest-neighbour lookup, and then I audited the harness itself and found where its own comparisons weren't yet valid."**

> **Authorship note before anything else.** `pyproject.toml` lists Kushagra Srivastava as author and a `kushagrathisside/...` repository URL. README line 298 credits the SO-100 fine-tuning runs to `AdityaKishore0698` and the ALOHA run to `chytra3` (M1 §1.4, gap 11). Decide your exact framing *before* the interview, and use it consistently. For example: "a collaborative project. The codebase design is X's; I ran the SO-100 experiments, and I did this audit." Everywhere below, "I" means "the project". Swap in "we" or "the author" wherever that's the truthful version.

---

## 17.1 Prerequisites — the absolute basics

- **Pitch.** A self-contained explanation sized to a time box. A 30-second pitch is *what and why*; 2 minutes adds *how and the result*; 5 minutes at a whiteboard adds *architecture and trade-offs*.
- **Answer skeleton.** The three or four beats an answer must hit, in order. Memorize skeletons, not scripts.
- **STAR.** Situation, Task, Action, Result: the standard shape for behavioural answers. Add a fifth beat, *Reflection* ("what I'd do differently").
- **Signal.** What an answer tells the interviewer about you beyond its content: judgement, honesty, depth, ownership.
- **The honest downgrade.** Volunteering the limit of a claim before you're asked. It's the highest-signal move in a technical interview, because it shows you know where the edges are.

---

## 17.2 Pitches

### 30 seconds

> "It's an experiment and evaluation layer for fine-tuning a vision-language-action robot policy, SmolVLA, on real LeRobot datasets. The organizing idea is that the model layer never owns the data. Every data intervention, like halving the dataset or dropping a regime, becomes a lineage-tracked version in a separate data engine, so every checkpoint traces back to raw episodes. On SO-100 pick-and-place, fine-tuning reached 2.6% of the mean-action baseline's error. But it exactly tied a nearest-neighbour lookup over robot state, and on bimanual ALOHA retrieval was twice as good. The harness was built to surface results like that instead of hiding them."

### 2 minutes

Beats, in order:
1. **The problem (15 s).** VLA fine-tuning results are easy to overclaim: offline error gets read as task success, data changes aren't traceable, and baselines are weak or missing.
2. **The design principle (20 s).** The model layer never owns data. Interventions are engine transforms, data is named by immutable version, and every run directory records the config, both repos' commits and the environment.
3. **The pipeline (30 s).** One command: build the intervention version, re-check the splits, profile, fine-tune with validation-only checkpoint selection, score against mean-action and nearest-neighbour through the same evaluator, decompose the error (tails, per episode, per phase, slices), produce a feedback artifact, then report and model card. Anything it can't measure, like rollout, it records as "unavailable".
4. **The results (25 s).** SO-100: test MSE 0.0290, then 0.0239 after continued training (an 18% improvement on the identical test split, outside the roughly 5% inference noise), equal to nearest-neighbour. ALOHA: 0.216 vs retrieval 0.106.
5. **The honest reading (20 s).** Offline, on single-setup data, robot state almost determines the action. And the ALOHA run saw about a ninth of an epoch, so 'undertrained' is the first hypothesis.
6. **What I'd fix next (10 s).** Interventions currently re-split the data, so they're scored on different test sets. Freezing one test split and making seeds real come first.

### 5 minutes at a whiteboard

Draw this, top to bottom, talking through each box (M1, M11, M14 §14.2):

```
  YAML (base ⊕ experiment) ──▶ ExperimentConfig (pydantic; validators: "declare the weak dimension")
                                     │
  ┌──────────────── ExperimentRunner.run (composition root, transaction script) ──────────────────┐
  │ 0 run dir + env + git(both repos) + resolved config                                            │
  │ 1 intervention ──▶ ENGINE: EpisodeSubsetFilter → clean/vK ; DatasetBuilder → training/vM       │
  │ 2 split re-check · profile (coverage, QC, "unavailable" where unmeasurable)                    │
  │ 3 Trainer ──callback──▶ engine fit:  val-only (32 frames) · checkpoint · verify · prune · final │
  │ 4 baselines (mean, NN-state) + SmolVLA through SAME Evaluator · tails · episodes · phase · slices│
  │ 5 per-frame outcomes · model artifact · feedback (engine FeedbackLoop) · model card            │
  │ 6 report (every section; "not run") · summary · JSONL index                                     │
  └────────────────────────────────────────────────────────────────────────────────────────────────┘
        └──── human picks next intervention from feedback (dashed arrow) ────▶ back to 1
```

Then spend the remaining two minutes on **three trade-offs** (pick based on the interviewer):
- **Lineage vs comparability.** Filter-then-split gives perfect provenance, but a different test set per intervention. The fix is split once, intervene on train only (M4 §4.5).
- **Validation cost vs selection quality.** 32 frames is cheap, but val MSE is about 13× below test. The fix is a frozen stratified subset (M6 §6.4).
- **Honesty markers vs completeness.** Rollout is "unavailable", and so is visual diversity. Nothing is fabricated, so the reports are incomplete by design (M7, M10).

---

## 17.3 Question bank

Format: **Q** · skeleton (the beats) · → pointer.

### A. Project and motivation

1. **What does this project do, in one sentence?** Experiment, evaluation and feedback layer for SmolVLA on versioned robot data; the model layer never owns the data. → M0, M1 A5
2. **Why build this instead of using LeRobot's training script?** LeRobot trains; this layer adds lineage-tracked interventions, baselines through the same evaluator, decompositions and honesty markers, and records it all per run. Admit that the step itself is the engine's. → M1 §1.3, M1 A3
3. **What did *you* specifically do?** Use your authorship framing; name concrete parts (the SO-100 runs, this audit, whichever modules are yours). → note above, M1 gap 11
4. **What's the headline result?** SO-100 0.0239, equal to NN; ALOHA 0.216 vs NN 0.106; 2.6% of mean-action; 18% from continued training. → M7 §7.2, README
5. **Is the model any good?** Learned the distribution; doesn't beat retrieval offline; probably undertrained on ALOHA; closed loop unmeasured. → M7 A5, M15 A2

### B. Data and interventions

6. **Why version data instead of passing episode IDs to training?** Immutable identity, lineage back to raw, recorded reasons; cost: catalog churn and the moving test set. → M4 A2
7. **Walk me through what `reduced` does to the catalog.** Seeded draw over sorted IDs → `EpisodeSubsetFilter` → `clean/vK` → `DatasetBuilder` → `training/vM`; lineage records kept and dropped IDs; tested. → M4 A1
8. **Are your intervention results comparable to the baseline?** No: filter-then-split means a different, smaller test set; the fix is to split once and intervene on train. → M4 A4
9. **Why episode length as the imbalance variable?** SO-100 has one task and the schema has no variation field; length is the only varying attribute; terciles. → M4 §4.3
10. **What does "imbalanced" actually do?** Removes the regime entirely, so it's an out-of-distribution test, not under-representation; its test set lacks that regime. → M4 §4.4
11. **How does rebalancing work, and what can't it do?** Sampling weights on existing episodes, nothing duplicated; can't close a coverage gap. → M4 §4.5
12. **How do you prevent train/test leakage?** Episode-level splits, a single label per episode, re-check, val-only selection; content-level leakage is unchecked, and NN strength is the evidence. → M5 A5
13. **Does your split check actually catch anything?** The overlap branch is near-tautological; it misses unassigned and empty splits; make `ok` mean a complete partition. → M5 A1, A2

### C. Training

14. **What do you own in the training loop?** A callback into the engine's `fit`: val-only validation, checkpoint, verify, select, prune, final, crash marker, GPU release. → M6 A1, M1 A3
15. **How do you choose the checkpoint, and how confident are you?** Min val MSE among verified checkpoints; 32 frames of a stochastic policy, val 13× below test; right on SO-100 because the curve was still falling. → M6 A5
16. **Explain the 0.0018 vs 0.0239 gap.** Likely first frames of one episode (easiest phase), winner's curse, 5-episode splits; unverified engine frame order. → M6 A4
17. **What happens if the job is killed?** SIGTERM doesn't raise, so no `crash.json`; periodic checkpoints survive; resume restarts warmup; retry blocked. → M6 A3
18. **Why not save optimizer state?** Disk (Adam moments about 3× size); a fresh warmup on resume; the engine reportedly supports it; flips for preemptible long runs. → M6 §6.3
19. **How much data does a run actually see?** 0.26 epoch SO-100, 0.11 ALOHA (per-frame assumption). → M15 A2
20. **Why batch size 4?** Motivated by 8 GB, but peak was 2.2–3.0 GB, so it's conservative; try 8–16. → M15 §15.4

### D. Evaluation and metrics

21. **Why two baselines, and what's nearest-neighbour exactly?** Mean-action is the floor; NN retrieves by robot state, no images or language; same evaluator call. → M7 A1
22. **What does tying NN tell you?** Offline, state nearly determines action on single-setup data; closed loop is where vision should matter. → M7 A5
23. **Why report tails and per-phase error?** Control fails at rare frames; continued training cut the max by 32%; error peaks mid-episode. → M7 §7.3–7.4
24. **How do your slices work, and what's wrong with them?** Each slice re-runs stochastic inference, so they don't reconcile, and it costs about 3× compute; predict once and slice. → M7 A2
25. **Would you defend "long episodes are easier"?** No: 1–2 episodes per slice, 5% inference noise, about 25% per-episode spread. → M7 A3
26. **How do you know a difference between runs is real?** Mostly you don't yet: single seed, 5 test episodes, no intervals; only continued training (same split, 18% vs 5% noise) is defensible. → M10 A5
27. **What's `success_rate`?** Fraction of frames with MAE below 0.1; a proxy, never called success; the tolerance isn't even wired to config. → M7 §7.1, M2 §2.6

### E. Feedback loop

28. **Walk me through your data flywheel. Does it work?** Plumbing yes; as a decision procedure, no: always recommends action, defect branch unreachable, runs on test. → M8 A5
29. **Why a percentile threshold?** Robust to model and normalization; can't say "nothing's wrong"; add an absolute gate. → M8 A1
30. **Why is the "data defect" explanation never produced?** Max per-episode failure rate ≈ 0.50 (SO-100) and 0.40 (ALOHA) vs a > 0.5 rule. → M8 A2
31. **What's wrong with running failure analysis on test?** Adaptive overfitting: decisions must use val; test is touched once. → M8 A3

### F. Systems and design

32. **Draw the architecture and the dependency graph.** Four layers, acyclic; runner as composition root; CLI the one violation; `rde` under everything. → M14 A1
33. **Name design patterns you used.** Template Method via callback, Transaction Script, Specification validators, Façade, explicit "unavailable". → M14 A2
34. **Weakest part of the low-level design?** Typed at the edges, dicts in the middle, schemaless on disk; the `n/a` git-key bug as proof. → M14 A5
35. **Why JSONL and run directories instead of MLflow?** Serverless, durable, query with DuckDB; durability is on you; flips at the second user. → M3 A2, M16 A3
36. **What happens if two runs start at once?** TOCTOU on the run directory, catalog version race, torn JSONL, GPU contention. → M14 A4
37. **How do you make a run reproducible six months later?** Resolved config plus version lineage plus both commits plus env; gaps: no lockfile, dirty diffs, guessed engine path, git-ignored artifacts, stochastic inference. → M3 A5
38. **What breaks first if this scales?** Comparison validity, then eval time, then disk; an evidence-limited system. → M15 §15.5, A5
39. **How did you deploy it, and what's the latency?** Not to a robot; a local loader that's never executed; 47–50 / 86 ms per frame is evaluator throughput, not control latency. → M9 A5
40. **Is your health check sufficient?** It checks presence and well-formedness, not compatibility; normalization checked on the wrong object. → M9 A1

### G. Process, testing, judgement

41. **How is it tested?** 56 tests, 52 torch-free, real engine, no mocks; lineage verified end to end; no cross-stage invariant tests. → M13 A1, A2
42. **Why not mock the engine?** Mocks agree with your assumptions; use real engine for the boundary and fakes for pure logic. → M13 A3
43. **Which decisions came from constraints, and which were arbitrary?** The capstone: GPU, no robot, tiny data, schema, engine; arbitrary thresholds; C3 and C6 not carried through. → M13 A5, M13 §13.6
44. **If you had one day?** Strict config, real seeds, non-destructive re-scoring, predict-once, truthfulness pass. → M16 A1
45. **Another month?** Validity, then ALOHA 5–10k steps, then a nested learning curve, then reliability; not MLflow, a server or a bigger model. → M16 A5
46. **What would you not build?** The anti-roadmap: automated feedback, more slices, a server; each with its flip. → M16 §16.7
47. **What's the most surprising thing you learned?** Evaluation took as long as training; under one epoch; the harness's own comparisons were invalid. → M15 §15.3–15.4

---

## 17.4 Behavioural framings (STAR + Reflection)

**Hardest bug.** *The action-queue bug* (engine-side, documented in the smoke test and API_REFERENCE).
- **S:** evaluation scores looked plausible but slightly off.
- **T:** SmolVLA's `select_action` keeps a stateful queue of predicted action chunks for one continuous rollout.
- **A:** in batched offline evaluation, that queue replayed stale actions across unrelated batches. The fix was switching to `predict_action_chunk` (a fresh forward pass per call), and the regression test used a deliberately non-divisor batch size (3) to force varying batch sizes.
- **R:** evaluation became correct.
- **Reflection:** "stateful APIs designed for control loops are dangerous in batch evaluation; now I check whether any model call carries hidden state."
- *If this wasn't your fix, use the audit instead:* "the hardest bug I *found* was that interventions re-split the data, so every comparison used a different test set." (M4 A5)

**What you'd do differently.** "Design the comparison protocol before the pipeline. I'd freeze one test split on day one, make the seed flow through to training, and put a bootstrap interval on every number. The mechanisms (lineage, provenance, honesty markers) were right; the statistics weren't carried through." (M13 §13.6, last table)

**Proudest.** "That the system refuses to fabricate. Rollout is recorded as unavailable with a reason, visual diversity is 'unavailable' instead of a fake score, every report section renders as 'not run' rather than disappearing, and the headline result is a tie with a nearest-neighbour baseline, reported as a tie." (M7, M10)

**A trade-off you're unsure about.** "Putting the model adapter and its training step inside the data engine. It keeps feature shapes and normalization in lockstep with the engine's training contract, which is why cross-embodiment loading worked. But it inverts the dependency: the data layer owns the model. With one model family I think it was right. With a second I'd split adapters into their own package, and I'm not sure where the line is." (M1 §1.3)

**Conflict or disagreement (if asked).** Use the docs-vs-code findings: "the docs said selection was enforced in a module that doesn't exist, and that predictions were deterministic. I treated docs as claims and verified each against the code, and corrected about two dozen." (M16 §16.3 D-table)

**Failure.** "My one-day plan would have been to add MLflow. Auditing showed the runs weren't comparable, so a dashboard would only have displayed incomparable numbers more nicely. I changed the plan to fix the protocol first." (M16 A3)

---

## 17.5 Phrases that hurt you, and what to say instead

| Don't say | Why it hurts | Say instead |
|---|---|---|
| "The model achieves 50–57% success." | `success_rate` is a tolerance proxy, offline | "56.9% of test frames are within 0.1 mean absolute error, an offline proxy, not task success." |
| "SmolVLA beats the baselines." | it ties NN on SO-100 and loses on ALOHA | "It's at 2.6% of mean-action error and ties a state nearest-neighbour lookup." |
| "I wrote the training loop." | `fit` lives in the engine | "The engine owns the gradient step; this layer owns validation, selection, checkpointing and crash handling around it." |
| "An independent check guarantees no leakage." | the overlap branch can't fire; content leakage unchecked | "Splits are episode-level and re-checked for disjointness; content-level leakage between same-session episodes isn't checked." |
| "Long episodes are easier." | 1–2 episodes per slice | "The slice infrastructure works; five test episodes are too few for slice conclusions." |
| "We ran three seeds." (if using `--seed`) | `--seed` doesn't seed | "Single seed. The seed flag currently only names the directory, which I'd fix first." |
| "It's deployed." | replay harness, never executed | "There's a local loader with health checks; it hasn't been run against a robot, and it takes dataset samples." |
| "Inference latency is 48 ms." | evaluator throughput per frame, batch 16 | "About 48 ms per frame of evaluator throughput at batch 16, excluding data loading. Not control-loop latency." |
| "Validation MSE was 0.0018." | 32 frames, about 13× below test | "Selected on validation; the honest number is test MSE 0.0239." |
| "The flywheel automatically improves the data." | human-gated, runs on test, always recommends action | "It produces ranked failures and a recommendation for a human; as a decision rule it needs to move to validation." |
| "It's fully reproducible." | no lockfile, dirty diffs, stochastic | "Every input is recorded; outputs reproduce within inference noise, a few percent." |
| "Coverage is 0.9." | occupied-bin evenness | "That metric is evenness among occupied bins. I'd renormalize it by the full bin count." |
| "Retrieval is just a strong baseline." | ALOHA ≈ 0.11 epoch | "Retrieval is strong on single-setup data, and the ALOHA run was undertrained at about a ninth of an epoch." |
| "No bugs that I know of." | there are ~200 gaps | "Here are the three I'd fix first, and why." |

---

## 17.6 Questions to ask them

1. "How do you evaluate policies before closed-loop testing, and how much do you trust offline metrics to rank them?"
2. "When you change a dataset, how do you track which models trained on which version?"
3. "What's your split policy for robot data: by episode, session, scene or operator?"
4. "How many seeds do you typically run before you believe a difference?"
5. "Where does the training loop live relative to the data pipeline in your stack, and who owns the model adapters?"
6. "What's your bottleneck today: GPU hours, data collection, or evaluation throughput?"
7. "How do you handle stochastic policies (diffusion or flow matching) when comparing checkpoints?"
8. "What would a successful first three months look like in this role?"

---

## 17.7 The ten sentences that carry the project

1. "The model layer never owns the data. Every intervention is a lineage-tracked dataset version, never a local copy."
2. "Offline, on single-setup data, robot state almost determines the next action. That's why a state nearest-neighbour lookup tied a VLA."
3. "The harness's job was to tell me the truth, and it did: the baseline was that strong."
4. "Test never influences checkpoint selection; the trainer's only validation call hard-codes the val split."
5. "I filter before I split, so each intervention got its own test set. The fix is to split once and intervene on train only."
6. "My ALOHA run saw about a ninth of an epoch; 'undertrained' comes before 'retrieval is better'."
7. "Every number I report is offline, and the code records rollout as unavailable rather than invent one."
8. "Evaluation took about as long as training, and half of it was redundant re-inference."
9. "I honoured the constraints in the mechanisms and didn't carry them through to the statistics."
10. "My P0 list is about one thing: making two numbers from this system comparable."

---

## 17.8 Quick-reference card: every number

**Datasets**

| | SO-100 pick-place | ALOHA static coffee |
|---|---|---|
| Source | `lerobot/svla_so100_pickplace` | `lerobot/aloha_static_coffee` |
| Robot / action dims | single arm, 6 | bimanual, 14 (state 14 joint + 14 effort = 28) |
| Cameras / fps | 2 / 30 | 4 / 50 (AV1) |
| Episodes after QC (train/val/test) | 48 (38/5/5); raw 50, dropped [9, 31] | 40 (32/4/4) |
| Test frames | 1,908 | 4,400 (all test episodes 1,100 frames) |
| Episode length | 326–475 frames, ≈ 13 s | — |
| Tasks | 1 | 1 (per config comment) |

**Results** (README Tables 2–3)

| Run | Steps | Test MSE | / mean-action | NN MSE | within 0.1 | Train time | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_full` (SO-100) | 1,000 | 0.0290 | 3.2% | 0.0239 | 49.8% | 324 s | 2.2 GB |
| `baseline_full_extended` | +1,000 | 0.0239 | 2.6% | 0.0239 | 56.9% | 385 s | 2.2 GB |
| `aloha_coffee_baseline` | 1,000 | 0.2157 | 21.4% | 0.1062 | 0.05% | 576 s | 3.0 GB |

Mean-action MSE 0.908 (SO-100), 1.010 (ALOHA). SO-100 2,000-step tails: median 0.089, p90 0.174, p99 0.272, max 0.375 (from 0.553). Per-joint MAE 0.063 (shoulder pan) → 0.157 (gripper). Phase: start 0.051, mid 0.144, end 0.049. Per-episode MAE 0.085–0.111 (SO-100), 0.296–0.405 (ALOHA). Val minimum 0.0018 at step 1,800. Re-scoring noise up to 5% (0.0305 vs 0.0290; 0.0232 vs 0.0239). Inference 47–50 ms/frame (SO-100), 86 ms (ALOHA) at batch 16. Host memory 3.0–3.3 GB. NN confidence–error correlation ≈ −0.79.

**Config** (`configs/base.yaml`): splits 0.8/0.1/0.1, seed 42, quantile · steps 1,000 · batch 4 · lr 1e-4 · wd 1e-5 · warmup 50 · clip 1.0 · ckpt/val every 100 · `val_batches` 8 (32 frames) · keep 3 · training seed 0 · eval batch 16 · tolerance 0.1 · slices `[per_task, per_length_bucket]` · baselines `[mean_action, nearest_neighbor]`. Extended run: lr 3e-5, warmup 20, `resume_from step_1000`. Schema defaults differ: steps 200, ckpt 50, eval batch 32.

**Derived** (M15): ≈ 3.1 steps/s; epochs seen ≈ 0.26 (SO-100), 0.53 (extended), 0.11 (ALOHA); eval ≈ 45% (SO-100) and 57% (ALOHA) of run time; run ≈ 10–11 min (SO-100), ≈ 22 min (ALOHA); checkpoints ≈ 4–9 GB per run (estimated).

**Thresholds:** feedback p90, ≥ 3 failures per episode, concentration > 0.5; terciles 33.3/66.7, ≥ 3 episodes; 10 histogram bins; oversample 3.0; split-sum tolerance 1e-6; ID digest 10 hex characters.

**Code and tests:** about 2,700 lines in `src`; 11 CLI commands; 56 tests (49 unit, 3 integration, 4 smoke), 52 light; 2 custom exceptions; 4 index files; 7 run subdirectories; 5 checkpoint files.

**Environment:** RTX 4060 Laptop 8 GB; ~10 GB VM RAM; lerobot 0.4.4; torch 2.10 (CUDA build); driver 592.82; CUDA 13.1; Python ≥ 3.10 declared, 3.11 used.

---

## 17.9 Glossary

| Term | Meaning (as used in this project) | Module |
|---|---|---|
| Action chunk | sequence of future actions predicted per call by SmolVLA | M1, M9 |
| Adapter (model input) | converts dataset samples to model tensors (`StateAdapter`, `VisionLanguageAdapter`) | M1, M7 |
| Adaptive overfitting | test set biased by decisions made from looking at it | M8 |
| Anti-roadmap | changes that would make the system worse now, with flip conditions | M16 |
| Atomic write | write to temp file, then `os.replace`; never half-written | M3 |
| Banker's rounding | Python `round` rounds .5 to even | M4 |
| Behaviour cloning | supervised regression from observation to demonstrated action | M1 |
| Bootstrap (over episodes) | resampling test episodes to get a confidence interval | M7 |
| Catalog | engine metadata store of versions and episodes | M1 |
| Characterization test | test pinning current (possibly wrong) behaviour | M13 |
| Checkpoint | saved weights plus processors plus metadata at a step (5 files) | M6, M9 |
| Clean version (`clean/vN`) | output of transforms (normalization, episode filtering) | M1, M4 |
| Closed-loop rollout | policy controls a robot or simulator; measures task success | M1 |
| Composition root | the one place collaborators are wired (`ExperimentRunner`) | M14 |
| Content-addressed | identified by a hash of its inputs | M2, M15 |
| Contract (training) | dict describing modalities: plane, shape, normalization | M1 |
| Coverage (profile) | normalized entropy of a 10-bin histogram; actually evenness among occupied bins | M5 |
| Deep merge | nested dicts merge; scalars and lists replace | M2 |
| DoF | degrees of freedom: independently controllable joints | M1 |
| Embodiment | the robot body: kinematics, cameras, action space | M1 |
| Engine (`rde`) | Episodic Data Engine, the sibling repo owning data, `fit`, evaluator, feedback | M1 |
| Episode | one recorded demonstration; the unit of splits and interventions | M1 |
| Epoch | one full pass over the training data | M15 |
| Evenness (Pielou) | entropy normalized by occupied categories | M5 |
| External plane | modality stored outside the frame table (images, video) | M1 |
| Fail fast | validate before expensive work | M11 |
| Feedback artifact | ranked failing episodes, explanations, recommended next action | M8 |
| Flow matching | generative method; makes SmolVLA's predictions stochastic | M1 |
| Frame | one timestep: images, state, action | M1 |
| Inline plane | numeric modality stored with the frame (state) | M1 |
| Intervention | a deliberate change to the training data (full, fraction, reduced, imbalanced, rebalanced, custom) | M2, M4 |
| JSONL | one JSON object per line; append-friendly | M3 |
| Lineage | recorded chain of operations producing a version | M1, M4 |
| LeRobot | Hugging Face robot-learning library and dataset format | M1 |
| Mean-action baseline | always predicts the training mean (MSE ≈ 1 normalized) | M7 |
| Nearest-neighbour baseline | returns the action of the closest training *state* | M7 |
| Normalized units | per-dimension (x − mean)/std | M1 |
| Offline evaluation | predicted vs demonstrated action on recorded data | M1 |
| Orphaned version | catalog version left by a failed run | M4 |
| Percentile threshold | flags the top (100 − p)% of frames by error, always about that fraction | M8 |
| Predict-once slicing | infer once, then group outcomes by slice | M7 |
| Quantile split mode | exact split sizes (inferred) | M2 |
| Rebalanced | upweight episodes by sampling weight; no exclusion | M4 |
| Replicate | a run differing only by random seed | M11 |
| Resolved config | merged config with all defaults, persisted per run | M2 |
| Run directory | `runs/<date>/<name>/seed_<n>/` | M3 |
| Slice | evaluation subset by task or length bucket | M7 |
| SmolVLA | small VLA from LeRobot; base `lerobot/smolvla_base` | M1 |
| `success_rate` | fraction of frames with MAE < 0.1; a proxy | M7 |
| Tautological check | a check whose failure can't be represented by its input | M5 |
| Template Method via callback | engine loop with an injected `on_step` hook | M6, M14 |
| Terciles | 33.3/66.7 percentile cut into short/medium/long | M4 |
| Training version (`training/vN`) | built dataset with splits and weights | M1, M4 |
| Transaction Script | one procedure running a whole operation | M11 |
| TOCTOU | race between checking a condition and acting on it | M3 |
| "Unavailable" marker | explicit record that something wasn't measured, with a reason | M5, M10, M12 |
| Validation-only selection | checkpoints compared on val, never test | M6 |
| VLA | vision-language-action model | M1 |
| Winner's curse | the minimum of noisy estimates is biased low | M6 |

---

## 17.10 Honest gaps to own before an interviewer finds them

1. **Authorship ambiguity** (`pyproject.toml` vs README credits). *Consequence:* any inconsistency in "I" vs "we" will be noticed. Settle the framing in advance.
2. **The three published runs predate the protocol fixes.** *Consequence:* present them as pipeline validation; the robust result is continued training on the identical split.
3. **Nothing in this manual was executed.** The engine wasn't available, so engine-side behaviour (Evaluator frame order, sampling unit, version-allocation atomicity) is inferred. *Consequence:* say "I didn't verify X" rather than guessing, if pressed.

---

## The shape to carry into the Data Provenance & Demo module

```
 30 s: what + principle + tie result     2 min: problem ▸ principle ▸ pipeline ▸ results ▸ honest read ▸ next fix
 5 min: whiteboard runner stages + 3 trade-offs (lineage vs comparability · val cost vs selection · honesty vs completeness)
 47 questions → module + appendix pointers      6 behavioural stories (STAR + reflection)
 14 phrase swaps · 8 questions to ask · 10 carrying sentences · numbers card · glossary
```

---

## Drill — answer from memory, files closed

1. Give the 30-second pitch, out loud, and time it.
2. *Why didn't you just* report the 56.9% "within tolerance" as the success rate? It's a bigger number.
3. Recite every number on the results table without looking.
4. *What breaks in production* if you say "we ran three seeds" and the interviewer asks to see the configs?
5. *Interviewer framing:* "Tell me about this project."

---

# Appendix A — Model answers to the drill

**A1. The 30-second pitch.** See §17.2. The test: under 35 seconds, mentions *lineage*, *the tie with nearest-neighbour* and *honesty markers*, and contains no unqualified "success".

*Why that answer lands:* in 30 seconds it gives a principle, a result and a stance, which is more memorable than a feature list.

**A2. Why not call 56.9% the success rate?**
"Because it isn't one. It's the fraction of test frames whose mean absolute action error is under 0.1, computed offline against recorded demonstrations. A policy can be within tolerance on most frames and still fail the grasp, and in closed loop small errors compound. The whole system is built so that offline accuracy is never presented as task success: the metric is labelled a proxy everywhere and rollout is recorded as unavailable. Calling it success would be the one claim that undoes everything else the project gets right."

*Why that answer lands:* it explains the technical difference, ties it to the project's design, and shows you won't trade credibility for a bigger number.

**A3. Reciting the numbers.** SO-100: 0.0290 → 0.0239, NN 0.0239, mean-action 0.908, 3.2% → 2.6%, 49.8% → 56.9% within tolerance, 324 s and 385 s, 2.2 GB. ALOHA: 0.2157, NN 0.1062, mean 1.010, 21.4%, 0.05%, 576 s, 3.0 GB. Splits 38/5/5 and 32/4/4; 1,908 and 4,400 test frames.

*Why that answer lands:* fluency with your own numbers is the cheapest evidence you did the work.

**A4. "We ran three seeds", then the configs.**
"They'd see that the `--seed` flag only changes the run directory name. The training seed in each resolved config is the same, zero. So the 'three seeds' are three runs with one seed, differing only through GPU nondeterminism and sampling noise, and any spread is an underestimate. That's why I'd never say it. The honest statement is 'single seed', followed by 'and the seed flag doesn't seed yet; that's the second fix on my list.'"

*Why that answer lands:* it shows you know exactly how the claim would collapse, and that the safe phrasing is already part of your vocabulary.

**A5. "Tell me about this project."** Deliver the 2-minute pitch (§17.2), then stop and offer a direction: "I can go deeper on the data-versioning design, the evaluation and what the nearest-neighbour tie means, or the flaws I found auditing it. Which is most useful?"

*Why that answer lands:* it's bounded, it has a result and a self-critique, and it hands control back with three strong branches, each of which you've prepared.

---

# Module 17 Summary — the one page to revise

**What this module is.** The speaking layer of the manual: three pitches, 47 questions with skeletons and pointers, six behavioural stories, phrase swaps, questions to ask, ten carrying sentences, a numbers card and a glossary.

**Five claims to make cold**
1. The model layer never owns the data; interventions are lineage-tracked versions.
2. SmolVLA ties state-NN on SO-100 (0.0239) and loses on ALOHA (0.216 vs 0.106).
3. Every number is offline; rollout is recorded as unavailable.
4. Continued training gave 18% on the identical split, outside the 5% noise. It's the robust result.
5. The fixes I'd make first all serve one goal: two numbers from this system being comparable.

**Decisions (as you'd narrate them)**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Lead with the tie | lead with the 2.6% ratio | the honest baseline | less flattering |
| Volunteer flaws | wait to be asked | the strongest signal | longer answers |
| "Proxy" every time | "success" | central honesty contract | smaller-sounding numbers |
| Offer three deep-dive branches | monologue | interviewer chooses, you're prepared | none |

**Volunteer unprompted:** authorship framing; the published runs predate the fixes; the engine-side behaviours you couldn't verify.

**Numbers and names:** see the §17.8 card. Memorize the results table and the derived epoch fractions.

**Sentences worth memorizing:** §17.7, all ten.
