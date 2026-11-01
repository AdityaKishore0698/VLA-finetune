# Module 18 — Data Provenance & the Demo

**Files covered:** `README.md` §1, §2.4, §4–5 and §10 (as claims), `docs/figures/results.svg`, `docs/diagrams/01-system-architecture.svg`, `configs/base.yaml`, `configs/experiments/aloha_coffee_baseline.yaml` (dataset identity), `tests/conftest.py` (synthetic data). Engine-side ingestion is described only from this repository's call sites and README; I couldn't read the engine.

**Where we are:** Every earlier module assumed data already in the catalog. This module answers where it actually comes from, how it gets in, what this repository owes the sibling engine, and how to *show* the project live without breaking anything, including what to say when something goes wrong in front of an interviewer.

**The one-line thesis:** The data's path from Hugging Face to a trained checkpoint is fully traceable through engine lineage. But **none of the artifacts behind the README's numbers live in this repository** (runs are git-ignored), so the demo is a walk through *pre-recorded* run directories plus a few safe, read-only commands. Two of the CLI's commands will silently damage those recordings if run during a demo.

---

## 18.1 Prerequisites — the absolute basics

- **Data provenance.** The documented origin and processing history of a dataset: where it came from, what was done to it, by what code, and when.
- **Hugging Face Hub dataset.** A versioned dataset repository addressed as `org/name` (here `lerobot/svla_so100_pickplace`, `lerobot/aloha_static_coffee`). Each has its own licence and dataset card. **I haven't checked either licence. Check before redistributing anything.**
- **LeRobot dataset format.** Per-frame tabular data (Parquet) plus camera streams stored as video (the ALOHA config's comment says AV1), with episode boundaries and metadata. The README says the test fixture uses "the real LeRobot v3.0 layout".
- **Ingest → transform → build.** The engine's three stages: ingest produces `raw/vN`, transforms (normalize, QC filtering, episode subsetting) produce `clean/vN`, and build produces `training/vN` with splits and weights. Module 1 has the vocabulary.
- **QC (quality control).** Automated checks on episodes (missing frames, duplicates, discontinuities). Failing episodes are excluded from clean versions or builds.
- **Pre-recorded demo.** Showing completed artifacts instead of running long jobs live. The standard for any ML demo longer than a minute.

---

## 18.2 Where the data comes from and how it gets in

```
 Hugging Face Hub                     episodic-data-engine (rde)                         vla-finetune (this repo)
 ─────────────────                    ────────────────────────────                       ────────────────────────
 lerobot/svla_so100_pickplace ──┐     rde run --config ../episodic-data-engine/          mtrain profile  (read-only)
 lerobot/aloha_static_coffee  ──┼──▶      configs/so100_pickplace.json                   mtrain train:
                                │       "download, ingest, QC, version"  (README §4)        build_intervention_dataset
                                │       ingest  ─▶ raw/v1   (50 episodes, SO-100)            → (EpisodeSubsetFilter) clean/vK
                                │       transform ─▶ clean/v1 (48; dropped [9, 31])           → DatasetBuilder        training/vM
                                │       QC records per episode                              Trainer / Evaluator / FeedbackLoop
                                └──     catalog DB + Parquet + video under settings.paths    runs/… (git-ignored)
```

**Verified from this repository:**
- **Dataset identities** are `so100_pickplace` and `aloha_coffee` (`configs/base.yaml`, `aloha_coffee_baseline.yaml`), each with input version `clean/v1`.
- **The SO-100 lineage chain**, as printed in README §1 by `rde version lineage so100_pickplace training/v6`:
  - `ingest [] -> raw/v1  episodes 0->50`
  - `transform ['raw/v1'] -> clean/v1  episodes 50->48  dropped=[9, 31]`
  - `build_training_dataset ['clean/v1'] -> training/v6  episodes 48->48`

  Each step carries a run ID, a fingerprint and the software version. The two dropped episodes are recorded *by ID*.
- **Splits after QC:** 48 episodes → 38/5/5 (SO-100); 40 → 32/4/4 (ALOHA) (README Table 1).
- **`training/v6`** means at least six training versions of SO-100 were built. That's consistent with Module 4's "every run mints a new version".

**Inferred, not verified:**
- **What the `clean/v1` transform did beyond dropping episodes 9 and 31.** The test fixture's `clean_version` applies `NormalizeVector("state.joint")`, `NormalizeVector("action.joint")` and then QC, which suggests the real `clean/v1` is normalized. The README's "MSE is in normalized action units" agrees.
- **Why episodes 9 and 31 were dropped** (presumably QC failures, perhaps missing or discontinuous frames). The engine's QC records hold the answer: `rde obs quality`, as `profile.py`'s note suggests.
- **ALOHA's raw episode count before QC**, and which ALOHA episodes (if any) were dropped. Not stated in this repo.
- **The engine ingest config's contents** (`../episodic-data-engine/configs/so100_pickplace.json`).

**The synthetic test data** (`tests/conftest.py`) is generated by `rde.testing.make_lerobot_dataset`: 8 episodes × 20 frames at 10 fps, two task strings, then ingested, normalized, QC'd and built exactly like real data. No network, and no mocks of the source format (README §9). It's the only data this repository can produce on its own.

---

## 18.3 The relationship to the sibling project

| Aspect | Engine (`episodic-data-engine`, `rde`) | This repo (`vla-finetune`, `model_training`) |
|---|---|---|
| Owns | ingestion, versions, lineage, QC, transforms, `DatasetBuilder`, `DatasetView`, `Evaluator`, `FeedbackLoop`, baselines, **`SmolVLAPolicy` including `fit`** | config, intervention *resolution*, training *procedure*, evaluation *decomposition*, feedback *packaging*, artifacts, reports, CLI |
| Dependency direction | imports nothing from this repo (README §5) | imports `rde` everywhere (Module 14) |
| Changes made *to* the engine for this project | three additive changes (README §5, ARCHITECTURE §4): **`SmolVLAPolicy.fit()` with checkpoint save, load and verify**; **an `episode_ids` selector on `Evaluator.evaluate()`**; **the `EpisodeSubsetFilter` transform** | — |
| Installed as | editable sibling checkout via `[tool.uv.sources]` (Module 1) | — |
| Version pin | **none** | — |
| Git provenance | recorded per run from a guessed path (Module 3) | recorded per run |

**What to say in an interview:** "Three engine changes were made for this project: real gradient fine-tuning with checkpointing, slice-level evaluation by episode ID, and the episode-filter transform interventions are built on. Everything else in the engine predates this layer." (Credit the engine's authorship per the Module 17 note.)

---

## 18.4 What's on disk versus in the repo

| Artifact | In this repository? | Where it lives |
|---|---|---|
| Source code, configs, docs, `results.svg`, architecture SVG | **yes** | git |
| Datasets (raw, clean, training versions) | no | engine data directory (`settings.paths`) |
| Run directories behind README Tables 2–4 (`runs/2026-09-03/...` etc.) | **no**, git-ignored (`runs`) | the experimenter's disk |
| Cross-run index | **no**, git-ignored (`experiments/*.jsonl`) | the experimenter's disk |
| Checkpoints (selected step 1,800 SO-100 / step 800 ALOHA) | no | inside the run directories |
| Base model weights | no | Hugging Face cache |
| Internal audit `docs/CODE_REVIEW.md` | **no**, git-ignored ("private record") | local only |

**Consequence:** every number in the README is backed by files that only exist on the machine that ran them. **Before any demo, copy those run directories somewhere safe** (a tarball and a second disk). Module 12 showed that one wrong command overwrites them, and they're the only copy.

---

## 18.5 The rehearsed demo script (about 8 minutes)

**Pre-flight checklist** (do this the day before, not on the call):
1. `cp -a runs/ runs.backup/` and `cp -a experiments/ experiments.backup/`. Make it a second copy, outside the repo.
2. Activate `../.venv-vla` (or `../.venv` for steps that don't need torch). Run `mtrain --help` once to warm imports.
3. Confirm the engine catalog still has `so100_pickplace@clean/v1` and `training/v6`: `mtrain inspect --dataset so100_pickplace --version training/v6`.
4. Know the three run paths by heart (or put them in shell variables): `BASE=runs/<date>/baseline_full/seed_0`, `EXT=runs/<date>/baseline_full_extended/seed_0`, `ALOHA=runs/<date>/aloha_coffee_baseline/seed_0`.
5. Open `docs/figures/results.svg` and `docs/diagrams/01-system-architecture.svg` in browser tabs.
6. Have a terminal font large enough to share.

**The script.** Each step lists what to run, what to point at, and what it proves.

| # | Do | Point at | What it proves | Time |
|---|---|---|---|---:|
| 1 | Show `01-system-architecture.svg` | the seven runner stages, and the left column of engine services | the design in one picture: procedure here, mechanism in the engine | 45 s |
| 2 | `rde version lineage so100_pickplace training/v6` | `dropped=[9, 31]`, three steps, fingerprints | **a trained model traces back to raw episodes, with every exclusion recorded** | 45 s |
| 3 | `mtrain profile --dataset so100_pickplace --version clean/v1` (scroll to `tasks`, `episode_length`, `coverage.visual_diversity`) | 1 task; 326–475 frames; `"status": "unavailable"` | **the imbalance variable came from the data, not a guess; unmeasurable things are marked unavailable** | 60 s |
| 4 | `mtrain build-intervention --dataset so100_pickplace --version clean/v1 --kind imbalanced --variable episode_length_bucket --variable-value short --target-config /tmp/demo_imbalanced.yaml` (**no `--apply`**) | the printed block of episode IDs and reason | **interventions are resolved from the real catalog, dry-run by default** | 45 s |
| 5 | `cat $BASE/split_integrity.json` | 38/5/5, all overlaps `[]`, `"ok": true` | **episode-level splits, re-checked before training** (say it's a cheap, near-tautological check; Module 5) | 30 s |
| 6 | Show `results.svg` panel (a), then `head $EXT/metrics/validation.jsonl` | the curve still falling at 1,000; resume point; selected step 1,800 | **selection on validation only; continued training kept improving** | 60 s |
| 7 | `mtrain compare --runs $BASE --runs $EXT --runs $ALOHA` | `mse` vs `nearest_neighbor_mse` columns | **the honest headline: 0.0290 → 0.0239 vs NN 0.0239; ALOHA 0.216 vs 0.106** | 60 s |
| 8 | `jq '.summary' $EXT/evaluation/test_overall.json` and `jq '.temporal_error.mean_error_by_progress_bucket' $EXT/evaluation/test_overall.json` | mid-episode peak (≈ 0.144) vs ends (≈ 0.05) | **error decomposed by phase: the grasp and transport phase is hard** | 45 s |
| 9 | `mtrain analyze-failures --run $EXT \| head -40` | 191/1,908 flagged, max episode 13.8%, "scattered" | **failures as evidence, not a verdict**. Then say, before they ask: at p90 with 5 episodes, "scattered" is the only possible output (Module 8) | 60 s |
| 10 | `mtrain rollout --run /tmp/demo_rollout_run` (an empty scratch directory, not a real run) | `"status": "unavailable"` | **the system refuses to invent a success rate** | 20 s |
| 11 | Close with the slide-free summary: "the harness told me a nearest-neighbour lookup was as good as my VLA, and auditing it told me which of its own comparisons weren't valid yet" | — | the project's real value | 30 s |

**Optional live run (no GPU needed, about 1–3 minutes, unmeasured here):** write a config with `training: {model: mean_action}` and a *new* `experiment_name` (for example `demo_live_<time>`), and run `mtrain train --config /tmp/demo_live.yaml`. It exercises intervention, split check, profile, baselines, report and index with no torch. That's exactly the path `test_runner_completes_baseline_only_experiment` covers. Show the new run directory tree, then `reports/experiment_report.md` with its "_not run_" and "unavailable" sections. **Point out that the report's model_training git line says `n/a`, and explain the bug** (Modules 3 and 10). Showing a known bug on purpose is a strong move.

### Never do these during a demo

| Command | Why |
|---|---|
| `mtrain test --run $EXT` or `mtrain evaluate --run …` | **overwrites** `test_overall.json` and `stratified.json` with new stochastic numbers and loses the length slices (Module 12) |
| `mtrain evaluate --split val …` | writes validation metrics into `test_overall.json` |
| `mtrain build-intervention … --apply --target-config configs/experiments/imbalanced.yaml` | erases the template's explanatory comments (Module 12) |
| `mtrain rollout --run $EXT` | harmless but appends a duplicate record to a published run; use a scratch directory |
| `mtrain train` with an existing experiment name the same day | `FileExistsError`; looks like a crash |
| `mtrain train --seed 1` "to show seeds" | it doesn't change the training seed (Module 11) |
| two `mtrain train` at once | run-directory, catalog and index races (Module 14) |

---

## 18.6 Recovery table: what to say when things go wrong

| What goes wrong | What you see | What to say | What to do |
|---|---|---|---|
| Engine catalog missing or moved | `LookupError: unknown version` | "Data is addressed by catalog version, never by path. The catalog for this machine isn't mounted, which is exactly the failure this design makes loud." | switch to the pre-recorded run directories (steps 5–9 need only files) |
| `rde` command not found | shell error | "The engine is a sibling package in a separate environment." | activate `../.venv-vla`; skip step 2 and show README §1's printed lineage instead |
| `import torch` slow or CUDA init hangs | long pause | "This is the heavy environment; the light one doesn't load torch at all, which is why the CLI imports lazily." | Ctrl-C; rerun in `../.venv` for read-only steps |
| `FileExistsError` on the live run | traceback | "Runs are write-once: it refuses to overwrite an existing run directory. A failed run blocks a same-day retry. That's a known gap, and the fix is a status file plus `--force` from the failed state." | change `experiment_name` and rerun |
| `JSONDecodeError` on `compare` or `validate` | traceback | "That's a partially written file. My writes aren't atomic yet, and the fix is temp-then-rename." | restore from `runs.backup/` |
| `compare` columns look misaligned | numbers under the wrong headers | "The terminal table builds headers from the first run and places values by position; the CSV output is keyed correctly." | rerun with `--out /tmp/c.csv` and `column -s, -t /tmp/c.csv` |
| The interviewer asks to run `mtrain test` | — | "I'd rather not on a published run. Re-scoring currently overwrites results, and SmolVLA is stochastic, so the numbers would move a few percent. That's the third fix on my list." | offer to run it against a copy: `cp -a $EXT /tmp/ext_copy && mtrain test --run /tmp/ext_copy` |
| A number on screen differs from the README | e.g. stratified slice MSE ≠ overall | "Slices re-run the stochastic model, so they don't reconcile exactly with overall; re-scoring moves MSE up to 5%." | cite README §2.3 |
| The interviewer asks "so does it work on the robot?" | — | "Unknown. There's no robot or simulator here, and the system records rollout as unavailable instead of estimating it. Offline it ties a nearest-neighbour lookup. Closed loop is where I'd expect vision to matter, and that's the experiment I'd most like to run." | — |
| Run directories were lost | nothing to show | "The artifacts are deliberately not in git because checkpoints are gigabytes; the figure in the README was generated from them by a script." | show `results.svg`, the README tables and the code paths |
| GPU out-of-memory on any live training | CUDA OOM | "Two models resident at once, or another process on the GPU. The trainer frees its model before evaluation for this reason, and the crash marker records what happened." | show `crash.json` in the run directory |

---

## 18.7 Honest gaps to own before an interviewer finds them

1. **No artifact behind the README's numbers is in the repository.** *Consequence:* results are reproducible only on the original machine, or by re-running. Back up before demos.
2. **Data licences aren't recorded anywhere in this repository**, and I didn't check them. *Consequence:* don't redistribute exports or datasets without checking each Hugging Face dataset card.
3. **The `clean/v1` transform contents and the reason for dropping episodes 9 and 31 aren't documented here.** They're engine-side. *Consequence:* "why were those dropped?" has to be answered from the engine's QC records, not from memory.
4. **ALOHA's pre-QC episode count and lineage aren't shown anywhere in this repo.** *Consequence:* SO-100 provenance is demonstrable; ALOHA's is asserted.
5. **The architecture diagram presents `DeployedModel` as an after-the-run step.** Module 9 showed it has never been executed. *Consequence:* if you point at it, say so.
6. **Two CLI commands damage the very artifacts a demo shows** (`evaluate`/`test`, `--apply`). *Consequence:* the demo script must avoid them, and you should say why. That turns a hazard into evidence of understanding.
7. **The live baseline-only run's duration is unmeasured here.** *Consequence:* rehearse it once and time it before relying on it.

---

## The shape to carry into Module 0

```
 HF Hub (lerobot/svla_so100_pickplace, lerobot/aloha_static_coffee)
   └─▶ rde run: download · ingest raw/v1 (50) · transform+QC clean/v1 (48; dropped 9, 31) · catalog
         └─▶ mtrain: intervention → clean/vK? → training/vN (38/5/5) → train → eval → feedback → reports
               └─▶ runs/ + experiments/  (git-ignored; only copy on the experimenter's disk → BACK UP)
 Demo: architecture ▸ lineage ▸ profile ▸ dry-run intervention ▸ split check ▸ val curve ▸ compare
       ▸ phase error ▸ failures (+ honest caveat) ▸ rollout "unavailable" ▸ one-sentence close
 Never: mtrain test/evaluate on published runs · --apply on templates · --seed "replicates" · concurrent runs
```

---

## Drill — answer from memory, files closed

1. Trace the SO-100 data from Hugging Face to `training/v6`, naming every version and the episodes dropped.
2. Which three changes were made to the engine for this project?
3. *Why didn't you just* commit the run directories so anyone can verify the README?
4. *What breaks in production:* mid-demo, the interviewer says "run the test command on that run so I can see it live." What do you do?
5. *Interviewer framing:* "Show me something."

---

# Appendix A — Model answers to the drill

**A1. The SO-100 data path.**
"It starts as the Hugging Face dataset `lerobot/svla_so100_pickplace`. The data engine's `rde run` downloads and ingests it as `raw/v1`, 50 episodes. A transform with quality control produces `clean/v1`, 48 episodes, with episodes 9 and 31 dropped and recorded by ID in the lineage. My experiment configs start from `clean/v1`. For the baseline, the intervention is 'full', so there's no extra filter, and the engine's builder produces `training/v6` with 38/5/5 episode-level splits. `rde version lineage` on that version prints all three steps with run IDs and fingerprints, so the model I selected traces back to exact raw episodes. What I can't tell you from memory is why 9 and 31 failed QC. That's in the engine's QC records."

*Why that answer lands:* it's exact on versions, counts and IDs, shows the verification command, and draws a clean line around what you don't know.

**A2. The three engine changes.**
"Real fine-tuning in `SmolVLAPolicy.fit`, with checkpoint save, load and verification, because the adapter previously had no training step. An `episode_ids` selector on `Evaluator.evaluate`, so slices can be scored by episode set. And the `EpisodeSubsetFilter` transform, which is what makes every data intervention a lineage-tracked engine operation instead of a local copy. They were additive, and the engine still imports nothing from this layer."

*Why that answer lands:* it's precise about scope and purpose, and it reinforces the one-way dependency.

**A3. Why not commit the run directories?**
"Checkpoints are on the order of a gigabyte each, with several per run, so committing them to git is the wrong tool. The figure script regenerates the README figure from them, and the per-frame outcomes and metrics are small. The real gap is that *nothing* behind the README is published. I'd publish the small artifacts (resolved configs, `dataset.json`, `split_integrity.json`, metrics, evaluation JSON, per-frame outcomes, git and environment files) as a release asset or a dataset repository, and publish the selected checkpoint to the Hub with its model card. Then anyone could verify every table without my disk."

*Why that answer lands:* it explains why the default is right for checkpoints, admits the actual gap, and proposes a concrete publication split by artifact size.

**A4. "Run the test command live."**
"I'd say: I'd rather not on that directory, because `mtrain test` currently overwrites the run's test results in place, and SmolVLA samples actions stochastically, so the published numbers would change by a few percent with no backup. That's one of the first fixes on my list. Then I'd offer to copy the run to a scratch directory and run it there, so we can watch it re-score and see the number move, which itself demonstrates the inference noise I mention in the README."

*Why that answer lands:* it turns a trap into a demonstration of deep knowledge of your own tool, and still satisfies the request safely.

**A5. "Show me something."**
"Three minutes, three things. First, lineage: this command shows the model's training data back to raw episodes, including the two QC drops. Second, the honest result: this comparison table shows fine-tuning cut error to 2.6% of the mean-action baseline but exactly tied a nearest-neighbour lookup on robot state. Third, the refusal to fabricate: rollout is recorded as unavailable, and the profile marks visual diversity unavailable instead of inventing a score. Then I'd show you the one bug I always point out myself: the report prints 'n/a' for this repo's git commit because of a key mismatch."

*Why that answer lands:* it's short, structured, and ends with a self-identified bug, which builds trust faster than any feature.

---

# Module 18 Summary — the one page to revise

**What this module is.** The data's provenance (Hugging Face → engine `raw/v1` → `clean/v1` → `training/vN`, with recorded drops), the exact boundary with the engine (three additive changes), what's on disk vs in git (no result artifacts in git), a rehearsed 8-minute demo built from read-only commands and pre-recorded runs, and a recovery table.

**Five claims to make cold**
1. SO-100: `lerobot/svla_so100_pickplace` → `raw/v1` (50) → `clean/v1` (48, dropped 9 and 31) → `training/v6` (38/5/5).
2. Three engine changes: `SmolVLAPolicy.fit` with checkpoints; an `episode_ids` selector on `Evaluator.evaluate`; `EpisodeSubsetFilter`.
3. Every README number is backed by git-ignored files on one machine. Back them up before demos.
4. The demo is read-only: lineage, profile, dry-run intervention, split check, validation curve, compare, phase error, failures, rollout.
5. Never run `mtrain test`/`evaluate` or `--apply` on published artifacts; offer a scratch copy.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Git-ignore runs | commit artifacts | checkpoints are GBs | README not verifiable from the repo |
| Pre-recorded demo | live training | runs take 10–22 min | less "live" |
| Read-only demo commands | full CLI tour | two commands damage artifacts | must explain why |
| Scratch copy for risky commands | refuse outright | satisfies the request safely | one extra step |

**Volunteer unprompted:** the backup step; why `mtrain test` is excluded; data licences unchecked; ALOHA lineage not shown in this repo; the diagram's `DeployedModel` box.

**Numbers and names:** 50 → 48 (dropped [9, 31]) → 38/5/5; ALOHA 40 → 32/4/4; `training/v6`; failures 191/1,908 (max episode 13.8%), ALOHA 440/4,400 (22.6%, 12.1%); demo about 8 minutes, 11 steps; fixture 8 × 20 frames at 10 fps, 2 tasks.

**Sentences worth memorizing**
- "This command traces the model back to raw episodes, including the two QC drops."
- "I won't run `test` on a published run: it overwrites in place, and the model is stochastic. Let me show you on a copy."
- "Every number in the README is backed by files that only live on the machine that ran them; publishing the small ones is on my list."
