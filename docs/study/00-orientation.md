# Module 0 — Orientation

**Files covered:** the whole repository at a glance. This module was written last, after Modules 1–18, so every statement here is a summary of something verified in a later module, with a pointer to it.

**Where we are:** This is the front door. It assumes nothing. By the end you should know what the system does, the one sentence that frames it, the twenty words you need before Module 1, the stack, the numbers to know cold, and a two-minute pitch you can give today.

**The one-line thesis:** `vla-finetune` is a thin, provenance-obsessed experiment layer that fine-tunes a robot vision-language-action model on versioned data and reports results honestly. Its most important result is a tie with a nearest-neighbour lookup, and its most important flaw is that its own interventions aren't yet scored on a shared test set.

---

## 0.1 Prerequisites — the absolute basics

The minimum vocabulary to start Module 1. Each term is defined fully in the module shown.

| Term | One-line meaning | Full treatment |
|---|---|---|
| **Episode** | one recorded demonstration of a robot task, a sequence of frames | M1 |
| **Frame** | one timestep: camera images, robot state (joint angles), commanded action | M1 |
| **Action** | the vector sent to the robot: 6 numbers (SO-100), 14 (ALOHA) | M1 |
| **Policy** | a function from observation to action; the trained model | M1 |
| **VLA** | vision-language-action model: images plus an instruction plus state in, actions out | M1 |
| **SmolVLA** | the small VLA used here (`lerobot/smolvla_base`); predicts action *chunks* with *flow matching*, so its outputs are random from call to call | M1 |
| **Fine-tuning** | continuing to train a pretrained model on new data | M1 |
| **Behaviour cloning** | training a policy to copy demonstrated actions (supervised regression) | M1 |
| **Offline evaluation** | comparing predicted vs recorded actions on held-out data, with the robot never moving. **The only evaluation here** | M1, M7 |
| **Closed-loop rollout** | letting the policy drive a robot or simulator and measuring task success. **Not available; recorded as "unavailable"** | M1, M12 |
| **Train / val / test split** | disjoint sets of *episodes*: train to learn, val to pick a checkpoint, test to report once | M1, M5 |
| **Baseline** | a trivial model to beat: *mean-action* (predict the average) and *nearest-neighbour* (copy the action of the closest training robot state) | M7 |
| **MSE (normalized)** | mean squared action error after per-dimension standardization; mean-action ≈ 1.0 | M1, M7 |
| **Dataset version** | an immutable, named snapshot in the data engine: `raw/v1`, `clean/v1`, `training/v6` | M1, M4 |
| **Lineage** | the recorded chain of operations that produced a version | M4, M18 |
| **Data intervention** | a deliberate change to the training data (reduce, imbalance, rebalance) to test what data contributes | M2, M4 |
| **The engine (`rde`)** | the sibling repository *Episodic Data Engine*: owns data, versions, the evaluator, failure detection and the SmolVLA training step | M1, M18 |
| **Run directory** | `runs/<date>/<name>/seed_<n>/`, holding everything one experiment produced | M3 |
| **Checkpoint** | saved model weights plus pre- and post-processors at a training step | M6, M9 |
| **"Unavailable" marker** | an explicit record that something wasn't measured, and why; never a fabricated value | M5, M10 |

---

## 0.2 What the system does, in one paragraph

Given a YAML experiment file, `mtrain train` asks the data engine to **build a new, lineage-tracked training dataset version** (optionally with an intervention such as "half the episodes" or "upweight short episodes"). It **re-checks the train/val/test split**, **profiles** the data, and **fine-tunes SmolVLA**, validating only on the val split and keeping the best verified checkpoint. It then **scores that checkpoint and two baselines on the test split through the same evaluator**, decomposes the error by tail, episode, phase of the episode and slice, runs the engine's **failure detector** to produce a recommended next data intervention for a human, and writes a **report, model card, per-frame outcomes and index rows**. Everything it can't measure is written down as unavailable (M11).

## 0.3 The framing sentence

> **"The model layer never owns the data."** Every dataset this package trains on is an immutable, catalog-resolved version produced by the engine, and even a data intervention is executed as a lineage-tracked transform, never a local copy (README; verified in M4).

And the honest counterweight to carry with it:

> **"The data layer owns the model."** The SmolVLA adapter and its gradient step (`fit`) live in the engine; this repository owns the *procedure* around them (M1 §1.3).

---

## 0.4 How the manual is organized (dependency order)

| # | Module | You'll be able to explain |
|---|---|---|
| 1 | Foundations | the domain, the engine boundary, packaging and two venvs |
| 2 | Configuration | the YAML merge, validators, three seeds, why typos vanish |
| 3 | Run directory and index | write-once runs, provenance capture, JSONL fragility |
| 4 | Data interventions | versions as interventions; **why each intervention gets a different test set** |
| 5 | Split integrity and profile | what the leak check can and can't catch; the coverage metric's flaw |
| 6 | Training loop | the callback around `fit`; **32-frame validation and the 13× val–test gap** |
| 7 | Offline evaluation | baselines through one evaluator; **the nearest-neighbour tie**; slices re-infer |
| 8 | Feedback artifact | the flywheel's plumbing; **why it always says "act" and runs on test** |
| 9 | Deployment | the artifact; a health check that doesn't cross-check; a replay harness |
| 10 | Reports and figures | "not run" discipline; dropped numbers; hard-coded figure markers |
| 11 | Orchestration | the runner; **`--seed` doesn't seed**; late validation; sweep flaws |
| 12 | CLI | lazy imports; **`mtrain test` overwrites published numbers** |
| 13 | Tests and the capstone | real-engine tests, missing invariants; **constraint → decision map** |
| 14 | Low-Level Design | layering, patterns, contracts, state machine, errors, idempotency |
| 15 | High-Level Design and scaling | latency budget; **under one epoch**; bottleneck order; the ladder |
| 16 | Fixes and roadmap | P0/P1/P2, the one-day shortlist, the anti-roadmap |
| 17 | Interview kit | pitches, 47 questions, stories, phrase swaps, numbers card, glossary |
| 18 | Data provenance and demo | HF → engine → run; the 8-minute demo; recovery table |

**Reading paths.**
- **Short on time:** read this module, then M17, M16 §16.5, and the summary page of M4, M6, M7, M8 and M11.
- **Before a systems interview:** M14, M15, M13 §13.6.
- **Before an ML or research interview:** M4, M6, M7, M8, M15 §15.4.

---

## 0.5 The stack

| Layer | Technology | Version / note | Where |
|---|---|---|---|
| Language | Python | ≥ 3.10 declared; 3.11 used | `pyproject.toml`, README |
| Packaging | hatchling, uv (sibling path source) | no lockfile | M1 |
| Config | YAML (PyYAML `safe_load`) + Pydantic v2 | `extra` not forbidden | M2 |
| CLI | Typer + rich | 11 commands, lazy imports | M12 |
| Data engine | `episodic-data-engine` (`rde`), editable sibling | unpinned | M1, M18 |
| Engine storage | catalog + Parquet + video; DuckDB queries | engine-side, partly inferred | M5, M18 |
| Model | SmolVLA via `lerobot` | lerobot 0.4.4; base `lerobot/smolvla_base` | M1, M18 |
| DL framework | PyTorch with CUDA | 2.10 (CUDA build); driver 592.82, CUDA 13.1 | M15 |
| Numerics / plots | NumPy, matplotlib (Agg) | — | M7, M10 |
| Tracking | run directories + 4 JSONL files | git-ignored | M3 |
| Tests | pytest, real engine fixtures | 56 tests, 52 torch-free | M13 |
| Hardware | RTX 4060 Laptop, 8 GB VRAM; ~10 GB VM RAM | single GPU, single process | M15 |
| Data | `lerobot/svla_so100_pickplace`, `lerobot/aloha_static_coffee` | licences not checked | M18 |

---

## 0.6 The numbers to know cold

**Data:** SO-100: 50 raw → 48 after QC (dropped 9, 31) → 38/5/5, 1,908 test frames, 6-dimensional action, 2 cameras at 30 fps, one task, 326–475 frames per episode. ALOHA: 40 → 32/4/4, 4,400 test frames, 14-dimensional action, 4 cameras at 50 fps, test episodes 1,100 frames each.

**Results (test MSE, normalized):**

| | SmolVLA | Nearest-neighbour | Mean-action | Ratio to mean |
|---|---:|---:|---:|---:|
| SO-100, 1,000 steps | 0.0290 | 0.0239 | 0.908 | 3.2% |
| SO-100, 2,000 steps | **0.0239** | **0.0239** | 0.908 | 2.6% |
| ALOHA, 1,000 steps | **0.2157** | **0.1062** | 1.010 | 21.4% |

**Training:** batch 4, lr 1e-4 (3e-5 when resumed), warmup 50 (20 when resumed), checkpoint and validation every 100 steps, validation on **32 frames**. 324 / 385 / 576 s. Peak GPU 2.2–3.0 GB of 8.

**The derived numbers that change the story:**
- val minimum **0.0018** vs test **0.0239**, about 13× (M6)
- **≈ 0.26 epoch** (SO-100) and **≈ 0.11 epoch** (ALOHA) per 1,000 steps (M15)
- re-scoring noise **up to 5%** vs the **18%** continued-training gain (M7)
- evaluation **≈ 45–57%** of run time, about half of it redundant (M15)
- failure rule flags **191 / 1,908** (10%), max episode **13.8%** vs the 0.5 cut (M8)

**Code:** about 2,700 lines in `src/`; 4 layers; 11 CLI commands; 56 tests; about 200 logged gaps → 7 P0 / 26 P1 / 28 doc corrections / about 60 P2 (M16).

---

## 0.7 The two-minute pitch

> "This is the experiment and evaluation layer for fine-tuning SmolVLA, a small vision-language-action robot policy, on real LeRobot datasets: SO-100 pick-and-place and bimanual ALOHA coffee.
>
> The design rests on one rule: the model layer never owns the data. Every data change, like halving the training set or dropping a regime, is executed as a lineage-tracked version in a separate data engine, so any checkpoint traces back to the exact raw episodes it learned from, including which ones QC removed.
>
> One command runs an experiment end to end. It builds the dataset version, re-checks the episode-level splits, fine-tunes with checkpoint selection on validation only, and scores the result against a mean-action and a nearest-neighbour baseline through the same evaluator. It breaks error down by tail, episode, phase and slice, produces a feedback artifact recommending the next data change, and writes a report where anything it can't measure, like closed-loop success, is recorded as unavailable rather than invented.
>
> The results are honest rather than flattering. On SO-100, fine-tuning reached 2.6% of the mean-action error, and continued training gave an 18% gain on an identical test split, but it exactly tied a lookup over robot state. On ALOHA, retrieval was twice as good. Since that run saw about a ninth of an epoch, I read that as undertrained before I read it as retrieval being better.
>
> Auditing the harness itself, the main thing I'd fix is that interventions re-split the data, so each one is scored on a different test set. Freezing one test split, propagating real seeds and adding bootstrap intervals come before any new infrastructure."

*(About 2 minutes at a normal pace. Adjust "I" and "we" to your actual role. See the M17 authorship note.)*

---

## 0.8 Honest gaps to own before an interviewer finds them

The five that matter most, each owned by a later module:
1. **Interventions are scored on different test sets** (filter-then-split). → M4, P0-1
2. **`--seed` doesn't seed; replicates are fake.** → M11, P0-2
3. **Checkpoint selection uses 32 validation frames** (val about 13× below test). → M6, P0-7
4. **The feedback loop runs on test, and can't say "no action".** → M8, P0-5
5. **`mtrain test` overwrites published results.** → M12, P0-6

And the scope facts: offline only, single seed, 5 and 4 test episodes, no CI, the engine unpinned and not in this repo, run artifacts not in git (M1, M18).

---

## The shape to carry into Module 1

```
   YAML ─▶ mtrain train ─▶ [ engine: version + lineage ] ─▶ train (val-only) ─▶ test vs 2 baselines
                                                                   │
           report · model card · per-frame outcomes · feedback ◀──┘   (rollout: "unavailable")

   principle: the model layer never owns the data      counterweight: the data layer owns the model
   result:    SmolVLA = state-NN on SO-100; NN 2× better on ALOHA (≈0.11 epoch → undertrained?)
   first fix: one frozen test split for every intervention
```

---

## Drill — answer from memory, files closed

1. In one sentence, what does this system do, and what's its organizing principle?
2. What's the difference between offline evaluation and closed-loop rollout, and which does this project do?
3. *Why didn't you just* report the 2.6%-of-mean-action number as the headline?
4. *What breaks in production* if someone treats the README's intervention comparisons as findings today?
5. *Interviewer framing:* "Give me the two-minute version."

---

# Appendix A — Model answers to the drill

**A1.** "It's an experiment, evaluation and feedback layer that fine-tunes SmolVLA on versioned robot datasets and reports results against baselines. Its principle is that the model layer never owns the data: every dataset, including every intervention, is an immutable, lineage-tracked version in a separate data engine."

*Why that answer lands:* it gives the what and the principle in under fifteen seconds, with the one word an interviewer will remember: lineage.

**A2.** "Offline evaluation compares predicted actions with recorded demonstrations on held-out episodes. The robot never moves, so errors can't compound. Closed-loop rollout lets the policy drive a robot or simulator and measures task success. This project only does offline evaluation, and it records rollout as unavailable rather than estimating it, because low offline error doesn't guarantee success once small mistakes push the robot off the demonstrated states."

*Why that answer lands:* it's precise about the mechanism (compounding error) and shows the project's honesty is a design choice.

**A3.** "Because the mean-action baseline is the weakest possible comparison. Any model that learned anything beats it. The informative comparison is the nearest-neighbour lookup over robot state, which SmolVLA ties on SO-100 and loses to on ALOHA. Leading with 2.6% would be technically true and misleading, and the whole point of scoring both baselines through the same evaluator is to report the one that matters."

*Why that answer lands:* it shows you understand baseline selection, and that you'd choose the less flattering framing deliberately.

**A4.** "They'd be comparing numbers measured on different test sets. Each intervention filters episodes and the engine then re-splits the remainder, so a 'reduced' run is scored on two or three test episodes that aren't even a subset of the baseline's five. Any difference could be which episodes landed in test. Combined with a single seed and a stochastic model, the comparison has no statistical footing. The fix is to freeze one test split and apply interventions to the train split only. Until then, they're pipeline validation, not findings."

*Why that answer lands:* it names the mechanism, quantifies it and gives the fix, which is the core of the project's methodological story in thirty seconds.

**A5.** Deliver §0.7.

*Why that answer lands:* it's bounded, it has a principle, a result, an honest interpretation and a next step, and it invites follow-up in exactly the areas you've prepared.

---

# Module 0 Summary — the one page to revise

**What this module is.** The entry point: what the system does, its framing sentence and counterweight, twenty terms, the reading map, the stack, the numbers, and the two-minute pitch.

**Five claims to make cold**
1. The model layer never owns the data; every intervention is a lineage-tracked engine version.
2. The data layer owns the model; `fit` is in the engine, and this repo owns the procedure around it.
3. Offline only: SmolVLA ties state-NN on SO-100 (0.0239) and loses on ALOHA (0.216 vs 0.106).
4. ALOHA saw about 0.11 epoch, so it's undertrained before it's out-retrieved; continued training gained 18% on an identical split.
5. First fix: one frozen test split for all interventions; then real seeds, frozen val subset, val-driven feedback, non-destructive re-scoring.

**Decisions (the headline four)**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Interventions as engine versions | local copies / ID lists | traceable to raw episodes | each intervention gets its own test split |
| Two baselines through one evaluator | model-only metrics | honest comparison | the result is a tie |
| "Unavailable" over estimates | proxy success numbers | never fabricate | incomplete-looking reports |
| Laptop, files, one process | infrastructure | one or two researchers, short runs | concurrency and durability gaps |

**Numbers and names:** §0.6, the results table and the five derived numbers.

**Sentences worth memorizing**
- "The model layer never owns the data, and the data layer owns the model."
- "Offline, a state lookup tied my VLA; the harness was built to tell me that."
- "My first fix is one frozen test split, because right now each intervention is graded on a different exam."
