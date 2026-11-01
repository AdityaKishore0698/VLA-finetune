# Module 8 — The feedback artifact: closing the loop

**Files covered:** `src/model_training/feedback/artifact.py` (`feedback/__init__.py` was covered in Module 1)

**Where we are:** Module 7 produced `test_overall`, a per-frame record of how the selected checkpoint did on the test split. This module turns that into the project's "flywheel" output: a structured artifact that ranks failing episodes, offers candidate explanations, and recommends the next data intervention. README Figure 1's loop arrow ("the feedback artifact's recommended action informs the next data intervention") begins here.

**The one-line thesis:** The artifact is a thin, honest-sounding wrapper over the engine's `FeedbackLoop`. Its percentile-relative failure threshold guarantees roughly 10% of frames always "fail", so with this repo's test sizes it **can't recommend "no action"**, and it **essentially can't produce its "data defect" explanation**. Worse, it runs on the **test split**, so following its advice and re-scoring on the same test set turns test into a tuning set.

---

## 8.1 Prerequisites — the absolute basics

- **Data flywheel.** The loop deploy (or evaluate) → find failures → fix the data → retrain → re-evaluate. Each turn is supposed to improve the model through better data.
- **Failure detection by threshold.** A frame "fails" if its error exceeds a threshold. **Absolute** threshold: error > 0.1. **Relative (percentile)** threshold: error above the 90th percentile of *this run's* errors. A relative threshold always flags about the same *fraction* of frames, by definition, however good or bad the model is.
- **Episode nomination.** Aggregating frame-level failures into a per-episode review: failure rate (flagged frames ÷ evaluated frames in that episode), mean error, and a priority score.
- **Data defect vs model limitation.** Two different causes of failure. A *defect* is a corrupted or mislabelled demonstration, so fix the data. A *limitation* is a regime the model hasn't learned, so get more coverage or a better model. The project's rule, stated in the code, is "a model failure does not automatically imply bad data."
- **Adaptive overfitting to the test set (the "garden of forking paths").** Every decision you make *by looking at* test results, such as which intervention to run next, makes subsequent test results optimistically biased. The test set has become part of your tuning loop. The textbook rule: decisions use validation, and test is touched once, at the end.
- **Quarantine.** Marking an episode for human review without deleting it.

---

## 8.2 The code

```python
def build_feedback_artifact(experiment_id, model_id, dataset_id, dataset_version, result, *,
                            settings=None, catalog=None,
                            error_percentile: float = 90.0, min_failures_per_episode: int = 3) -> dict:
    """Detect failures via the real flywheel, then package the structured artifact."""
    loop = FeedbackLoop(settings=settings, catalog=catalog)
    report = loop.detect(result, error_percentile=error_percentile,
                         min_failures_per_episode=min_failures_per_episode, persist=True)
    selection = loop.build_retraining_selection(report)
```

**Decision: wrap the engine's `FeedbackLoop` instead of reimplementing detection.**
- *Why:* the module docstring says the engine's detection and nomination logic is reused, and this module adds "what that module deliberately leaves to a human: candidate explanations ... and a concrete recommended next action". It's a single implementation of detection.
- *Cost:* every important number (the threshold, `failure_rate`, priority) is computed in code I can't read here. The behaviour below is derived from `docs/FAILURE_ANALYSIS.md` §2 and the explanation strings, so it's partly **unverified**.

**`persist=True`: evaluation writes to the shared catalog.** Unlike every `Evaluator` call in Modules 6 and 7 (`persist=False`), this one *persists* the failure report into the engine. So every SmolVLA run mutates the engine's state as a side effect of evaluation. That's intended (it's how a curator later finds flagged episodes with `rde` tools), but it means:
- re-running an experiment writes a second set of failure records for the same episodes;
- a crashed run after this point leaves engine records with no completed run directory pointing at them (Module 3's non-transactional index, again);
- `mtrain evaluate` re-scores a run *without* re-running feedback (Module 12), so after a re-evaluation, `feedback_artifact.json` and the engine's records describe the *old* predictions.

**`error_percentile=90.0` and `min_failures_per_episode=3` are hard-coded defaults.** No config field reaches them. The runner calls this with defaults only (Module 11).

### Why a relative threshold, and what it costs

`FAILURE_ANALYSIS.md` §2 defends the choice: "against percentiles computed from the current evaluation run, not a fixed absolute threshold; a fixed threshold would flag either everything or nothing the moment the model or the normalization scheme changes." That's a real argument: an absolute 0.1 would flag about half the frames for SmolVLA on SO-100 (README: 49.8–56.9% within tolerance) and nearly all of them on ALOHA (0.05%).

**The cost is arithmetic.** At the 90th percentile, *by definition* about 10% of evaluated frames are flagged, in every run, for every model.
- SO-100 test: 1,908 frames, so about 190 flagged frames spread over 5 episodes of about 380 frames each.
- An episode is reviewed if it has at least 3 flagged frames. With about 190 flagged frames across 5 episodes, **every episode will almost certainly have 3 or more**. So `report.reviews` is non-empty in essentially every real run.
- Therefore the `else` branch ("no episode crossed the failure-concentration threshold ... no evidence") and the recommendation `"no action indicated by this evaluation."` are **effectively unreachable** on this project's datasets. **The artifact will recommend an intervention for a perfect model too**, because a perfect model's top 10% of errors are still "failures".

### The two explanations, and why one almost never fires

```python
if report.reviews:
    concentrated = [r for r in report.reviews if r.failure_rate > 0.5]
    if concentrated:
        candidate_explanations.append(
            f"{len(concentrated)} episode(s) fail on more than half their evaluated frames -- "
            "concentrated, episode-local failure is evidence worth checking for a data/collection "
            "defect ... (a model failure does not automatically imply bad data).")
    scattered = [r for r in report.reviews if r.failure_rate <= 0.5]
    if scattered:
        candidate_explanations.append(
            f"{len(scattered)} episode(s) fail on a minority of their evaluated frames -- more "
            "consistent with a model/coverage limitation ... than an episode-specific data defect.")
```

**The epistemic framing is good**: concentrated failure suggests a defect, scattered failure suggests a limitation, and neither is asserted. `FAILURE_ANALYSIS.md` §5 says "Neither pattern is presented as a conclusion". That's the right posture, and it's worth quoting.

**But combine `failure_rate > 0.5` with a 90th-percentile threshold.** If `failure_rate` is flagged ÷ evaluated frames *for that episode* (which the explanation text, "fail on more than half their evaluated frames", implies), then the *maximum* failure rate any episode can reach is about

`min(1, 0.10 × total_frames / episode_frames)` ≈ `0.10 × 1,908 / 380` ≈ **0.50** on SO-100,

and that only happens if *every* flagged frame in the whole test set lands in that one episode. On ALOHA (4 episodes of 1,100 frames), it's `0.10 × 4,400 / 1,100` = **0.40**. **So "concentrated" (> 0.5) is mathematically unreachable on ALOHA and needs a perfect worst case on SO-100.** The "data defect" explanation can, in practice, never be produced here. The artifact will say "scattered → model/coverage limitation" every time. That isn't evidence. It's a consequence of choosing p90 and 0.5 together. (It's reachable only when an episode is much shorter than average, or when there are many episodes and one is truly anomalous.)

**Confirmed by the README's own numbers (added during the Data Provenance module).** README §2.4 reports that on the 2,000-step SO-100 run the p90 rule flags **191 of 1,908** test frames (that's 10.0%, as the arithmetic predicts), "every test episode contains some of them", and **no episode exceeds 13.8%** (range 6.5–13.8%), so the artifact says "scattered". On ALOHA, 440 of 4,400 are flagged, and the worst episodes reach 22.6% and 12.1%. Both datasets sit far below the 0.5 cut, exactly as the bound says they must. The README presents "scattered → model or coverage limitation" as a *finding*. It's the only output these thresholds can produce at these sizes.

**The two thresholds should be expressed relative to each other.** For example, call an episode concentrated if its failure rate exceeds k× the expected base rate of `(100 − p)/100`, or test it with a binomial test against that base rate. That's the one-line fix that makes both branches reachable.

### The recommendation is a template

```python
recommended = (
    "no action indicated by this evaluation."
    if not report.reviews else
    "inspect the top-ranked episode(s) below for corruption/ambiguity before assuming a data fix "
    "will help; if the pattern holds and traces to a real shared attribute (task, length bucket), "
    "build a targeted correction with "
    "`mtrain build-intervention --kind rebalanced --variable <attribute> --variable-value <value>` "
    "and retrain, then compare against this run on the same fixed test slice."
)
```

- **It's one fixed string with placeholders.** Nothing analyses the reviews to see whether failures share a task or length bucket, or names the attribute and value. `FAILURE_ANALYSIS.md` §6 says the field "names a concrete next step whenever the evidence supports one". It names the *same* step, with `<attribute>` and `<value>` left blank, whenever any episode is reviewed, which (above) is always. The recommendation is good *advice*; it isn't *output*.
- **"Compare against this run on the same fixed test slice" is impossible** for any excluding intervention (Module 4: the test set moves). For `rebalanced`, the recommended kind, the test set plausibly *is* identical, because rebalancing excludes nothing. So the specific advice is internally consistent. That's worth noticing: rebalancing is the only intervention the loop can evaluate fairly.
- **"Inspect ... before assuming a data fix will help"** puts a human gate on the loop, matching README Figure 1's dashed "decision taken by a person" arrow. Good.

### The returned artifact

```python
return {
    "experiment_id": ..., "model_id": ..., "dataset_id": ..., "dataset_version": ...,
    "affected_slice": {"split": result.split, "frames_evaluated": result.frames_evaluated},
    "failure_summary": report.summary(),
    "ranked_episode_reviews": [r.to_dict() for r in report.reviews],
    "candidate_explanations": candidate_explanations,
    "recommended_next_action": recommended,
    "retraining_selection": selection,
    "rde_artifacts": report.artifacts,
}
```

- **`affected_slice` is always the whole evaluated split** (`"test"`), never a slice. The name promises a localization the code doesn't do.
- **`retraining_selection`** comes from the engine's `build_retraining_selection(report)`. By `FAILURE_ANALYSIS.md` §5, flagged episodes are "upweighted -- never dropped -- via the `episode_weights` mechanism", so this is presumably a weights proposal. Nothing in this repo *consumes* it. `mtrain build-intervention` computes weights from `--variable` and `--variable-value`, not from this field (Module 12). The engine's own nominated selection is computed, stored and ignored.
- **The integration test** (`test_prediction_to_evaluator_to_metrics_to_feedback_artifact`) checks shape only: every ranked episode ID is real, and `recommended_next_action` exists. It doesn't check that either explanation branch can fire, which is exactly the gap above.

---

## 8.3 The methodological problem: the loop runs on test

The runner calls `build_feedback_artifact(..., test_overall, ...)` (Module 11), on the **test** split's outcomes. The loop the docs describe (`FAILURE_ANALYSIS.md` §6) is:

```
train → evaluate on TEST → feedback picks a weak regime FROM TEST → rebalance → retrain → compare on TEST
```

Each turn of that loop **chooses the next dataset by looking at test errors**, then **judges the choice on the same test errors**. With 5 test episodes, an intervention tuned to the specific episodes that happened to fail in test will tend to look like an improvement on those same episodes, whether or not it generalizes. This is adaptive overfitting, and it's the single most important methodological correction in the project. Training-time selection is strict about val (Module 6), but data-curation selection isn't.

**The fix keeps everything else:** run `build_feedback_artifact` on a **validation** evaluation (a full val pass of the selected checkpoint, scored once). Drive interventions from val failures, and keep test untouched until a final comparison. If the val split is too small to localize failures (5 episodes), that's itself the finding: the dataset is too small to run a data flywheel on and also keep an honest test set.

---

## 8.4 Docs vs code in `FAILURE_ANALYSIS.md`

| Claim | Reality in the code |
|---|---|
| §4: "predictions are deterministic given the checkpoint and the frame, they are reproducible on demand" | **False for SmolVLA.** README §2.3 says flow-matching sampling makes re-scoring differ by up to 5%. The doc's justification for *not* storing predicted and ground-truth action vectors rests on a false premise. Re-running `mtrain evaluate` gives *different* predictions. |
| §3: per-action-dimension error is "surfaced per episode via `episode_metrics`" | `episode_metrics` returns `mae/mse/rmse/max_error` per episode, with **no per-dimension output** (Module 7). Per-dimension error is in each frame's `per_dim_error` in `test_outcomes.jsonl`, but nothing aggregates it per episode. |
| §3: highest-error episodes come from "`episode_metrics`, sorted by mean squared error" | `episode_metrics` sorts by **episode ID** (`sorted(by_ep.items())`). |
| §3 Table 1 lists stratification dimensions "this project adds" | None of them appear in the feedback artifact. They're separate fields of `test_overall.json`. |
| §6: the recommendation "names a concrete next step whenever the evidence supports one" | A fixed template with blank placeholders, emitted whenever any episode is reviewed (effectively always). |
| §2: SmolVLA's constant confidence gives `confidence_used: false` | Engine-side; unverifiable here. It's consistent with `model_card.py`'s "confidence is an uncalibrated constant". |

---

## 8.5 Honest gaps to own before an interviewer finds them

1. **The loop runs on test.** Failures that drive the next intervention come from the test split, and the intervention is then judged on the same test split. *Consequence:* adaptive overfitting. Data curation tunes on test.
2. **"No action" is effectively unreachable.** A 90th-percentile threshold always flags about 10% of frames, which spread across 4–5 episodes always exceed 3 per episode. *Consequence:* the artifact recommends an intervention for any model, including a perfect one.
3. **The "concentrated / data defect" branch is mathematically near-unreachable** (maximum per-episode failure rate ≈ 0.50 on SO-100 and 0.40 on ALOHA, versus a > 0.5 rule). *Consequence:* the artifact always says "scattered → model limitation", by construction, not by evidence.
4. **The recommendation is a static template** with unfilled `<attribute>`/`<value>` placeholders. *Consequence:* "names a concrete next step" overclaims; no analysis links reviews to attributes.
5. **`FAILURE_ANALYSIS.md` claims predictions are deterministic.** They're stochastic. *Consequence:* the stated reason for not persisting predicted and target vectors is false, and "re-run to reproduce" doesn't reproduce.
6. **Several `FAILURE_ANALYSIS.md` §3 claims don't match `episode_metrics`** (no per-dimension output, sorted by ID). *Consequence:* doc overclaims.
7. **`persist=True` writes failure records into the engine on every run**, with no dedupe and no refresh on `mtrain evaluate`. *Consequence:* engine records go stale or duplicate relative to the run directory.
8. **`retraining_selection` is computed and never consumed.** *Consequence:* the engine's nominated weights are dead output, and `build-intervention` ignores them.
9. **`affected_slice` is always the whole split.** *Consequence:* the field name overclaims localization.
10. **Thresholds (`error_percentile`, `min_failures_per_episode`) aren't configurable.** *Consequence:* you can't tune them per dataset without a code change.
11. **The integration test checks shape, not that either branch is reachable.** *Consequence:* gaps 2–3 are invisible to the suite.

---

## The shape to carry into Module 9

```
 test_overall (EvaluationResult on TEST)  ← should be VAL for curation decisions
        │
        ▼
 rde FeedbackLoop.detect(p90, min 3/episode, persist=True)  ──▶ engine failure records (side effect)
        │   ~10% of frames flagged by definition
        ▼
 report.reviews (failure_rate, mean error, priority)    loop.build_retraining_selection → (unused)
        │
        ├─ failure_rate > 0.5 → "data defect?"     ← max ≈ 0.50 (SO-100) / 0.40 (ALOHA): ~never
        ├─ failure_rate ≤ 0.5 → "model/coverage limitation"   ← always
        └─ recommended = fixed template "build-intervention --kind rebalanced --variable <…>"
        ▼
 failures/feedback_artifact.json ──▶ model card "Known failure modes", report, human decision
                                        │ (dashed arrow: a person picks the next intervention)
                                        ▼
                               Module 4: next training/vN
```

---

## Drill — answer from memory, files closed

1. Why does the feedback loop use a percentile threshold instead of an absolute one, and what does that choice make impossible?
2. Show, with numbers, why the "concentrated failure" explanation essentially never appears on SO-100 or ALOHA.
3. *Why didn't you just* run failure analysis on the test set? That's where the final model is evaluated.
4. *What breaks in production:* you run the loop five times (feedback, rebalance, retrain, compare) and report the fifth model's test MSE. What's wrong with that number?
5. *Interviewer framing:* "Walk me through your data flywheel. Does it work?"

---

# Appendix A — Model answers to the drill

**A1. Percentile vs absolute.**
"An absolute threshold depends on the model and the normalization. At 0.1 mean absolute error, SmolVLA would flag about half the SO-100 frames and nearly all of ALOHA's, so the flag would carry no information about *which* frames are unusually bad. A 90th-percentile threshold adapts to the run and always isolates the worst tenth. The cost is that it can never say 'nothing is wrong'. Ten percent of frames fail by definition, for a terrible model and a perfect one alike. With 1,900 test frames over five episodes, every episode gets well over the three-failure minimum, so the 'no action' branch is effectively dead and the artifact always recommends an intervention. I'd keep the relative threshold for *ranking*, and add an absolute or baseline-relative gate for *whether to act*, for example only recommending action if the flagged frames' error is worse than the nearest-neighbour baseline's on the same frames."

*Why that answer lands:* it defends the original choice with real numbers, then names the exact capability the choice destroys, and proposes a two-threshold fix that keeps the benefit.

**A2. Concentrated failure is unreachable.**
"At p90, about 10% of all test frames are flagged: about 190 of SO-100's 1,908. An episode is about 380 frames. Even if every one of the 190 flagged frames landed in a single episode, its failure rate would be 190 / 380, which is 0.5, and the rule needs strictly more than 0.5. On ALOHA it's 440 flagged frames over 1,100-frame episodes, so at most 0.4. So the explanation that points at a data defect can't fire, and every run reports 'scattered failure, consistent with a model limitation'. That sounds like evidence, but it's a consequence of picking p90 and 0.5 independently. The fix is to define 'concentrated' relative to the 10% base rate, for example with a binomial test per episode."

*Why that answer lands:* it's a clean back-of-envelope proof that a code path is dead. Few things are more convincing in an interview than showing a branch can't execute.

**A3. Why not run it on test?**
"Because the feedback artifact *decides* what to do next, and anything that makes decisions has to use data you're not going to report on. If test failures pick which regime to upweight and test then judges whether the upweighting helped, the test set has become a tuning set. With five test episodes, an intervention fit to the particular episodes that failed will look like an improvement on exactly those episodes. My training loop is strict about this, since checkpoint selection only ever sees val, but the data-curation loop isn't, and that's the most important methodological fix in the project. Run the feedback on a full val evaluation and keep test for the final comparison. If val is too small to localize failures, that tells me the dataset is too small for a flywheel *and* an honest test set."

*Why that answer lands:* it identifies adaptive overfitting precisely, contrasts it with the place the project gets it right, and turns the small-data constraint into an insight instead of an excuse.

**A4. Five turns of the loop.**
"The fifth model's test MSE is optimistically biased, because each of the five interventions was chosen by looking at test failures and kept or rejected by test MSE. That's five rounds of selection on a five-episode test set, and the reported number is the winner of a search. It isn't an estimate of performance on new episodes. On top of that, the test set may not even be the same across those runs: rebalancing keeps it fixed, but any intervention that excludes episodes reshuffles the split. To report honestly, I'd either hold out a fresh test set that the loop never saw, or report the fifth model alongside the first on a split none of the decisions touched, with a bootstrap interval over episodes."

*Why that answer lands:* it names selection bias over repeated rounds, connects it to the moving-test-set issue from another module, and gives an actionable reporting protocol.

**A5. "Walk me through your data flywheel. Does it work?"**
"Mechanically, yes: every stage is a real persisted artifact. A run's test predictions go through the engine's failure detector, which flags the top 10% of frames, rolls them up into ranked episode reviews, and writes them to the catalog for curators. My layer adds candidate explanations, careful not to equate model failure with bad data, plus a recommended next step: rebalance toward a shared attribute and retrain. That becomes a new, lineage-tracked dataset version. As a *decision procedure*, it doesn't work yet, for three reasons. It always recommends action, because the percentile threshold always flags 10%. The 'data defect' explanation can't fire at these test sizes. And it runs on test, so following it overfits the test set. I'd move it onto validation, add an absolute or baseline-relative gate for whether to act, and make the concentration test relative to the base rate. The plumbing is the hard part, and that's done; the decision logic needs those three changes."

*Why that answer lands:* it separates "the plumbing works" from "the decisions are sound", gives specific reasons for the second, and ends with a clear, bounded fix list. That's credible, not defensive.

---

# Module 8 Summary — the one page to revise

**What this module is.** A wrapper that sends test-set outcomes through the engine's `FeedbackLoop` (p90 threshold, at least 3 failures per episode, persisted to the catalog), then attaches templated explanations ("concentrated → check data" or "scattered → model limitation") and a fixed recommendation to rebalance and retrain. The plumbing is sound; the decision logic is degenerate at these data sizes and runs on the wrong split.

**Five claims to make cold**
1. Detection is the engine's. This layer adds explanations framed so that "model failure does not imply bad data."
2. A p90 threshold flags about 10% of frames by definition, so "no action" is effectively unreachable.
3. Concentrated (> 0.5) can't occur: the maximum is about 0.50 (SO-100) and 0.40 (ALOHA).
4. Feedback runs on test, so the curation loop tunes on test. Move it to val.
5. Rebalancing is the one intervention that keeps the test set fixed, so it's the only one the loop can evaluate fairly.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Wrap engine `FeedbackLoop` | reimplement | single detection implementation | thresholds and metrics unverifiable here |
| Percentile threshold (p90) | absolute threshold | adapts to model and normalization | can never say "nothing is wrong" |
| Fixed 0.5 concentration cut | base-rate-relative test | simple | unreachable at p90 with 4–5 episodes |
| Templated explanation and recommendation | attribute analysis of reviews | honest hedging, a human in the loop | no concrete attribute or value produced |
| `persist=True` | in-memory only | curators find flags via `rde` | side effects, duplicates, stale after re-evaluation |
| Run on test outcomes | run on val | test is what's evaluated at the end | adaptive overfitting of test |

**Volunteer unprompted:** the test-set loop; both dead branches, with the arithmetic; `FAILURE_ANALYSIS.md`'s "deterministic predictions" claim contradicts README §2.3; `retraining_selection` is unused.

**Numbers and names:** `error_percentile = 90.0`, `min_failures_per_episode = 3`, concentration `> 0.5`; SO-100 1,908 frames, 5 episodes, about 190 flagged, max rate ≈ 0.50; ALOHA 4,400 frames, 4 × 1,100, max rate 0.40; priority = failure_rate × (1 + mean_error) per `FAILURE_ANALYSIS.md`; output at `failures/feedback_artifact.json`.

**Sentences worth memorizing**
- "A percentile threshold is good for ranking failures and useless for deciding whether there are any."
- "My training loop never selects on test; my data-curation loop did, and that's the fix I'd make first."
- "At p90 with five test episodes, 'concentrated failure' can't happen. The branch is dead by arithmetic."
- "The plumbing of the flywheel is real and persisted; the decision logic on top of it needs to move to validation and get an absolute gate."
