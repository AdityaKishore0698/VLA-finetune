# Module 7 — Offline evaluation: baselines, tails, time and slices

**Files covered:** `src/model_training/evaluation/offline.py` (`evaluation/__init__.py` was covered in Module 1)

**Where we are:** Module 6 produced a selected checkpoint. This module turns it into the numbers the project is judged by. It scores two baselines through the same evaluator as SmolVLA, then breaks the test error down by tail, by episode, by phase of the episode, and by named slice. Everything in README §2, the headline results, comes out of these five functions plus the engine's `Evaluator`.

**The one-line thesis:** The decompositions are the right ones to report. Baselines are scored through the same call, tails come before means, error is broken down by phase, and metrics that can't be computed are refused. But each slice **re-runs a stochastic model** instead of slicing the predictions already made. The slices are tiny (1–2 episodes), and nothing carries an uncertainty estimate, so the finer-grained claims in the README are mostly noise.

---

## 7.1 Prerequisites — the absolute basics

- **Per-frame error, as defined here.** For one frame, `FrameOutcome.error` is the **mean absolute error across action dimensions** (per `docs/METRICS.md` §4 and the `error_distribution` docstring). Each frame produces one scalar.
- **Split-level MSE vs "per-episode MSE".** The split's `mse` squares *each dimension's* error and then averages. This module's per-episode `"mse"` squares the *per-frame mean absolute error* and averages. Those are different quantities: by Jensen's inequality `(mean |e|)² ≤ mean e²`, so the per-episode number is systematically smaller. `METRICS.md` §4 states this honestly.
- **Percentiles / tail metrics.** p90 is the error below which 90% of frames fall. Tails matter for control because one wildly wrong action can end a task even when the average is low.
- **Normalized episode progress.** `frame_index / (episode_length − 1)`, from 0 at the start to 1 at the end, so episodes of different lengths line up by phase.
- **Stratified evaluation.** Reporting metrics separately on predefined subsets (strata or slices) instead of only in aggregate. Aggregates can hide a subgroup that fails.
- **Mean-action baseline.** It always predicts the training-set mean action. In normalized units its MSE is about the variance, roughly 1.0 (README: 0.908 SO-100, 1.010 ALOHA). Any model must beat it.
- **Nearest-neighbour baseline (behaviour retrieval).** Per `docs/API_REFERENCE.md`, `NearestNeighborPolicy` paired with `StateAdapter`: find the training frame whose **robot state** is closest to the query frame's state, and return that frame's action. **It uses no images and no language.** It's described there as "a genuinely strong baseline on single-task data". (This is from the docs, because the engine code isn't here, but the integration test's `StateAdapter` plus `nearest_neighbor` pairing corroborates it.)
- **Stochastic inference.** SmolVLA samples from noise each call (flow matching, Module 1). Scoring the same frames twice gives different errors.
- **Sampling uncertainty.** With 5 test episodes, a metric's spread over *which* episodes happened to be in test is large. A **bootstrap over episodes** (resampling episodes with replacement and recomputing the metric) is the standard way to put an interval on it.

---

## 7.2 `evaluate_baselines`: same call, different inputs

```python
def evaluate_baselines(view, names, *, split="test", train_split="train",
                       batch_size=32, max_frames=None) -> dict[str, EvaluationResult]:
    """Fit and score each baseline through the exact call used for the model
    under test, so the comparison is apples-to-apples."""
    ev = Evaluator()
    out: dict[str, EvaluationResult] = {}
    for name in names:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"unknown baseline model {name!r}; known: {sorted(MODEL_REGISTRY)}")
        model, adapter = build_model(name, view.contract)
        model.fit(view, split=train_split)
        out[name] = ev.evaluate(view, model, adapter, split=split, batch_size=batch_size,
                                max_frames=max_frames, persist=False)
    return out
```

**Decision: score baselines through the identical `Evaluator.evaluate` call, in every run.**
- *Why:* the metric code, frame selection, normalization and `max_frames` cap are shared, so a ratio to the baseline is meaningful. `METRICS.md` says it directly: "a model that cannot outperform the training-set mean has, by definition, learned nothing that this metric can detect."
- *What "apples-to-apples" does and doesn't mean:* the **evaluation** is identical; the **inputs** aren't. `build_model` returns each model's own adapter: `StateAdapter` for both baselines, `VisionLanguageAdapter` for SmolVLA. That's correct, since a baseline can't use images, but it changes what the comparison *says* (next point).
- *Fitted on the same `train` split of the same `training/vN`,* so for an intervention run, the baselines see the same reduced or filtered training pool as SmolVLA. They don't receive `episode_weights`, so in a rebalanced run the baselines are unweighted. That's a small asymmetry, worth knowing.
- *The `KeyError` for an unknown name* is tested (`test_evaluate_baselines_rejects_unknown_model`). It's raised *before* any work, with the list of known names.
- *Refit every run* (k-NN index built over all training frames each time). That's cheap on CPU and avoids stale caches.

**What the headline result actually means.** README Table 2: SO-100 test MSE is SmolVLA **0.0239** and nearest-neighbour **0.0239**. ALOHA: SmolVLA **0.2157**, nearest-neighbour **0.1062**. Since the nearest-neighbour baseline looks up the closest *robot state* and replays its action, the tie says:

> On these single-setup datasets, **robot state alone nearly determines the next action**, and a 450M-parameter vision-language-action model, fine-tuned for 2,000 steps, adds nothing measurable *offline* over a state-space lookup table, and does worse on the harder bimanual task.

That isn't a failure of the evaluation; it's what the evaluation is *for*. Offline, though, it can't tell you what the image and language inputs buy in **closed loop**, where retrieval typically breaks down once the robot leaves the demonstrated states. That's the strongest argument for the rollout evaluation the project can't run. Have this paragraph ready; it's the most interesting sentence the project can say.

**Doc vs code:** `METRICS.md` §2 says baseline-relative metrics (`mse_ratio`, `mae_ratio`) "are computed by `rde.evaluation.Evaluator.compare` and by `model_training.evaluation.offline.evaluate_baselines`". **`evaluate_baselines` computes no ratio**; it returns raw results. The only ratio in this repo is calculated in the report renderer (`m["mse"] / mean_mse`, Module 10). Whether `Evaluator.compare` exists in the engine is unverifiable here, and nothing in `src/` calls it.

---

## 7.3 `error_distribution` and `episode_metrics`

```python
def error_distribution(result: EvaluationResult) -> dict[str, Any]:
    """Median / p90 / p95 / p99 of per-frame mean-absolute error."""
    if not result.outcomes:
        return {"status": "no outcomes"}
    errs = np.array([o.error for o in result.outcomes])
    return {"median": float(np.median(errs)), "p90": float(np.percentile(errs, 90)),
            "p95": float(np.percentile(errs, 95)), "p99": float(np.percentile(errs, 99)),
            "min": float(errs.min()), "max": float(errs.max())}
```

- **Tails, not just means.** README §2.1 uses these: continued training cut p99 by 7% and the worst frame by 32% (0.553 → 0.375), while the median moved less. That's a genuinely useful observation, because tail frames are where control fails.
- **`float(...)` casts** turn NumPy scalars into Python floats, so the JSON is exact. This line is what keeps Module 3's `default=str` stringification hazard from applying here.
- **The empty guard** returns a status dict instead of crashing `np.percentile` on an empty array. `plot_error_distribution` then skips plotting because the keys are missing (Module 10). That's consistent.
- **The p99 of 1,908 correlated frames** is set by about 19 frames, which may all come from one difficult moment in one episode. `METRICS.md` §3 says so ("statistically noisy; the frame count should always be reported"). The returned dict doesn't include the frame count, though, so the caveat relies on the reader.

```python
def episode_metrics(result: EvaluationResult) -> list[dict[str, Any]]:
    by_ep: dict[int, list[float]] = {}
    for o in result.outcomes:
        by_ep.setdefault(o.episode_id, []).append(o.error)
    out = []
    for ep, errs in sorted(by_ep.items()):
        a = np.asarray(errs)
        out.append({"episode_id": ep, "frames": len(a), "mae": float(a.mean()),
                    "mse": float((a ** 2).mean()), "rmse": float(np.sqrt((a ** 2).mean())),
                    "max_error": float(a.max())})
    return out
```

- **The `"mse"` and `"rmse"` keys are misnamed.** They're computed from per-frame *mean absolute* error, so they're "mean squared MAE", not MSE in the sense every other `mse` in the system uses. `METRICS.md` §4 discloses it, which is to its credit, but a JSON consumer or a `compare --metric mse` reader won't see the doc. Rename to `mean_sq_frame_mae` or compute true per-episode MSE from `per_dim_error` (which `FrameOutcome` carries). README §2.2 quotes per-episode **MAE** only, which is the correctly named one.
- **`sorted(by_ep.items())`** makes the output order deterministic by episode ID, so diffs between runs are stable.
- **Per-episode is the right grain for a 5-episode test set.** README: SO-100 per-episode MAE spans 0.085–0.111. That spread (about 25%) is the honest measure of how much a single test-split number depends on which episodes are in test. Use it as your uncertainty argument (7.6).

---

## 7.4 `temporal_error`: error by phase of the episode

```python
def temporal_error(result, view, buckets: int = 10) -> dict[str, Any]:
    if not result.outcomes:
        return {"status": "no outcomes"}
    episodes_touched = {o.episode_id for o in result.outcomes}
    frame_index: dict[int, int] = {}
    ep_len: dict[int, int] = {}
    for ep_id in episodes_touched:
        table = view.get_episode_frames(ep_id, columns=["frame_id", "frame_index"])
        d = table.to_pydict()
        ep_len[ep_id] = max(1, len(d["frame_id"]))
        for fid, fidx in zip(d["frame_id"], d["frame_index"]):
            frame_index[int(fid)] = int(fidx)

    bucket_errors: list[list[float]] = [[] for _ in range(buckets)]
    unresolved = 0
    for o in result.outcomes:
        fidx = frame_index.get(o.frame_id)
        if fidx is None:
            unresolved += 1
            continue
        length = ep_len[o.episode_id]
        progress = fidx / (length - 1) if length > 1 else 0.0
        b = min(buckets - 1, max(0, int(progress * buckets)))
        bucket_errors[b].append(o.error)
    return {"buckets": buckets, "progress_bucket_edges": [...],
            "mean_error_by_progress_bucket": [round(float(np.mean(b)), 6) if b else None for b in bucket_errors],
            "n_by_bucket": [len(b) for b in bucket_errors], "unresolved_frames": unresolved}
```

- **Why the re-read.** The docstring explains it: `FrameOutcome` has `frame_id` but not `frame_index`, so the function fetches the small `frame_id → frame_index` table once per touched episode. It reads two narrow columns, not images. That's the right cost profile.
- **`columns=[...]`** projects only the needed columns. Without it, the engine might materialize full frames (state, action and image references) for every episode.
- **`length = len(d["frame_id"])` assumes `frame_index` runs from 0 to length−1 with no gaps.** If an episode has dropped frames (the profile reports `episodes_with_missing_or_discontinuous_frames`, Module 5), `frame_index` can exceed `length − 1`, so `progress > 1`, and the clamp `min(buckets − 1, …)` silently piles those frames into the last bucket. Using `max(frame_index)` instead of the row count would be robust to gaps.
- **`max(1, …)` and `if length > 1 else 0.0`** guard single-frame episodes against division by zero.
- **The `min/max` clamp** handles `progress == 1.0` exactly (the last frame), which would otherwise compute bucket 10.
- **`unresolved_frames` is reported rather than hidden.** That's the honest-metadata pattern.
- **Weighting:** every bucket pools frames across episodes, so longer episodes contribute more frames per bucket. With SO-100's 326–475-frame spread, that's mild.
- **With a `max_frames` cap and first-N frame selection (Module 6's open question), later buckets would be empty.** The function correctly returns `None` and `n = 0` for them instead of a fabricated mean.

**What it found:** README §2.2 says SO-100 error is lowest at the start (0.051) and end (0.049), and peaks mid-episode (0.144 at 50–60% progress). That's plausible physically, since the middle is the grasp and transport phase. It's also the evidence behind Module 6's hypothesis that start-of-episode validation frames are unrepresentatively easy.

---

## 7.5 `stratified_evaluate`: slices that re-run the model

```python
ev = Evaluator()
out: dict[str, EvaluationResult] = {
    "overall": ev.evaluate(view, model, adapter, split=base_split, batch_size=batch_size,
                            max_frames=max_frames, persist=False)
}
for name in slices or []:
    if name == "per_task":
        for task_id in sorted({e.task_id for e in view.episodes if e.task_id is not None}):
            ep_ids = view.episode_ids(split=base_split, task_id=task_id)
            if not ep_ids:
                continue
            out[f"task_{task_id}"] = ev.evaluate(
                view, model, adapter, split=f"{base_split}:task_{task_id}",
                episode_ids=ep_ids, batch_size=batch_size, max_frames=max_frames, persist=False)
    elif name == "per_length_bucket":
        lengths = {e.episode_id: e.num_frames for e in view.episodes
                   if e.attributes.get("split") == base_split}
        if len(lengths) >= 3:
            t1, t2 = np.percentile(list(lengths.values()), [33.3, 66.7])
            groups = {"short": [], "medium": [], "long": []}
            for eid, n in lengths.items():
                groups["short" if n <= t1 else "long" if n > t2 else "medium"].append(eid)
            for bucket_name, ids in groups.items():
                if ids:
                    out[f"length_{bucket_name}"] = ev.evaluate(..., episode_ids=ids, ...)
    else:
        raise ValueError(f"unknown evaluation slice {name!r}; supported: 'per_task', 'per_length_bucket' ...")
return out
```

**Decision: define slices from real per-episode metadata only, and reject anything else.**
- *Why:* the docstring says "Slice names are resolved from real per-episode metadata, never invented". Asking for `"variation"` raises (`test_stratified_evaluate_rejects_unknown_slice`). This is the same "refuse to fabricate" discipline as the profile.
- *The split label* `f"{base_split}:task_{id}"` is passed to `evaluate` with explicit `episode_ids`. Per `docs/API_REFERENCE.md`, `episode_ids` "takes precedence over `split` for frame selection" and `split` becomes a label. That's subtle and correct: the slice is restricted to test episodes by construction, because `ep_ids` came from `episode_ids(split=base_split, ...)`. `test_stratified_evaluate_per_task_and_length_bucket` asserts that every slice's episodes are a subset of overall's.

**Problem 1: every slice is a fresh stochastic inference pass.** Each `ev.evaluate(...)` calls `model.predict` again on frames that `"overall"` already predicted. For SmolVLA, whose outputs are sampled, **a slice's frames get different predictions from the same frames inside `"overall"`.** Consequences:
- **Slices don't reconcile with overall.** A frame-weighted average of the three length slices won't equal the overall MSE, beyond rounding.
- **The README's noise estimate *is* this effect.** README §2.3: "The stratified pass re-scores all 1,908 SO-100 test frames, and its MSE differs from the main pass by up to 5% (0.0305 against 0.0290 ...)". SO-100 has a single task, so `task_0` is *every* test episode, which makes it a full re-scoring of overall with fresh noise. That's actually a lucky natural experiment: it's the only measurement of inference variance in the project. But it was produced by redundancy, not design.
- **Cost.** With `base.yaml`'s `slices: [per_task, per_length_bucket]` on single-task SO-100, test inference runs over overall (1×) + task_0 (1×) + three length slices (about 1× together), so **about 3× the necessary inference**. At about 50 ms per frame over 1,908 frames, that's about 95 s per pass, or about 3 minutes extra per run.
- **Config inconsistency.** `aloha_coffee_baseline.yaml` drops `per_task` because "a 'per_task' slice would be an exact duplicate of 'overall' -- pure redundant video-decode cost". **SO-100 is also single-task** (`EXPERIMENTS.md` §3), but `base.yaml` keeps `per_task` for it. Same reasoning, opposite configuration.
- **The fix:** predict once, then slice the `outcomes` of `"overall"` by `episode_id` in Python. Every slice then becomes a partition of one set of predictions, reconciles exactly with overall, and costs zero extra inference. If you want an inference-noise estimate, measure it *deliberately* (score overall twice with different seeds), not by accident.

**Problem 2: the length slices hold 1–2 episodes each.** The test split has 5 episodes (SO-100) or 4 (ALOHA). The 33.3/66.7 percentiles of 5 lengths, with `<= t1` short and `> t2` long, give groups of about 2/1/2. README §2.2 claims "long SO-100 episodes are easier (MSE 0.0206) than short (0.0290) and medium (0.0311) ones". That's a comparison of **one or two episodes per group**, each scored once by a stochastic model whose re-scoring moves MSE by about 5%, among episodes whose per-episode MAE already varies by about 25% (0.085–0.111). **That claim isn't supported by the data's resolution.** It might be true, but you can't tell from this. In an interview, don't repeat it as a finding. Say that the slices exist and that the test set is too small for them to be informative.
- For ALOHA, the README correctly notes the length slice is uninformative because "all four of its test episodes are 1,100 frames long and fall into one bucket". With ties, `<= t1` puts everything into "short" (the Module 4 tie collapse). That's a correct caveat in the README.
- **`len(lengths) >= 3` silently skips** the length slices for test splits under 3 episodes. A `reduced` run with 2–3 test episodes (Module 4's moving test set) would silently get **no length slices, with no note**.

**Problem 3: within-split cut points.** Here terciles are computed over the *test split's own* 5 lengths; in Module 4's intervention they're computed over the *whole dataset*. So `length_short` at evaluation time and `"short"` at intervention time are different sets (Module 4 gap 4). They also differ *between runs* whenever the test split changes, which is every excluding intervention. A "short slice" isn't a fixed slice.

**`docs/EXPERIMENTS.md` §6 claims** "A slice is a frozen episode-identifier list, produced identically on every invocation". That's true for *one* `training/vN` (same test episodes, same lengths, same cut points), and false across interventions, and the cut points themselves depend on the 5 test lengths. "Fixed evaluation slices" in the docs means "deterministic given the version", not "the same slice across experiments".

---

## 7.6 What isn't here: uncertainty

Nothing in this module (or anywhere in the repo) produces a confidence interval, a standard error, or a multi-seed aggregate. With 5 test episodes, one training seed and a stochastic policy, a single test MSE has three variance sources: *which episodes are in test*, *the training seed*, and *inference sampling*. The project measured the third once (about 5%), by accident. The README's "18% improvement from continued training" sits outside that 5% inference band, but it isn't tested against episode variance or seed variance.

The cheapest honest addition is an **episode-level bootstrap**. Resample the 5 test episodes with replacement 1,000 times, and recompute MSE from the stored per-frame `per_dim_error` (already written to `predictions/test_outcomes.jsonl` in Module 11). That gives an interval at zero inference cost. A **paired** comparison between two runs on the *same* test episodes (only possible for full and rebalanced runs, Module 4) is stronger still: per-episode differences cancel the episode-difficulty variance.

---

## 7.7 Honest gaps to own before an interviewer finds them

1. **Slices re-run stochastic inference instead of slicing one prediction set.** *Consequence:* slices don't reconcile with overall, cost about 3× the necessary inference on SO-100, and the project's only inference-noise estimate is an accident of a redundant slice.
2. **Length slices contain 1–2 test episodes, so the README's "long episodes are easier" is below the data's resolution.** *Consequence:* a claim that can't survive an interviewer's "how many episodes?"
3. **There's no uncertainty estimate anywhere** (no bootstrap, no seeds, no intervals). *Consequence:* no difference in the README is statistically qualified except against the accidental 5% noise band.
4. **The nearest-neighbour baseline is state-only retrieval, and it ties or beats SmolVLA.** *Consequence:* this isn't a bug. It's the key interpretive fact, and you must frame it as "offline, state nearly determines action on single-setup data".
5. **Per-episode `"mse"`/`"rmse"` are mean squared per-frame MAE, not MSE.** *Consequence:* a misnamed key that's disclosed only in `METRICS.md`.
6. **`base.yaml` keeps `per_task` for single-task SO-100**, which the ALOHA config explicitly removes as redundant. *Consequence:* about 1 wasted full test pass per SO-100 run.
7. **`METRICS.md` §2 says `evaluate_baselines` computes ratios.** It doesn't. *Consequence:* a doc overclaim. The ratio lives only in the report renderer.
8. **Length slices are silently skipped below 3 test episodes**, with no note. *Consequence:* small-pool intervention runs lose slices invisibly.
9. **`temporal_error` assumes contiguous `frame_index`.** *Consequence:* episodes with gaps pile frames into the last bucket.
10. **Baselines don't receive `episode_weights` in rebalanced runs.** *Consequence:* a minor asymmetry in "apples-to-apples".
11. **Tail percentiles don't carry their frame count**, despite `METRICS.md` saying it should always be reported. *Consequence:* the noise caveat depends on the reader.
12. **`success_tolerance` is accepted by the engine's `Evaluator.evaluate` (per the API doc) but never passed.** *Consequence:* this confirms that Module 2's dead config field could be wired in one line.

---

## The shape to carry into Module 8

```
 selected checkpoint ──▶ SmolVLAPolicy + VisionLanguageAdapter
 training/vN view ─────┐
                       ├─ evaluate_baselines: build_model(name) → (policy, StateAdapter); fit(train); Evaluator(test)
                       │       mean_action (≈1.0 MSE)   nearest_neighbor (state-space retrieval; ties SmolVLA on SO-100)
                       │
                       └─ stratified_evaluate(test, slices)
                              overall  ───────── Evaluator(test)                    ← pass 1
                              task_k   ───────── Evaluator(episode_ids=…)           ← re-inference (dup on 1-task data)
                              length_* ───────── Evaluator(episode_ids=…)           ← re-inference, 1–2 episodes each
                              (within-split terciles; skipped if < 3 episodes)
                                 │
                 overall.outcomes ├─ error_distribution  (median/p90/p95/p99/min/max of per-frame MAE)
                                  ├─ episode_metrics     (mae ✓, "mse"/"rmse" = squared MAE ✗)
                                  └─ temporal_error      (10 progress buckets; re-reads frame_index)
                                 ▼
                 FeedbackLoop.detect(overall)  → Module 8
```

---

## Drill — answer from memory, files closed

1. What exactly does the nearest-neighbour baseline do, and what does it mean that it ties SmolVLA on SO-100?
2. *Why didn't you just* compute slice metrics by filtering the overall predictions?
3. The README says long SO-100 episodes are easier than short ones. Would you defend that claim?
4. *What breaks in production:* you add a third slice type and run on a dataset with 40 tasks. What happens to evaluation time, and why?
5. *Interviewer framing:* "Your model ties a nearest-neighbour lookup. Why should I believe it learned anything useful?"

---

# Appendix A — Model answers to the drill

**A1. The nearest-neighbour baseline.**
"It's behaviour retrieval over robot state. During fit it indexes every training frame's state vector and action. At test time it finds the training frame whose state is closest to the query's state and replays that action. No images, no language. It's scored through exactly the same evaluator call as SmolVLA. On SO-100 both reach 0.0239 test MSE, so offline, on a single-setup, single-task dataset, the robot's current joint configuration almost determines the demonstrated next action, and the vision-language model adds nothing measurable on top of a lookup table. On ALOHA, retrieval is twice as good. That's the honest result. What offline evaluation can't show is closed-loop behaviour, where retrieval usually fails as soon as the robot drifts into states the demos never visited. That's where vision should earn its keep, and it's exactly the experiment I couldn't run without a robot or simulator."

*Why that answer lands:* it explains the mechanism precisely, states the uncomfortable result without spin, and identifies exactly what experiment would distinguish the models.

**A2. Why not slice the overall predictions?**
"I should have. Each slice calls the evaluator again, which reruns SmolVLA, and since SmolVLA samples actions from noise, the same frame gets a different prediction inside a slice than inside overall. So slices don't reconcile with the overall number, and on SO-100, which has a single task, the per-task slice is a complete second pass over the test set. Together with the length slices, that's about three times the inference I needed. Funnily enough, that redundant pass is the only place I measured inference noise (about 5%). The right design predicts once, slices the per-frame outcomes by episode ID, and measures inference noise deliberately by scoring twice with different seeds."

*Why that answer lands:* it admits the flaw directly, explains the stochastic mechanism that makes it more than a performance issue, and turns the accidental noise estimate into a point about deliberate measurement.

**A3. "Long episodes are easier"?**
"No, not as a finding. The test split has five episodes, and the length terciles of five values put about two episodes in short, one in medium and two in long. Each is scored once by a stochastic model whose re-scoring moves MSE by about five percent, and per-episode error already varies by about twenty-five percent across the five episodes. A difference between one and two episodes at that noise level isn't evidence. I'd say the slicing infrastructure works, and that this test set is too small for slice-level conclusions. To make a real claim, I'd want a bootstrap interval over episodes at minimum, and ideally a larger held-out set or several seeds."

*Why that answer lands:* it retracts the overreach yourself, with arithmetic, before the interviewer can. Volunteering a downgrade buys credibility for every claim you keep.

**A4. A third slice type on a 40-task dataset.**
"Evaluation time scales with the number of slices, because every slice is a separate inference pass. `per_task` alone would add 40 passes. They're disjoint subsets, so together about one more full pass, plus 40 rounds of evaluator overhead and model batching inefficiency on small slices. Add a third slice type and every new axis costs another pass. And because the passes are stochastic, the slices wouldn't sum to overall. If slices ever overlapped, like task crossed with length, the cost would grow with the number of combinations. The fix is the same: one inference pass, with slicing as a cheap group-by over stored per-frame outcomes. Then any number of slice definitions is essentially free."

*Why that answer lands:* it reasons about scaling from the code's structure, catches the stochastic-reconciliation issue again, and gives a fix that changes the complexity class.

**A5. "Your model ties a nearest-neighbour lookup. Why believe it learned anything?"**
"Offline, I'd say it learned the demonstrated state-to-action mapping about as well as memorizing it does. It's at 2.6% of the mean-action error, so it clearly learned the task distribution, but it isn't better than retrieval here, and on ALOHA it's worse. I wouldn't claim more than that. What would justify more is a test where retrieval should fail: held-out scenes or object positions, or closed-loop rollouts where the policy's own errors push it off the demonstrated states. That's where a model using vision and language should beat a state-space lookup. Without a robot or simulator, I made the system report rollout as unavailable rather than imply it. The value of the project at this stage is that the evaluation harness told me the truth, including that the baseline is this strong."

*Why that answer lands:* it concedes exactly what the evidence supports, names the experiment that would change the conclusion, and reframes the project's value around rigorous measurement, which is defensible, rather than model performance, which isn't yet.

---

# Module 7 Summary — the one page to revise

**What this module is.** Five functions over the engine's `Evaluator`: score baselines through the identical call, then decompose test error into tails, per-episode, per-phase and named slices, each resolved only from real metadata. The decompositions are right; the slicing re-runs inference, and nothing carries uncertainty.

**Five claims to make cold**
1. Baselines go through the same `Evaluator.evaluate` call. Mean-action is about 1.0 MSE in normalized units; nearest-neighbour is state-only retrieval.
2. SmolVLA ties nearest-neighbour on SO-100 (0.0239 vs 0.0239) and loses on ALOHA (0.216 vs 0.106). Offline, state nearly determines action on single-setup data.
3. Tails and phases are reported, not just means. Continued training cut the worst frame by 32%, and mid-episode error is about 3× start and end error.
4. Each slice is a fresh stochastic pass, so slices don't reconcile with overall. The fix is to predict once and slice outcomes.
5. With 5 test episodes and 1–2 per length slice, slice-level claims are below resolution, and there's no bootstrap or multi-seed interval.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Same evaluator for baselines | separate baseline scoring | comparable ratios | inputs still differ (state vs vision and language) |
| Slices from real metadata only; reject others | proxy "variation" slices | refuse to fabricate | only task and length available |
| Slice by re-calling `evaluate` | slice stored outcomes | reuses the evaluator's metric code | about 3× inference; non-reconciling stochastic slices |
| Within-split length terciles | dataset-level cut points | adapts to the split | slices move across runs; differ from intervention buckets |
| Per-frame MAE as the error scalar | per-dimension MSE | one scalar per frame | per-episode "mse" is misnamed |
| Re-read `frame_index` per episode | add `frame_index` to outcomes | no evaluator change | assumes contiguous indices |

**Volunteer unprompted:** the redundant `per_task` on SO-100 and the accidental 5% noise estimate; the "long episodes are easier" overreach; no intervals; the misnamed per-episode MSE; `METRICS.md`'s ratio claim.

**Numbers and names:** SO-100 test 1,908 frames from 5 episodes; ALOHA 4,400 frames from 4 episodes (all 1,100 frames); SmolVLA/mean-action 3.2% → 2.6% (SO-100), 21.4% (ALOHA); NN 0.0239 / 0.1062; mean-action 0.908 / 1.010; per-episode MAE 0.085–0.111 (SO-100), 0.296–0.405 (ALOHA); temporal 0.051 start / 0.144 mid / 0.049 end; re-scoring noise up to 5%; 10 progress buckets; terciles at the 33.3/66.7 percentiles, `>= 3` episodes required; NN confidence–error correlation ≈ −0.79 (METRICS.md).

**Sentences worth memorizing**
- "Offline, on single-setup data, the robot's state almost determines the next action; that's why a state-space lookup ties a VLA."
- "The harness's job was to tell me the truth, and it did: the baseline was that strong."
- "Every slice re-ran a stochastic model, so my slices didn't add up to my overall number. Predict once, slice the outcomes."
- "Five test episodes can't support slice-level claims; I'd want a bootstrap over episodes before saying long episodes are easier."
