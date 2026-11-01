# Module 4 — Data interventions as lineage

**Files covered:** `src/model_training/data/interventions.py` (the `data/__init__.py` façade was covered in Module 1)

**Where we are:** Module 2 validated a `DatasetIntervention`, and Module 3 gave the run a directory. This module is the first one that *changes the world*: it turns "halve the data" or "drop the short episodes" into new, immutable dataset versions inside the engine's catalog. This is the project's central idea, the one the README's organizing principle rests on ("Even a data intervention ... is executed as a lineage-tracked transform inside the engine, never as a local copy").

**The one-line thesis:** Every intervention becomes a real `clean/vN` (an episode filter) plus a real `training/vN` (a split assignment plus sampling weights), so the provenance is impeccable. But because the filter runs **before** the split, every excluding intervention is scored on a **different, smaller test set** than the baseline it's meant to be compared against. The lineage design quietly undermines the comparisons the experiments exist to make.

---

## 4.1 Prerequisites — the absolute basics

- **Data-centric ML / data intervention.** Holding the model and training recipe fixed and changing *the data*, to measure what the data contributes. Here the four primary interventions are *full*, *reduced* (fewer episodes, chosen at random), *imbalanced* (deliberately skewed along one attribute) and *rebalanced* (oversampling under-represented episodes).
- **Quantity vs coverage.** Two different hypotheses about why more data helps. *Quantity:* more examples of the same thing reduce variance. *Coverage:* examples of *different* situations extend what the model can handle. A random 50% subset changes quantity while preserving coverage in expectation. Removing every "short" episode changes coverage. The experiment set is designed to tell the two apart (the configs' `hypothesis: quantity_vs_coverage` tag).
- **Oversampling / importance weighting.** Instead of duplicating data, give some examples a higher probability of being drawn into a training batch. A weight of 3.0 means "drawn about three times as often as a weight-1.0 episode", in expectation, under a weighted sampler. Nothing is copied on disk.
- **Stratum / slice.** A subset of the evaluation data defined by an attribute (task, length bucket). Covered in depth in Module 7.
- **Terciles / percentiles.** The 33.3rd and 66.7th percentiles split a sorted list into three roughly equal groups. `np.percentile` uses linear interpolation between data points by default.
- **Banker's rounding.** Python's `round()` rounds halves to the nearest *even* integer: `round(2.5) == 2`, `round(3.5) == 4`. This matters whenever `fraction × count` lands on `.5`.
- **`numpy.random.default_rng(seed)`.** A modern NumPy random generator (PCG64). With the same seed and the same input array it produces the same draws on the same NumPy version. `rng.choice(ids, size=n, replace=False)` draws `n` distinct elements.
- **Nested subsets.** In a data-efficiency curve, the 25% subset being contained in the 50% subset (and that in the 75%) reduces variance between curve points, because each point adds data to the last instead of swapping it out. The standard way to get them is to take prefixes of one random permutation.
- **Orphaned version.** A dataset version created in the catalog by a run that then failed, so no experiment refers to it.

---

## 4.2 The version pipeline this module drives

```
 input version (config.dataset.version, e.g. clean/v1)
      │
      ├─ kind = full / rebalanced, or a subset equal to every episode
      │        → no transform; clean_version := input_version
      │
      └─ kind = fraction / reduced / imbalanced / custom with a real subset
               → next_version(CLEAN) = clean/vK
                 TransformPipeline([EpisodeSubsetFilter(subset_ids, reason)]).run(input → clean/vK)
      ▼
 next_version(TRAINING) = training/vM
 DatasetBuilder.build(clean_version → training/vM,
                      splits=train/val/test fractions, seed=42, split_mode=quantile,
                      episode_weights = {...} or None)
```

**Decision: an intervention is two engine operations, never a local copy.**
- *What:* episode selection is an engine *transform* (`EpisodeSubsetFilter`), which records `dropped_episode_ids` in lineage. Split assignment and weights are an engine *build*.
- *Why:* the README's lineage output shows exactly what this buys:
  `transform ['raw/v1'] -> clean/v1 episodes 50->48 dropped=[9, 31]`. Any `training/vN` can be traced back to raw episodes, with the reason for each drop. `test_reduced_intervention_selects_half_and_records_lineage` checks that the transform run's `output_episode_ids` and `dropped_episode_ids` equal what this module computed. **That test verifies the core claim end to end**, and it's the strongest test in the repo.
- *Alternative:* filter episode IDs at training time (pass `episode_ids=` to `fit`) with no new version, or copy a subset to a new directory.
- *Cost:* (1) every run writes to the shared catalog, even a `full` run, which still allocates a new `training/vN`; (2) the order "filter, *then* split" has a consequence big enough to get its own section (4.5).

---

## 4.3 `compute_length_bucket`

```python
def compute_length_bucket(episode_rows: list[dict[str, Any]]) -> dict[int, str]:
    """short/medium/long by within-dataset episode-length terciles.
    ... Same bucketing method as ``model_training.data.profile``'s coverage section, so
    the two stay comparable.
    """
    if len(episode_rows) < 3:
        return {r["episode_id"]: "medium" for r in episode_rows}
    lengths = [r["num_frames"] for r in episode_rows]
    t1, t2 = np.percentile(lengths, [33.3, 66.7])
    return {
        r["episode_id"]: ("short" if r["num_frames"] <= t1 else "long" if r["num_frames"] > t2 else "medium")
        for r in episode_rows
    }
```

**Why length is a variable at all.** `docs/EXPERIMENTS.md` §3 says the primary dataset `so100_pickplace` has *one* task instruction across all 48 episodes, so `task_id` has nothing to select between. Trajectory length "326 to 475 frames per episode" is the only real per-episode attribute that varies. The schema has no "variation" field (`profile.py`'s note says so too). This is an honest choice made from what the data actually contains, and it's the right story to tell: "I couldn't find a meaningful variation axis in the schema, so I used the one that exists, and I said so."

Line by line:
- **`< 3` → all `"medium"`.** Terciles of one or two values are meaningless, so every episode gets the neutral bucket. Without this guard `np.percentile` would still return numbers and produce a meaningless split. Fine.
- **`33.3` and `66.7`, not `100/3` and `200/3`.** These are rounded magic numbers. They're harmless, but they're approximations of a tercile, not a definition.
- **`<= t1` → short, `> t2` → long.** The asymmetric boundaries put ties at the low edge into "short". **With heavily tied lengths the buckets collapse.** The test-file comment admits the synthetic fixture's equal-length episodes all bucket to `"short"`. A real case: the README says all four ALOHA test episodes are 1,100 frames. If ALOHA's whole dataset is tied that way (I can't verify the full distribution), then `t1 = t2 = 1100`, every episode is "short", and:
  - `imbalanced, variable_value: short` drops *everything*, and the run fails loudly with "resolved to zero episodes". Good.
  - `imbalanced, variable_value: long` (or `medium`) drops *nothing*, the subset equals the full set, and **a full-data run proceeds labelled "imbalanced"** (the same silent-no-op class as Module 2 gap 4).
- **The docstring's comparability claim is false.** It says the method is the "Same bucketing method as `model_training.data.profile`'s coverage section". Grep for `percentile` or `tercile` in `profile.py` finds **nothing**. The profile reports the engine's duration *histogram*, not terciles. The only other tercile code is in `evaluation/offline.py:171`, and that one computes terciles **within the evaluated split**, not the whole dataset (`docs/EXPERIMENTS.md` Table 2 confirms: "within-split episode-length terciles"). So the `"short"` you dropped at intervention time (whole-dataset cut points) and the `length_short` slice you score at test time (test-split cut points, from 5 episodes) **are different sets with different thresholds**. Any claim that "dropping short episodes hurt the short slice" is comparing two different definitions of short.

---

## 4.4 `resolve_episode_subset`

```python
kind = intervention.kind
if kind == "full":
    return sorted(all_episode_ids), "full dataset, no episode selection"
if intervention.episode_ids is not None:
    return sorted(intervention.episode_ids), intervention.reason or f"explicit episode_ids ({kind})"
```

- **`full` short-circuits before `episode_ids`.** `kind: full, episode_ids: [1, 2]` returns every episode. That matches the docstring "explicit `episode_ids` always wins" *only for non-full kinds*. That's a silent override of user input (Module 2 gap 12).
- **Explicit IDs aren't validated against the dataset.** `episode_ids: [999]` returns `[999]`. The caller computes `excluded = set(all_ids) - {999}` (everything) and asks the engine's filter for an episode that doesn't exist. What `EpisodeSubsetFilter` does then is **unverified**. It might raise, or it might produce an empty version that fails later in the build. Duplicates (`[3, 3, 5]`) survive `sorted` and make `subset_ids == sorted(all_ids)` false even when every episode is listed, so they'd create an unnecessary `clean/vK`.
- **`reason` is always recorded.** It's either user-supplied or generated. It travels into the engine's transform (`EpisodeSubsetFilter(subset_ids, reason=reason)`), and so into lineage. That's the auditable "why" for every dropped episode, and it's a genuinely good detail.

### Fraction and reduced

```python
if kind in ("fraction", "reduced"):
    frac = intervention.fraction if intervention.fraction is not None else 0.5
    rng = np.random.default_rng(intervention.seed)
    ids = sorted(all_episode_ids)
    # frac=0.0 is a legitimate, explicit request (a zero-data anchor
    # point on a data-efficiency sweep) and must be honored as 0
    # episodes, not silently floored to 1 ...
    n = round(frac * len(ids))
    if frac > 0:
        n = max(1, n)
    chosen = sorted(int(i) for i in rng.choice(ids, size=n, replace=False))
    return chosen, intervention.reason or f"seeded random {frac:.0%} draw over all episodes (seed={intervention.seed})"
```

- **`sorted(all_episode_ids)` before drawing** makes the draw independent of the order the catalog returns rows in. Without it, the same seed could select different episodes if the catalog's row order changed. This is the non-obvious line that makes `test_reduced_intervention_is_deterministic_given_seed` meaningful.
- **`int(i)`** converts NumPy integers to Python `int`, so they serialize cleanly into JSON and catalog records (Module 3's `default=str` would otherwise be the only safety net).
- **The 0.5 fallback is unreachable** for validated configs (Module 2 gap 13).
- **Banker's rounding:** `round(0.5 × 5) = 2`, `round(0.5 × 7) = 4`. For the real dataset (48 episodes) every multiple of 1/48 is exact, so this only matters in sweeps over small sets. It's still a "why is my 50% run using 2 of 5?" question you should be able to answer.
- **The floor at 1** (`max(1, n)` for `frac > 0`) prevents "1% of 3 episodes" from silently meaning "no data". That's sensible.

**Doc and test vs code: the zero-fraction anchor can't run.** The comment insists `frac=0.0` "must be honored as 0 episodes", and `test_reduced_intervention_at_zero_fraction_is_honored` asserts that `resolve_episode_subset` returns `[]`. But the only caller, `build_intervention_dataset`, does this next:

```python
subset_ids, reason = resolve_episode_subset(intervention, all_ids, task_of, length_bucket_of)
if not subset_ids:
    raise ValueError(f"intervention kind={intervention.kind!r} resolved to zero episodes")
```

So a data-fraction sweep including `0.0` (the "zero-data anchor" the comment names as the use case) **raises before training**. The test passes because it tests the helper in isolation, never the path a sweep actually takes. That's a textbook unit-tests-pass, system-fails gap. The honest resolution is to decide what a zero-data point *means*. Most likely it's "evaluate the pretrained `smolvla_base` with no fine-tuning", which doesn't need a dataset version at all, and should be implemented as its own path, not as an empty filter.

**Draws at different fractions aren't nested.** `rng.choice(ids, size=n, replace=False)` with the same seed at `n = 12` and `n = 24` isn't guaranteed to return a 12-subset of the 24-subset. NumPy's `Generator.choice` without replacement doesn't promise prefix consistency across sizes, and I haven't checked which algorithm it selects for these sizes. *Fix:* `perm = rng.permutation(ids); chosen = sorted(perm[:n])` gives nested subsets by construction, which is the standard design for data-efficiency curves.

### Imbalanced

```python
if kind == "imbalanced":
    var, val = intervention.variable, intervention.variable_value
    if var == "task_id":
        keep = sorted(e for e in all_episode_ids if task_of_episode.get(e) != val)
        return keep, intervention.reason or f"drop every episode with task_id={val!r}"
    if var == "episode_length_bucket" and length_bucket_of_episode is not None:
        keep = sorted(e for e in all_episode_ids if length_bucket_of_episode.get(e) != val)
        return keep, intervention.reason or f"drop every episode in length bucket {val!r}"
    raise ValueError(f"... no automatic rule for variable={var!r}; supported: 'task_id', ...")
```

- **"Imbalanced" here means *removed entirely*, not *under-represented*.** Every episode with the chosen value is dropped. The `imbalanced.yaml` hypothesis template says "Skewing the demonstration set to *under-represent* `<variable>=<value>`". Zero coverage isn't under-representation. It turns the experiment into an *out-of-distribution* test ("can it handle a regime it never saw?"), which is a different and arguably more interesting question, but not the one the config states. Under-representation would keep, say, 20% of those episodes. Name the semantics correctly in an interview.
- **The unsupported-variable path raises**, and it's tested (`test_resolve_episode_subset_unknown_variable_raises`, plus the end-to-end `test_unknown_dataset_variation_raises` in `test_failure_modes.py`). The design rule "never guess" is enforced here.
- **The `None`/mistyped-value path doesn't raise** (Module 2 gap 4): `!= None` keeps everything.
- **The module docstring is stale.** It says imbalance today uses "`task_id`, the one grouping key this dataset's schema actually carries". The code also supports `episode_length_bucket`, and the function's own docstring two definitions down explains that `so100_pickplace` is single-task, so `task_id` imbalance has nothing to select there. The two docstrings disagree, and the code and `EXPERIMENTS.md` side with the length bucket.

---

## 4.5 `build_intervention_dataset`, and the test set that moves

```python
manifest = catalog.get_manifest(dataset_id, input_version)
if manifest is None:
    raise LookupError(f"unknown version: {dataset_id}@{input_version}")
episode_rows = catalog.get_episodes(dataset_id, input_version)
all_ids = [e["episode_id"] for e in episode_rows]
task_of = {e["episode_id"]: e["task_id"] for e in episode_rows}
length_bucket_of = compute_length_bucket(episode_rows)
if not all_ids:
    raise ValueError(f"{dataset_id}@{input_version} has no episodes to build from")
```

- **The `LookupError` for an unknown version** is tested (`test_intervention_on_unknown_input_version_raises`). It's the right exception type, matching the engine's `get_dataset` behaviour asserted in `test_invalid_dataset_version_raises`.
- **Every episode in the input version is a candidate.** There's no filter on QC status here. The conftest comment says `DatasetBuilder.build()` "admits only PASSED/WARNING by default", so failed-QC episodes are presumably dropped *at build time*, after this module has computed `episode_ids` and `excluded_episode_ids`. If that's right (unverified), then `dataset.json`'s `episode_ids` can list episodes that never made it into the training version. The engine's `build_summary` would show the true count, but this module's own fields wouldn't. For `so100_pickplace`, whose `clean/v1` already dropped QC failures (episodes 9 and 31), it probably doesn't bite.

### Rebalanced: weights, never exclusion

```python
if intervention.kind == "rebalanced":
    weights = dict(intervention.episode_weights or {})
    if not weights and intervention.variable is not None and intervention.variable_value is not None:
        target = _target_episodes(intervention.variable, intervention.variable_value)
        if not target:
            raise ValueError(f"no episodes with {intervention.variable}={intervention.variable_value!r} to upweight")
        weights = {e: intervention.oversample_factor for e in target}
    if not weights:
        raise ValueError("rebalanced intervention resolved to empty episode_weights -- "
                         "nothing would change relative to a full-data run")
    subset_ids = sorted(all_ids)
    excluded: list[int] = []
    clean_version = input_version
```

**Decision: rebalance by sampling weight, not by duplication or synthesis.**
- *Why:* `docs/EXPERIMENTS.md` §4 states the constraint plainly: the project collects no new data, so a rebalanced run "can only redistribute training attention over data that already exists". Weights are honest about that; duplicating episodes on disk would just be a heavier, less traceable way to get the same effect.
- *Alternative:* physically duplicate episodes in a new `clean/vK`, which would be visible in lineage as copies, or use loss re-weighting instead of sampling re-weighting.
- *Cost:* the effect depends on the engine's weighted sampler in `fit`, which isn't verifiable here. Sampling with replacement at weight 3.0 also means those episodes' frames repeat within an epoch, which raises overfitting risk on exactly the regime you're trying to help.

**Double guard against a no-op**, the counterpart of the gap imbalanced has: both "`variable` set but no matching episodes" and "nothing resolved" raise. Note that `_target_episodes` returns `[]` for an *unsupported* variable too, so the error says "no episodes with camera=dim to upweight" when the truth is "camera isn't a supported variable". It's a misleading message, though it's at least loud.

**Explicit weights aren't validated.** Keys naming nonexistent episodes have no effect, silently. Weights ≤ 0 pass (Module 2 gap 15). Weights are also attached to *every* targeted episode in the pool, including those that end up in val and test. The runner only reads weights for `train` IDs (Module 11), so this is harmless, but it does mean `dataset.json`'s `episode_weights` overstates which episodes were actually upweighted during training.

**`docs/EXPERIMENTS.md` §4 overclaims:** "Every experiment report generated by the system states this limitation explicitly in its 'Limitations' section." Grep for `rebalanc` in `experiments/` and `reports/` finds **nothing**. The runner's limitations list covers only GPU, torch and non-SmolVLA model notes (Module 11). A rebalanced report would *not* contain the caveat. The caveat exists only in the YAML comment and in the docs.

### The excluding path

```python
else:
    subset_ids, reason = resolve_episode_subset(intervention, all_ids, task_of, length_bucket_of)
    if not subset_ids:
        raise ValueError(f"intervention kind={intervention.kind!r} resolved to zero episodes")
    excluded = sorted(set(all_ids) - set(subset_ids))
    weights = intervention.episode_weights or {}
    if subset_ids == sorted(all_ids):
        clean_version = input_version
    else:
        clean_version = next_version(catalog, dataset_id, VersionStage.CLEAN)
        TransformPipeline([EpisodeSubsetFilter(subset_ids, reason=reason)],
                          settings=settings, catalog=catalog).run(dataset_id, input_version, clean_version)

training_version = next_version(catalog, dataset_id, VersionStage.TRAINING)
build = DatasetBuilder(settings=settings, catalog=catalog).build(
    dataset_id, clean_version, training_version,
    splits=splits.as_fractions(), seed=splits.seed, split_mode=splits.mode,
    episode_weights=weights or None,
)
```

- **`subset_ids == sorted(all_ids)` → reuse the input version.** No pointless transform for a filter that drops nothing. It's sensible, and it's exactly why the silent-no-op imbalanced case (4.4) passes through invisibly: no new `clean` version, and `excluded_episode_ids == []`. The only trace is `kind: imbalanced` alongside zero exclusions in `dataset.json`.
- **`weights or None`** turns an empty dict into `None`, presumably so the engine skips weighting entirely rather than recording empty weights. That's inferred from the call site.
- **`next_version` then `run` isn't atomic.** Two concurrent runs can both be allocated `training/v7`. Whether the engine's catalog rejects the second write is **unverified**. If it doesn't, two experiments claim the same version.
- **Failures leave orphans.** If `DatasetBuilder.build` fails after the transform has succeeded, `clean/vK` exists with no experiment pointing at it. And because the runner calls this function *after* `make_run_dir` (Module 3), the run directory exists too, blocking a retry. There's no cleanup, and none is possible, since versions are immutable by design. The right fix is to make orphans *discoverable* (a run status that links to its versions), not deletable.
- **Version proliferation.** Every run allocates a fresh `training/vN`, even `full` runs with identical config and seed, which produce byte-for-byte equivalent splits if the builder is deterministic. The README's `training/v6` shows this in practice. Content-addressing a build (hash of input version, subset, splits, seed and weights) would let identical requests reuse one version.

### The finding that matters most: the test set depends on the intervention

`DatasetBuilder.build` assigns splits over **`clean_version`**, which for any excluding intervention is the *filtered* subset. Splits are therefore assigned after filtering:

| Run | Episode pool split | Test episodes (0.8/0.1/0.1, quantile) |
|---|---|---|
| `baseline_full` | 48 | 5 (README: 38/5/5) |
| `reduced_data` (fraction 0.5) | 24 | about 2–3, and **not** a subset of the baseline's 5 |
| `imbalanced` (drop "short", about 16 of 48) | about 32 | about 3, with **no short episodes at all** |
| `rebalanced` (weights only) | 48 | 5; plausibly identical to the baseline's if the builder is deterministic in pool and seed (unverified) |

Consequences:
1. **`reduced` vs `baseline_full` compares MSE on different test episodes, of different sizes.** With 2–3 test episodes, the per-episode spread (README: SO-100 per-episode MAE 0.085–0.111 across five episodes) is of the same order as the effect you'd be trying to detect.
2. **The imbalanced run can't measure the regime it removed.** Its test split contains no short episodes, so there's no `length_short` slice for it to fail on. The experiment designed to "give the rebalanced experiment a real weak regime to target" (its `objective:`) can't observe that weak regime.
3. **The feedback artifact's recommended action**, "retrain, then compare against this run on the same fixed test slice" (Module 8), is impossible for any excluding intervention.

**What the docs say.** `docs/EXPERIMENTS.md` §6 *does* acknowledge this, and that's to its credit: "Comparing two different dataset interventions is, necessarily, a comparison across two different `training/vN` versions' test splits, since an intervention can change which episodes are assigned to `test`". Two corrections to that sentence:
- It isn't *necessary*. It follows from filtering before splitting. Split the full pool **once** (a frozen `training/v_base`), then apply interventions *only to its train split* (filter or weight train episodes, inherit val and test unchanged). Every intervention is then scored on the identical test set by construction. The engine's lineage model supports this shape: a transform whose input is the split training version.
- It says "each experiment's `dataset.json` records exactly which ... episode identifiers were evaluated". **It doesn't.** `dataset.json`'s `episode_ids` is the *whole* selected pool (train + val + test). `split_integrity.json` stores split *sizes* and overlaps, not membership. The evaluated test episodes can only be recovered from `predictions/test_outcomes.jsonl` (one `episode_id` per frame, Module 11), or by reopening the training version.

This is the gap to raise yourself in an interview. Finding it in your own experimental design, and having the "split once, intervene on train only" fix ready, is worth more than any of the features.

---

## 4.6 Honest gaps to own before an interviewer finds them

1. **Filter-then-split moves the test set.** Every excluding intervention is evaluated on a different, smaller test split than `baseline_full`, and the imbalanced run's test split doesn't contain the removed regime. *Consequence:* the core quantity-vs-coverage comparisons aren't like-for-like, and the imbalanced → rebalanced story can't be measured as designed.
2. **The zero-fraction anchor fails** (the resolver returns `[]`, the builder raises), despite the comment and a passing unit test. *Consequence:* a sweep including 0.0 crashes, and the test gives false confidence.
3. **Imbalanced means total removal, not under-representation**, contrary to the config template's hypothesis. *Consequence:* the experiment measures out-of-distribution generalization, so it must be described that way.
4. **Length buckets differ between intervention (whole-dataset terciles) and evaluation (within-split terciles)**, and the "same method as profile.py" docstring is false (profile has no tercile code). *Consequence:* "short" means different things at train time and test time.
5. **Tied lengths collapse the buckets** (`<= t1` puts ties in "short"). *Consequence:* on a dataset with uniform episode lengths, imbalancing "long" or "medium" silently runs full data.
6. **Fraction draws aren't nested across fractions.** *Consequence:* data-efficiency curves carry extra between-point variance from swapping episodes. Prefixes of a single permutation fix it.
7. **Explicit `episode_ids` and `episode_weights` aren't validated** against the dataset (unknown IDs, duplicates, non-positive weights). *Consequence:* engine-side behaviour is undefined here; duplicates create needless versions.
8. **The rebalanced "limitation in every report" claim is false** (`EXPERIMENTS.md` §4). *Consequence:* the report for a rebalanced run omits the key caveat about what the result can't show.
9. **`dataset.json` doesn't record test membership**, contrary to `EXPERIMENTS.md` §6. *Consequence:* the evaluated episodes are recoverable only from the predictions file.
10. **Version allocation isn't atomic, failed runs orphan versions, and identical requests always mint new versions.** *Consequence:* catalog clutter, plus a possible version collision under concurrency (unverified engine behaviour).
11. **The module docstring is stale** (it says `task_id` is "the one grouping key"). *Consequence:* the docs disagree with themselves; the code and `EXPERIMENTS.md` use the length bucket.
12. **A misleading error for unsupported rebalance variables** ("no episodes with X=Y"). *Consequence:* it slows down diagnosis.
13. **Weights are recorded for val and test episodes too.** *Consequence:* `dataset.json` overstates which episodes were upweighted in training.

---

## The shape to carry into Module 5

```
 DatasetIntervention ──▶ build_intervention_dataset(dataset_id, input_version, splits)
                          │ catalog.get_manifest  (LookupError if unknown)
                          │ catalog.get_episodes  → all_ids, task_of, length_bucket_of (whole-dataset terciles)
                          │
        ┌─────────────────┴─────────────────────────┐
        │ rebalanced                                 │ full / fraction / reduced / imbalanced / custom
        │  weights = explicit | {target: factor}     │  subset = resolve_episode_subset(...)
        │  raise if empty                            │  raise if empty   (← kills frac=0.0)
        │  clean = input                             │  subset == all ? clean = input
        │                                            │                : clean/vK = EpisodeSubsetFilter
        └─────────────────┬─────────────────────────┘
                          ▼
            training/vM = DatasetBuilder.build(clean, splits, seed=42, quantile, weights)
                          │  splits assigned over the FILTERED pool  ← test set moves per intervention
                          ▼
            InterventionResult(clean_version, training_version, episode_ids (pool, not test),
                               excluded_episode_ids, episode_weights, selection_reason, build_summary)
```

---

## Drill — answer from memory, files closed

1. Trace what happens to the catalog when you run `reduced.yaml` against `so100_pickplace@clean/v1`. Which versions are created, and what does the lineage record?
2. *Why didn't you just* pass a list of episode IDs to `fit()` instead of creating new dataset versions?
3. *What breaks in production:* someone runs `run_data_fraction_sweep` with `[0.0, 0.25, 0.5, 1.0]`.
4. Is `reduced_data`'s test MSE comparable to `baseline_full`'s? Explain precisely, and say what you'd change.
5. *Interviewer framing:* "Tell me about a flaw in your own experimental design."

---

# Appendix A — Model answers to the drill

**A1. Running `reduced.yaml`.**
"The intervention is `reduced` at fraction 0.5 with seed 0. The module reads the 48 episodes of `clean/v1` from the catalog, sorts the IDs so the draw doesn't depend on row order, and uses a seeded NumPy generator to draw 24 without replacement. Because that's a strict subset, it allocates the next clean version, say `clean/v2`, and runs the engine's `EpisodeSubsetFilter` transform from `clean/v1` to `clean/v2`. The lineage for that transform records the 24 kept and 24 dropped IDs and the reason string, 'seeded random 50% draw over all episodes (seed=0)'. Then it allocates the next training version and calls the engine's builder, which assigns train/val/test over those 24 episodes with split seed 42 in quantile mode. So the catalog gains two versions, and `rde version lineage` on the training version walks back through the filter to raw. The unit test checks that the transform's recorded kept and dropped IDs equal exactly what my code computed."

*Why that answer lands:* it's concrete down to the version names and the recorded reason, and it ends with how the claim is verified, not just asserted.

**A2. Why not pass episode IDs to `fit()`?**
"Because then the experiment's data would exist only as an argument in one process. The question 'what exactly did this model train on' would depend on my run directory surviving and being read correctly. As engine versions, every intervention is an immutable, named dataset with lineage back to raw episodes and a recorded reason for every exclusion, so the data has an identity independent of my code. The cost is real. Every run writes to the shared catalog, identical requests mint duplicate versions, failures leave orphaned versions, and, the big one, because I filter before splitting, the split is recomputed over each filtered pool. If I were redoing it, I'd keep versions but change the order: split the full pool once, then express interventions as transforms on the train split only."

*Why that answer lands:* it defends the decision on provenance grounds, then volunteers that the same decision causes the design flaw, and gives the fix without abandoning the principle.

**A3. A sweep with 0.0.**
"The 0.0 point fails before training. The resolver honours zero, returning an empty list, and there's a unit test that checks exactly that. But the only caller, `build_intervention_dataset`, raises 'resolved to zero episodes' on an empty subset, so the first sweep point throws and the sweep stops. The unit test passes because it tests the helper in isolation, not the path the sweep uses. Even if it got through, a zero-episode version can't be split or trained on. What a zero-data anchor should really mean is 'evaluate the pretrained base checkpoint with no fine-tuning', and that needs its own code path, not an empty filter. Separately, the other fractions' subsets aren't nested, so I'd draw them as prefixes of a single permutation."

*Why that answer lands:* it identifies the contradiction between the helper and the caller, explains why the test missed it, and redefines the experiment correctly. That shows judgement, not just bug-spotting.

**A4. Is `reduced` comparable to `baseline_full`?**
"Not on a like-for-like basis. Splits are assigned after the filter, so the baseline is scored on 5 test episodes out of 48, and the reduced run on about 2 or 3 out of 24, which aren't even guaranteed to be a subset of the baseline's 5. With per-episode error varying by roughly 25% across episodes, a difference on 2–3 episodes is mostly noise about which episodes landed in test. The docs acknowledge the test sets differ, but call it necessary, and it isn't. I'd split the full 48 once and freeze that, then apply every intervention only to the train split, so every run is scored on the same 5 test episodes. And because those 5 are few, I'd report per-episode results and run several training seeds before claiming a quantity effect."

*Why that answer lands:* a precise "no", with the mechanism, the magnitude of the problem and the fix. Correcting your own docs' "necessarily" is exactly the kind of rigour the question is fishing for.

**A5. "Tell me about a flaw in your own experimental design."**
"The data interventions are implemented as engine transforms that filter episodes, and the training-set builder then assigns splits over whatever's left. That gives perfect lineage, but it means each intervention gets its own test split. So 'reduced vs full' compares different test episodes, and worse, the imbalanced run, which drops every short episode, has no short episodes in its test set, so it can't show the failure it was designed to create. I found it by tracing what the builder's split input actually was, not by looking at results. The fix keeps the provenance model and changes the order: split once on the full pool, then intervene only on the train split, so every experiment shares one frozen test set. I'd also define the length buckets once, on the full dataset, and use the same cut points for the intervention and the evaluation slice, because today they're computed differently."

*Why that answer lands:* it's a real, non-trivial flaw, explained mechanistically, discovered by reasoning, with a fix that preserves the system's core principle. That's the most valuable answer type in a research-engineering interview.

---

# Module 4 Summary — the one page to revise

**What this module is.** The translator from a declared data intervention to engine operations: an optional `EpisodeSubsetFilter` transform into `clean/vK`, then a `DatasetBuilder.build` into `training/vM` with splits and optional sampling weights. Provenance is excellent; filtering before splitting makes cross-intervention test sets incomparable.

**Five claims to make cold**
1. Interventions are engine transforms with recorded reasons and dropped IDs, never local copies, and a test verifies the lineage matches.
2. Rebalancing uses sampling weights over existing episodes. Nothing is duplicated or synthesized, so it can't close a coverage gap.
3. "Imbalanced" removes a regime entirely. It's an out-of-distribution test, not under-representation.
4. Splits are assigned after filtering, so every excluding intervention has its own smaller test set. The fix is to split once and intervene on train only.
5. The length variable exists because `so100_pickplace` has one task. The intervention and evaluation buckets use different cut points.

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Engine versions per intervention | IDs passed to `fit` | immutable identity, lineage, reasons | catalog churn, orphans, duplicate versions |
| Filter → build (split) | split → intervene on train | uses existing transform/build shape | test set changes per intervention |
| Sampling weights | duplication / synthesis | honest about no new data | repeated frames; sampler unverifiable here |
| Length-bucket terciles | a `task_id` variable | only varying attribute in so100 | tie collapse; mismatched with eval terciles |
| Seeded `rng.choice` over sorted IDs | unseeded / unsorted | deterministic per seed | not nested across fractions |
| Reuse input when subset = all | always transform | no pointless versions | hides silent no-op imbalances |

**Volunteer unprompted:** the moving test set; the 0.0 anchor that can't run despite its test; total removal vs under-representation; the bucket definitions don't match; `EXPERIMENTS.md` overclaims the rebalanced limitation and `dataset.json` test membership.

**Numbers and names:** so100 has 48 episodes, 326–475 frames, one task; terciles at the 33.3/66.7 percentiles; `< 3` episodes → all "medium"; default oversample 3.0; split seed 42, quantile; `EpisodeSubsetFilter`, `TransformPipeline`, `DatasetBuilder.build`, `next_version`, `VersionStage.CLEAN/TRAINING`; README lineage `raw/v1 → clean/v1 (50→48, dropped [9, 31]) → training/v6`.

**Sentences worth memorizing**
- "An intervention is a dataset version with a reason, not an argument to a training call."
- "I filter before I split, so each intervention gets its own test set. The fix is to split once and intervene on train only."
- "Dropping every short episode isn't imbalance; it's an out-of-distribution test, and its test set can't contain the regime it removed."
- "Rebalancing redistributes attention over data I already have; it can't close a coverage gap, and I don't claim it does."
