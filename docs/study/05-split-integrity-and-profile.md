# Module 5 — Pre-training guards: split integrity and the dataset profile

**Files covered:** `src/model_training/data/split_integrity.py`, `src/model_training/data/profile.py`

**Where we are:** Module 4 produced a `training/vN` version. Before a single gradient step, the runner does two things to it: it re-checks that the splits don't leak, and it profiles the version (statistics, coverage, QC findings) for the report. Both are framed in the docs as *guards*, evidence that the data is fit to train on. This module asks how much each one actually guards.

**The one-line thesis:** The split check is honest code that checks the one property that can't fail, and passes the properties that *can* fail (unassigned episodes, empty splits). The profile is careful about saying what it can't measure, but its headline "coverage" number is normalized by the number of *occupied* bins, so it rewards a dimension that uses two bins exactly as much as one that uses all ten.

---

## 5.1 Prerequisites — the absolute basics

- **Data leakage (split leakage).** Information from the evaluation data reaching training. At episode level, that's the same episode ID in two splits. At content level, it's *different* episodes that are near-duplicates (recorded twice, or the same session and scene with trivially different motion). ID-level checks can only catch the first kind.
- **Partition.** A division of a set into disjoint subsets that together cover it. "Splits form a partition" means: no overlap, *and* nothing left over.
- **Defense in depth / independent verification.** Re-checking an invariant with code that doesn't share a path with the code that established it. It's only valuable if the invariant can actually be violated in the representation you check.
- **Tautological check.** A check whose failure condition can't be represented by the data it reads. It always passes, and it proves nothing.
- **Test double (fake).** A minimal stand-in object with just the methods a function under test calls. `_FakeView` in the tests is one.
- **Shannon entropy.** For a probability distribution `p₁…p_k`, `H = −Σ pᵢ log₂ pᵢ` bits. It's 0 when all mass is on one outcome, and `log₂ k` when spread evenly over `k` outcomes.
- **Normalized entropy (evenness).** `H / H_max`. The value of a normalized entropy depends entirely on **what you pick as H_max**: `log₂(number of possible bins)` measures how much of the possible space is used evenly, while `log₂(number of occupied bins)` measures only how even the occupied bins are (ecologists call this *Pielou's evenness*). The two answer different questions.
- **Equal-width histogram.** Split `[min, max]` into `B` equal intervals and count values per interval. Here `B = 10` (`_HIST_BINS`).
- **Coefficient of variation (CV).** `std / mean`, a scale-free measure of spread. *Not* `(max − min) / mean`. For a roughly normal sample of 48 values, the expected range is about 4.5 standard deviations, so range/mean comes out around 4.5× the CV.
- **DuckDB.** An in-process analytical SQL engine that queries Parquet and Arrow directly. The engine's `explorer.sql(...)` runs SQL against a version's `frames` table (inferred from the query text here). DuckDB **list indexing is 1-based**.
- **SQL injection.** Building SQL by string interpolation lets a crafted value change the query's meaning. Identifier quoting (`"col"` with embedded `"` doubled) is the defence for column names; parameters are the defence for values.

---

## 5.2 `verify_split_integrity`

```python
class SplitLeakageError(Exception):
    """Raised when any two of train/val/test share at least one episode id."""

@dataclass
class SplitIntegrityReport:
    dataset_id: str
    version: str
    split_sizes: dict[str, int]
    train_val_overlap: list[int]
    train_test_overlap: list[int]
    val_test_overlap: list[int]
    unassigned_episodes: list[int]

    @property
    def ok(self) -> bool:
        return not (self.train_val_overlap or self.train_test_overlap or self.val_test_overlap)

    def to_dict(self) -> dict:
        return {**self.__dict__, "ok": self.ok}
```

```python
def verify_split_integrity(view: DatasetView, *, raise_on_failure: bool = True) -> SplitIntegrityReport:
    train = set(view.episode_ids(split="train"))
    val = set(view.episode_ids(split="val"))
    test = set(view.episode_ids(split="test"))
    all_ids = {e.episode_id for e in view.episodes}

    report = SplitIntegrityReport(
        ...,
        split_sizes={"train": len(train), "val": len(val), "test": len(test)},
        train_val_overlap=sorted(train & val),
        train_test_overlap=sorted(train & test),
        val_test_overlap=sorted(val & test),
        unassigned_episodes=sorted(all_ids - train - val - test),
    )
    if raise_on_failure and not report.ok:
        raise SplitLeakageError(...)
    return report
```

**What it does:** it reads split membership back from the materialized view, intersects every pair, and raises on any overlap. The runner calls it with the default `raise_on_failure=True`, right after opening the training version (Module 11), and writes `split_integrity.json`. The README quotes that file as evidence: "Before any training step, an independent check confirms that no episode appears in more than one split, and the run aborts if one does."

**Why it's framed as valuable** (module docstring): the builder "already guarantees this by construction", and this check "exists to catch a *real* leak regardless of where it would come from -- including a bug this project's own intervention builder introduces -- by re-deriving the partition independently".

**What it can actually catch.** Look at how split membership is stored. `evaluation/offline.py` reads it as `e.attributes.get("split") == base_split` (Module 7). So **each episode carries one `split` attribute with one value.** An episode can't simultaneously have `split = "train"` and `split = "test"` in that representation. If `view.episode_ids(split=s)` filters on that attribute (very likely, though I can't verify the engine), then `train ∩ test` is *empty by construction of the data structure*, not just by the builder's correctness. The test file says as much in its fake's docstring:

> "used to prove the checker actually detects an engineered overlap, since a real `DatasetView` built by rde's own (correct) builder cannot produce one."

So **the overlap check is close to tautological**. It verifies a property the storage format makes unrepresentable. It isn't harmful, and it's cheap (three set operations over at most 48 IDs), but "independent re-verification" overstates it. The only way it could fire is if the engine stored membership as three separate ID lists, which isn't how this repository reads it.

**What can actually go wrong, and passes:**
1. **Unassigned episodes don't fail the check.** `ok` ignores `unassigned_episodes`. An episode with no split attribute is silently excluded from train, val *and* test, so it's lost data, and the report says `"ok": true`. The partition property ("covers the set") isn't enforced, only disjointness.
2. **Empty splits don't fail the check.** A version with `val` empty (for example from `splits.val: 0.0`, which Module 2 showed validates) or a tiny pool after a `reduced` intervention passes. An empty val split disables checkpoint selection (Module 6), and an empty test split makes the headline metric meaningless. Neither is caught here.
3. **Split sizes aren't compared with the config.** The report records `split_sizes` but never checks them against `config.splits`. A builder that put 46/1/1 for a requested 0.8/0.1/0.1 would pass.
4. **Content-level leakage is out of scope**, and that's the leak that actually matters in single-setup robot datasets. Episodes from one session in one scene are highly correlated across splits. The README itself concedes nearest-neighbour retrieval "is a strong baseline when test episodes come from the same setup". Nothing here (or anywhere) detects near-duplicate episodes across splits.

The **`raise_on_failure=False` escape hatch** is used only by tests. The production path always raises, which is right. The `test_split_leakage_report_can_be_inspected_without_raising` test exists to show the report is introspectable.

**The tests use a hand-written fake** (`_FakeView`, which exposes only `episodes` and `episode_ids`). That's good test design: the only way to exercise the failure branch is to construct an impossible view. It also proves the point above. **The failure branch is reachable only with a fake.**

**What I'd change.** Keep the disjointness check (it's free) and make `ok` mean the property that can fail:

```python
ok = no_overlaps and not unassigned and all(split_sizes[s] > 0 for s in ("train", "val", "test"))
```

Also compare sizes against the requested fractions within ±1 episode for `quantile` mode, and frame the README sentence as "verifies the version is a complete, non-empty, disjoint partition". Content-level leakage would need episode embeddings or state-trajectory hashing. That's out of scope; name it as a known limitation.

---

## 5.3 `build_dataset_profile`, part 1: reuse, don't recompute

```python
explorer = DatasetExplorer(settings=settings, catalog=catalog)

overview = explorer.overview(dataset_id, version)
durations = explorer.duration_distribution(dataset_id, version)
tasks = explorer.task_summary(dataset_id, version)
modalities = explorer.modality_summary(dataset_id, version)
dims = explorer.dimensionality(dataset_id, version)
qc = explorer.qc_summary(dataset_id, version)
lineage = explorer.lineage(dataset_id, version)
```

**Decision: package the engine's explorer outputs instead of recomputing.**
- *Why:* the module docstring says "this module's only real job is packaging those into the profile shape and adding the two things `DatasetExplorer` does not compute: coverage/diversity scores and the train/val/test split distribution". It's a single source of truth for statistics, so the profile and `rde`'s own CLI can't disagree.
- *Cost:* the profile's correctness is the explorer's correctness, and the profile's *shape* depends on undocumented dict keys (`durations["duration_s"]["max"]`, `dims[k]["per_dimension"][i]["min"]`, `lineage["chain"][-1]["inputs"]`). An engine change to any of those keys raises `KeyError` here, and nothing pins the engine version (Module 1).
- `explorer.overview` on an unknown version raises `LookupError` (`test_profile_of_unknown_version_raises`), so that error surfaces before any of the fallbacks below are reached.

```python
parent_version = lineage["chain"][-1]["inputs"][0] if lineage["chain"] and lineage["chain"][-1]["inputs"] else None
```

The parent is the *first* input of the *last* lineage step. That's correct for this project's single-input transforms and builds. `test_profile_to_dict_is_json_shaped` asserts `parent_version == "clean/v1"` for the fixture's training version. For a multi-input operation (a merge), it would silently report only one parent.

```python
manifest = catalog.get_manifest(dataset_id, version)
contract = manifest.attributes.get("training_contract") if manifest else None
if not contract:
    from rde.dataset_builder.builder import build_training_contract
    contract = build_training_contract(manifest) if manifest else {"observation": {}, "action": {}}
```

- **Fallback contract:** a `clean/vN` version has no stored training contract (only builds create one, by inference), so the profile derives one on the fly. That's why `mtrain profile` works on `clean/v1`, which is exactly what `EXPERIMENTS.md`'s workflow needs before choosing an imbalance variable.
- The `if manifest else {...}` empty-contract branch is effectively **unreachable**: a missing manifest means an unknown version, and `explorer.overview` already raised `LookupError` above.

```python
act_key = contract.get("action", {}).get("key")
state_keys = [k for k, v in contract.get("observation", {}).items() if v.get("plane") == "inline"]
image_keys = [k for k, v in contract.get("observation", {}).items() if v.get("plane") == "external"]
```

**Inline vs external is the modality test used everywhere** (Module 1 vocabulary). Note that "state" means *every* inline modality. For ALOHA that's `state.joint` *and* `force.effort` (the smoke test's 14 + 14 = 28 dimensions). So "state-space coverage" for ALOHA includes effort sensors, not just joint positions.

### Splits: "unassigned" isn't a split

```python
if manifest is not None:
    engine = DataEngine(settings)
    try:
        with engine.get_dataset(dataset_id, version) as view:
            raw_splits = view.splits()
        if set(raw_splits) - {"unassigned"}:
            splits = raw_splits
    except Exception as exc:
        notes.append(f"could not resolve splits for {dataset_id}@{version}: {exc}")
if not splits:
    notes.append(f"{version!r} carries no split assignment (it is not a training/vN version) -- "
                 "splits reported empty, not fabricated")
```

- **The `"unassigned"` bucket is the subtle line.** The engine's `view.splits()` reports episodes with no split attribute under the key `"unassigned"`. A clean version is *entirely* unassigned, which means "no split exists yet", not "a split called unassigned". Without this line, the profile of `clean/v1` would report `{"unassigned": 48}` as if it were a real partition. It's tested (`test_profile_on_clean_version_has_no_splits`).
- **But on a training version with *some* unassigned episodes, `raw_splits` keeps the `"unassigned"` key**, so the profile reports it as a fourth split alongside train, val and test, while `split_integrity.json` reports it separately and still says `ok`. That's a second place unassigned episodes would show up and still not stop anything.
- **The broad `except Exception` plus a note** means profiling never fails because of the split lookup. Then the "no split assignment (it is not a training/vN version)" note is *also* appended, which is false if the cause was an exception on a training version. Two notes, one of them misleading.
- **The `with ... as view`** context manager is used here, while the runner and CLI use explicit `try/finally: view.close()`. Both are correct; the codebase just isn't consistent.

---

## 5.4 `build_dataset_profile`, part 2: coverage

```python
_HIST_BINS = 10

def _shannon_entropy_bits(counts: list[int]) -> float | None:
    total = sum(counts)
    if total <= 0:
        return None
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)

def _normalized_entropy(counts: list[int]) -> float | None:
    """0 (all mass in one bucket) .. 1 (uniform across every non-empty bucket)."""
    h = _shannon_entropy_bits(counts)
    if h is None:
        return None
    k = sum(1 for c in counts if c > 0)
    if k <= 1:
        return 0.0
    return round(h / math.log2(k), 4)
```

**The normalization is by `k`, the number of *occupied* bins, not by `_HIST_BINS`.** Work two examples through:

| Dimension's 10-bin counts | H (bits) | k | Normalized entropy reported | log₂(10)-normalized |
|---|---|---|---|---|
| `[50, 0, 0, 0, 0, 0, 0, 0, 0, 50]` (bimodal, e.g. a gripper that's open or closed) | 1.0 | 2 | **1.0** | 0.30 |
| `[10]*10` (uniform across the whole range) | 3.32 | 10 | **1.0** | 1.0 |
| `[90, 10, 0, …]` | 0.47 | 2 | 0.47 | 0.14 |

**A dimension using two of ten bins scores the same "perfect coverage" as one spread across all ten.** The metric is *evenness among occupied bins* (Pielou's J), not coverage. Range-filling, which is what "coverage" means in ordinary speech and in the report's label "action-space coverage", is ignored, because the empty bins *are* the gaps and they're excluded from the denominator.

**What the docs say:**
- The function's docstring is accurate: "uniform across every non-empty bucket".
- The methodology string written into every profile is half accurate: "0 = all mass in one bin, 1 = uniform across every non-empty bin. **Measures how evenly the *observed* range is used**". The first sentence is right. The second is wrong: the observed range is `[min, max]` split into ten bins, and empty interior bins are exactly unused range.
- The module docstring says coverage is "a real, computable proxy for 'how much of the observed range gets used'". That's the same overclaim.

**Fix (one token):** `return round(h / math.log2(_HIST_BINS), 4)` for per-dimension coverage. Keep the evenness variant if you want it, under its real name. For *task* entropy, `k`-normalization is arguably fine, because there's no "possible task" denominator.

**Edge behaviour, for completeness:**
- `k <= 1 → 0.0`: a single task (so100) gives task entropy and instruction diversity of `0.0`. That's correct and honest, and it's how the profile *shows* that `task_id` can't be the imbalance variable.
- `total <= 0 → None`: an empty version produces `None`, not a fake number.
- `round(..., 4)`: four decimals, display precision only.

### The per-dimension histogram query

```python
def _dim_coverage(explorer, dataset_id, version, column, index, lo, hi):
    if hi <= lo:
        return {"normalized_entropy": None, "reason": "zero-range (constant) dimension"}
    col = '"' + column.replace('"', '""') + '"'
    sql = (
        f"SELECT LEAST({_HIST_BINS - 1}, GREATEST(0, "
        f"CAST(FLOOR({_HIST_BINS} * ({col}[{index + 1}] - {lo}) / ({hi} - {lo})) AS INTEGER))) "
        f"AS bucket, COUNT(*) AS n FROM frames GROUP BY 1"
    )
    try:
        rows = explorer.sql(dataset_id, version, sql)
    except Exception as exc:  # pragma: no cover
        return {"normalized_entropy": None, "reason": f"histogram query failed: {exc}"}
```

Every non-obvious piece:
- **`hi <= lo` guard:** without it, a constant dimension divides by zero inside DuckDB. It's reported as `None` with a reason, which is the project's "unavailable, not fabricated" pattern applied well.
- **Identifier quoting** (`'"' + column.replace('"', '""') + '"'`): column names like `action.joint` contain a dot, which DuckDB would otherwise parse as `table.column`. Doubling embedded quotes is the correct SQL escape. This is *necessary for correctness*, not just security.
- **`[index + 1]`:** DuckDB lists are 1-indexed; Python's `enumerate` is 0-indexed. Without the `+ 1`, every dimension would be histogrammed as its neighbour, and the last would read out of bounds (NULL).
- **`LEAST(9, GREATEST(0, …))` clamp:** a value exactly equal to `hi` computes bucket 10, and the clamp folds it into bucket 9. It also silently folds anything *outside* `[lo, hi]` into the edge bins. If the explorer's min/max were ever computed on a different population than the `frames` table (for example, raw vs normalized values), every value would pile up in bins 0 and 9 with no error.
- **`lo` and `hi` are interpolated as Python float literals.** That's safe from injection because they're numbers from the engine. A `nan` or `inf` would produce invalid SQL and fall into the `except`. Parameter binding would be cleaner.
- **NULL values** produce `bucket IS NULL`, and `if b is not None` skips them. Missing values are silently excluded from coverage. That's defensible, but it isn't reported.
- **Cost:** one full scan of `frames` per dimension. For ALOHA that's 14 action plus 28 state dimensions, so 42 scans per profile. That's seconds at this data size. The video plane isn't touched, which matters: only the inline columns are read.

### Two smaller mislabels

```python
"episode_length_cv": (
    round(length_frames.get("mean", 0) and
          (float(durations["duration_s"]["max"] - durations["duration_s"]["min"])
           / max(durations["duration_s"]["mean"], 1e-9)), 4)
    if durations else None
),
```

- **It isn't a coefficient of variation.** It's `(max − min) / mean` of duration. For a roughly normal sample of 48 episodes, range ≈ 4.5σ, so this reports about **4.5× the true CV**. The label invites exactly the wrong reading.
- **`length_frames.get("mean", 0) and (...)`** uses Python's `and` as a guard: if the frames mean is 0 (or missing), the value is `0`; otherwise it's the ratio. It guards on the **frame** mean but divides by the **duration** mean. That works only because both are zero together. The `max(..., 1e-9)` guard is then redundant. It's convoluted for what should be `statistics.pstdev(lengths) / mean(lengths)`.
- **`task_coverage.normalized_entropy` and `instruction_diversity.diversity_normalized_entropy` are the same number**, computed twice from `task_episode_counts`. The profile presents one fact as two metrics.

### The parts that are genuinely good

```python
"visual_diversity": {"status": "unavailable",
                     "reason": "requires an image embedding/similarity model; ..."},
"episode_diversity": {"status": "partial", ..., "reason_partial": "limited to task-id and length variability; ..."},
```

```python
missing_data = {
    "source": "rde QC finding records (qc_records) for this version -- only failing/warning "
              "checks are persisted as rows (passing checks leave no row), so a check's "
              "absence below means either it found nothing or has not been run against "
              f"{dataset_id}@{version} specifically; cross-check with `rde obs quality`",
    ...
}
```

- **Refusing to compute visual diversity** without an embedding model, and saying so in the data, is the right call. A pixel-variance "diversity score" would look rigorous and mean nothing.
- **The QC caveat is the best-written line in the module.** It names a real ambiguity in absence-of-evidence ("no row" means "passed *or* never ran") that most tools hide. Quote it in an interview as an example of honest metadata.
- `variation_note` states that the schema has no variation field and that `task_id` and length are proxies "rather than an invented 'variation' axis". It's consistent with Module 4.

### What population is profiled

The runner profiles **`training_version`**, the whole post-intervention version, *including val and test episodes* (Module 11). So the report's "action-space coverage" and "Dataset statistics (from profile)" describe train ∪ val ∪ test, not what the model trained on. For a "quantity vs coverage" study, you'd want coverage of the **train split**, ideally next to coverage of the **test split**, so you can say whether test states fall inside the trained range.

**`to_dict` lists all 20 fields by hand** instead of `dataclasses.asdict(self)`. Add a field and forget to add it to `to_dict`, and it silently vanishes from `dataset_profile.json`. `asdict` would deep-copy nested dicts, which is a cost here, but `{f.name: getattr(self, f.name) for f in fields(self)}` would avoid both problems.

---

## 5.5 Honest gaps to own before an interviewer finds them

1. **The overlap check is near-tautological.** Split is a single attribute per episode, so an overlap can't be represented, and the failure branch is reachable only with a fake view. *Consequence:* "an independent check confirms no leakage" overstates what's verified.
2. **`ok` ignores unassigned episodes, empty splits, and sizes vs config.** *Consequence:* the properties that *can* actually fail (lost episodes, an empty val split that disables selection) pass with `"ok": true`.
3. **No content-level leakage detection.** *Consequence:* near-duplicate episodes from the same session can sit in train and test. This is the realistic leak in single-setup datasets, and it's consistent with nearest-neighbour retrieval being so strong.
4. **"Coverage" is normalized by occupied bins.** *Consequence:* a two-bin bimodal dimension scores 1.0, the same as a uniform one. "Action-space coverage" in every report measures evenness, not range use, and the methodology string misdescribes it.
5. **`episode_length_cv` is range/mean, about 4.5× the true CV at n = 48.** *Consequence:* a mislabelled number in every profile.
6. **The profile covers train ∪ val ∪ test, not the train split.** *Consequence:* it can't answer "did training cover the test states?", which is what a coverage study needs.
7. **Dependence on undocumented explorer dict keys, with no engine pin.** *Consequence:* an engine refactor breaks profiling with `KeyError`.
8. **The histogram clamp silently folds out-of-range values into the edge bins, and NULLs are dropped unreported.** *Consequence:* a min/max mismatch produces confident but wrong coverage.
9. **Duplicate metric** (task entropy = instruction diversity). *Consequence:* one fact is presented as two.
10. **A misleading note after a split-lookup exception** ("it is not a training/vN version"). *Consequence:* it misdiagnoses.
11. **`to_dict` is hand-listed.** *Consequence:* new fields can silently disappear from the persisted profile.
12. **`parent_version` takes the first input only.** *Consequence:* wrong for future multi-input operations.

---

## The shape to carry into Module 6

```
 training/vN (DatasetView)
   │
   ├─ verify_split_integrity(view)                 runner: raise_on_failure=True
   │     train/val/test = view.episode_ids(split=…)
   │     overlaps  → SplitLeakageError   (unrepresentable if split is one attribute/episode)
   │     unassigned / empty / size-vs-config → NOT checked → ok=true
   │     → split_integrity.json
   │
   └─ build_dataset_profile(dataset_id, training_version)     (all splits pooled)
         DatasetExplorer: overview · durations · tasks · modalities · dims · qc · lineage
         contract = stored | build_training_contract(manifest)
         splits   = view.splits() minus pure-"unassigned"
         coverage = per-dim DuckDB 10-bin histogram → H / log2(#occupied bins)   (evenness!)
         visual_diversity = "unavailable" (honest)   episode_length_cv = range/mean (mislabel)
         → dataset_profile.json → report + task/length figures
```

---

## Drill — answer from memory, files closed

1. What exactly does `verify_split_integrity` check, and what would have to be true about the storage format for it to ever fail on a real dataset?
2. *Why didn't you just* trust the builder's guarantee and skip the check, or, alternatively, why not make the check stronger?
3. Compute the profile's reported normalized entropy for a gripper dimension whose values fall half in the lowest bin and half in the highest. What's wrong with that number?
4. *What breaks in production:* a builder bug leaves three episodes with no split attribute. What happens across the pipeline?
5. *Interviewer framing:* "How do you know your test set isn't leaking into training?"

---

# Appendix A — Model answers to the drill

**A1. What the split check verifies.**
"It reads the train, val and test episode IDs back from the materialized training version, intersects each pair, and raises `SplitLeakageError` if any intersection is non-empty. It also records split sizes and any episode in none of the three. For the overlap branch to fire on real data, the engine would have to be able to store an episode in two splits at once. But split is a single attribute on each episode, and the evaluation code reads it as `attributes['split']`, so an overlap isn't representable. The test file says so itself: it uses a fake view because the real builder can't produce an overlap. So the overlap check is cheap insurance against a representation change, not a meaningful independent verification."

*Why that answer lands:* it describes the check precisely, then shows you've reasoned about *whether it can fail*. That's the move that separates real validation from ceremony.

**A2. Skip it or strengthen it?**
"I'd keep it, since three set intersections over 48 IDs are free and it documents the invariant in the run directory, but I'd make it check what can actually go wrong. Today `ok` ignores unassigned episodes and empty splits, and those are the realistic failures. An empty val split silently disables checkpoint selection, and unassigned episodes silently vanish from training and evaluation. So `ok` should mean a complete, non-empty partition whose sizes match the configured fractions within one episode. The harder leak, near-duplicate episodes from the same recording session ending up on both sides, needs content-level checks like trajectory hashing or embedding similarity. I'd list that as a known limitation rather than imply the ID check covers it."

*Why that answer lands:* it rejects both extremes, specifies the stronger invariant exactly, and distinguishes ID-level from content-level leakage. That's a sign of real data-quality experience.

**A3. The gripper example.**
"Two occupied bins with equal mass give 1 bit of entropy. The code normalizes by the log of the number of *occupied* bins, log₂ 2 = 1, so it reports 1.0, the same score as a dimension spread uniformly across all ten bins. Normalized by the ten possible bins it would be about 0.30. So the number measures evenness among occupied bins, not how much of the range is used. The empty bins are exactly the coverage gaps, and they're excluded from the denominator. The function's docstring says 'uniform across every non-empty bucket', which is accurate, but the methodology text and the report label call it coverage of the observed range, which it isn't. The fix is to normalize by log₂ of the bin count."

*Why that answer lands:* you do the arithmetic out loud, find the conceptual error, separate the accurate docstring from the inaccurate label, and give a one-token fix. Interviewers love a worked number.

**A4. Three unassigned episodes.**
"The split check computes them as unassigned but still reports `ok: true`, so nothing stops. The training loop asks for train IDs and never sees them. Validation and test ask for their splits and never see them. So three episodes silently drop out of the experiment. The profile, which runs over the whole version, would include them in its statistics, and would also show an extra 'unassigned' split, so the profile and the actual training data disagree. `dataset.json` would still list them among the selected episodes. The only place you'd notice is comparing split sizes against the episode count. The fix is to make unassigned episodes fail the integrity check, because a partition that doesn't cover the set isn't a partition."

*Why that answer lands:* it traces the failure through every downstream consumer, including the inconsistency between artifacts, and ties the fix to a crisp definition.

**A5. "How do you know your test set isn't leaking into training?"**
"At the episode level: splits are assigned per episode, never per frame, so neighbouring near-identical frames can't straddle the boundary, and each episode carries exactly one split label. A check before training reads the splits back and confirms they're disjoint. Selection uses validation only; the trainer hard-codes the val split for scoring. The test set is touched once, after a checkpoint is chosen. At the content level I can't claim it isn't leaking. Both datasets are single-setup recordings, so test episodes are close relatives of training episodes, and you can see it in the results: frame-level nearest-neighbour retrieval ties fine-tuned SmolVLA on SO-100. That's the honest framing. These are in-distribution held-out episodes, not a generalization test, and I'd add a held-out-session or held-out-scene split before claiming generalization."

*Why that answer lands:* it gives the real protections, then volunteers the limitation, backed by evidence from your own results, and names the fix that would justify a stronger claim.

---

# Module 5 Summary — the one page to revise

**What this module is.** Two pre-training steps over the new `training/vN`: a split-disjointness check that raises on overlap, and a profile that packages the engine's explorer statistics and adds per-dimension histogram "coverage" plus honest "unavailable" markers. Both are useful records, and both are weaker guards than the docs imply.

**Five claims to make cold**
1. The split check runs before training and raises on overlap. Overlap is essentially unrepresentable because split is one attribute per episode.
2. `ok` doesn't cover unassigned episodes, empty splits, or sizes vs config. Those are the failures that can actually happen.
3. ID-level checks can't detect content-level leakage. Single-setup data makes held-out episodes in-distribution, and nearest-neighbour's strength shows it.
4. Profile coverage is entropy normalized by *occupied* bins, so it's evenness, not range use. The fix is to normalize by log₂(10).
5. The profile refuses to fabricate visual diversity, and documents that missing QC rows mean "passed or never ran".

**Decisions**

| Chose | Over | Because | Cost |
|---|---|---|---|
| Re-check splits from the view | trust the builder | defense in depth, recorded evidence | checks an unrepresentable failure; misses real ones |
| Explorer reuse | recomputing stats | single source of truth with `rde` CLI | coupled to undocumented dict keys |
| 10 equal-width bins plus entropy | no coverage metric / k-NN coverage | cheap, one DuckDB pass per dimension | occupied-bin normalization; clamp hides range mismatch |
| "unavailable" visual diversity | pixel-variance proxy | no embedding model; refuse to fake | profile is numeric-only |
| Fallback contract for clean versions | require a training version | profile before choosing an intervention | none |
| Profile `training_version` | profile train split only | one call, the whole version | coverage includes test episodes |

**Volunteer unprompted:** the tautological overlap branch; `ok` ignores unassigned and empty splits; the occupied-bin normalization; `episode_length_cv` is range/mean; coverage pools all splits; there's no content-level leak check.

**Numbers and names:** `_HIST_BINS = 10`; DuckDB 1-based list index (`[index + 1]`); clamp `LEAST(9, GREATEST(0, …))`; `k <= 1 → 0.0`; ALOHA state = 28 dimensions (joint 14 + effort 14), so 42 histogram queries; range of n = 48 normal samples ≈ 4.5σ; README split 38/5/5, all `overlap: []`, `ok: true`.

**Sentences worth memorizing**
- "A check you can only make fail with a fake object isn't verifying anything the data can do."
- "A partition has to be disjoint *and* complete; I only checked the half that couldn't break."
- "My 'coverage' metric was really evenness: dividing by occupied bins hides exactly the gaps."
- "These are held-out episodes from the same setup, which is why retrieval is such a strong baseline. It's not a generalization test."
