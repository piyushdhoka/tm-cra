# TM-CRA → Neurocomputing: Codex Master Engineering and Experiment Plan

**Repository:** `https://github.com/piyushdhoka/tm-cra.git`  
**Target venue:** *Neurocomputing* (Elsevier), regular research article  
**Project direction:** **TM-CRA — Token-Matched Capability Recovery Assessment for Depth-Pruned Small Language Models**  
**Primary goal:** turn the current repository and experimental evidence into a reproducible, defensible, reviewer-ready study in which TM-CRA is the methodological contribution and RAH is a planner/case study used to demonstrate why fair recovery accounting matters.

---

# 0. Instructions to Codex — read this before changing anything

You are working on a research repository that already contains completed experiments used in an earlier IEEE Access manuscript. **Do not treat this as a green-field refactor.** Preserve scientific provenance.

## Non-negotiable rules

1. **Audit before editing.** Inspect the full repository, current git status, current branch, existing results, configs, scripts, tests, and README before making changes.
2. **Do not delete, rewrite, or silently modify old result JSON files.** Existing results are research evidence. New experiments must go into a versioned namespace such as `results/neurocomputing_v1/` or another explicit experiment root.
3. **Do not change old result values to make them agree with the paper.** If manuscript claims and repository evidence disagree, record the disagreement and fix the manuscript/metadata later.
4. **Do not optimize for statistical significance.** Never change baseline allocations, evaluation sizes, pruning cuts, seeds, or analysis rules after inspecting new test/OOD results merely to improve a p-value.
5. **Do not run expensive jobs automatically.** Implement and test the pipeline, but do not launch any training job above 100k tokens or any full experiment matrix without explicit human approval.
6. **Test/dev separation is sacred.** Allocation decisions may use only the development split and pilot data specified by the protocol. Test data and OOD test data must never affect allocation, scope selection, hyperparameters, pruning severity, or stopping decisions.
7. **Every number reported in the future paper must be reproducible by script.** No manually typed table values.
8. **Every new run must capture full provenance.** Hardware, software, model revision, dataset revision/hash, git commit, config hash, seed, dtype, realized tokens, timing, and memory must be saved with the result.
9. **Do not claim equivalence from a nonsignificant difference.** If a CI includes zero, say “no difference was detected.” Only use “equivalent/non-inferior” if a defensible equivalence/non-inferiority test and margin were pre-specified.
10. **Keep RAH and TM-CRA conceptually separate.** TM-CRA is the evaluation protocol. RAH is one planner investigated under the protocol. The paper must not depend on RAH “winning.”

At the end of each coding phase, output:

- files changed;
- why each change was required scientifically;
- tests executed and results;
- backward-compatibility impact;
- exact commands for smoke tests;
- exact commands for full experiments, but **do not execute full experiments**;
- unresolved risks.

---

# 1. Scientific objective

The new paper should answer the following central question:

> **When a post-pruning recovery strategy spends training tokens to discover how to heal a model, does its apparent advantage remain when that planning expenditure is charged to the method and compared against a simple recovery baseline given the same total training-token budget?**

The study should additionally establish:

1. which deployment-relevant capabilities are most fragile under depth pruning;
2. how capability recovery differs across model families;
3. whether a sophisticated pilot-based planner beats simple zero-pilot allocation rules;
4. whether apparent recovery generalizes to out-of-domain tasks rather than merely benchmark-aligned evaluations;
5. how uncertain pilot-derived recovery curves are across seeds;
6. whether structural depth reduction produces measurable latency/throughput/VRAM benefits;
7. whether conclusions remain qualitatively stable at another pruning severity.

The paper should **not** claim that RAH is a universally superior recovery algorithm. A valid final result may be that RAH does not outperform simple alternatives once planning cost is included.

---

# 2. Current repository state that must be preserved and audited

At the time this plan was written, the public repository contains:

- `rah/` — pruning, healing, recovery fitting, allocation, evaluation, statistics;
- `scripts/00_check_env.py` through `scripts/10_compute_matched.py`;
- `configs/` with Llama, Qwen, SmolLM2, and OLMo model configurations;
- `results/` with per-run outputs;
- `tests/`;
- a README presenting TM-CRA as an eight-step protocol.

Current main models:

- `unsloth/Llama-3.2-1B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `allenai/OLMo-2-0425-1B-Instruct`

Current main pruning counts:

- Llama: remove 2 / 16 blocks;
- Qwen: remove 7 / 28 blocks;
- SmolLM2: remove 4 / 24 blocks;
- OLMo: remove 2 / 16 blocks.

Current capabilities:

- `fmt`: structured output;
- `tool`: function calling;
- `inst`: instruction following;
- `math`: mathematical reasoning;
- `safe`: safety refusal / over-refusal;
- `know`: knowledge/calibration.

Current healing pools:

- `fmt`, `tool`, `inst`, `math`, `safe`, `general`.

Current main healing budget:

- 1,000,000 training tokens.

Current pilot design:

- 6 pools × budgets `[30k, 100k, 300k]` = 18 recovery pilots;
- 3 scope pilots × 300k = 3 scope pilots;
- total pilot training = **3.48M tokens/model**.

Current main seeds:

- `[0, 1, 2]` for core methods.

---

# 3. Mandatory codebase problems to fix before new experiments

## P0 — scientific/reproducibility blockers

These are blocking issues. Do not start final A100 experiments until all P0 items are resolved and tested.

### P0.1 — configured dtype is ignored during training

**Current issue:** `configs/base.yaml` defines `dtype: bfloat16`, and `rah/utils.py::torch_dtype()` contains fallback logic, but `rah/healing.py::train()` currently uses:

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
```

This hard-codes BF16 and bypasses the configured/resolved dtype.

### Required fix

- Resolve runtime dtype once from config/hardware.
- Use the resolved dtype consistently in model loading, autocast, finalization, and metadata.
- If configured BF16 is unsupported, explicitly fall back to FP16 only when the protocol allows it and record the fallback.
- For FP32, disable autocast rather than forcing a half dtype.
- Store both `requested_dtype` and `resolved_dtype` in every run manifest.
- Add tests for BF16-capable, BF16-incapable, FP16, and FP32 resolution paths without requiring a real GPU where possible.

**Important:** confirmatory paper runs should use one standardized precision policy. T4 smoke runs may use FP16, but do not combine them silently with A100 BF16 confirmatory runs.

---

### P0.2 — hardware/software provenance is inconsistent

The README currently claims all runs used NVIDIA A100 40 GB GPUs. Existing checked-in environment files do not support that statement:

- Llama environment records RTX 4060 Laptop, PyTorch 2.5.1 + CUDA 12.1;
- Qwen environment records RTX 4060 Laptop while its manual `device_note` says RTX 4070;
- SmolLM2 environment records A100-SXM4-40GB while its `device_note` says RTX 4060 Laptop;
- OLMo environment records A100-SXM4-40GB while its `device_note` says RTX 4060 Laptop.

### Required fix

- Treat `torch.cuda.get_device_name()` and actual runtime detection as the authoritative hardware record.
- Remove stale manual hardware claims from model configs, or clearly rename them as non-authoritative notes.
- Do not overwrite historical `env.json` files. Preserve them as evidence.
- Add a provenance audit script, e.g. `scripts/audit_provenance.py`, which compares:
  - README hardware claims;
  - model config notes;
  - environment JSON;
  - per-run manifests;
  - requirements/lock file versions.
- The audit must fail non-zero for contradictory claims in the **new Neurocomputing experiment namespace**.
- Update README only after the scientific team decides how old hardware should be disclosed.

---

### P0.3 — dependency story is inconsistent

`requirements.txt` currently instructs PyTorch 2.5.1/CUDA 12.1, while some A100 environment records contain PyTorch 2.11.0/CUDA 13.0. The README also states PyTorch 2.11.0.

### Required fix

Create a reproducible environment strategy. Preferred options:

1. `requirements-core.txt` for Python packages + documented PyTorch install command;
2. `requirements-lock-a100.txt` / `environment-a100.yml` for the confirmatory environment;
3. optional `environment-t4.yml` for smoke/benchmark work;
4. or a Docker/Apptainer environment if practical.

Record:

- Python;
- PyTorch;
- CUDA runtime;
- transformers;
- PEFT;
- datasets;
- accelerate;
- scipy;
- lm-eval;
- tokenizer/model revisions.

Do not force old historical results to pretend they were generated in the new environment.

---

### P0.4 — public statistics do not match the README/manuscript description

`rah/stats.py`, `scripts/09_headline_tests.py`, and `scripts/10_compute_matched.py` currently perform item-level bootstrapping after pooling seeds through per-item means. The README says the study uses seed-aware paired bootstrap tests and Holm correction.

### Required fix

Implement explicit statistical APIs for:

1. item-level paired bootstrap;
2. two-level / hierarchical seed-aware bootstrap;
3. Holm-Bonferroni family-wise correction;
4. confidence intervals and raw/adjusted p-values;
5. per-seed descriptive statistics.

The new functions must have unit tests using synthetic data with known behavior.

#### Statistical implementation requirements

- Preserve the existing item bootstrap as a named analysis, not the only analysis.
- Implement the exact seed-aware procedure intended by the manuscript before changing it.
- Keep seed IDs associated with runs. Do not destroy seed identity prematurely by averaging.
- Produce both raw and Holm-adjusted p-values.
- Store the exact correction family in output metadata.
- Output machine-readable JSON/CSV as well as Markdown.
- Never label a nonsignificant difference as equivalence.
- If later adding equivalence/non-inferiority tests, require a pre-specified and scientifically justified margin; do not invent a margin after seeing outcomes.

#### New confirmatory families

Before new OOD results are viewed, freeze which comparisons/metrics are confirmatory. Suggested primary metrics:

- capped mean retention;
- worst-capability retention.

Suggested primary fixed-budget comparisons:

- RAH vs Uniform;
- RAH vs Damage-Proportional;
- RAH vs Worst-First;
- Worst-First vs Uniform.

Suggested token-matched comparisons:

- RAH total-cost condition vs Uniform with equal total training tokens.

Do not automatically expand the correction family every time a plot is created. Exploratory analyses must be labeled exploratory.

---

### P0.5 — “compute-matched” terminology is scientifically incorrect

Current file/output names include `10_compute_matched.py` and `compute_matched.csv`, but the comparison equalizes **training tokens**, not wall-clock time, FLOPs, energy, or GPU-hours.

### Required fix

Rename new analysis code to **token-matched**, e.g.:

- `scripts/10_token_matched.py`;
- `token_matched.csv`;
- `token_matched.md`.

For backward compatibility, either keep `10_compute_matched.py` as a thin deprecated wrapper with a warning, or preserve it unchanged for historical reproduction and add the correctly named replacement.

Never write “compute-matched” in new manuscript-facing outputs unless actual compute is measured and equalized.

---

### P0.6 — per-run provenance is insufficient

Current results store training steps/tokens, evaluation time, summaries, and records, but the pipeline does not reliably store full experiment provenance.

### Required run manifest

For every new run, save a `run_meta` object containing at least:

```yaml
schema_version:
experiment_id:
git_commit:
git_dirty:
config_path:
config_sha256:
start_utc:
end_utc:
wall_seconds_total:
train_seconds:
eval_seconds_indomain:
eval_seconds_ood:
python_version:
platform:
torch_version:
cuda_runtime:
cudnn_version:
gpu_name:
gpu_uuid_or_index:
gpu_memory_total_gb:
requested_dtype:
resolved_dtype:
tf32_matmul:
model_id:
model_revision:
tokenizer_revision:
seed:
prune_level:
removed_blocks:
scope:
requested_budget_tokens:
requested_tokens_per_pool:
realized_tokens_per_pool:
realized_total_tokens:
optimizer_steps:
trainable_parameters:
peak_memory_allocated_gb:
peak_memory_reserved_gb:
dataset_manifest_hash:
```

Use UTC timestamps.

Add a schema/version field so future changes are detectable.

---

### P0.7 — requested pool quotas are stored, but realized per-pool tokens are not explicit

`build_mixture()` may exceed each quota by up to one example. Current result files record requested `tokens_per_pool` and total seen tokens, but the exact realized tokens by pool are not preserved.

### Required fix

- Preserve pool identity in each mixed example, or compute realized counts before shuffling.
- Record requested and realized tokens per pool.
- Report token-matching using **realized total tokens** as a sensitivity check, while retaining the planned budget definition.

---

### P0.8 — model and dataset revisions are not frozen strongly enough

Hugging Face IDs alone are insufficient for long-term reproduction because upstream repositories can change.

### Required fix

- Resolve and store model commit SHA/revision.
- Pin tokenizer revision.
- Pin dataset revision where supported.
- For generated JSONL evaluation/pool files, create SHA256 checksums.
- Add `data/manifest.json` containing source, split, revision, item count, license/source note, and checksum.
- New result manifests must include the data-manifest hash.

---

### P0.9 — new outputs must not contaminate historical results

Create a versioned experiment namespace, for example:

```text
experiments/
  neurocomputing_v1.yaml
results/
  neurocomputing_v1/
    llama/
    qwen/
    smollm/
    olmo/
    analysis/
```

Do not reuse old output paths for new confirmatory runs.

---

# 4. Strongly recommended code improvements

## P1.1 — one frozen Neurocomputing protocol file

Create `experiments/neurocomputing_v1.yaml` containing every decision that must be frozen before results:

- models;
- model revisions;
- main/light cuts;
- in-domain datasets;
- OOD datasets;
- seeds;
- healing budget;
- pilot budgets;
- scopes;
- Worst-First allocation rule;
- metrics;
- confirmatory comparisons;
- multiplicity families;
- efficiency benchmark settings;
- run matrix;
- experiment version.

Add a SHA256 hash of this protocol to every new result.

The protocol should be committed/tagged **before** final new test/OOD runs.

Suggested tag:

```text
tmcra-neurocomputing-prerun-v1
```

Do not alter this tag after seeing outcomes. If methodology must change, create `v2` and document why.

---

## P1.2 — allow multiple evaluation suites in one healed-model session

Current `heal_and_evaluate()` heals a fresh pruned model, evaluates the regular suite, writes results, and deletes the model. This means adding an OOD benchmark later would require retraining because healed checkpoints are not saved.

### Required change

Refactor the evaluation stage so a single healed model can run:

1. in-domain test suite;
2. OOD test suite;
3. optional additional diagnostics;

before model deletion.

Suggested result structure:

```json
{
  "summary": { ... legacy in-domain summary ... },
  "records": [ ... legacy records ... ],
  "evaluations": {
    "indomain": {"summary": {}, "records": [], "seconds": 0.0},
    "ood": {"summary": {}, "records": [], "seconds": 0.0}
  },
  "run_meta": {}
}
```

Maintain backward compatibility for old analysis code if practical.

Optional: allow checkpoint/adapter saving via explicit config, default OFF. The final confirmatory run should evaluate all frozen suites before deleting the model, so checkpoint saving is not required for correctness.

---

## P1.3 — reliable timing and VRAM logging

Add instrumentation around training and evaluation:

- `time.perf_counter()`;
- `torch.cuda.reset_peak_memory_stats()` before the measured section;
- `torch.cuda.synchronize()` before/after GPU timing boundaries;
- `torch.cuda.max_memory_allocated()`;
- `torch.cuda.max_memory_reserved()`.

Record training and evaluation separately.

Do not use these timings for cross-GPU method comparisons unless hardware/software is identical.

---

## P1.4 — environment validation should be explicit, not based on stale comments

Improve `00_check_env.py` so it:

- records actual GPU;
- records CUDA/cuDNN;
- records resolved dtype;
- records model revision;
- records git commit/dirty state;
- checks critical dependency versions against the selected experiment profile;
- prints a warning if the current environment differs from the frozen confirmatory environment;
- never overwrites historical evidence without versioning.

---

## P1.5 — paper outputs must be generated, not hand-built

Add a final export script, e.g.:

```text
scripts/14_export_neurocomputing.py
```

It should generate:

- CSV tables;
- Markdown tables;
- LaTeX-ready tables if convenient;
- figure PDFs;
- a machine-readable `paper_numbers.json`;
- a provenance summary identifying which result files feed each table.

No paper number should need manual transcription.

---

# 5. New baseline: Worst-First / Bottleneck-First allocation

## Why this is necessary

The current manuscript already identifies a simpler alternative explanation: the largest worst-case gains often come from giving a large allocation to math / the weakest capability. A reviewer can reasonably ask whether a cheap zero-pilot policy would capture the same benefit as RAH.

The new baseline must answer that directly.

## Baseline definition

Use only the **development diagnosis after pruning**. No pilots, no test data.

1. Compute development retention `r_c(M_p)` for each capability.
2. Find the capability with minimum retention.
3. Map it to its healing pool using `own_pool`.
4. Allocate **50% of the total budget as an extra bottleneck mass** and distribute the other 50% uniformly across all pools.

For `P` pools and bottleneck pool `p*`:

```text
base = 0.5 * B / P
allocation[p] = base for every p
allocation[p*] += 0.5 * B
```

With six pools, the bottleneck receives approximately 58.33% and each other pool approximately 8.33%.

Name in code/paper:

```text
worst_first
```

or

```text
bottleneck_first
```

Pick one name and use it consistently.

### Important methodological rule

The 50/50 rule is frozen **before** new OOD/test results. Do not sweep 30/40/50/60/70% and choose the best test outcome. If a sensitivity sweep is later desired, it must be exploratory and use development data only.

### Tie handling

If multiple capabilities share the same minimum retention:

- use a deterministic tie rule based on the frozen capability order, or
- split the 50% bottleneck mass equally across tied pools.

The latter is preferable; add a unit test.

### Code locations

Likely changes:

- `rah/allocate.py` — implement `worst_first(...)`;
- `scripts/06_fit_allocate.py` — add allocation to output;
- `configs/base.yaml` / Neurocomputing protocol — include as core method for new experiment namespace;
- `scripts/07_main.py` or a new Neurocomputing runner — allow running it for seeds 0,1,2;
- analysis/export scripts — include it in tables/figures/tests;
- tests — budget conservation, tie behavior, mapping of `know -> general`, deterministic output.

---

# 6. OOD recovery evaluation — mandatory for the strongest paper

The current study uses healing data closely aligned with some evaluation tasks. The new version should test whether “recovery” transfers beyond those aligned evaluations.

## OOD benchmark A — mathematical reasoning

### Training/aligned capability

- healing math data: GSM8K-derived pool;
- current aligned test: GSM8K.

### OOD test

- **SVAMP**.

### Requirements

- Use a deterministic exact-answer scorer consistent with the benchmark definition.
- Do not use an LLM-as-judge.
- Pin source/revision and store item checksum.
- Use a frozen subset only if full evaluation is too slow; if subsampling, choose the subset by deterministic seed **before** method results are evaluated and record exact IDs.
- Do not use SVAMP for allocation, recovery fitting, scope selection, or hyperparameter tuning.

Output capability name should distinguish it from aligned math, e.g. `math_ood_svamp`.

---

## OOD benchmark B — instruction following

### Training/aligned capability

- current instruction healing data uses verifiable constraints;
- aligned evaluation: IFEval.

### OOD test

- **IFBench** using its deterministic/verifiable evaluation implementation.

### Requirements

- Prefer official checker code / official dataset source.
- Pin revision and store checksums.
- No LLM judge for the primary score.
- Never expose OOD results to the allocation stage.

Output capability name such as `inst_ood_ifbench`.

---

## OOD metrics

For each OOD benchmark report at minimum:

1. absolute unpruned score;
2. absolute pruned score;
3. absolute healed score;
4. retention relative to unpruned;
5. recovery fraction where interpretable;
6. per-seed result;
7. bootstrap CI over items;
8. seed-aware sensitivity analysis for core comparisons.

Avoid ratios when the unpruned baseline is near zero; add guards and report absolute scores prominently.

---

# 7. Efficiency/deployment benchmark — mandatory, but no training required

The paper motivates depth pruning as a deployment-efficiency technique. The new version should measure actual efficiency instead of only structural block removal.

## Compare

For each of the four model families:

- unpruned model;
- main-cut pruned model.

Optional:

- light-cut pruned model;
- a healed main-cut model, if checkpoint access is convenient.

Because healing does not restore deleted blocks, the main structural inference cost should be similar before/after healing; the key comparison is unpruned vs pruned architecture.

## Hardware

A T4 is acceptable and useful for this deployment benchmark **if all compared conditions use the same T4 environment**. Record hardware precisely and keep these inference measurements separate from A100 training results.

## Measure

At minimum:

- original and pruned block count;
- parameter count;
- model weight memory;
- peak VRAM;
- prefill latency;
- decode latency per generated token;
- end-to-end latency;
- tokens/second;
- KV-cache memory if feasible;
- percentage improvement relative to unpruned.

## Frozen benchmark protocol

Suggested primary benchmark settings:

```yaml
prompt_lengths: [128, 512, 1024]
new_tokens: 128
batch_sizes: [1, 4]
warmup_repeats: 20
measured_repeats: 100
decoding: greedy
use_cache: true
```

If T4 memory cannot support batch 4 for a condition, record OOM as a result and keep batch 1 as the cross-model primary condition.

Use synthetic/token-controlled prompts or a frozen prompt set so sequence lengths are comparable. Do not mix tokenizers without recording realized token lengths.

## Timing correctness

- use `torch.inference_mode()`;
- synchronize CUDA around timing;
- separate prefill from decode when possible;
- warm up before measuring;
- report median and percentile spread, not only one mean;
- store raw repetition timings.

Create e.g.:

```text
scripts/12_benchmark_efficiency.py
results/neurocomputing_v1/efficiency/
```

---

# 8. Pilot stability experiment — strongly recommended

The original planner fits recovery curves from pilots that are currently run once per family. The new paper should quantify whether the most important slow-recovery conclusion is stable across pilot seeds without repeating the entire 21-pilot suite.

## Targeted design

Repeat only **math recovery pilots** for two additional seeds.

Current pilot seed is effectively one run. Add two more seeds for:

- 30k tokens;
- 100k tokens;
- 300k tokens.

Across all four models:

```text
4 models × 3 budgets × 2 new seeds = 24 short pilot runs
```

Additional training-token total:

```text
4 × 2 × (30k + 100k + 300k) = 3.44M tokens
```

## Analyze

- per-seed math retention at each pilot budget;
- fitted `a_math` and `tau_math` across pilot seeds;
- whether the qualitative classification “fast/slow/no early recovery” changes;
- uncertainty band for the recovery curve;
- whether the RAH allocation would materially change under alternate pilot seeds.

Do not silently average the pilots and refit the original confirmatory planner. Treat this as a stability/sensitivity study unless a new protocol version is explicitly defined.

---

# 9. Selected RAH OOD runs — recommended

To show how the planner behaves beyond aligned benchmarks without exploding compute, run OOD evaluation for RAH on two representative families:

- Llama-3.2-1B;
- Qwen2.5-1.5B.

Use seeds 0, 1, 2:

```text
2 models × 1 method × 3 seeds = 6 full 1M-token runs
```

Each run should evaluate both in-domain and OOD suites in the same healed-model session.

These runs also serve as part of the hardware/software reproducibility audit if executed in the standardized A100 environment.

---

# 10. Cross-environment reproducibility gate

Because historical Llama/Qwen results were produced in a different recorded environment from SmolLM2/OLMo, perform a targeted audit before relying on all historical values in the new paper.

## Reuse new runs where possible

The new Uniform and selected RAH A100 runs already provide reproducibility evidence.

Additionally rerun, at minimum:

- Uniform 4.48M, seed 0, Llama;
- Uniform 4.48M, seed 0, Qwen.

That is **2 long token-matched audit runs**.

Generate a comparison report between historical and standardized-environment results:

- per-capability absolute score;
- capped mean retention;
- worst retention;
- seed-0 historical vs new;
- new result relative to historical seed spread where available.

### Stop/go rule

Do **not** automatically declare a pass from one arbitrary threshold. Produce the report and flag material discrepancies. If the new standardized run changes the qualitative paper conclusion or falls clearly outside historical seed variability, stop and investigate before running the full matrix.

If the audit fails, rerun affected core conditions under one standardized environment instead of mixing unexplained incompatible numbers.

---

# 11. Optional second pruning severity — high-value extension

If the core study is stable and compute remains, test whether the conclusions depend on the main pruning cut.

Use existing `light` pruning specifications rather than introducing an entirely new pruning algorithm.

Recommended models:

- Llama;
- Qwen.

Methods:

- Uniform;
- Worst-First.

Seeds:

- 0, 1, 2.

Budget:

- 1M tokens.

Run count:

```text
2 models × 2 methods × 3 seeds = 12 full runs
```

Evaluate in-domain + OOD in the same session.

Treat this as robustness/generalization evidence, not a new method-selection sweep.

If results differ from the main cut, that is scientifically useful: TM-CRA may reveal severity-dependent recovery behavior.

---

# 12. Do not add a second full pruning algorithm unless required

A second pruning criterion may eventually strengthen generality, but it is **not the first use of GPU budget**. First complete:

1. reproducibility fixes;
2. simple baseline;
3. OOD evaluation;
4. efficiency benchmark;
5. pilot stability;
6. second severity.

Only consider another pruning algorithm if the paper still looks too specific after those additions or reviewers explicitly request it.

---

# 13. Exact run-count plan

## Tier 0 — engineering only

No expensive training.

Tasks:

- P0/P1 code fixes;
- statistics;
- OOD loaders/scorers;
- efficiency benchmark implementation;
- provenance system;
- tests;
- 30k–100k smoke jobs only.

Use Colab T4 for smoke tests if needed.

---

## Tier 1 — minimum serious Neurocomputing extension

### A. Uniform + Worst-First, all models, all 3 seeds, 1M

```text
4 models × 2 methods × 3 seeds = 24 full healing runs
```

Training tokens:

```text
24M
```

Every run evaluates:

- original in-domain suite;
- SVAMP OOD;
- IFBench OOD.

### B. Efficiency benchmark

No healing training.

Conditions:

```text
4 models × 2 structural states = 8 primary inference conditions
```

with repeated timing across frozen prompt/batch settings.

### Tier-1 training total

```text
24 full runs
24M training tokens
```

This is the minimum package that directly addresses the simple-baseline and OOD-generalization weaknesses.

---

## Tier 2 — recommended strong Neurocomputing package

Add:

### C. RAH OOD on Llama + Qwen

```text
2 × 3 = 6 full 1M runs
6M tokens
```

### D. Math pilot stability

```text
4 models × 3 budgets × 2 additional seeds = 24 short runs
3.44M tokens
```

### E. Uniform 4.48M standardized-environment audit

```text
2 models × 1 seed = 2 long runs
8.96M tokens
```

### Tier-2 cumulative total

Full/long training jobs:

```text
24 + 6 + 2 = 32
```

Short pilot jobs:

```text
24
```

Total training jobs:

```text
56
```

Total nominal new training tokens:

```text
24M + 6M + 8.96M + 3.44M = 42.40M tokens
```

This is the preferred strong package if the A100 budget is available.

---

## Tier 3 — optional robustness package

Add second pruning severity:

```text
12 full 1M runs = 12M tokens
```

Cumulative:

```text
68 total training jobs
54.40M nominal training tokens
```

Do not go to Tier 3 until Tier 1/2 results and reproducibility gates are clean.

---

# 14. Compute planning

These are planning ranges only. The repository currently lacks authoritative training wall-clock measurements, so the first new run must calibrate real throughput.

## Approximate A100 40GB planning ranges

- 30k pilot: ~2–5 min training;
- 100k pilot: ~3–8 min;
- 300k pilot + dev eval: ~8–20 min;
- 1M heal + existing evaluation: ~15–35 min;
- 4.48M heal + existing evaluation: ~50–120 min;
- OOD evaluation adds additional time depending on final item counts.

## Approximate T4 planning ranges

- 30k: ~5–15 min;
- 100k: ~10–25 min;
- 300k: ~25–60 min;
- 1M: ~45–110 min;
- 4.48M: multiple hours.

Use T4 for:

- code validation;
- 30k/100k smoke runs;
- evaluator/scorer validation;
- efficiency benchmarking;
- memory testing.

Prefer A100 for confirmatory healing because the final study should use a standardized precision/training environment.

## Mandatory runtime calibration

After the first 300k and first 1M A100 jobs:

- record train/eval wall time;
- update a generated `compute_estimate.md` based on measured throughput;
- estimate remaining GPU-hours from actual tokens/sec / updates/sec;
- do not retroactively change the experiment plan simply because a condition is slow.

Expected rough order for the recommended Tier-2 package is **roughly tens of A100 GPU-hours**, not hundreds. Replace this estimate with measured values as soon as instrumentation exists.

---

# 15. What if the new results are “not satisfactory”?

The study must be designed so negative results are informative. Do not define success as “RAH wins.”

## Scenario A — Worst-First beats or matches RAH

**Interpretation:** strong support for TM-CRA. A trivial zero-pilot allocation captures gains that appeared to justify expensive planning.

**Paper direction:** emphasize that fair accounting and simple baselines are essential before claiming planner benefit.

**Neurocomputing viability:** potentially stronger, provided the result is robust and the study is broad/reproducible.

---

## Scenario B — RAH clearly beats Worst-First at fixed budget

**Interpretation:** recovery modeling contains information beyond simply identifying the weakest capability.

Then ask whether the benefit survives total-token matching.

**Neurocomputing viability:** strong if both fixed-budget and token-matched results are reported honestly.

---

## Scenario C — Uniform remains best once pilot cost is counted

**Interpretation:** this is the central TM-CRA result, not a failure.

**Paper direction:** sophisticated planning can look favorable under a nominal healing budget but lose its practical advantage after accounting for planning tokens.

Do not claim equivalence unless tested; say no advantage was detected / uniform was competitive / specific conditions favored uniform as supported by evidence.

---

## Scenario D — OOD recovery is much weaker than aligned recovery

**Interpretation:** highly interesting. “Recovery” may be benchmark-aligned rather than capability-general.

This can become a major Neurocomputing contribution:

> Post-pruning healing can restore evaluation-aligned scores without restoring broader capability transfer.

Do not hide this result.

---

## Scenario E — OOD recovery tracks aligned recovery well

**Interpretation:** strengthens the claim that healing restores a broader capability rather than only the benchmark.

Also useful.

---

## Scenario F — pilot curves vary heavily across seeds

**Interpretation:** pilot-based planners may be unstable because small pilot evaluations introduce estimation noise.

This strengthens the case for uncertainty-aware evaluation and weakens overconfident planner claims.

Do not average away the instability.

---

## Scenario G — pilot curves are stable

**Interpretation:** strengthens the recovery-dynamics evidence and lets the paper make more confident descriptive claims about slow/fast recovery.

---

## Scenario H — second pruning severity changes the conclusions

**Interpretation:** recovery and planning usefulness are severity-dependent.

This is still valuable. The paper should narrow its generalization and explicitly show the interaction.

---

## Scenario I — corrected statistics eliminate previous significance

**Interpretation:** the old inferential claim was too strong. Update the paper.

Do **not** search for alternative tests until something becomes significant.

The study may remain Neurocomputing-worthy if the protocol, capability diagnosis, token accounting, OOD findings, and efficiency evidence remain substantive.

---

## Scenario J — key historical results cannot be reproduced

This is a stop condition.

Actions:

1. freeze further expensive runs;
2. compare environment/config/data hashes;
3. inspect model/dataset revision drift;
4. verify pruning set;
5. verify dtype behavior;
6. verify seed handling;
7. verify teacher data;
8. verify scorer versions;
9. rerun minimal diagnostic cases;
10. only resume the matrix once the discrepancy is understood.

If the corrected result changes the paper conclusion, the corrected result wins.

---

## When would the project become weak for Neurocomputing?

The Neurocomputing case becomes materially weaker if **all** of the following occur:

- token matching does not change interpretation;
- no meaningful capability-specific recovery pattern is stable;
- OOD results add no new insight;
- simple baselines and planners are indistinguishable with very large uncertainty;
- pruning provides little realized efficiency benefit;
- conclusions are highly unstable to seed/severity;
- and the protocol does not expose a practically important evaluation issue.

If that happens, do not manufacture a positive result. Options are:

1. broaden the protocol to another pruning severity/criterion;
2. increase evaluation precision where uncertainty is the limiting factor;
3. reframe as a careful negative/measurement study for a more suitable venue;
4. target a strong Q2 journal rather than overstating the work.

A negative outcome alone does **not** make the work unsuitable. A study becomes weak when it does not support a clear, generalizable insight.

---

# 16. Statistics plan for the new paper

## Primary metrics

Keep:

1. **capped mean retention**;
2. **worst-capability retention**.

Also report:

- uncapped mean retention;
- per-capability retention;
- absolute benchmark scores;
- safety over-refusal and harmful refusal;
- knowledge ECE;
- OOD absolute and retention scores.

## Required analyses

### Item-level analysis

Paired bootstrap over common item IDs within capability.

### Seed-aware analysis

Preserve the three independent training seeds. Implement a hierarchical/two-level bootstrap consistent with the predeclared statistical method.

### Multiple testing

Use Holm correction for pre-specified confirmatory families.

### Descriptive uncertainty

Report per-seed spread (SD and/or range) for primary metrics.

### Effect sizes

Report differences and CIs, not only p-values.

### Negative claims

“Nonsignificant” must not be written as “equivalent.”

### OOD

Treat OOD results as a distinct analysis family. Freeze tests before examining final results.

---

# 17. Analysis-code cleanup

## `scripts/08_analyze.py`

Refactor so that it does not silently use seed-averaged item scores for every inferential statement.

Separate:

- descriptive aggregation;
- item-bootstrap inference;
- seed-aware inference;
- multiplicity correction.

Avoid one script doing everything implicitly.

Suggested modules:

```text
rah/stats.py
rah/analysis.py
scripts/09_fixed_budget_tests.py
scripts/10_token_matched.py
scripts/11_ood_tests.py
scripts/13_pilot_stability.py
scripts/14_export_neurocomputing.py
```

Keep historical scripts if needed for exact reproduction.

## Plotting

Generate publication-quality vector PDFs.

New key figures:

1. TM-CRA protocol schematic data export (diagram can be finalized separately);
2. fixed-budget vs token-matched primary metrics;
3. in-domain vs OOD recovery;
4. efficiency vs retention trade-off;
5. math pilot stability / recovery-curve uncertainty;
6. optional pruning-severity comparison.

Do not use visual design choices to conceal uncertainty.

---

# 18. Potential issue in allocation diagnostics

Review `scripts/06_fit_allocate.py` carefully. The `predicted` output currently appears to call the main fitted recovery model on every allocation, including variants that may have been generated from a different curve/proxy model. Confirm whether this is intentional.

If not:

- store the prediction from the model actually responsible for that allocation;
- label cross-model predictions separately;
- add tests so diagnostic predictions cannot silently misrepresent an ablation.

This may not affect training results, but manuscript-facing diagnostics must be accurate.

---

# 19. Dataset/evaluation integrity tests

Add automated checks for:

- no overlapping item IDs between dev and test;
- no OOD test item used in healing pools;
- no evaluation item text duplicated in healing data where detectable;
- stable dataset checksums;
- correct item counts;
- valid metadata for every scorer;
- deterministic OOD subsampling;
- every method within a comparison sees identical evaluation IDs;
- every seed of a method uses the same evaluation set;
- no allocation script imports/reads test or OOD result files.

Create a script such as:

```text
scripts/audit_data_leakage.py
```

and make it part of the release checklist.

---

# 20. Safety evaluation caveat

The existing safety scorer uses a phrase-based refusal heuristic. Do not silently promote this into a precise safety metric.

For the Neurocomputing revision:

- preserve the existing metric for comparability;
- clearly label it as heuristic;
- if feasible, manually validate a stratified sample of safety outputs or add a stronger deterministic/validated classifier;
- do not let a new subjective LLM judge become the sole safety metric unless the methodology is carefully validated.

This is secondary to the core experiments and should not consume the first GPU budget.

---

# 21. Tests to add

At minimum add unit/integration tests for:

## Allocation

- uniform sums to budget;
- damage-proportional sums to budget;
- Worst-First sums to budget;
- Worst-First tie handling;
- `know -> general` mapping;
- no negative quotas;
- deterministic allocation.

## Dtype/runtime

- BF16 requested + supported;
- BF16 requested + unsupported fallback;
- FP16;
- FP32;
- resolved dtype written to manifest.

## Statistics

- paired bootstrap sign on synthetic positive/negative effects;
- seed-aware bootstrap uses seed structure;
- Holm correction monotonicity;
- missing seed handling fails loudly for confirmatory analyses;
- common item-ID pairing;
- no accidental seed pooling before seed-aware test.

## Provenance

- run manifest required fields;
- config hash stable;
- git commit captured;
- historical result not overwritten;
- new namespace used.

## OOD

- SVAMP scorer known examples;
- IFBench checker known examples;
- OOD evaluation cannot be loaded by allocation code;
- frozen IDs/checksums.

## Efficiency

- benchmark warm-up excluded from statistics;
- CUDA synchronization used where applicable;
- raw timing output saved;
- output summary reproducible from raw timings.

## Data integrity

- dev/test disjoint;
- benchmark/pool overlap audit;
- expected counts and hashes.

Run:

```bash
pytest -q
```

and make the final release require a clean test suite.

---

# 22. Recommended new repository structure

Do not over-refactor, but the following additions are recommended:

```text
experiments/
  neurocomputing_v1.yaml

rah/
  analysis.py
  provenance.py
  benchmark.py
  evaluation/
    ood_svamp.py
    ood_ifbench.py

scripts/
  audit_provenance.py
  audit_data_leakage.py
  09_fixed_budget_tests.py
  10_token_matched.py
  11_ood_tests.py
  12_benchmark_efficiency.py
  13_pilot_stability.py
  14_export_neurocomputing.py
  run_neurocomputing.sh

results/
  neurocomputing_v1/
    <model>/
      diagnosis/
      main/
      pilots_stability/
      budget/
      ood/
    efficiency/
    analysis/

paper_artifacts/
  tables/
  figures/
  paper_numbers.json
```

Preserve old paths so old results remain reproducible.

---

# 23. Run launcher requirements

Create a launcher that can:

- list planned jobs without running them (`--dry-run`);
- estimate nominal tokens;
- filter by model/method/seed;
- resume safely;
- never overwrite a completed result with different config hash;
- fail if protocol hash differs from the result namespace;
- log start/end status;
- produce a job manifest CSV.

Example desired behavior:

```bash
python scripts/run_neurocomputing.py --tier 1 --dry-run
python scripts/run_neurocomputing.py --model llama --method worst_first --seed 0 --smoke --budget 30000
python scripts/run_neurocomputing.py --tier 1 --execute
```

The `--execute` flag should be required for expensive jobs.

---

# 24. Smoke-test plan before A100 runs

On T4 or any available CUDA GPU:

1. environment audit;
2. unit tests;
3. dataset/OOD build + checksum;
4. unpruned/pruned OOD evaluator sanity check;
5. 30k Uniform run on one model;
6. 30k Worst-First run on one model;
7. one 30k RAH-compatible run if needed;
8. verify result manifests;
9. verify timing/VRAM fields;
10. verify in-domain + OOD evaluation occurs before model deletion;
11. verify rerunning skips an identical finished job;
12. verify changed config refuses to reuse an incompatible result;
13. efficiency benchmark dry run with ~3 warm-ups / ~5 repetitions.

Do not use smoke-test scores in the paper.

---

# 25. A100 execution order

Once A100 is available, do not immediately launch the entire matrix.

## Gate 1 — calibration

Run one 300k job and one 1M job.

Confirm:

- BF16 actually active;
- no OOM;
- no CPU/shared-memory spill;
- train time recorded;
- peak VRAM plausible;
- outputs deterministic enough;
- OOD evaluation works.

## Gate 2 — reproducibility sample

Run new Uniform / selected RAH conditions for Llama/Qwen and compare to historical results.

If major unexplained discrepancy exists, stop.

## Gate 3 — Tier 1 matrix

Run 24 full Uniform/Worst-First jobs.

## Gate 4 — inspect pipeline integrity, not significance

Verify:

- no missing jobs;
- no corrupted files;
- identical protocol hashes;
- correct seed counts;
- correct evaluation IDs;
- no leakage;
- no anomalous training failures.

Do not change methods based on which one wins.

## Gate 5 — Tier 2

Run selected RAH OOD, math pilot stability, and token-matched audit.

## Gate 6 — Tier 3 only if justified

Run second severity.

---

# 26. Manuscript-facing outputs Codex must support

The final paper should be able to generate the following tables directly from the repository.

## Main paper tables

### Table 1 — Models and pruning conditions

- checkpoint + revision;
- parameters;
- original blocks;
- removed blocks main/light;
- percentage depth reduction.

### Table 2 — Damage after pruning

- unpruned absolute scores;
- main-cut retention;
- chance-corrected diagnostic retention where used;
- capability ranks.

### Table 3 — Fixed-budget recovery at 1M

Core:

- Uniform;
- Damage-Proportional;
- Worst-First;
- RAH;
- RAH-capped if retained.

Report per-capability and primary metrics, three seeds where core.

### Table 4 — Token-matched comparison

- RAH planning + final heal total training tokens;
- Uniform with equal total training tokens;
- realized token totals;
- capped mean;
- worst retention;
- CIs / adjusted tests.

### Table 5 — OOD generalization

- GSM8K vs SVAMP recovery;
- IFEval vs IFBench recovery;
- Uniform, Worst-First, selected RAH;
- absolute scores and retention.

### Table 6 — Efficiency

- parameters;
- blocks;
- peak VRAM;
- prefill latency;
- decode latency/token;
- tokens/s;
- percentage speedup.

### Table 7 — Pilot stability

- math recovery at 30k/100k/300k across pilot seeds;
- fitted curve variability;
- allocation sensitivity.

Optional table:

- second-severity robustness.

---

# 27. Paper claim hierarchy

The code and analysis should support claims in this order.

## Primary contribution

**TM-CRA is a fair evaluation protocol for post-pruning capability recovery that distinguishes nominal healing budget from total adaptation cost.**

## Empirical demonstration

Charging pilot training tokens can change the practical comparison between planning-based recovery and simple baselines.

## Capability result

Depth pruning damages capabilities unevenly, with math/function calling showing strong fragility in the current four-family study.

## Generalization result

Determine whether aligned recovery transfers to SVAMP/IFBench.

## Stability result

Determine how stable the most important recovery-curve inference is across pilot seeds.

## Deployment result

Quantify realized efficiency improvements rather than assuming block deletion translates directly into deployment speed.

Do not invert this hierarchy and make the paper “RAH: a new superior method.”

---

# 28. Neurocomputing-specific publication goal

The target is a **regular research article**, not primarily the Software Track.

The manuscript should present TM-CRA as a methodological/experimental contribution to learning-system evaluation, model compression, resource allocation, and efficient neural computation.

The code release is supporting evidence for reproducibility, not the sole publication contribution.

Before submission, ensure:

- concise informative title;
- single corresponding author chosen by the team;
- single-column submission manuscript unless the live guide changes;
- numbered sections;
- abstract with concrete quantitative findings and no hype;
- ≤6 keywords if required by the current live guide;
- 3–5 highlights in a separate file if required/encouraged, each within the current character limit;
- CRediT author statement;
- funding declaration;
- competing-interest declaration;
- data/code availability statement;
- generative-AI disclosure consistent with current Elsevier policy;
- recent 2026 pruning/compression literature included;
- GitHub release/tag matching the submitted paper;
- preferably archive the release (e.g. Zenodo) for a DOI;
- all tables regenerate from scripts;
- all repo/manuscript hardware claims agree.

Always re-check the live Neurocomputing Guide for Authors immediately before submission because publisher requirements can change.

---

# 29. arXiv policy / sequencing

Elsevier generally permits authors to share preprints, including arXiv. However, arXiv should not be used as a substitute for experimental readiness.

Recommended sequence:

1. finish P0 code fixes;
2. freeze protocol;
3. complete confirmatory experiments;
4. resolve authorship/order;
5. run full reproducibility audit;
6. freeze paper numbers;
7. create versioned GitHub release;
8. prepare manuscript;
9. internal coauthor review;
10. post stable preprint to arXiv;
11. submit essentially the same scientific version to Neurocomputing shortly afterward.

Do not post the old IEEE manuscript as the final TM-CRA preprint if the Neurocomputing study is going to contain materially different methods/results.

---

# 30. README changes after experiments are complete

Do not prematurely rewrite results.

Eventually update README to include:

- accurate hardware provenance;
- exact environment instructions;
- TM-CRA protocol diagram;
- new Worst-First baseline;
- OOD benchmarks;
- efficiency benchmark;
- statistical methodology;
- how to reproduce each paper table;
- experiment tag/release;
- arXiv link after posting;
- paper citation;
- warning that historical results were generated under multiple environments if they remain in the repository.

Remove any blanket claim such as “all runs used A100” unless every manuscript-facing run truly did.

---

# 31. Acceptance-readiness gates

Before recommending Neurocomputing submission, generate a report with PASS/WARN/FAIL for each gate.

## Gate A — provenance

- [ ] every manuscript run has manifest;
- [ ] no unexplained hardware contradiction;
- [ ] model/data revisions frozen;
- [ ] git/config hashes recorded.

## Gate B — reproducibility

- [ ] all tests pass;
- [ ] key historical result audit completed;
- [ ] no unexplained major discrepancy;
- [ ] every table regenerates from raw results.

## Gate C — statistics

- [ ] seed structure preserved;
- [ ] hierarchical analysis implemented;
- [ ] Holm correction implemented;
- [ ] confirmatory families frozen;
- [ ] no equivalence language from nonsignificance.

## Gate D — scientific breadth

- [ ] four model families;
- [ ] six deployment capabilities;
- [ ] Worst-First simple baseline;
- [ ] OOD math;
- [ ] OOD instruction following;
- [ ] deployment efficiency measurements;
- [ ] pilot stability or clearly justified omission.

## Gate E — interpretation

At least one strong, reproducible insight should exist beyond “we ran many benchmarks,” for example:

- token matching changes the practical interpretation;
- simple no-pilot allocation competes with expensive planning;
- aligned recovery does/does not generalize OOD;
- pilot uncertainty materially affects planner behavior;
- recovery behavior changes with pruning severity;
- realized efficiency differs from structural compression expectations.

If none exists, the paper needs more scientific breadth before a high-confidence Q1 submission.

## Gate F — submission package

- [ ] manuscript matches live Neurocomputing guide;
- [ ] title/abstract contribution clear in first page;
- [ ] related work current through submission date;
- [ ] author order/CRediT agreed;
- [ ] code/data statements complete;
- [ ] disclosures complete;
- [ ] highlights prepared;
- [ ] cover letter explicitly explains Neurocomputing fit.

---

# 32. Definition of done for Codex

The engineering phase is complete only when Codex can demonstrate all of the following **without running the expensive full matrix**:

1. repository tests pass;
2. historical results remain untouched;
3. new experiment namespace exists;
4. frozen Neurocomputing protocol exists;
5. dtype is correctly resolved and recorded;
6. per-run manifest is generated;
7. Worst-First allocation is implemented and tested;
8. SVAMP evaluator is implemented and tested;
9. IFBench evaluator is implemented and tested;
10. one healed model can run in-domain + OOD evaluation in one session;
11. training wall time and peak VRAM are recorded;
12. seed-aware bootstrap exists and is tested;
13. Holm correction exists and is tested;
14. token-matched naming is correct;
15. provenance audit exists;
16. data-leakage audit exists;
17. efficiency benchmark exists and has a dry-run mode;
18. experiment launcher can print an exact run matrix and nominal token total;
19. final export script can consume mock/smoke results and produce expected table schemas;
20. documentation explains exact A100 commands;
21. no >100k-token job was launched without explicit approval.

---

# 33. First action Codex should take

Do **not** immediately modify files.

First produce an audit report containing:

```text
A. Current git commit and dirty state
B. Full repository tree relevant to experiments
C. Current experiment/result schema
D. Exact dtype path from config -> model load -> autocast -> finalize
E. Current statistics path and whether seed structure is preserved
F. Current environment/version inconsistencies
G. Current file naming that says compute-matched
H. Current output overwrite/resume behavior
I. Current dataset/revision/checksum behavior
J. Existing tests and missing coverage
K. Proposed minimal patch sequence
L. Any disagreement between this master plan and the current repository
```

Only after that audit, implement changes in small commits/phases.

---

# 34. Suggested implementation phases/commits

Prefer small reviewable commits such as:

### Commit 1 — provenance and experiment namespace

- run manifest;
- git/config hashes;
- new result root;
- no scientific behavior change.

### Commit 2 — dtype/runtime correctness

- resolved dtype;
- timing;
- VRAM;
- environment audit.

### Commit 3 — statistics

- hierarchical bootstrap;
- Holm correction;
- tests;
- token-matched naming.

### Commit 4 — Worst-First

- allocator;
- config;
- tests;
- dry-run matrix.

### Commit 5 — OOD evaluation

- SVAMP;
- IFBench;
- manifests/checksums;
- evaluate-all-suites flow.

### Commit 6 — efficiency benchmark

- timing protocol;
- raw outputs;
- summary script.

### Commit 7 — pilot stability support

- extra pilot seeds;
- stability analysis.

### Commit 8 — export/release tooling

- paper tables;
- figures;
- audit reports;
- README instructions.

Do not mix unrelated refactors with scientific changes.

---

# 35. Final principle

The goal is **not** to make every experiment look positive.

The goal is to make the following statement defensible to a skeptical Neurocomputing editor/reviewer:

> We introduce a reproducible protocol for evaluating capability recovery after depth pruning under both nominal healing budgets and total adaptation-token budgets. Across multiple small instruction-tuned model families, we compare planning-based and simple recovery policies, quantify seed uncertainty, test out-of-domain recovery, and measure realized deployment efficiency. The protocol is useful regardless of whether a sophisticated planner wins, because it exposes when apparent recovery gains depend on uncharged planning cost, benchmark alignment, adaptation scope, or noisy pilot estimates.

If the repository and experiments support that sentence honestly, the project has a coherent Neurocomputing-level story. If they do not, report why rather than tuning the study until they appear to.

