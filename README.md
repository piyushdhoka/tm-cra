<div align="center">

# Depth-Pruned SLM Recovery

### Capability damage and healing in depth-pruned small language models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Models](https://img.shields.io/badge/models-Llama%20%C2%B7%20Qwen%20%C2%B7%20SmolLM2%20%C2%B7%20OLMo-8a2be2.svg)

</div>

---

Depth pruning deletes whole Transformer blocks to make a small language model cheaper to run, and a short fine-tuning
run (*healing*) decides what comes back. This repository holds the code, evaluation data, per-item results and
analysis scripts for measuring which capabilities break under depth pruning, how they recover under different healing
data mixtures and adaptation scopes, and what pilot-based planning of the healing mixture costs.

## Setup

| | |
| --- | --- |
| Models | `unsloth/Llama-3.2-1B-Instruct` (16 blocks), `Qwen/Qwen2.5-1.5B-Instruct` (28), `HuggingFaceTB/SmolLM2-1.7B-Instruct` (24), `allenai/OLMo-2-0425-1B-Instruct` (16) |
| Pruning | Block Influence; remove 2 (Llama, OLMo), 7 (Qwen) or 4 (SmolLM2) blocks |
| Capabilities | Structured output (JSONSchemaBench), function calling (BFCL v3), instruction following (IFEval), math (GSM8K), safety refusal (XSTest), knowledge (MMLU) |
| Healing | 6 self-distilled data pools; LoRA on all blocks, LoRA at the cut, or full tuning of the last 3 blocks; 1M-token budget |
| Allocation rules | Uniform, damage-proportional, RAH (21 pilot heals, 3.48M tokens) and RAH-capped |
| Control | Uniform healing with 4.48M tokens, the planner's heal plus its pilots |
| Statistics | 3 seeds per core method; item-level and seed-aware paired bootstraps; Holm correction |

## Repository structure

```
.
├── rah/                         # library: pruning, healing, recovery curves, allocation, evaluation, statistics
├── scripts/                     # pipeline, one numbered script per stage
│   ├── 00_check_env.py … 07_main.py         # GPU: data, teacher targets, pruning, pilots, allocation, heals
│   ├── 08_analyze.py … 10_token_matched.py   # tables and item-level tests
│   ├── 11_refusal_validation.py, 12_rescore_safety.py
│   ├── 13_seed_aware_tests.py   # seed-aware tests, Holm correction, breakage agreement
│   ├── 14_paper_numbers.py      # collects every summary number into results/analysis/paper_numbers.json
│   └── 15_paper_figures.py      # summary figures
├── configs/                     # one settings file per model
├── data/
│   ├── eval/, pools/            # evaluation items and healing-pool prompts
│   └── teacher/llama/           # recovered Llama teacher targets + SHA256SUMS
├── results/
│   ├── {llama,qwen,smollm,olmo}/   # every run with per-item scores
│   └── analysis/                # summary tables, tests and refusal validation
├── tests/                       # unit tests (pytest)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/piyushdhoka/depth-pruned-slm-recovery.git
cd depth-pruned-slm-recovery
pip install -r requirements.txt        # install PyTorch for your CUDA version first
```

## Reproducing the analysis

### 1. Recompute the summary tables and tests (CPU, minutes)

```bash
python scripts/08_analyze.py --models qwen llama smollm olmo   # per-model tables and diagnostic figures
python scripts/09_headline_tests.py                            # item-level tests at the fixed 1M budget
python scripts/10_token_matched.py                             # item-level tests against the 4.48M-token heal
python scripts/13_seed_aware_tests.py                          # seed-aware tests, Holm correction, breakage agreement
python scripts/14_paper_numbers.py                             # results/analysis/paper_numbers.json
pytest -q tests/test_stats_seed_aware.py tests/test_recovery_allocate.py
```

### 2. Re-run the experiments (GPU)

Each stage script takes the model name (`llama`, `qwen`, `smollm` or `olmo`), reads its settings from `configs/`, and
skips runs that have already finished, so an interrupted stage resumes where it stopped.

```bash
bash scripts/run_all.sh llama              # stages 0-7 (Windows: .\scripts\run_all.ps1 -Model llama)
python scripts/07_main.py --model llama --only uniform --budgets 4480000 --sweep-seeds 1 2   # token-matched seeds
python scripts/11_refusal_validation.py score    # compare the refusal classifier with the blind labels
python scripts/12_rescore_safety.py              # re-score stored safety responses (CPU)
```

To study another model, add a settings file to `configs/` and run the same commands with its name.

### Hardware

| Model | Runs | GPU | PyTorch |
| --- | --- | --- | --- |
| Llama-3.2-1B | diagnosis, pilots, 1M heals, 4.48M seed 0 | RTX 4060 laptop GPU (8 GB) | 2.5.1 |
| Llama-3.2-1B | RAH-capped seeds 1–2, 4.48M seeds 1–2 | NVIDIA L4 (Lightning AI) | not recorded |
| Qwen2.5-1.5B | diagnosis, pilots, 1M heals, budget sweep | RTX 4060 laptop GPU (8 GB) | 2.5.1 |
| Qwen2.5-1.5B | 4.48M seeds 0–2 | NVIDIA L4 (Lightning AI) | not recorded |
| SmolLM2-1.7B, OLMo-2-1B | all runs | NVIDIA A100 (40 GB) | 2.11.0 |

All runs used transformers 4.56.2, PEFT 0.21.2 and lm-evaluation-harness 0.4.13. In `results/<model>/env.json`, the
`gpu` field is the device detected by stage 0; the `device_note` field was copied from a config comment and is wrong
for Qwen, SmolLM2 and OLMo.

## Notes on the data

- **Teacher targets.** Healing targets are written by the unpruned model (stage 3). The original Llama targets are in
  `data/teacher/llama/` with `SHA256SUMS`; copy them to `artifacts/llama/teacher/` to reuse them. The other three
  models' original targets were not preserved, so rerunning stage 3 produces new targets that may differ slightly.
- **Safety scores.** On 27 October 2026 the refusal classifier was fixed to recognize refusals written with a
  typographic apostrophe, and all stored safety responses were re-scored (stage 12). The planner's Llama allocations
  in `results/llama/fit/` were computed before the fix and are kept as they were used.

## License

Code and released results are under the [MIT License](LICENSE). Model weights are not redistributed and remain under
their original licenses; the datasets keep their own licenses.

## Contact

Piyush Dhoka (piyushdhoka007@gmail.com) · Vedant Pandhare (vedant11734@gmail.com)
