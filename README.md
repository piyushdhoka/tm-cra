# TM-CRA: Token-Matched Capability Recovery Assessment for Depth-Pruned Small Language Models

Code, per-item results, and analysis for the paper
**"A Token-Matched Protocol for Measuring Capability-Specific Damage and Recovery in Depth-Pruned Small Language Models"**
by Piyush Dhoka, Vedant Pandhare, Varun Inamdar, and Aadarsh Pathre (Vishwakarma University, Pune).

> arXiv link: *to be added after posting.*

Depth pruning removes whole Transformer blocks from a language model to make it smaller and faster, but it damages
capabilities unevenly. A short fine-tuning run, called **healing**, decides what comes back, and its token budget has to
be split across data that target different capabilities. Methods that plan this split are usually compared at a fixed
healing budget, which never charges them for the training tokens they spend on planning.

**TM-CRA** is an eight-step evaluation protocol that fixes this. It diagnoses six deployment-relevant capabilities
(structured output, function calling, instruction following, math, safety refusal, and knowledge), heals every
allocation rule with identical data, recipe, and seeds, and adds a **token-matched control**: a uniform heal that receives
the planner's pilot tokens as extra training. It then compares methods with seed-aware paired bootstrap tests and Holm
correction, with the primary metrics fixed in a public commit (`73e8582`) before any test result existed.

We validated TM-CRA on four instruction-tuned models: Llama-3.2-1B, Qwen2.5-1.5B, SmolLM2-1.7B, and OLMo-2-1B. The
planner under test is **RAH** (recoverability-aware healing), which plans the data mixture and adaptation scope from 21
short pilot heals.

## Key findings

- **What breaks.** Function calling and math collapse in every model. Once chance floors are corrected for, which of the
  other capabilities look robust depends on the model.
- **What is trained matters as much as the data.** Fully tuning only the last three blocks instead of adding low-rank
  adapters to all blocks ties on one model and loses 0.05–0.18 of capped mean retention on the other three. One
  300k-token scope pilot catches this.
- **At a fixed 1M-token budget**, planning never significantly hurts mean retention and lifts the weakest capability
  (math) by 0.17 on Llama and 0.14 on OLMo.
- **At equal training tokens**, simply healing longer with uniform data is never significantly worse than planning, and
  beats it on the weakest capability on Qwen and SmolLM2 (by 0.11 and 0.24). Short pilots cannot see how slowly math
  recovers, so pilot-based planning only makes sense when the healing budget itself is fixed.

| Model | Uniform | Damage-proportional | RAH | RAH-capped | Uniform, 4.48M tokens |
|---|---|---|---|---|---|
| Llama-3.2-1B | 0.82 / 0.38 | 0.83 / 0.43 | 0.84 / 0.54 | 0.83 / 0.55 | 0.86 / 0.58 |
| Qwen2.5-1.5B | 0.62 / 0.21 | 0.61 / 0.19 | 0.64 / 0.19 | – | 0.65 / 0.29 |
| SmolLM2-1.7B | 0.77 / 0.15 | 0.76 / 0.14 | 0.77 / 0.10 | 0.78 / 0.10 | 0.79 / 0.34 |
| OLMo-2-1B | 0.74 / 0.21 | 0.77 / 0.31 | 0.75 / 0.29 | 0.76 / 0.35 | 0.78 / 0.37 |

*Capped mean retention / worst-capability retention, relative to the unpruned model, on the test split, averaged over
three seeds. The first four columns heal with 1M tokens; RAH also spends 3.48M tokens on pilots, and the last column
gives uniform healing that same 4.48M-token total. On Qwen, RAH-capped gives the same allocation as RAH.*

## Repository layout

| Path | Contents |
|---|---|
| `rah/` | The library: Block-Influence pruning (`influence.py`), healing (`healing.py`), recovery curves (`recovery.py`), allocation rules (`allocate.py`), the evaluation suite (`evaluation/`), the linear patch (`linear_patch.py`), and statistics (`stats.py`) |
| `scripts/` | The pipeline, one numbered script per stage (`00_check_env.py` to `10_compute_matched.py`) |
| `configs/` | One settings file per model (checkpoint, cut, budgets) |
| `data/` | Evaluation items and healing prompts |
| `results/` | Every run with per-item scores; summary tables and figures in `results/analysis/` |
| `tests/` | Unit tests (`pytest`) |

## Reproducing the results

Install PyTorch for your CUDA version first (the runs used PyTorch 2.11.0), then:

```bash
pip install -r requirements.txt
```

Each stage script takes the model name (`llama`, `qwen`, `smollm`, or `olmo`), reads its settings from `configs/`, and
skips runs that have already finished, so an interrupted stage can be relaunched.

| Protocol step | Command | What it does |
|---|---|---|
| Steps 2–5 | `bash scripts/run_all.sh llama` (Windows: `.\scripts\run_all.ps1 -Model llama`) | Environment check, evaluation items, pool prompts, teacher targets, pruning and diagnosis, pilots, allocation, and the fixed-budget heals |
| Step 6 | `python scripts/07_main.py --model llama --only uniform --budgets 4480000 --sweep-seeds 1 2` | Adds seeds of the token-matched uniform heal (seed 0 is run by stage 07) |
| Step 7 | `python scripts/09_headline_tests.py` and `python scripts/10_compute_matched.py` | Paired bootstrap tests at the fixed budget and against the token-matched heal |
| Step 8 | `python scripts/08_analyze.py --models qwen llama smollm olmo` | Writes all summary tables (CSV and Markdown) and figures to `results/analysis/` |

To apply TM-CRA to another model, add a settings file for it to `configs/` with its checkpoint and cut, and run the
same commands with its name.

**Hardware.** All runs used NVIDIA A100 40 GB GPUs.
environment of each model's runs is recorded in `results/<model>/env.json`.

## Citation

```bibtex
@misc{dhoka2026tmcra,
  title  = {A Token-Matched Protocol for Measuring Capability-Specific Damage and Recovery in Depth-Pruned Small Language Models},
  author = {Dhoka, Piyush and Pandhare, Vedant and Inamdar, Varun and Pathre, Aadarsh},
  year   = {2026},
  howpublished = {\url{https://github.com/piyushdhoka/tm-cra}}
}
```

## License

The code is released under the [MIT License](LICENSE). The models and datasets it uses keep their own licenses.
