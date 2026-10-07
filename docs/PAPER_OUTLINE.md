# Paper Outline: Recoverability-Aware Healing (IEEE Access)

**Title (fixed 2026-10-07):** *Recoverability-Aware Healing: Pilot-Guided Allocation of Data Mixture and Adaptation Scope for Depth-Pruned Small Language Models*
**Manuscript type:** Research Article. **Target length:** 12–14 template pages (IEEE Access strongly recommends < 20).
**Template:** official IEEE Access LaTeX template (2026-05-13), copied into `paper/`.

---

## 1. Claims the paper makes (each backed by a table or figure)

| # | Claim | Evidence |
|---|---|---|
| C1 | After Block-Influence depth pruning of 1–2B instruct models, capabilities break in a consistent order across four families: function calling and math first, safety and knowledge last. | Kendall's W = 0.92 (4 families); `damage.csv` |
| C2 | Damage does not predict recovery *speed* (ρ(damage, τ) = 0.43, 0.03, −0.14, −0.77), and recovery *ceilings* track damage on only two of four families (ρ = 0.77, 0.67, −0.09, 0.43). Damage alone is therefore not enough to plan healing. | `h1.json`, Fig. 3 |
| C3 | At an equal 1M-token budget, RAH is never worse than uniform healing on capped mean retention (significant on Llama +0.031, p = 0.014, and OLMo-capped +0.025, p = 0.043) and beats damage-proportional allocation on Llama (worst case +0.12, p = 0.02) and Qwen (mean +0.024, p = 0.039), tying on SmolLM and OLMo. | `headline_tests.csv`, Table III/IV |
| C4 | RAH raises the worst-retained capability by +0.17 (Llama) and +0.14 (OLMo) over uniform healing. | `headline_tests.csv` |
| C5 | RAH cuts over-refusal: Qwen 0.59 → 0.06, SmolLM 0.21 → 0.08 (capped), Llama 0.16 → 0.09. | `main.csv` |
| C6 | Adaptation scope matters and is family-specific: full tuning of the last three blocks is best on Llama but harmful on Qwen, SmolLM and OLMo (capped mean 0.46 / 0.59 / 0.69 vs ≥ 0.74 for all-layer LoRA). Scope selection from pilots avoids this. | `main.csv`, scope pilots |
| C7 | Budget sweep (Qwen): RAH loses at 0.25M tokens, wins at 1M, ties at 4M: pilots pay off at moderate budgets. | `budget.csv` |
| C8 | Negative results, reported as such: the worst-case (max-min) objective is worse than uniform on three of four families; the pilot-free proxy ≈ uniform; the training-free linear patch is far behind. | `main.csv` |
| C9 | The whole pipeline runs on an 8 GB laptop GPU; code, teacher-data recipe, per-item predictions and all results are public. | repo |

**We will not claim:** that "most damaged is not most recoverable" (not supported on Llama/Qwen); that capability-aware data selection is new (PASER); that math/code breaking first is new; that cut-adjacent tuning or self-distillation is new; state-of-the-art compression.

---

## 2. Section plan (about 13 pages)

| Section | Pages | Content |
|---|---|---|
| I. Introduction | 1.25 | Small instruct models as agents; depth pruning; healing has a budget; current rules (uniform, damage-proportional); our proposal: plan from cheap pilots; contributions (4 bullets); Fig. 1 overview |
| II. Related Work | 1 | Depth-pruning criteria; healing recipes and scopes; capability-aware recovery (PASER, limits-of-layer-pruning, smart-home breakage); data-mixture optimisation (DoReMi, Sheared-LLaMA DBL); self-distillation targets |
| III. Problem Setting | 0.75 | Notation; retention r_c; capped mean and worst case (pre-registered); budget B; scopes (all-LoRA, cut-LoRA, last-k) |
| IV. Recoverability-Aware Healing | 2 | Pilots (6 pools × 3 budgets + 3 scope pilots); recovery curves g_c(x) = a_c(1 − e^{−x/τ_c}); transfer matrix T; concave allocation (sum, max-min, capped); scope choice; safeguards (τ lower bound, trust region, objective cap); pilot-free proxy; Algorithm 1; cost accounting |
| V. Experimental Setup | 1.5 | Models and cuts (Table I; the Llama k = 4 collapse motivates k = 2); evaluation suite (Table II); healing pools and teacher QC; baselines; metrics and statistics (pre-registration dates); hardware (RTX 4060 laptop; A100 for SmolLM/OLMo; L4 for Llama capped seeds 1–2) |
| VI. Results | 4 | A. breakage order; B. damage vs recoverability; C. main comparison (Table III, Fig. 7); D. worst case and over-refusal; E. scope; F. budget sweep; G. ablations (no transfer, no scope, hyperbolic curve); H. negative results |
| VII. Discussion | 1 | When RAH helps and when it does not (OLMo); pilot overhead vs gain; threats to validity (1–2B only, evaluation subsets, one pruning criterion, string-match safety classifier, two GPU types) |
| VIII. Conclusion | 0.4 | |
| Acknowledgment | | Includes the IEEE-required AI-use disclosure |
| References | | about 45, all 2025–2026 depth-pruning/healing work cited and verified |
| Biographies | | all authors, below references |

---

## 3. Figures (vector PDF, IEEE sizes: 3.5 in column / 7.16 in page)

| Fig. | Content | Source |
|---|---|---|
| 1 | Pipeline overview: prune → diagnose → pilots → fit → allocate → heal | new drawing |
| 2 | Retention after pruning, per capability × family (breakage order) | `damage.csv` |
| 3 | Damage vs recovery ceiling (4 panels) with τ annotations | restyle `*_damage_vs_ceiling.pdf` |
| 4 | Fitted recovery curves per capability (4 panels) | restyle `*_recovery_curves.pdf` |
| 5 | Transfer matrices (4 heat maps) | restyle `*_transfer.pdf` |
| 6 | Token allocation chosen by each method (stacked bars per family) | `allocations.json` |
| 7 | Retention by method and capability (4 panels) | restyle `*_retention_by_method.pdf` |
| 8 | Worst-case retention and over-refusal by method | `main.csv` |
| 9 | Budget sweep on Qwen | `budget.csv` |

## 4. Tables (Roman numerals, title above)

I models and pruning levels · II evaluation suite · III main results per family (capped mean, worst case, over-refusal, ECE; 3 seeds) · IV paired bootstrap on headline metrics · V scope pilots · VI ablations · VII compute cost (pilots vs full heal, GPU-hours)

---

## 5. Open items before the draft is complete

- Author names, affiliations, e-mails, ORCID of the submitting author, and 100–150-word biographies (needed from the authors).
- Verify every 2025–2026 citation (arXiv IDs, authors, venues) with web lookups before it enters `references.bib`.
- Regenerate all figures in IEEE style (Times/Helvetica, 8–10 pt labels, no gray backgrounds).
- Pull pilot and heal wall-clock times from the logs for Table VII.
