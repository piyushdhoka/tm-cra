"""Stage 14: every number the arXiv paper reports, in one file, plus the LaTeX bodies of its result tables (CPU only).

Reads results/<model>/ and results/analysis/ (run stages 8-10 and 13 first) and writes
  results/analysis/paper_numbers.json   all values quoted in the paper text
  arxiv/tables/*.tex                    table bodies included by arxiv/main.tex

  python scripts/14_paper_numbers.py
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd

from rah import allocate as A
from rah.config import REPO, load_config
from rah.recovery import RecoveryModel

MODELS = ["llama", "qwen", "smollm", "olmo"]
LABEL = {"llama": "Llama-3.2-1B", "qwen": "Qwen2.5-1.5B", "smollm": "SmolLM2-1.7B", "olmo": "OLMo-2-1B"}
CHECKPOINT = {"llama": "unsloth/Llama-3.2-1B-Instruct", "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
              "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "olmo": "allenai/OLMo-2-0425-1B-Instruct"}
PARAMS_B = {"llama": 1.24, "qwen": 1.54, "smollm": 1.71, "olmo": 1.48}       # from the model cards
GPU = {"llama": "RTX 4060 (laptop); L4", "qwen": "RTX 4060 (laptop); L4", "smollm": "A100", "olmo": "A100"}
CAPS = ["fmt", "tool", "inst", "math", "safe", "know"]
CAP_NAME = {"fmt": "Struct.", "tool": "Tool", "inst": "Instr.", "math": "Math", "safe": "Safety", "know": "Knowl."}
CHANCE = {"fmt": 0.0, "tool": 0.0, "inst": 0.0, "math": 0.0, "safe": 0.5, "know": 0.25}
METHODS = [("uniform", "main", "uniform_s*.json", "Uniform", "1M"),
           ("damage_prop", "main", "damage_prop_s*.json", "Damage-prop.", "1M"),
           ("rah_sum", "main", "rah_sum_s*.json", "RAH", "1M + 3.48M"),
           ("rah_capped", "main", "rah_capped_s*.json", "RAH-capped", "1M + 3.48M"),
           ("last_k_uniform", "main", "last_k_uniform_s*.json", "Uniform, last-$k$", "1M"),
           ("linear_patch", "main", "linear_patch_s*.json", "Linear patch", "0"),
           ("uniform_4480k", "budget", "uniform_b4480000_s*.json", "Uniform", "4.48M")]
ANALYSIS = REPO / "results" / "analysis"
TABLES = REPO / "arxiv" / "tables"


def J(p):
    return json.loads(p.read_text(encoding="utf-8"))


def chance_corrected(score: float, base: float, kappa: float) -> float:
    if base - kappa <= 1e-9:
        raise ValueError("unpruned score at or below its chance floor: chance-corrected retention is undefined")
    return (score - kappa) / (base - kappa)


def primary(r: dict) -> tuple[float, float]:
    v = np.array([r[c] for c in CAPS])
    return float(np.minimum(v, 1.0).mean()), float(v.min())


def method_block(res, base, pattern_dir, pattern) -> dict | None:
    runs = [J(f) for f in sorted((res / pattern_dir).glob(pattern))]
    if not runs:
        return None
    score = {c: float(np.mean([r["summary"][c] for r in runs])) for c in CAPS}
    ret = {c: score[c] / base[c] for c in CAPS}
    cm, worst = primary(ret)
    per_seed = [primary({c: r["summary"][c] / base[c] for c in CAPS}) for r in runs]
    sd = (lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"))
    tokens = runs[0].get("tokens_per_pool", {})
    return {"seeds": len(runs), "score": score, "retention": ret, "capped_mean": cm, "worst": worst,
            "worst_cap": min(CAPS, key=lambda c: ret[c]),
            "capped_mean_seed_sd": sd([p[0] for p in per_seed]), "worst_seed_sd": sd([p[1] for p in per_seed]),
            "per_seed": [{"capped_mean": p[0], "worst": p[1]} for p in per_seed],
            "unsafe_refused": float(np.mean([r["summary"]["safe_harmful_refusal"] for r in runs])),
            "benign_refused": float(np.mean([r["summary"]["safe_over_refusal"] for r in runs])),
            "know_ece": float(np.mean([r["summary"]["know_ece"] for r in runs])),
            "scope": runs[0].get("scope"), "train_tokens": sum(tokens.values()) if tokens else 0.0,
            "tokens_per_pool": tokens,
            "realized_tokens": [r.get("train", {}).get("tokens") for r in runs]}


def family_numbers(name: str) -> dict:
    res = REPO / "results" / name
    cfg = load_config(name)
    diag = res / "diagnosis"
    base = J(diag / "eval_base_test.json")["summary"]
    pruned = J(diag / "eval_main_test.json")["summary"]
    light = J(diag / "eval_light_test.json")["summary"]
    spec = J(diag / "prune_specs.json")
    out = {"label": LABEL[name], "checkpoint": CHECKPOINT[name], "params_b": PARAMS_B[name], "gpu": GPU[name],
           "blocks": spec["main"]["n_layers"], "removed_main": spec["main"]["removed"],
           "removed_light": spec["light"]["removed"],
           "unpruned": {c: base[c] for c in CAPS},
           "unpruned_refusal": {"unsafe": base["safe_harmful_refusal"], "benign": base["safe_over_refusal"],
                                "know_ece": base["know_ece"]},
           "pruned": {c: pruned[c] for c in CAPS},
           "pruned_refusal": {"unsafe": pruned["safe_harmful_refusal"], "benign": pruned["safe_over_refusal"],
                              "know_ece": pruned["know_ece"]},
           "retention_main": {c: pruned[c] / base[c] for c in CAPS},
           "retention_light": {c: light[c] / base[c] for c in CAPS},
           "retention_main_corrected": {c: chance_corrected(pruned[c], base[c], CHANCE[c]) for c in CAPS},
           "teacher_kept": {p: v["kept"] for p, v in J(res / "teacher_stats.json").items()},
           "teacher_pass_rate": {p: v["passed_qc"] / v["prompts"] for p, v in J(res / "teacher_stats.json").items()},
           "methods": {}}
    for key, d, pat, _, _ in METHODS:
        blk = method_block(res, base, d, pat)
        if blk:
            out["methods"][key] = blk
    alloc = J(res / "fit" / "allocations.json")
    B = alloc["budget"]
    out["allocation_share"] = {m: {p: t / B for p, t in v["tokens"].items()} for m, v in alloc["methods"].items()
                               if m in ("uniform", "damage_prop", "rah_sum", "rah_capped")}
    out["allocation_scope"] = {m: v["scope"] for m, v in alloc["methods"].items()}
    out["budget_sweep_share"] = {b: {m: {p: t / float(b) for p, t in v["tokens"].items()} for m, v in ms.items()}
                                 for b, ms in alloc["budget_sweep"].items()}
    # RAH-capped recomputed exactly as stage 6 does (trust region 2 x largest pilot, objective capped at 1.0);
    # on Qwen it was not run because it equals RAH's allocation, which this records
    fc = cfg["fit"]
    capped = A.rah(RecoveryModel.from_dict(J(res / "fit" / "recovery.json")[fc["curve"]]), B, "sum",
                   max_per_pool=fc["trust_factor"] * max(cfg["pilots"]["budgets"]), pred_cap=fc["objective_cap"])
    out["rah_capped_recomputed_share"] = {p: t / B for p, t in capped.items()}
    out["rah_capped_equals_rah"] = bool(all(abs(capped[p] - alloc["methods"]["rah_sum"]["tokens"][p]) < 1.0
                                            for p in capped))
    base_dev = J(diag / "eval_base_dev.json")["summary"]
    out["scope_pilots_dev"] = {s: {"retention": r, "capped_mean": primary(r)[0], "worst": primary(r)[1]}
                               for s, r in alloc["scope_pilots"].items()}
    # recovery fit (exponential curves, development split) and its check against test retention
    fit = RecoveryModel.from_dict(J(res / "fit" / "recovery.json")["exp"])
    out["fit"] = {"a": fit.a, "tau": fit.tau, "r0_dev": fit.r0, "rmse": J(res / "fit" / "recovery.json")["exp"].get("fit_rmse")}
    checks = {}
    for key in ("uniform", "damage_prop", "rah_sum", "rah_capped", "uniform_4480k"):
        m = out["methods"].get(key)
        if not m or m["scope"] != "all_lora":
            continue          # the curves describe all-LoRA healing only
        pred = fit.predict(m["tokens_per_pool"])
        obs = m["retention"]
        pc, _ = primary(pred)
        checks[key] = {"mae": float(np.mean([abs(pred[c] - obs[c]) for c in CAPS])),
                       "capped_mean_pred": pc, "capped_mean_obs": m["capped_mean"],
                       "math_pred": pred["math"], "math_obs": obs["math"],
                       "tool_pred": pred["tool"], "tool_obs": obs["tool"],
                       "math_effective_tokens": fit.effective_tokens(m["tokens_per_pool"])["math"]}
    out["prediction_check"] = checks
    out["math_pilots_dev"] = []
    for b in cfg["pilots"]["budgets"]:
        r = J(res / "pilots" / f"recovery_math_{b}.json")["summary"]
        out["math_pilots_dev"].append({"budget": b, "retention": r["math"] / base_dev["math"]})
    return out


def fmt_p(p: float, bound: bool = False) -> str:
    if bound:
        return f"$\\le${p:.3f}"
    if p == 0.0:
        return "$<$0.001"
    return f"{p:.3f}"


def write_tables(nums: dict, sig: pd.DataFrame):
    TABLES.mkdir(parents=True, exist_ok=True)
    # Table: models, cuts, hardware
    def ranges(idx: list[int]) -> str:
        spans, out = [], []
        for i in sorted(idx):
            if spans and i == spans[-1][1] + 1:
                spans[-1][1] = i
            else:
                spans.append([i, i])
        for a, b in spans:
            out.append(str(a) if a == b else (f"{a}, {b}" if b == a + 1 else f"{a}--{b}"))
        return ", ".join(out)

    lines = []
    for n in MODELS:
        f = nums["families"][n]
        k = len(f["removed_main"])
        lines.append(f"{f['label']} & \\texttt{{{f['checkpoint']}}} & {f['params_b']:.2f} & {f['blocks']} & "
                     f"{ranges(f['removed_main'])} ({100 * k / f['blocks']:.1f}\\%) & {f['gpu']} \\\\")
    (TABLES / "models.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: damage after pruning
    lines = []
    for n in MODELS:
        f = nums["families"][n]
        rk = nums["breakage"]["chance_corrected"]["ranks"][n]
        lines.append(f"\\multirow{{3}}{{*}}{{{f['label']}}} & unpruned score & "
                     + " & ".join(f"{f['unpruned'][c]:.2f}" for c in CAPS) + " \\\\")
        lines.append(" & retention & " + " & ".join(f"{f['retention_main'][c]:.2f}" for c in CAPS) + " \\\\")
        lines.append(" & chance-corrected & " + " & ".join(f"{f['retention_main_corrected'][c]:.2f}" for c in CAPS)
                     + " \\\\")
        lines.append("\\midrule" if n != MODELS[-1] else "")
    (TABLES / "damage.tex").write_text("\n".join(l for l in lines if l) + "\n", encoding="utf-8")

    # Table: main results
    order = ["uniform", "damage_prop", "rah_sum", "rah_capped", "uniform_4480k"]
    names = {k: (lab, tok) for k, _, _, lab, tok in METHODS}
    lines = []
    for n in MODELS:
        f = nums["families"][n]
        rows = [k for k in order if k in f["methods"]]
        best_c = max(f["methods"][k]["capped_mean"] for k in rows)
        best_w = max(f["methods"][k]["worst"] for k in rows)
        for i, k in enumerate(rows):
            m = f["methods"][k]
            lab, tok = names[k]
            fam = f"\\multirow{{{len(rows)}}}{{*}}{{{f['label']}}}" if i == 0 else ""
            cm = f"{m['capped_mean']:.2f}"
            wr = f"{m['worst']:.2f}"
            cm = f"\\textbf{{{cm}}}" if abs(m["capped_mean"] - best_c) < 1e-12 else cm
            wr = f"\\textbf{{{wr}}}" if abs(m["worst"] - best_w) < 1e-12 else wr
            caps = " & ".join(f"{m['retention'][c]:.2f}" for c in CAPS)
            lines.append(f"{fam} & {lab} & {tok} & {caps} & {cm} & {wr} & {m['benign_refused']:.2f} \\\\")
        if n != MODELS[-1]:
            lines.append("\\midrule")
    (TABLES / "main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: significance
    meth = {"rah_sum": "RAH", "rah_capped": "RAH-capped"}
    vs = {"uniform": "uniform", "damage_prop": "damage-prop.", "uniform_4.48M": "uniform 4.48M"}
    lines = []
    for fam, title in (("fixed_budget", "Fixed budget (1M tokens each; 28 tests)"),
                       ("token_matched", "Token-matched (4.48M training tokens each; 14 tests)")):
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{title}}}}} \\\\")
        g = sig[sig.family_of_tests == fam]
        for (model, method, other), h in g.groupby(["model", "method", "vs"], sort=False):
            cells = []
            for metric in ("capped_mean", "worst"):
                r = h[h.metric == metric].iloc[0]
                cells += [f"{r['diff']:+.3f}", fmt_p(r["p_item_holm"], r["p_item_holm_is_bound"]),
                          fmt_p(r["p_seed_holm"], r["p_seed_holm_is_bound"])]
            lines.append(f"{LABEL[model]} & {meth[method]} vs {vs[other]} & " + " & ".join(cells) + " \\\\")
        if fam == "fixed_budget":
            lines.append("\\midrule")
    (TABLES / "significance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    nums = {"families": {n: family_numbers(n) for n in MODELS}}
    nums["breakage"] = J(ANALYSIS / "breakage_agreement.json")
    steps = {"1M": [], "4.48M": []}
    for n in MODELS:
        res = REPO / "results" / n
        for f in (res / "main").glob("*_s*.json"):
            r = J(f)
            if r.get("train"):                               # the linear patch is not trained
                steps["1M"].append(r["train"]["steps"])
        steps["4.48M"] += [J(f)["train"]["steps"] for f in (res / "budget").glob("uniform_b4480000_s*.json")]
    nums["optimizer_steps"] = {k: [min(v), max(v)] for k, v in steps.items()}
    sig = pd.read_csv(ANALYSIS / "significance.csv")
    nums["significance"] = sig.to_dict(orient="records")
    val = J(ANALYSIS / "refusal_validation" / "summary.json")
    nums["refusal_validation"] = {k: val[k] for k in ("labels", "all", "unsafe", "benign")}
    budget = pd.read_csv(ANALYSIS / "budget.csv")
    nums["qwen_budget_sweep"] = (budget[budget.model == "qwen"].groupby(["method", "budget"])["mean_ret"]
                                 .agg(["mean", "count"]).reset_index().to_dict(orient="records"))
    (ANALYSIS / "paper_numbers.json").write_text(json.dumps(nums, indent=1), encoding="utf-8")
    write_tables(nums, sig)
    for n in MODELS:
        f = nums["families"][n]
        print(n, {k: (round(v["capped_mean"], 3), round(v["worst"], 3), v["seeds"]) for k, v in f["methods"].items()})


if __name__ == "__main__":
    main()
