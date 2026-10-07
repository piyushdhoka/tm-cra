"""Generate every LaTeX table body in paper/tables/ from results/ so no number is typed by hand.

Run from the repo root:  venv/Scripts/python paper/make_tables.py
Tables (bodies only; captions live in main.tex):
  tab_models      Table I    model families, cuts, devices
  tab_evalsuite   Table II   evaluation suite (static facts from rah/data/eval_sets.py + configs)
  tab_pools       Table III  healing pools and teacher-QC yield
  tab_damage      Table IV   retention after pruning (main cut, light cut) and breakage ranks
  tab_fit         Table V    fitted recovery ceilings and time-scales
  tab_main        Table VI   per-capability retention of the four allocation methods (3 seeds)
  tab_headline    Table VII  paired bootstrap on the headline metrics
  tab_scope       Table VIII scope pilots and the chosen scope
  tab_ablation    Table IX   ablations, variants and training-free baseline (1 seed)
  tab_cost        Table X    what the pilots cost
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES, OUT = REPO / "results", REPO / "paper" / "tables"
OUT.mkdir(exist_ok=True)

MODELS = ["llama", "qwen", "smollm", "olmo"]
NAME = {"llama": "Llama-3.2-1B", "qwen": "Qwen2.5-1.5B", "smollm": "SmolLM2-1.7B", "olmo": "OLMo-2-1B"}
CAPS = ["fmt", "tool", "inst", "math", "safe", "know"]
CAPNAME = {"fmt": "Struct.", "tool": "Tool", "inst": "Instr.", "math": "Math", "safe": "Safety", "know": "Knowl."}
POOLS = ["fmt", "tool", "inst", "math", "safe", "general"]
POOLNAME = {"fmt": "fmt", "tool": "tool", "inst": "inst", "math": "math", "safe": "safe", "general": "general"}
METHOD = {"uniform": "Uniform", "damage_prop": "Damage-prop.", "rah_sum": "RAH", "rah_capped": "RAH-capped",
          "rah_maxmin": "RAH-maxmin", "rah_proxy": "RAH-proxy", "last_k_uniform": "Last-$k$ + uniform",
          "linear_patch": "Linear patch", "rah_no_transfer": "RAH, no transfer", "rah_no_scope": "RAH, no scope choice",
          "rah_hyp": "RAH, hyperbolic curve"}
SCOPENAME = {"all_lora": "all-LoRA", "cut_lora": "cut-LoRA", "last_k": "last-$k$", "none": "--"}


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def f2(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, rows, header=None, spec=None, mid=None):
    """rows: list of lists of strings (already formatted); mid: row indices after which to put \\midrule."""
    lines = []
    if spec:
        lines.append("\\begin{tabular}{" + spec + "}")
    lines.append("\\toprule")
    if header:
        lines += [h + " \\\\" for h in header]
        lines.append("\\midrule")
    for i, r in enumerate(rows):
        lines.append(" & ".join(r) + " \\\\")
        if mid and i in mid:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    if spec:
        lines.append("\\end{tabular}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name, f"({len(rows)} rows)")


main = pd.read_csv(RES / "analysis" / "main.csv")
head = pd.read_csv(RES / "analysis" / "headline_tests.csv")
dmg = pd.read_csv(RES / "analysis" / "damage.csv")
budget = pd.read_csv(RES / "analysis" / "budget.csv")
summary = J(RES / "analysis" / "summary.json")
facts = J(REPO / "paper" / "model_facts.json") if (REPO / "paper" / "model_facts.json").exists() else {}


def row_of(m, meth):
    r = main[(main.model == m) & (main.method == meth)]
    return None if r.empty else r.iloc[0]


# ---------------------------------------------------------------- Table I: models
rows = []
for m in MODELS:
    spec = J(RES / m / "diagnosis" / "prune_specs.json")
    env = J(RES / m / "env.json")
    f = facts.get(m, {})
    L = spec["main"]["n_layers"]
    rem = spec["main"]["removed"]
    light = spec["light"]["removed"]
    rows.append([NAME[m], f.get("checkpoint", "--"), f.get("params", "--"), str(L),
                 ", ".join(map(str, rem)) + f" ({100 * len(rem) / L:.1f}\\%)",
                 ", ".join(map(str, light)), f.get("device", env.get("gpu", "--"))])
write("tab_models", rows, header=["Family & Checkpoint & Params & Blocks & Removed (main) & Removed (light) & Hardware"],
      spec="llrrllll")

# ---------------------------------------------------------------- Table II: evaluation suite (static)
base = J(REPO / "configs" / "base.yaml".replace(".yaml", ".yaml")) if False else None
cfg = __import__("yaml").safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))
nt, nd, mnt = cfg["data"]["eval_test"], cfg["data"]["eval_dev"], cfg["eval"]["max_new_tokens"]
suite = [
    ("fmt", "Structured output", "JSONSchemaBench (GitHub-easy)", "schema-valid JSON"),
    ("tool", "Function calling", "BFCL v3 simple + multiple", "AST match (name + arguments)"),
    ("inst", "Instruction following", "IFEval", "strict prompt-level accuracy"),
    ("math", "Mathematical reasoning", "GSM8K test", "exact final answer"),
    ("safe", "Safety refusal", "XSTest (100 unsafe + 100 benign)", "refuse unsafe, comply benign"),
    ("know", "Knowledge", "MMLU (all subjects)", "letter log-likelihood accuracy; ECE"),
]
rows = [[c, name, src, str(nt[c]), str(nd[c]), str(mnt.get(c, "--")) if c != "know" else "--", score]
        for c, name, src, score in suite]
write("tab_evalsuite", rows, header=["Key & Capability & Benchmark & Test & Dev & Max new tokens & Score"],
      spec="llllrrl")

# ---------------------------------------------------------------- Table III: pools and teacher QC
rows = []
pool_src = {"fmt": "JSON instances for glaive function schemas", "tool": "glaive-function-calling-v2",
            "inst": "Alpaca-cleaned + verifiable constraint", "math": "GSM8K train",
            "safe": "PKU-SafeRLHF harmful + Alpaca benign", "general": "Alpaca-cleaned"}
qc = {"fmt": "schema-valid", "tool": "known function + JSON args", "inst": "IFEval checker passes",
      "math": "final answer correct", "safe": "none", "general": "none"}
for p in POOLS:
    kept = [J(RES / m / "teacher_stats.json")[p]["kept"] for m in MODELS]
    rate = [J(RES / m / "teacher_stats.json")[p]["passed_qc"] / J(RES / m / "teacher_stats.json")[p]["prompts"] for m in MODELS]
    rows.append([p, pool_src[p], qc[p]] + [f"{k} ({100 * r:.0f}\\%)" for k, r in zip(kept, rate)])
write("tab_pools", rows, header=["Pool & Prompt source & Teacher QC & " + " & ".join(NAME[m] for m in MODELS)],
      spec="lllrrrr")

# ---------------------------------------------------------------- Table IV: damage and breakage ranks
rows, ranks = [], {}
for m in MODELS:
    d = dmg[dmg.model == m].set_index("cap")
    base_s = J(RES / m / "diagnosis" / "eval_base_test.json")["summary"]
    ranks[m] = d["retention_main"].rank().to_dict()
    rows.append([NAME[m], "unpruned score"] + [f2(base_s[c]) for c in CAPS])
    rows.append(["", "retention, light cut"] + [f2(d.loc[c, "retention_light"]) for c in CAPS])
    rows.append(["", "retention, main cut"] + [f2(d.loc[c, "retention_main"]) for c in CAPS])
    rows.append(["", "breakage rank (1 = worst)"] + [f"{int(ranks[m][c])}" for c in CAPS])
mean_rank = {c: np.mean([ranks[m][c] for m in MODELS]) for c in CAPS}
rows.append(["All", f"mean rank ($W$ = {summary['kendalls_w_breakage_order']:.2f})"] + [f"{mean_rank[c]:.2f}" for c in CAPS])
write("tab_damage", rows, header=["Family & Quantity & " + " & ".join(CAPNAME[c] for c in CAPS)], spec="llrrrrrr",
      mid=[3, 7, 11, 15])

# ---------------------------------------------------------------- Table V: fitted curves
rows = []
for m in MODELS:
    fit = J(RES / m / "fit" / "recovery.json")["exp"]
    h1 = J(RES / m / "fit" / "h1.json")
    rows.append([NAME[m], "damage $1-r_c$"] + [f2(1 - fit["r0"][c]) for c in CAPS])
    rows.append(["", "ceiling $a_c$"] + [f2(fit["a"][c]) for c in CAPS])
    rows.append(["", "time-scale $\\tau_c$ (k tokens)"] + [("n/a" if fit["a"][c] < 0.005 else f"{fit['tau'][c] / 1e3:.0f}") for c in CAPS])
    rows.append(["", "fit RMSE"] + [f2(fit["fit_rmse"][c], 3) for c in CAPS])
    rows.append(["", "$\\rho$(damage, $a_c$) / $\\rho$(damage, $\\tau_c$)",
                 f"\\multicolumn{{6}}{{c}}{{{h1['spearman_damage_vs_ceiling']:.2f} ($p$ = {h1['p']:.2f}) / "
                 f"{h1['spearman_damage_vs_tau']:.2f} ($p$ = {h1['p_tau']:.2f})}}"])
write("tab_fit", rows, header=["Family & Quantity & " + " & ".join(CAPNAME[c] for c in CAPS)], spec="llrrrrrr",
      mid=[4, 9, 14])

# ---------------------------------------------------------------- Table VI: main results per capability
rows, mid = [], []
core = ["uniform", "damage_prop", "rah_sum", "rah_capped"]
for m in MODELS:
    avail = [x for x in core if row_of(m, x) is not None]
    best_cm = max(row_of(m, x)["mean_ret_capped"] for x in avail)
    best_w = max(row_of(m, x)["worst_ret"] for x in avail)
    for i, meth in enumerate(avail):
        r = row_of(m, meth)
        cm, w = f2(r["mean_ret_capped"]), f2(r["worst_ret"])
        if abs(r["mean_ret_capped"] - best_cm) < 5e-4:
            cm = "\\textbf{" + cm + "}"
        if abs(r["worst_ret"] - best_w) < 5e-4:
            w = "\\textbf{" + w + "}"
        rows.append([NAME[m] if i == 0 else "", METHOD[meth]] + [f2(r[c]) for c in CAPS]
                    + [cm, f2(r["mean_ret"]), w, f2(r["safe_over_refusal"]), f2(r["know_ece"])])
    if meth == "rah_sum" and m == "qwen":
        rows.append(["", "RAH-capped", "\\multicolumn{11}{l}{identical allocation to RAH on this family (see Fig.~6)}"])
    mid.append(len(rows) - 1)
write("tab_main", rows, header=["Family & Method & " + " & ".join(CAPNAME[c] for c in CAPS)
                                + " & Capped mean & Mean & Worst & Over-refusal & ECE"],
      spec="llrrrrrrrrrrr", mid=mid[:-1])

# ---------------------------------------------------------------- Table VII: headline paired tests
rows, mid = [], []
for m in MODELS:
    ours_list = ["rah_sum"] + (["rah_capped"] if (head[(head.model == m) & (head.ours == "rah_capped")].shape[0] > 0) else [])
    first = True
    for ours in ours_list:
        for vs in ["uniform", "damage_prop"]:
            h = head[(head.model == m) & (head.ours == ours) & (head.vs == vs)].set_index("metric")
            if h.empty:
                continue
            cm, w = h.loc["capped_mean"], h.loc["worst"]
            rows.append([NAME[m] if first else "", METHOD[ours] + " vs " + METHOD[vs],
                         f"{cm['diff']:+.3f} [{cm['ci_low']:+.3f}, {cm['ci_high']:+.3f}]", pval(cm["p"]),
                         f"{w['diff']:+.3f} [{w['ci_low']:+.3f}, {w['ci_high']:+.3f}]", pval(w["p"])])
            first = False
    mid.append(len(rows) - 1)
write("tab_headline", rows, header=["Family & Comparison & $\\Delta$ capped mean [95\\% CI] & $p$ & $\\Delta$ worst case [95\\% CI] & $p$"],
      spec="llrrrr", mid=mid[:-1])

# ---------------------------------------------------------------- Table VIII: scope pilots
rows, mid = [], []
for m in MODELS:
    al = J(RES / m / "fit" / "allocations.json")
    sp = al["scope_pilots"]
    chosen = al["methods"]["rah_sum"]["scope"]
    lk = row_of(m, "last_k_uniform")
    un = row_of(m, "uniform")
    for i, s in enumerate(["all_lora", "cut_lora", "last_k"]):
        vals = [sp[s][c] for c in CAPS]
        rows.append([NAME[m] if i == 0 else "", SCOPENAME[s] + (" (chosen)" if s == chosen else "")]
                    + [f2(v) for v in vals] + [f2(np.mean(np.minimum(vals, 1.0))), f2(min(vals))])
    rows.append(["", "1M check: last-$k$ / all-LoRA, uniform mix",
                 f"\\multicolumn{{8}}{{l}}{{capped mean {f2(lk['mean_ret_capped'])} / {f2(un['mean_ret_capped'])}, "
                 f"worst {f2(lk['worst_ret'])} / {f2(un['worst_ret'])} (test split)}}"])
    mid.append(len(rows) - 1)
write("tab_scope", rows, header=["Family & Scope & " + " & ".join(CAPNAME[c] for c in CAPS)
                                 + " & Capped mean & Worst"], spec="llrrrrrrrr", mid=mid[:-1])

# ---------------------------------------------------------------- Table IX: ablations and other variants
rows, mid = [], []
variants = ["rah_sum", "rah_no_transfer", "rah_no_scope", "rah_hyp", "rah_maxmin", "rah_proxy", "last_k_uniform", "linear_patch"]
for m in MODELS:
    first = True
    for v in variants:
        r = row_of(m, v)
        if r is None:
            continue
        al = J(RES / m / "fit" / "allocations.json")["methods"].get(v, {})
        mix = al.get("tokens", {})
        share = " / ".join(f"{100 * mix.get(p, 0) / 1e6:.0f}" for p in POOLS) if mix else "--"
        rows.append([NAME[m] if first else "", METHOD[v] + (" (3 seeds)" if int(r["seeds"]) == 3 else ""),
                     SCOPENAME.get(r.get("scope", al.get("scope", "none")) if False else al.get("scope", "none"), "--") if v != "linear_patch" else "--",
                     share, f2(r["mean_ret_capped"]), f2(r["worst_ret"]), f2(r["safe_over_refusal"])])
        first = False
    mid.append(len(rows) - 1)
write("tab_ablation", rows, header=["Family & Variant & Scope & Mixture \\% (fmt/tool/inst/math/safe/general) & Capped mean & Worst & Over-refusal"],
      spec="lllcrrr", mid=mid[:-1])

# ---------------------------------------------------------------- Table X: cost of the pilots
rows = []
pb = cfg["pilots"]["budgets"]
n_rec = len(POOLS) * len(pb)
rec_tokens = len(POOLS) * sum(pb)
scope_tokens = len(cfg["pilots"]["scopes"]) * cfg["pilots"]["scope_budget"]
for m in MODELS:
    pil = [J(f) for f in sorted((RES / m / "pilots").glob("*.json"))]
    ev = np.mean([p["eval_seconds"] for p in pil])
    tr = {p["scope"]: p["train"]["n_trainable"] for p in pil}
    full = [J(f) for f in sorted((RES / m / "main").glob("rah_sum_s*.json"))]
    test_ev = np.mean([r["eval_seconds"] for r in full])
    rows.append([NAME[m], f"{n_rec} + {len(cfg['pilots']['scopes'])}", f"{(rec_tokens + scope_tokens) / 1e6:.2f}",
                 f"{(rec_tokens + scope_tokens) / cfg['healing']['budget_tokens']:.1f}$\\times$",
                 f"{tr.get('all_lora', 0) / 1e6:.1f} / {tr.get('cut_lora', 0) / 1e6:.1f} / {tr.get('last_k', 0) / 1e6:.0f}",
                 f"{ev:.0f}", f"{test_ev:.0f}"])
write("tab_cost", rows, header=["Family & Pilot runs & Pilot tokens (M) & vs. one 1M heal & Trainable params (M): all-LoRA / cut-LoRA / last-$k$ & Dev eval (s) & Test eval (s)"],
      spec="lrrrrrr")

# ---------------------------------------------------------------- budget sweep (inline numbers for the text)
bs = budget.pivot(index="budget", columns="method", values="mean_ret")
print(bs.round(3))
print("done")
