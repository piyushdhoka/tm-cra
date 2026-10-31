"""Stage 13: seed-aware significance tests, Holm correction, and agreement of the breakage order (CPU only).

Two pre-specified families of tests on the two primary metrics (capped mean and worst-case retention):

  fixed budget   RAH and RAH-capped against uniform and damage-proportional allocation, all at 1M tokens
                 (28 tests: RAH-capped was not run on Qwen, where it gives the same allocation as RAH)
  token-matched  RAH and RAH-capped at 1M tokens (plus 3.48M pilot tokens) against uniform healing at 4.48M tokens
                 (14 tests)

For every test it reports the item-level paired bootstrap of stages 9 and 10 (seeds pooled through per-item means,
read from headline_tests.csv and token_matched.csv) and the two-level bootstrap of rah.stats.two_level_bootstrap
(seeds resampled, then items), each with Holm-adjusted p-values within its family. A bootstrap p-value of 0 means
p < 2/n_boot (the two-sided p is twice a tail fraction); the Holm value is then computed from that bound and marked
as an upper bound.

It also measures how consistently the four families order capability damage: Kendall's W with the tie correction,
on raw and on chance-corrected retention, with a permutation p-value (each family's ranking permuted independently).

  python scripts/13_seed_aware_tests.py
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rah.config import REPO, load_config
from rah.stats import holm_bounded, kendalls_w_ties, two_level_bootstrap

MODELS = ["llama", "qwen", "smollm", "olmo"]
OURS, BASELINES, METRICS = ("rah_sum", "rah_capped"), ("uniform", "damage_prop"), ("capped_mean", "worst")
TOKEN_MATCHED = "uniform_b4480000"
CHANCE = {"fmt": 0.0, "tool": 0.0, "inst": 0.0, "math": 0.0, "safe": 0.5, "know": 0.25}
N_BOOT, N_PERM = 2000, 100_000
OUT = REPO / "results" / "analysis"


def J(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load_runs(res, pattern: str) -> list[dict]:
    return [J(f) for f in sorted(res.glob(pattern))]


def aligned(runs_a: list[dict], runs_b: list[dict], caps: list[str]):
    """Per-run score vectors on the item IDs that every run of both methods shares, in one fixed order."""
    def by_id(run, c):
        return {r["id"]: r["score"] for r in run["records"] if r["cap"] == c}
    ids = {c: sorted(set.intersection(*[set(by_id(r, c)) for r in runs_a + runs_b])) for c in caps}

    def vec(run):
        d = {c: by_id(run, c) for c in caps}
        return {c: np.array([d[c][i] for i in ids[c]], dtype=float) for c in caps}
    return [vec(r) for r in runs_a], [vec(r) for r in runs_b]


def test_rows() -> pd.DataFrame:
    head = pd.read_csv(OUT / "headline_tests.csv")
    tm = pd.read_csv(OUT / "token_matched.csv")
    rows = []
    for name in MODELS:
        res = REPO / "results" / name
        caps = load_config(name)["capabilities"]
        base = J(res / "diagnosis" / "eval_base_test.json")["summary"]
        runs = {m: load_runs(res / "main", f"{m}_s*.json") for m in OURS + BASELINES}
        runs["tm"] = load_runs(res / "budget", f"{TOKEN_MATCHED}_s*.json")
        for ours in OURS:
            if not runs[ours]:
                continue
            for other in BASELINES + ("tm",):
                a, b = aligned(runs[ours], runs[other], caps)
                seed = two_level_bootstrap(a, b, base, caps, n_boot=N_BOOT, seed=0)
                for metric in METRICS:
                    if other == "tm":
                        t = tm[(tm.model == name) & (tm.method == f"{ours}@1M")].iloc[0]
                        lab = "capped" if metric == "capped_mean" else "worst"
                        item = {"diff": t[f"d_{lab}_vs_cm"], "ci_low": t[f"d_{lab}_lo"],
                                "ci_high": t[f"d_{lab}_hi"], "p": t[f"p_{lab}"]}
                        family, vs = "token_matched", "uniform_4.48M"
                    else:
                        h = head[(head.model == name) & (head.ours == ours) & (head.vs == other)
                                 & (head.metric == metric)].iloc[0]
                        item = {k: h[k] for k in ("diff", "ci_low", "ci_high", "p")}
                        family, vs = "fixed_budget", other
                    s = seed[metric]
                    rows.append({"family_of_tests": family, "model": name, "method": ours, "vs": vs,
                                 "metric": metric, "seeds_method": len(runs[ours]), "seeds_vs": len(runs[other]),
                                 "diff": float(item["diff"]), "ci_low": float(item["ci_low"]),
                                 "ci_high": float(item["ci_high"]), "p_item": float(item["p"]),
                                 "seed_diff": s["diff"], "seed_ci_low": s["ci_low"], "seed_ci_high": s["ci_high"],
                                 "p_seed": s["p"]})
    df = pd.DataFrame(rows)
    for col in ("p_item", "p_seed"):
        df[f"{col}_holm"], df[f"{col}_holm_is_bound"] = np.nan, False
    for fam, g in df.groupby("family_of_tests"):
        for col in ("p_item", "p_seed"):
            adj, bound = holm_bounded(g[col].tolist(), N_BOOT)
            df.loc[g.index, f"{col}_holm"] = adj
            df.loc[g.index, f"{col}_holm_is_bound"] = bound
    return df


def breakage_agreement() -> dict:
    out = {}
    for label, corrected in (("raw", False), ("chance_corrected", True)):
        ranks, ret = [], {}
        for name in MODELS:
            caps = load_config(name)["capabilities"]
            diag = REPO / "results" / name / "diagnosis"
            base, pruned = J(diag / "eval_base_test.json")["summary"], J(diag / "eval_main_test.json")["summary"]
            k = {c: (CHANCE[c] if corrected else 0.0) for c in caps}
            if any(base[c] - k[c] <= 1e-9 for c in caps):
                raise ValueError(f"{name}: an unpruned score is at or below its chance floor; retention is undefined")
            r = [(pruned[c] - k[c]) / (base[c] - k[c]) for c in caps]
            ret[name] = dict(zip(caps, r))
            ranks.append(rankdata(r))                       # rank 1 = most damaged (lowest retention)
        R = np.array(ranks)
        w = kendalls_w_ties(R)
        rng = np.random.default_rng(0)
        perm = np.array([kendalls_w_ties(np.array([rng.permutation(row) for row in R])) for _ in range(N_PERM)])
        out[label] = {"kendalls_w_tie_corrected": w, "permutation_p": float((perm >= w - 1e-12).mean()),
                      "n_permutations": N_PERM, "mean_rank": dict(zip(caps, R.mean(0).tolist())),
                      "ranks": {n: dict(zip(caps, r.tolist())) for n, r in zip(MODELS, R)},
                      "retention": ret}
    return out


def main():
    df = test_rows()
    df.to_csv(OUT / "significance.csv", index=False)
    (OUT / "significance.md").write_text(df.round(4).to_markdown(index=False), encoding="utf-8")
    agree = breakage_agreement()
    (OUT / "breakage_agreement.json").write_text(json.dumps(agree, indent=1), encoding="utf-8")
    cols = ["family_of_tests", "model", "method", "vs", "metric", "diff", "p_item", "p_item_holm", "p_seed",
            "p_seed_holm"]
    print(df[cols].round(3).to_string(index=False))
    for k, v in agree.items():
        print(f"breakage order ({k}): W = {v['kendalls_w_tie_corrected']:.3f}, permutation p = {v['permutation_p']:.4f}")


if __name__ == "__main__":
    main()
