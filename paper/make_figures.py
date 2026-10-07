"""Generate every data figure of the paper as vector PDF in paper/figures/ (IEEE sizes: 3.5 in column, 7.16 in page).

Run from the repo root:  venv/Scripts/python paper/make_figures.py
Design rules (dataviz skill): fixed categorical slot order, thin marks, hairline recessive grid, legend for >= 2 series,
text in ink tokens (never the series colour), one axis per panel, a validated palette.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
RES, OUT = REPO / "results", REPO / "paper" / "figures"
OUT.mkdir(exist_ok=True)

MODELS = ["llama", "qwen", "smollm", "olmo"]
NAME = {"llama": "Llama-3.2-1B", "qwen": "Qwen2.5-1.5B", "smollm": "SmolLM2-1.7B", "olmo": "OLMo-2-1B"}
CAPS = ["fmt", "tool", "inst", "math", "safe", "know"]
CAPNAME = {"fmt": "structured output", "tool": "function calling", "inst": "instruction following",
           "math": "math", "safe": "safety", "know": "knowledge"}
POOLS = ["fmt", "tool", "inst", "math", "safe", "general"]

# validated categorical slots (light surface): blue, orange, aqua, yellow, magenta, green
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# methods: the baseline is the de-emphasis gray; the three coloured slots validate (adjacent, light)
MCOL = {"uniform": "#8a8984", "damage_prop": "#eb6834", "rah_sum": "#2a78d6", "rah_capped": "#1baf7a"}
MNAME = {"uniform": "Uniform", "damage_prop": "Damage-prop.", "rah_sum": "RAH", "rah_capped": "RAH-capped"}
FCOL = dict(zip(MODELS, SLOT[:4]))                     # families as series (adjacent use only)
PCOL = dict(zip(POOLS, SLOT))                          # pools as stacked segments (adjacent use)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "axes.titlecolor": INK,
})
COL_W, PAGE_W = 3.5, 7.16


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


main = pd.read_csv(RES / "analysis" / "main.csv")
dmg = pd.read_csv(RES / "analysis" / "damage.csv")
budget = pd.read_csv(RES / "analysis" / "budget.csv")
fits = {m: J(RES / m / "fit" / "recovery.json")["exp"] for m in MODELS}
allocs = {m: J(RES / m / "fit" / "allocations.json") for m in MODELS}
gains = {m: J(RES / m / "fit" / "pilot_gains.json") for m in MODELS}


def row(m, meth):
    r = main[(main.model == m) & (main.method == meth)]
    return None if r.empty else r.iloc[0]


# ---------------------------------------------------------------- Fig. 2: what breaks (retention after the main cut)
fig, ax = plt.subplots(figsize=(COL_W, 2.1))
x = np.arange(len(CAPS))
w = 0.19
for k, m in enumerate(MODELS):
    d = dmg[dmg.model == m].set_index("cap")
    vals = [min(d.loc[c, "retention_main"], 1.3) for c in CAPS]
    ax.bar(x + (k - 1.5) * w, vals, width=w * 0.9, color=FCOL[m], label=NAME[m], linewidth=0)
ax.axhline(1.0, color=INK, lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([CAPNAME[c].replace(" ", "\n") for c in CAPS])
ax.set_ylabel("Retention after pruning")
ax.set_ylim(0, 1.32)
ax.legend(ncol=2, loc="upper left", handlelength=1.0, columnspacing=1.0)
style(ax)
save(fig, "fig2_breakage")

# ---------------------------------------------------------------- Fig. 3: damage vs recovery ceiling (4 panels)
fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.9), sharey=True)
for ax, m in zip(axes, MODELS):
    f, h1 = fits[m], J(RES / m / "fit" / "h1.json")
    d = [1 - f["r0"][c] for c in CAPS]
    a = [f["a"][c] for c in CAPS]
    ax.plot([-0.1, 1.2], [-0.1, 1.2], color=GRID, lw=0.6)
    ax.scatter(d, a, s=18, color=SLOT[0], zorder=3, linewidths=1.2, edgecolors="#fcfcfb")
    for c, xx, yy in zip(CAPS, d, a):
        ax.annotate(c, (xx, yy), textcoords="offset points", xytext=(3, 2), fontsize=6.5, color=INK2)
    ax.set_title(f"{NAME[m]}  ($\\rho$ = {h1['spearman_damage_vs_ceiling']:.2f})")
    ax.set_xlabel("Damage $1 - r_c$")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.05, 1.2)
    style(ax, grid_axis="both")
axes[0].set_ylabel("Recovery ceiling $a_c$")
fig.tight_layout(w_pad=0.6)
save(fig, "fig3_damage_vs_ceiling")

# ---------------------------------------------------------------- Fig. 4: recovery curves, one panel per capability, families as lines
fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, 3.4), sharex=True)
xs = np.linspace(0, 6e5, 200)
for ax, c in zip(axes.flat, CAPS):
    for m in MODELS:
        f = fits[m]
        tau = f["tau"][c]
        y = f["r0"][c] + f["a"][c] * (1 - np.exp(-xs / tau))
        ax.plot(xs / 1e3, np.minimum(y, 1.3), color=FCOL[m], lw=1.6, label=NAME[m])
        own = {"know": "general"}.get(c, c)
        pts = [(int(b), f["r0"][c] + g[c]) for b, g in gains[m][own].items()]
        ax.scatter([b / 1e3 for b, _ in pts], [min(v, 1.3) for _, v in pts], s=9, color=FCOL[m], zorder=3,
                   linewidths=0.8, edgecolors="#fcfcfb")
    ax.axhline(1.0, color=INK, lw=0.5)
    ax.set_title(CAPNAME[c])
    ax.set_ylim(-0.02, 1.32)
    style(ax)
for ax in axes[1]:
    ax.set_xlabel("Own-pool healing tokens (k)")
for ax in axes[:, 0]:
    ax.set_ylabel("Retention")
axes[1, 1].legend(loc="lower right", handlelength=1.2)          # the safety panel has free space below the curves
fig.tight_layout(h_pad=0.8, w_pad=0.6)
save(fig, "fig4_recovery_curves")

# ---------------------------------------------------------------- Fig. 5: transfer matrices (sequential blue)
fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 2.0))
for ax, m in zip(axes, MODELS):
    f = fits[m]
    T = np.array([[f["T"][c][p] for p in POOLS] for c in CAPS])
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CAPS)):
        for j in range(len(POOLS)):
            v = T[i, j]
            if 0.005 < v < 1:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                        color="#ffffff" if v > 0.6 else INK)
    ax.set_xticks(range(len(POOLS)))
    ax.set_xticklabels(POOLS, rotation=45, ha="right")
    ax.set_yticks(range(len(CAPS)))
    ax.set_yticklabels(CAPS if ax is axes[0] else [])
    ax.set_title(NAME[m])
    ax.set_xlabel("Healing pool")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("Capability")
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("Transfer $T_{cp}$")
cb.outline.set_visible(False)
save(fig, "fig5_transfer")

# ---------------------------------------------------------------- Fig. 6: allocations chosen by each method (stacked bars)
fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.9), sharey=True)
meths = ["uniform", "damage_prop", "rah_sum", "rah_capped"]
for ax, m in zip(axes, MODELS):
    al = allocs[m]["methods"]
    ms = [x for x in meths if x in al]
    for i, meth in enumerate(ms):
        tok = al[meth]["tokens"]
        bottom = 0.0
        for p in POOLS:
            v = tok.get(p, 0.0) / 1e6 * 100
            if v > 0.3:
                ax.bar(i, v, bottom=bottom, color=PCOL[p], width=0.62, linewidth=0.8, edgecolor="#fcfcfb")
            bottom += v
        if al[meth]["scope"] != "all_lora":                 # all-LoRA unless annotated (see caption)
            ax.text(i, 102, {"cut_lora": "cut-LoRA", "last_k": "last-$k$"}[al[meth]["scope"]],
                    ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([MNAME[x] for x in ms], rotation=35, ha="right")
    ax.set_title(NAME[m])
    ax.set_ylim(0, 115)
    style(ax)
axes[0].set_ylabel("Share of 1M tokens (%)")
handles = [Patch(color=PCOL[p], label=p) for p in POOLS]
axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.0, title="Pool")
fig.tight_layout(w_pad=0.5)
save(fig, "fig6_allocations")

# ---------------------------------------------------------------- Fig. 7: retention by method and capability (4 panels, CIs)
fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 3.8), sharey=True)
for ax, m in zip(axes.flat, MODELS):
    ms = [x for x in meths if row(m, x) is not None]
    w = 0.8 / len(ms)
    for k, meth in enumerate(ms):
        r = row(m, meth)
        ys = np.array([min(r[c], 1.6) for c in CAPS])
        lo = np.array([r[f"{c}_lo"] for c in CAPS])
        hi = np.array([min(r[f"{c}_hi"], 1.6) for c in CAPS])
        xx = np.arange(len(CAPS)) + (k - (len(ms) - 1) / 2) * w
        ax.bar(xx, ys, width=w * 0.9, color=MCOL[meth], label=MNAME[meth], linewidth=0)
        ax.errorbar(xx, ys, yerr=[np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)], fmt="none", ecolor=INK2,
                    elinewidth=0.5, capsize=0)
        for xi, c in zip(xx, CAPS):                          # a bar clipped at the axis top shows its true value
            if r[c] > 1.6:
                ax.text(xi, 1.61, f"{r[c]:.1f}", ha="center", va="bottom", fontsize=5.5, color=INK2)
    ax.axhline(1.0, color=INK, lw=0.6)
    ax.set_xticks(np.arange(len(CAPS)))
    ax.set_xticklabels(CAPS)
    ax.set_title(NAME[m])
    ax.set_ylim(0, 1.65)
    style(ax)
for ax in axes[:, 0]:
    ax.set_ylabel("Retention (test, 3 seeds)")
axes[0, 0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.0, 1.32), handlelength=1.0, columnspacing=1.2)
fig.tight_layout(h_pad=0.8, w_pad=0.6)
save(fig, "fig7_retention_by_method")

# ---------------------------------------------------------------- Fig. 8: headline metrics by method
fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.9))
metrics = [("mean_ret_capped", "Capped mean retention"), ("worst_ret", "Worst-case retention"),
           ("safe_over_refusal", "Over-refusal (lower is better)")]
for ax, (key, lab) in zip(axes, metrics):
    for k, meth in enumerate(meths):
        ys = [row(m, meth)[key] if row(m, meth) is not None else np.nan for m in MODELS]
        xx = np.arange(len(MODELS)) + (k - 1.5) * 0.19
        ax.bar(xx, ys, width=0.17, color=MCOL[meth], label=MNAME[meth], linewidth=0)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([NAME[m].split("-")[0] for m in MODELS])
    ax.set_title(lab)
    style(ax)
axes[0].set_ylim(0.5, 0.9)
axes[1].set_ylim(0, 0.6)
axes[2].set_ylim(0, 0.65)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, handlelength=1.0, columnspacing=1.5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(w_pad=0.8)
save(fig, "fig8_headline")

# ---------------------------------------------------------------- Fig. 9: budget sweep (Qwen)
fig, ax = plt.subplots(figsize=(COL_W, 1.9))
q = budget[budget.model == "qwen"]
pts = {"uniform": [], "rah_sum": []}
for meth in pts:
    for b in [250000, 1000000, 4000000]:
        if b == 1000000:
            v = row("qwen", meth)["mean_ret"]
        else:
            v = q[(q.method == meth) & (q.budget == b)]["mean_ret"].iloc[0]
        pts[meth].append((b, v))
for meth, series in pts.items():
    ax.plot([b / 1e6 for b, _ in series], [v for _, v in series], marker="o", ms=4, lw=1.6, color=MCOL[meth],
            label=MNAME[meth], markeredgecolor="#fcfcfb", markeredgewidth=0.8)
ax.set_xscale("log")
ax.set_xticks([0.25, 1, 4])
ax.set_xticklabels(["0.25M", "1M (3 seeds)", "4M"])
ax.set_xlabel("Healing budget (tokens)")
ax.set_ylabel("Mean retention")
ax.legend(loc="lower right", handlelength=1.4)
style(ax)
save(fig, "fig9_budget_sweep")
print("all figures written to", OUT)
