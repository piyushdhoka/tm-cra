"""Stage 15: the figures of the arXiv paper, drawn from results/analysis/paper_numbers.json (CPU only).

  python scripts/15_paper_figures.py      # arxiv/figures/fig*.pdf (paper) and docs/figures/fig*.png (README)

Palette: the first three categorical slots of a colour-blind-checked palette (blue, orange, aqua), validated for
all-pairs use, plus neutral greys. Every series also has its own marker shape and a legend entry, so no figure
depends on colour alone.
"""
import _bootstrap  # noqa: F401
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from rah.config import REPO

OUT = REPO / "arxiv" / "figures"
README_FIGS = REPO / "docs" / "figures"
MODELS = ["llama", "qwen", "smollm", "olmo"]
CAPS = ["fmt", "tool", "inst", "math", "safe", "know"]
CAP_NAME = {"fmt": "Structured\noutput", "tool": "Function\ncalling", "inst": "Instruction\nfollowing",
            "math": "Math", "safe": "Safety\nrefusal", "know": "Knowledge"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
    "text.color": INK, "axes.titlecolor": INK, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    """Vector PDF for the paper (arxiv/figures/) and a PNG copy for the README (docs/figures/)."""
    OUT.mkdir(parents=True, exist_ok=True)
    README_FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    fig.savefig(README_FIGS / name.replace(".pdf", ".png"), dpi=200, facecolor="#ffffff")
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 1: study design
def fig_design(N):
    fig = plt.figure(figsize=(6.8, 2.7))
    ax = fig.add_axes([0, 0, 1, 1])            # the drawing fills the whole figure, so box widths are as planned
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    PAD = 6                                    # pixels kept clear between any label and its box outline

    def inside(text, frame) -> bool:
        t = text.get_window_extent(renderer)
        return (t.x0 >= frame.x0 + PAD and t.x1 <= frame.x1 - PAD and
                t.y0 >= frame.y0 + PAD and t.y1 <= frame.y1 - PAD)

    boxes = []                                 # (frame, title label, body label or None)

    def box(x, y, w, h, title, sub="", edge=AXIS, fill="#f9f9f7"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                               linewidth=1.0, edgecolor=edge, facecolor=fill)
        ax.add_patch(patch)
        if sub:
            head = ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=7.6,
                           fontweight="bold")
            body = ax.text(x + w / 2, y + h * 0.34, sub, ha="center", va="center", fontsize=6.6, color=INK2,
                           linespacing=1.2)
        else:
            head = ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7.6,
                           fontweight="bold", linespacing=1.2)
            body = None
        boxes.append((patch.get_window_extent(renderer), head, body))

    def fits(frame, head, body) -> bool:
        labels = [t for t in (head, body) if t is not None]
        if not all(inside(t, frame) for t in labels):
            return False
        # the title must stay clear above the description
        return body is None or head.get_window_extent(renderer).y0 >= body.get_window_extent(renderer).y1 + 2

    def fit_all():
        """One title size and one description size for every box. Titles shrink only while some title does not
        fit, descriptions only while some description does not fit, and both while a title crowds its
        description, until every label sits inside its box, clear of the outline and of the other label."""
        heads = [h for _, h, _ in boxes]
        bodies = [b for _, _, b in boxes if b is not None]
        while True:
            head_bad = any(not inside(h, f) for f, h, _ in boxes)
            body_bad = any(b is not None and not inside(b, f) for f, _, b in boxes)
            gap_bad = not all(fits(*b) for b in boxes) and not (head_bad or body_bad)
            if not (head_bad or body_bad or gap_bad):
                return
            shrink = (heads if head_bad else []) + (bodies if body_bad else []) + (heads + bodies if gap_bad else [])
            if min(t.get_fontsize() for t in shrink) <= 5.0:
                raise ValueError("a label does not fit its box; widen the boxes")
            for t in set(shrink):
                t.set_fontsize(t.get_fontsize() - 0.1)

    def arrow(x0, y0, x1, y1, color=MUTED):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=1.0, color=color, shrinkA=0, shrinkB=0))

    mid, bh = 20, 13
    box(0.5, mid - bh / 2, 12.5, bh, "Instruction-\ntuned\nmodel")
    box(15.5, mid - bh / 2, 14.5, bh, "Prune", "drop the least\ninfluential\nblocks")
    box(32.5, mid - bh / 2, 13.5, bh, "Diagnose", "6 capabilities\n(dev split)")
    arrow(13.2, mid, 15.3, mid)
    arrow(30.2, mid, 32.3, mid)
    lanes = [(33, "Fixed budget, no planning", "uniform or damage-\nproportional mixture", "1M", INK2),
             (20, "RAH: plan from pilots", "21 pilot heals (3.48M tokens)\n→ curves → mixture + scope",
              "1M", BLUE),
             (7, "Token-matched control", "uniform mixture, also\ngiven the pilot tokens", "4.48M", INK)]
    heal = []
    for y, title, sub, tok, col in lanes:
        box(49, y - 5.6, 28.5, 11.2, title, sub, edge=col, fill="#ffffff")
        heal.append(ax.text(78.8, y, f"heal\n{tok}", ha="left", va="center", fontsize=7, color=col,
                            fontweight="bold", linespacing=1.1))
        arrow(46.2, mid, 48.8, y, color=col)
        arrow(85.2, y, 87.3, mid, color=col)
    box(87.5, mid - bh / 2, 12.2, bh, "Test", "6 capabilities\n3 seeds")
    fit_all()
    for t in heal:                             # the token labels match the box titles
        t.set_fontsize(boxes[0][1].get_fontsize())
    save(fig, "fig1_design.pdf")


# --------------------------------------------------------------------------- Figure 2: what breaks
def fig_breakage(N):
    order = sorted(CAPS, key=lambda c: N["breakage"]["raw"]["mean_rank"][c])
    cmap = LinearSegmentedColormap.from_list("ret", RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"wspace": 0.06})
    for ax, key, title in ((axes[0], "retention_main", "Retention after pruning"),
                           (axes[1], "retention_main_corrected", "Chance-corrected retention")):
        M = np.array([[N["families"][n][key][c] for c in order] for n in MODELS])
        ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ink = "#ffffff" if np.clip(v, 0, 1) > 0.55 else INK
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=ink)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CAP_NAME[c].replace("\n", " ") for c in order], fontsize=7, rotation=32,
                           ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([N["families"][n]["label"] for n in MODELS] if ax is axes[0] else [])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MODELS)), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=2)
        ax.tick_params(which="minor", length=0)
    save(fig, "fig2_breakage.pdf")


# --------------------------------------------------------------------------- Figure 3: adaptation scope
def fig_scope(N):
    fig, ax = plt.subplots(figsize=(4.4, 1.95))
    ys = np.arange(len(MODELS))[::-1]
    for y, n in zip(ys, MODELS):
        m = N["families"][n]["methods"]
        a, k = m["uniform"]["capped_mean"], m["last_k_uniform"]["capped_mean"]
        ax.plot([a, k], [y, y], color=GRID, linewidth=2.5, zorder=1)
        ax.scatter([a], [y], s=46, color=BLUE, marker="o", zorder=3, edgecolor="#ffffff", linewidth=1)
        ax.scatter([k], [y], s=46, color=ORANGE, marker="D", zorder=3, edgecolor="#ffffff", linewidth=1)
        ax.text(min(a, k) - 0.012, y, f"{k - a:+.2f}", ha="right", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([N["families"][n]["label"] for n in MODELS])
    ax.set_xlim(0.38, 0.9)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_xlabel("Capped mean retention (test, uniform mixture, 1M tokens)")
    grid(ax, "x")
    ax.scatter([], [], s=40, color=BLUE, marker="o", label="LoRA on all blocks (3 seeds)")
    ax.scatter([], [], s=40, color=ORANGE, marker="D", label="full tuning of last 3 blocks (1 seed)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.27), ncol=2, handletextpad=0.3, columnspacing=1.2)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig3_scope.pdf")


# --------------------------------------------------------------------------- Figure 4: planning vs healing longer
def fig_token_matched(N):
    series = [("uniform", INK2, "o", "Uniform"), ("damage_prop", ORANGE, "s", "Damage-proportional"),
              ("rah_sum", BLUE, "D", "RAH"), ("rah_capped", AQUA, "^", "RAH-capped")]
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.6), sharex=True, gridspec_kw={"hspace": 0.18, "wspace": 0.35})
    for col, n in enumerate(MODELS):
        f = N["families"][n]["methods"]
        for row, metric in enumerate(("capped_mean", "worst")):
            ax = axes[row, col]
            grid(ax)
            u1, u4 = f["uniform"], f["uniform_4480k"]
            ax.plot([0, 1], [u1[metric], u4[metric]], color=INK2, linewidth=1.5, zorder=2)
            for key, color, marker, _ in series:
                if key not in f:
                    continue
                m = f[key]
                sd = m[f"{'capped_mean' if metric == 'capped_mean' else 'worst'}_seed_sd"]
                x = {"uniform": 0.0, "damage_prop": -0.13}.get(key, 1.0 + (0.1 if key == "rah_sum" else 0.2))
                if key in ("rah_sum", "rah_capped"):        # the same heal, as usually reported (pilots not counted)
                    x0 = 0.12 if key == "rah_sum" else 0.22
                    ax.scatter([x0], [m[metric]], s=26, facecolor="#ffffff", edgecolor=color, marker=marker,
                               linewidth=1.1, zorder=3)
                    ax.add_patch(FancyArrowPatch((x0 + 0.04, m[metric]), (x - 0.05, m[metric]), arrowstyle="-|>",
                                                 mutation_scale=6, color=color, linewidth=0.7, alpha=0.55,
                                                 zorder=2))
                ax.errorbar([x], [m[metric]], yerr=[sd], fmt="none", ecolor=color, elinewidth=1, capsize=0,
                            zorder=3)
                ax.scatter([x], [m[metric]], s=30, color=color, marker=marker, edgecolor="#ffffff",
                           linewidth=0.8, zorder=4)
            ax.errorbar([1.0], [u4[metric]], yerr=[u4[f"{metric}_seed_sd"]], fmt="none", ecolor=INK, elinewidth=1,
                        capsize=0, zorder=4)
            ax.scatter([1.0], [u4[metric]], s=36, color=INK, marker="o", edgecolor="#ffffff", linewidth=0.8,
                       zorder=5)
            if row == 0:
                ax.set_title(N["families"][n]["label"], fontsize=8.5)
            if col == 0:
                ax.set_ylabel("Capped mean\nretention" if metric == "capped_mean" else "Worst-capability\nretention")
            ax.set_xlim(-0.35, 1.4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["1M", "4.48M"])
            shown = [f[k][metric] for k in ("uniform", "damage_prop", "rah_sum", "rah_capped", "uniform_4480k")
                     if k in f]
            lo, hi = min(shown) - 0.05, max(shown) + 0.05
            if metric == "capped_mean":
                ax.set_ylim(np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20)
            else:
                ax.set_ylim(0, 0.65)
    for ax in axes[1]:
        ax.set_xlabel("training tokens charged", fontsize=7.5)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="-" if k == "uniform" else "none", markersize=5,
                          label=lab) for k, c, m, lab in series]
    handles.append(plt.Line2D([], [], color=INK2, marker="D", markerfacecolor="#ffffff", linestyle="none",
                              markersize=5, label="RAH variants with pilot tokens not counted"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03), handletextpad=0.3,
               columnspacing=1.0, fontsize=7.3)
    save(fig, "fig4_token_matched.pdf")


# --------------------------------------------------------------------------- Figure 5: pilot curves vs reality
def fig_curves(N):
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.1), sharey=True, gridspec_kw={"wspace": 0.12})
    x = np.linspace(0, 1.1e6, 400)
    for ax, n in zip(axes, MODELS):
        f = N["families"][n]
        a, tau, r0 = f["fit"]["a"]["math"], f["fit"]["tau"]["math"], f["fit"]["r0_dev"]["math"]
        y = np.minimum(1.2, r0 + a * (1 - np.exp(-x / tau)))
        ax.axvspan(3e5, 1.1e6, color="#f0efec", zorder=0)
        inside = x <= 3e5
        ax.plot(x[inside] / 1e3, y[inside], color=BLUE, linewidth=1.8, zorder=2)
        ax.plot(x[~inside] / 1e3, y[~inside], color=BLUE, linewidth=1.8, linestyle=(0, (3, 2)), zorder=2)
        px = [p["budget"] for p in f["math_pilots_dev"]]
        py = [p["retention"] for p in f["math_pilots_dev"]]
        ax.scatter(np.array(px) / 1e3, py, s=22, facecolor="#ffffff", edgecolor=BLUE, linewidth=1.1, zorder=3)
        for key, marker, color in (("uniform", "o", INK2), ("uniform_4480k", "s", INK)):
            c = f["prediction_check"][key]
            ax.scatter([c["math_effective_tokens"] / 1e3], [c["math_obs"]], s=34, marker=marker, color=color,
                       edgecolor="#ffffff", linewidth=0.8, zorder=4)
        ax.set_title(f["label"], fontsize=8.5)
        ax.set_xlim(0, 1100)
        ax.set_ylim(0, 1.1)
        grid(ax)
        ax.set_xlabel("effective math tokens (k)", fontsize=7.5)
    axes[0].set_ylabel("Math retention")
    axes[0].text(310, 1.02, "extrapolated", fontsize=6.8, color=INK2)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=1.8, label="curve fitted to pilots"),
               plt.Line2D([], [], color=BLUE, marker="o", markerfacecolor="#ffffff", linestyle="none", markersize=5,
                          label="math pilots (dev)"),
               plt.Line2D([], [], color=INK2, marker="o", linestyle="none", markersize=5, label="uniform 1M (test)"),
               plt.Line2D([], [], color=INK, marker="s", linestyle="none", markersize=5, label="uniform 4.48M (test)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.97), handletextpad=0.3)
    save(fig, "fig5_curves.pdf")


# --------------------------------------------------------------------------- Figure 6: refusal operating points
def fig_refusal(N):
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.05), sharey=True, gridspec_kw={"wspace": 0.12})
    pts = [("unpruned", INK, "o", "Unpruned", True), ("pruned", MUTED, "X", "Pruned", False),
           ("uniform", BLUE, "o", "Uniform 1M", False), ("rah_sum", ORANGE, "D", "RAH 1M", False),
           ("uniform_4480k", AQUA, "s", "Uniform 4.48M", False)]
    for ax, n in zip(axes, MODELS):
        f = N["families"][n]
        ax.add_patch(plt.Rectangle((0, 0.8), 0.2, 0.2, color="#f0efec", zorder=0))
        for key, color, marker, _, hollow in pts:
            if key == "unpruned":
                b, u = f["unpruned_refusal"]["benign"], f["unpruned_refusal"]["unsafe"]
            elif key == "pruned":
                b, u = f["pruned_refusal"]["benign"], f["pruned_refusal"]["unsafe"]
            else:
                b, u = f["methods"][key]["benign_refused"], f["methods"][key]["unsafe_refused"]
            ax.scatter([b], [u], s=34, marker=marker, zorder=3, linewidth=1.1,
                       facecolor="#ffffff" if hollow else color, edgecolor=color if hollow else "#ffffff")
        ax.set_xlim(-0.03, 0.8)
        ax.set_ylim(0.2, 1.03)
        ax.set_title(f["label"], fontsize=8.5)
        grid(ax, "both")
        ax.set_xlabel("benign prompts refused", fontsize=7.5)
    axes[0].set_ylabel("unsafe prompts refused")
    axes[0].text(0.01, 0.975, "ideal", fontsize=6.8, color=INK2, va="top")
    handles = [plt.Line2D([], [], color=c, marker=m, markerfacecolor="#ffffff" if h else c, linestyle="none",
                          markersize=5, label=lab) for _, c, m, lab, h in pts]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.97), handletextpad=0.3)
    save(fig, "fig6_refusal.pdf")


def main():
    N = json.loads((REPO / "results" / "analysis" / "paper_numbers.json").read_text(encoding="utf-8"))
    fig_design(N)
    fig_breakage(N)
    fig_scope(N)
    fig_token_matched(N)
    fig_curves(N)
    fig_refusal(N)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
