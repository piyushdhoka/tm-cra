"""Statistics: bootstrap CIs on retention, paired bootstrap between methods, rank agreement."""
import numpy as np
from scipy.stats import spearmanr


def bootstrap_mean_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def retention_ci(item_scores, base_score: float, n_boot: int = 2000, seed: int = 0):
    """Retention = mean(item_scores)/base_score with a bootstrap CI over items (baseline held fixed)."""
    m, lo, hi = bootstrap_mean_ci(item_scores, n_boot, seed)
    return m / base_score, lo / base_score, hi / base_score


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) over paired items, 95% CI, and two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(bs, 0.025)),
            "ci_high": float(np.quantile(bs, 0.975)), "p": float(min(1.0, p))}


def spearman(x, y):
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def kendalls_w(rank_matrix) -> float:
    """Kendall's coefficient of concordance for m raters (rows) ranking n items (cols), no ties correction."""
    R = np.asarray(rank_matrix, dtype=float)
    m, n = R.shape
    if n < 2 or m < 2:
        return float("nan")
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def kendalls_w_ties(rank_matrix) -> float:
    """Kendall's W with the standard correction for tied (average) ranks: the denominator loses
    m * sum over raters and tie groups of (t^3 - t)."""
    R = np.asarray(rank_matrix, dtype=float)
    m, n = R.shape
    if n < 2 or m < 2:
        return float("nan")
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    ties = sum(float(t ** 3 - t) for row in R for t in np.unique(row, return_counts=True)[1])
    return float(12 * S / (m ** 2 * (n ** 3 - n) - m * ties))


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order, monotone and capped at 1."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return [float(x) for x in adj]


def holm_bounded(pvals, n_boot: int) -> tuple[list[float], list[bool]]:
    """Holm-adjust bootstrap p-values. A bootstrap p of 0 only says p < 2 / n_boot (the two-sided p is twice a tail
    fraction), so it is replaced by that bound before adjusting and flagged; flagged adjusted values are upper bounds."""
    bound = [float(p) == 0.0 for p in pvals]
    adj = holm([2.0 / n_boot if b else float(p) for p, b in zip(pvals, bound)])
    return adj, bound


def _primary(retention: np.ndarray) -> tuple[float, float]:
    return float(np.minimum(retention, 1.0).mean()), float(retention.min())


def two_level_bootstrap(a_runs: list[dict], b_runs: list[dict], base: dict, caps: list[str],
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """Seed-aware paired bootstrap of the primary metrics (capped mean and worst-case retention), A minus B.

    a_runs / b_runs: one dict per training seed, capability -> per-item scores aligned on the same item order
    for every run of both methods. Each replicate resamples the seeds of A and of B independently (with
    replacement), then one shared set of items per capability (paired across methods), averages the resampled
    seeds per item and computes retention against the fixed unpruned score base[c]."""
    for c in caps:
        sizes = {len(r[c]) for r in a_runs + b_runs}
        if len(sizes) != 1:
            raise ValueError(f"capability '{c}': runs are not aligned on one item set (sizes {sorted(sizes)})")
    A = {c: np.stack([r[c] for r in a_runs]) for c in caps}          # seeds x items
    B = {c: np.stack([r[c] for r in b_runs]) for c in caps}
    base_v = np.array([base[c] for c in caps], dtype=float)
    na, nb = len(a_runs), len(b_runs)

    def stats(sa, sb, idx):
        ra = np.array([A[c][sa][:, idx[c]].mean() for c in caps]) / base_v
        rb = np.array([B[c][sb][:, idx[c]].mean() for c in caps]) / base_v
        (ca, wa), (cb, wb) = _primary(ra), _primary(rb)
        return ca - cb, wa - wb

    full = {c: np.arange(A[c].shape[1]) for c in caps}
    d0 = stats(np.arange(na), np.arange(nb), full)
    rng = np.random.default_rng(seed)
    bs = np.empty((n_boot, 2))
    for k in range(n_boot):
        sa, sb = rng.integers(0, na, na), rng.integers(0, nb, nb)
        idx = {c: rng.integers(0, A[c].shape[1], A[c].shape[1]) for c in caps}
        bs[k] = stats(sa, sb, idx)
    out = {}
    for j, name in enumerate(("capped_mean", "worst")):
        col = bs[:, j]
        p = 2 * min((col <= 0).mean(), (col >= 0).mean())
        out[name] = {"diff": float(d0[j]), "ci_low": float(np.quantile(col, 0.025)),
                     "ci_high": float(np.quantile(col, 0.975)), "p": float(min(1.0, p))}
    return out
