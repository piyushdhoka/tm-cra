"""Seed-aware (two-level) bootstrap, Holm correction and tie-corrected Kendall's W."""
import numpy as np
import pytest

from rah.stats import holm, holm_bounded, kendalls_w_ties, two_level_bootstrap


# --------------------------------------------------------------------------- Holm
def test_holm_known_values():
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> raised to 0.06 (monotone); original order kept
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_holm_empty():
    assert holm([]) == []


# --------------------------------------------------------------------------- Kendall's W with ties
def test_kendalls_w_ties_hand_computed():
    # rank sums 2.5, 3.5, 6 -> S = 6.5; denominator 2^2 (27 - 3) - 2 * (2^3 - 2) = 84; W = 12 * 6.5 / 84
    assert kendalls_w_ties([[1, 2, 3], [1.5, 1.5, 3]]) == pytest.approx(13 / 14)


def test_kendalls_w_ties_identical_tied_rankings_is_one():
    assert kendalls_w_ties([[1.5, 1.5, 3, 4], [1.5, 1.5, 3, 4]]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- two-level bootstrap
CAPS = ["x", "y"]
BASE = {"x": 1.0, "y": 1.0}


def _runs(per_seed: list[dict]) -> list[dict]:
    return [{c: np.asarray(v, dtype=float) for c, v in run.items()} for run in per_seed]


def test_two_level_detects_constant_positive_effect():
    rng = np.random.default_rng(1)
    b = [{c: rng.uniform(0.0, 0.4, 50) for c in CAPS} for _ in range(3)]
    a = [{c: run[c] + 0.2 for c in CAPS} for run in b]
    res = two_level_bootstrap(_runs(a), _runs(b), BASE, CAPS, n_boot=500, seed=0)
    assert res["capped_mean"]["diff"] == pytest.approx(0.2)
    assert res["worst"]["diff"] == pytest.approx(0.2)
    assert res["capped_mean"]["p"] < 0.01
    assert res["capped_mean"]["ci_low"] > 0.0


def test_two_level_respects_seed_variability():
    # Every item scores the same within a seed, so an item-only bootstrap would be certain (p = 0).
    # Across seeds A averages 0.5 against B's 0.45 with a spread of 0.2-0.8: seed resampling must not call it.
    a = [{c: np.full(50, s) for c in CAPS} for s in (0.2, 0.5, 0.8)]
    b = [{c: np.full(50, 0.45) for c in CAPS} for _ in range(3)]
    res = two_level_bootstrap(_runs(a), _runs(b), BASE, CAPS, n_boot=1000, seed=0)
    assert res["capped_mean"]["diff"] == pytest.approx(0.05)
    assert res["capped_mean"]["p"] > 0.05
    assert res["capped_mean"]["ci_low"] < 0.0 < res["capped_mean"]["ci_high"]


def test_two_level_single_seed_method():
    a = [{"x": np.ones(20), "y": np.ones(20)}]
    b = [{"x": np.zeros(20), "y": np.full(20, 0.5)} for _ in range(3)]
    res = two_level_bootstrap(_runs(a), _runs(b), BASE, CAPS, n_boot=200, seed=0)
    assert res["capped_mean"]["diff"] == pytest.approx(0.75)   # mean(1, 1) - mean(0, 0.5)
    assert res["worst"]["diff"] == pytest.approx(1.0)           # min(1, 1) - min(0, 0.5)


def test_two_level_capped_mean_caps_retention_at_one():
    a = [{"x": np.full(10, 0.9), "y": np.full(10, 0.9)}]
    b = [{"x": np.full(10, 0.5), "y": np.full(10, 0.5)}]
    res = two_level_bootstrap(_runs(a), _runs(b), {"x": 0.6, "y": 1.0}, CAPS, n_boot=50, seed=0)
    # A: x retention 1.5 -> capped at 1, y 0.9 -> capped mean 0.95; B: 0.8333 and 0.5 -> 0.6667
    assert res["capped_mean"]["diff"] == pytest.approx(0.95 - (0.5 / 0.6 + 0.5) / 2)


def test_two_level_rejects_misaligned_items():
    a = [{"x": np.ones(10), "y": np.ones(10)}]
    b = [{"x": np.ones(9), "y": np.ones(10)}]
    with pytest.raises(ValueError):
        two_level_bootstrap(_runs(a), _runs(b), BASE, CAPS, n_boot=10, seed=0)


# --------------------------------------------------------------------------- Holm with bootstrap-resolution bounds
def test_holm_bounded_replaces_zero_by_two_over_n_boot():
    # p = 2 * min(tail fractions): the smallest non-zero value is 2 / n_boot, so p = 0 means p < 2 / n_boot
    adj, bound = holm_bounded([0.0, 0.01, 0.5], n_boot=2000)
    assert bound == [True, False, False]
    assert adj == pytest.approx([3 * 0.001, 2 * 0.01, 0.5])
