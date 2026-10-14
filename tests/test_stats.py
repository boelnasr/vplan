"""W3.2 interval statistics: Wilson score intervals and the paired bootstrap over tasks (spec §14)."""
import math

import numpy as np
import pytest

from vplan.evaluate.stats import Z95, PairedDifference, paired_bootstrap, wilson


def test_wilson_plan_acceptance_value():
    low, high = wilson(7, 10)
    assert low == pytest.approx(0.397, abs=1e-3) and high == pytest.approx(0.892, abs=1e-3)


def test_wilson_matches_the_closed_form_at_one_half():
    # p = 1/2: center 1/2, half-width z * sqrt(1/(4n) + z^2/(4n^2)) / (1 + z^2/n), computed by hand for n = 100.
    z2 = Z95 * Z95
    half = Z95 * math.sqrt(0.25 / 100 + z2 / 40000) / (1 + z2 / 100)
    low, high = wilson(50, 100)
    assert Z95 == pytest.approx(1.959964, abs=1e-6)
    assert low == pytest.approx(0.5 - half, abs=1e-12) and high == pytest.approx(0.5 + half, abs=1e-12)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)


def test_wilson_all_or_nothing_touches_the_bound_exactly():
    low, high = wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson(10, 10)
    assert high == 1.0 and low == pytest.approx(0.7225, abs=1e-4)


def test_wilson_is_symmetric_in_successes_and_failures():
    low, high = wilson(3, 10)
    mirror_low, mirror_high = wilson(7, 10)
    assert low == pytest.approx(1 - mirror_high, abs=1e-12) and high == pytest.approx(1 - mirror_low, abs=1e-12)


def test_wilson_with_no_trials_is_the_uninformative_unit_interval():
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_accepts_numpy_integers():
    assert wilson(np.int64(7), np.int64(10)) == wilson(7, 10)


@pytest.mark.parametrize("k, n", [(-1, 5), (6, 5), (1, 0), (0, -1)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)


@pytest.mark.parametrize("k, n", [(0.5, 2), (1, 2.0), (True, 2), ("1", 2)])
def test_wilson_rejects_non_integer_counts(k, n):
    with pytest.raises(TypeError):
        wilson(k, n)


# ---- paired bootstrap ----------------------------------------------------------------------------

A = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
B = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0]


def test_bootstrap_is_deterministic_for_a_given_seed():
    first = paired_bootstrap(A, B, n_boot=2000, seed=3)
    assert paired_bootstrap(A, B, n_boot=2000, seed=3) == first
    assert paired_bootstrap(A, B) == paired_bootstrap(A, B, n_boot=10_000, seed=0)


def test_bootstrap_returns_the_observed_mean_difference_and_n():
    result = paired_bootstrap(A, B)
    assert isinstance(result, PairedDifference)
    assert result.n == 12 and result.mean == pytest.approx((8 - 5) / 12)
    assert result.low <= result.mean <= result.high


def test_bootstrap_hand_checked_case():
    # d = a - b = (1, 1, 1, 1, 0): a resampled mean is Binomial(5, 0.8) / 5. P(mean <= 0.2) = 0.0067 and
    # P(mean <= 0.4) = 0.058 straddle 2.5%, and P(mean < 1) = 0.67 is below 97.5%, so the 95% percentile
    # interval is (0.4, 1.0) unless 10,000 draws stray by about 20 standard errors.
    result = paired_bootstrap([1, 1, 1, 1, 1], [0, 0, 0, 0, 1])
    assert result.mean == pytest.approx(0.8)
    assert (result.low, result.high) == pytest.approx((0.4, 1.0), abs=1e-12)


@pytest.mark.parametrize("a, b, value", [([1, 1, 1], [0, 0, 0], 1.0), ([0, 0], [1, 1], -1.0),
                                         ([1, 0, 1], [1, 0, 1], 0.0)])
def test_bootstrap_of_constant_differences_is_degenerate(a, b, value):
    assert tuple(paired_bootstrap(a, b))[:3] == pytest.approx((value, value, value))


def test_bootstrap_interval_agrees_with_the_normal_approximation_on_a_larger_sample():
    # 200 tasks: d = +1 on 40, -1 on 20, 0 on 140. mean 0.1, sd sqrt(0.30 - 0.01), se = sd / sqrt(200).
    a = [1] * 40 + [0] * 20 + [1] * 70 + [0] * 70
    b = [0] * 40 + [1] * 20 + [1] * 70 + [0] * 70
    result = paired_bootstrap(a, b)
    se = math.sqrt(0.29 / 200)
    assert result.mean == pytest.approx(0.1)
    assert result.low == pytest.approx(0.1 - Z95 * se, abs=0.01)
    assert result.high == pytest.approx(0.1 + Z95 * se, abs=0.01)


def test_bootstrap_accepts_booleans_and_numpy_arrays():
    assert paired_bootstrap([True, False, True], [False, False, True]) == paired_bootstrap(
        np.array([1, 0, 1]), np.array([0, 0, 1]))


@pytest.mark.parametrize("a, b", [([1, 0], [1]), ([], []), ([2, 0], [1, 0]), ([0.5, 1], [0, 1]),
                                  ([1, 0], [None, 0]), ([1, -1], [0, 0]), ([[1], [0]], [[0], [0]]),
                                  (["1", "0"], ["0", "0"])])
def test_bootstrap_rejects_unequal_lengths_empty_and_non_binary_inputs(a, b):
    with pytest.raises(ValueError):
        paired_bootstrap(a, b)


@pytest.mark.parametrize("n_boot", [0, -5])
def test_bootstrap_rejects_a_non_positive_resample_count(n_boot):
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [0, 0], n_boot=n_boot)
