# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""The stochastic tail of :meth:`Lindbladian.pc_step_arr` / :meth:`BlockLindbladian.pc_step_arr`
(``tail_mode="stochastic"``, ``tail_threshold``, ``tail_rule``, ``tail_seed``) and its account.

Conventions checked here:

* ``tail_mode=None`` / ``"drop"`` is the deterministic trim: the same basis and coefficients as a call without the keyword.
* ``"stochastic"``: strings with ``|c| >= tau`` and protected words are untouched; a string below ``tau`` is kept at
  ``sign(c) * tau`` or removed; exact zeros are removed; the account identities hold step by step.
* Unbiasedness: on the complete basis of a 3-site chain (the exact in-basis exponential, admission never short of room)
  the replica mean of every coefficient after several stochastic steps is the untruncated one within five standard errors.
* The seed reproduces a trajectory; two seeds differ.
* The rules: ``abs`` / ``rel`` / ``budget`` set the threshold the account reports.
* The block core takes the same keywords.
"""

from __future__ import annotations

import inspect
import itertools

import numpy as np
import pytest

from ppvm import STEP_FEATURES, BlockLindbladian, Lindbladian
from ppvm.lindblad import TAIL_MODES, TAIL_RULES, _basis_to_codes, _codes_to_basis


def chain_h(n: int, scale: float = 0.5):
    h = []
    for i in range(n - 1):
        for p in "XYZ":
            t = ["I"] * n
            t[i] = t[i + 1] = p
            h.append(("".join(t), scale * (1.0 + 0.1 * i)))
    for i in range(n):
        t = ["I"] * n
        t[i] = "Z"
        h.append(("".join(t), 0.3 * (1 + i % 2)))
    return h


def as_map(basis_arr, coeffs):
    return dict(zip(_codes_to_basis(basis_arr), coeffs))


def full_strings(n):
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]


def test_tail_features_announced():
    assert "tail" in STEP_FEATURES
    assert TAIL_MODES == ("drop", "stochastic") and TAIL_RULES == ("abs", "rel", "budget")
    params = inspect.signature(Lindbladian.pc_step_arr).parameters
    assert {"tail_mode", "tail_threshold", "tail_rule", "tail_seed"} <= set(params)
    assert params["tail_mode"].default is None
    bparams = inspect.signature(BlockLindbladian.pc_step_arr).parameters
    assert {"tail_mode", "tail_threshold", "tail_rule", "tail_seed"} <= set(bparams)


def test_drop_mode_is_the_plain_step():
    n, dt = 4, 0.05
    L = Lindbladian(n, chain_h(n), [("IIZI", 0.1)])
    start = _basis_to_codes(["ZIII"], n)
    c = np.array([1.0])
    b0, c0 = start.copy(), c.copy()
    b1, c1 = start.copy(), c.copy()
    b2, c2 = start.copy(), c.copy()
    for _ in range(3):
        b0, c0 = L.pc_step_arr(b0, c0, dt, 40, drop_tol=1e-10)
        b1, c1 = L.pc_step_arr(b1, c1, dt, 40, drop_tol=1e-10, tail_mode="drop")
        b2, c2, st = L.pc_step_arr(b2, c2, dt, 40, drop_tol=1e-10, tail_mode=None, return_stats=True)
        assert st["tail_mode"] == "drop"
        assert st["tail_threshold_used"] == 0.0
        assert st["n_strings_rounded"] == st["n_strings_kept_by_chance"] == 0
        assert st["tail_mass_rounded"] == st["tail_mass_kept"] == st["tail_mass_removed"] == 0.0
    assert np.array_equal(b0, b1) and np.array_equal(c0, c1)
    assert np.array_equal(b0, b2) and np.array_equal(c0, c2)


def test_stochastic_rounding_values_and_account():
    n, dt = 4, 0.05
    L = Lindbladian(n, chain_h(n), [])
    strings = full_strings(n)
    basis = _basis_to_codes(strings, n)
    rng = np.random.default_rng(3)
    c0 = rng.normal(size=len(strings)) * 0.05
    c0[0] = 0.0                                   # the identity, an exact zero: removed
    c0[5] = 0.9                                   # one string far above tau: never randomised
    protected = _basis_to_codes([strings[7]], n)  # a protected string below tau: never randomised
    c0[7] = 0.001
    # H = 0 would make the step the identity; use an H but a tiny dt so most coefficients are near c0 -- the account is
    # exact whatever the step did, so compare against the corrector output through a drop-mode twin on the same basis
    b_ref, c_ref = L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 10_000_000, drop_tol=0.0, protected_arr=protected)
    ref = as_map(b_ref, c_ref)
    tau = 0.04
    seen_kept = seen_removed = False
    for seed in range(20):
        b, c, st = L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 10_000_000, drop_tol=0.0, protected_arr=protected,
                                 tail_mode="stochastic", tail_threshold=tau, tail_rule="abs", tail_seed=seed,
                                 return_stats=True)
        got = as_map(b, c)
        assert st["tail_mode"] == "stochastic" and st["tail_threshold_used"] == tau
        assert strings[0] not in got                                    # the exact zero is gone
        assert got[strings[5]] == ref[strings[5]]                       # above tau: untouched
        assert got[strings[7]] == ref[strings[7]]                       # protected: untouched
        below = [s for s in strings if abs(ref[s]) < tau and s != strings[7]]
        assert st["n_strings_rounded"] == len(below)
        assert st["n_strings_rounded"] == st["n_strings_kept_by_chance"] + st["trim_dropped"]
        mass = sum(ref[s] ** 2 for s in below)
        l1 = sum(abs(ref[s]) for s in below)
        assert abs(st["tail_mass_rounded"] - mass) < 1e-15
        assert abs(st["tail_l1_rounded"] - l1) < 1e-15
        assert abs(st["tail_mass_kept"] - st["n_strings_kept_by_chance"] * tau * tau) < 1e-15
        assert abs(st["tail_mass_removed"] - st["trim_dropped_l2sq"]) < 1e-18
        kept_before = sum(ref[s] ** 2 for s in below if s in got)
        assert abs(st["tail_mass_rounded"] - st["tail_mass_removed"] - kept_before) < 1e-15
        for s in below:
            if s in got:
                assert got[s] == np.sign(ref[s]) * tau
                seen_kept = True
            else:
                seen_removed = True
        for s in strings:
            if abs(ref[s]) >= tau:
                assert got[s] == ref[s]
    assert seen_kept and seen_removed


def test_stochastic_tail_is_unbiased_on_the_complete_basis():
    n, dt, steps, R = 3, 0.07, 3, 400
    L = Lindbladian(n, chain_h(n), [("IIZ", 0.15)])
    strings = full_strings(n)
    basis = _basis_to_codes(strings, n)
    rng = np.random.default_rng(11)
    c0 = rng.normal(size=len(strings)) * 0.3
    c0[0] = 0.0
    b, c = basis.copy(), c0.copy()
    for _ in range(steps):
        b, c = L.pc_step_arr(b, c, dt, 10_000_000, drop_tol=0.0)
    exact = as_map(b, c)
    vals = np.zeros((R, len(strings)))
    rounded = 0
    for r in range(R):
        b, c = basis.copy(), c0.copy()
        for k in range(steps):
            b, c, st = L.pc_step_arr(b, c, dt, 10_000_000, drop_tol=0.0, tail_mode="stochastic", tail_threshold=0.4,
                                     tail_rule="rel", tail_seed=r * 1000 + k, return_stats=True)
            rounded += st["n_strings_rounded"]
            assert st["room_refused_terms"] == 0
        got = as_map(b, c)
        vals[r] = [got.get(s, 0.0) for s in strings]
    assert rounded > 100
    mean = vals.mean(axis=0)
    se = vals.std(axis=0, ddof=1) / np.sqrt(R)
    ex = np.array([exact.get(s, 0.0) for s in strings])
    assert np.all(np.abs(mean - ex) <= 5.0 * se + 1e-12), np.max(np.abs(mean - ex) / np.maximum(se, 1e-300))
    assert se.max() > 1e-4, "the rounding must leave a measurable variance"


def test_seed_reproduces_and_two_seeds_differ():
    n, dt = 4, 0.05
    L = Lindbladian(n, chain_h(n), [])
    strings = full_strings(n)
    basis = _basis_to_codes(strings, n)
    c0 = np.random.default_rng(1).normal(size=len(strings)) * 0.05

    def run(seed):
        b, c = basis.copy(), c0.copy()
        for k in range(2):
            b, c = L.pc_step_arr(b, c, dt, 10_000_000, drop_tol=0.0, tail_mode="stochastic", tail_threshold=0.03,
                                 tail_seed=seed + k)
        return as_map(b, c)

    a, b2, c = run(5), run(5), run(6)
    assert a == b2
    assert a != c


def test_threshold_rules_and_validation():
    n = 3
    L = Lindbladian(n, chain_h(n), [])
    strings = full_strings(n)
    basis = _basis_to_codes(strings, n)
    c0 = np.linspace(-0.5, 0.5, len(strings))
    _, c_ref, st_abs = L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 10_000_000, drop_tol=0.0, tail_mode="stochastic",
                                     tail_threshold=0.1, tail_rule="abs", tail_seed=1, return_stats=True)
    assert st_abs["tail_threshold_used"] == 0.1
    _, _, st_rel = L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 10_000_000, drop_tol=0.0, tail_mode="stochastic",
                                 tail_threshold=0.5, tail_rule="rel", tail_seed=1, return_stats=True)
    assert abs(st_rel["tail_threshold_used"] - 0.5 * 0.5) < 1e-9
    slots = 20
    _, _, st_b = L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, slots, drop_tol=0.0, admit_basis=10_000_000,
                               tail_mode="stochastic", tail_rule="budget", tail_seed=1, return_stats=True)
    tau = st_b["tail_threshold_used"]
    assert tau > 0
    # the corrector's output of the twin step gives the magnitudes the budget rule was solved on
    _, c_twin = L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 10_000_000, drop_tol=0.0)
    expected = np.minimum(1.0, np.abs(c_twin) / tau).sum()
    assert abs(expected - slots) < 1e-9, (expected, slots)
    _, _, st_fit = L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 10_000_000, drop_tol=0.0,
                                 tail_mode="stochastic", tail_rule="budget", tail_seed=1, return_stats=True)
    assert st_fit["tail_threshold_used"] == 0.0 and st_fit["n_strings_kept_by_chance"] == 0
    with pytest.raises(ValueError):
        L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 100, tail_mode="sometimes")
    with pytest.raises(ValueError):
        L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 100, tail_mode="stochastic", tail_rule="median")
    with pytest.raises(ValueError):
        L.pc_step_arr(basis.copy(), c0.copy(), 1e-9, 100, tail_mode="stochastic", tail_threshold=-1.0)


def test_block_core_takes_the_tail():
    n = 3
    h = chain_h(n)
    B = BlockLindbladian(n, 2, [h, h], None, None, exchange=[[0.0, 0.2], [0.3, 0.0]], matchings=None)
    strings = full_strings(n)
    basis = _basis_to_codes(strings, n)
    species = np.zeros(len(strings), dtype=np.uint8)
    c0 = np.random.default_rng(2).normal(size=len(strings)) * 0.1
    b, sp, c, st = B.pc_step_arr(basis, species, c0, 0.05, 10_000_000, drop_tol=0.0, tail_mode="stochastic",
                                 tail_threshold=0.05, tail_seed=3, return_stats=True)
    assert st["tail_mode"] == "stochastic" and st["n_strings_rounded"] > 0
    assert st["n_strings_rounded"] == st["n_strings_kept_by_chance"] + st["trim_dropped"]
    assert len(b) == len(sp) == len(c)
    b2, sp2, c2 = B.pc_step_arr(basis, species, c0, 0.05, 10_000_000, drop_tol=0.0, tail_mode="stochastic",
                                tail_threshold=0.05, tail_seed=3)
    assert np.array_equal(b, b2) and np.array_equal(c, c2)
