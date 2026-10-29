# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""The admission weight cap (``max_weight``) and the per-site depolarizing
channel (``depolarize_eps``) of :meth:`Lindbladian.pc_step_arr` /
:meth:`BlockLindbladian.pc_step_arr`, and the step account (``return_stats``).

Conventions checked here:

* ``depolarize_eps = eps`` with ``H = 0`` damps a string of Pauli weight ``w``
  as ``exp(-eps * w * t)`` exactly (the channel is diagonal in the Pauli
  basis and never leaks); the identity is untouched.
* With ``H != 0`` the generator route equals the explicit Lindblad channel with
  jumps ``sqrt(eps/4) * {X_i, Y_i, Z_i}`` on every site (both exact on the
  complete basis).
* ``max_weight = l`` never admits a string of weight ``> l``; the refused
  leakage is counted in ``weight_dropped_terms`` / ``weight_dropped_l1``.
* With both knobs at their defaults the step is the plain step, bit for bit,
  and the ``return_stats`` route returns the same basis and coefficients.
* On the block core the weight of a row is taken over the physical sites.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ppvm import STEP_FEATURES, BlockLindbladian, Lindbladian
from ppvm.lindblad import _basis_to_codes, _codes_to_basis


def weight(s: str) -> int:
    return sum(ch != "I" for ch in s)


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


def explicit_depolarizing(n: int, eps: float):
    """Jump list of the explicit channel: rate eps/4 for each of X_i, Y_i, Z_i."""
    out = []
    for i in range(n):
        for p in "XYZ":
            t = ["I"] * n
            t[i] = p
            out.append(("".join(t), eps / 4.0))
    return out


def as_map(basis_arr, coeffs):
    return dict(zip(_codes_to_basis(basis_arr), coeffs))


def test_step_features_announced():
    assert "max_weight" in STEP_FEATURES
    assert "depolarize_eps" in STEP_FEATURES
    assert "step_stats" in STEP_FEATURES
    import inspect

    params = inspect.signature(Lindbladian.pc_step_arr).parameters
    assert {"max_weight", "depolarize_eps", "return_stats"} <= set(params)


def test_depolarize_h0_decay_is_exp_eps_w_t():
    n, eps, dt = 6, 0.41, 0.05
    L = Lindbladian(n, [], [])
    strings = ["IIIIII", "ZIIIII", "XYIIII", "IZZXII", "YXZIIZ", "XYZXYI", "XYZXYZ"]
    basis = _basis_to_codes(strings, n)
    coeffs = np.ones(len(strings))
    big = 10_000_000
    for step in range(1, 11):
        basis, coeffs, st = L.pc_step_arr(
            basis, coeffs, dt, big, drop_tol=0.0, depolarize_eps=eps, return_stats=True
        )
        assert st["depolarize_eps"] == eps
        assert st["max_weight"] is None
        assert st["admitted1"] == st["admitted2"] == 0
        assert st["basis_len"] == len(strings)
        assert st["basis_max_weight"] == 6
        t = step * dt
        got = as_map(basis, coeffs)
        for s in strings:
            want = np.exp(-eps * weight(s) * t)
            assert abs(got[s] - want) <= 1e-12 * max(want, 1e-300), (s, got[s], want)


def test_depolarize_matches_explicit_jumps_with_h():
    n, eps, dt = 3, 0.23, 0.07
    h = chain_h(n)
    deph = [("IIZ", 0.15)]
    L_gen = Lindbladian(n, h, deph)
    L_exp = Lindbladian(n, h, deph + explicit_depolarizing(n, eps))
    strings = ["".join(p) for p in itertools.product("IXZY", repeat=n)]
    basis = _basis_to_codes(strings, n)
    rng = np.random.default_rng(5)
    c0 = rng.normal(size=len(strings))
    b1, c1 = basis.copy(), c0.copy()
    b2, c2 = basis.copy(), c0.copy()
    for _ in range(4):
        b1, c1 = L_gen.pc_step_arr(b1, c1, dt, 10_000_000, drop_tol=0.0, depolarize_eps=eps)
        b2, c2 = L_exp.pc_step_arr(b2, c2, dt, 10_000_000, drop_tol=0.0)
    m1, m2 = as_map(b1, c1), as_map(b2, c2)
    assert len(m1) == len(m2) == 4**n
    diff = max(abs(m1[s] - m2[s]) for s in strings)
    assert diff < 1e-12, diff
    # and it is not the eps = 0 trajectory
    b3, c3 = basis.copy(), c0.copy()
    for _ in range(4):
        b3, c3 = L_gen.pc_step_arr(b3, c3, dt, 10_000_000, drop_tol=0.0)
    m3 = as_map(b3, c3)
    assert max(abs(m1[s] - m3[s]) for s in strings) > 1e-3


def test_depolarize_with_locality_step_matches_plain_step():
    n, eps, dt = 5, 0.3, 0.05
    L = Lindbladian(n, chain_h(n), [("IIZII", 0.1)])
    start = _basis_to_codes(["ZIIII"], n)
    c = np.array([1.0])
    bp, cp = start.copy(), c.copy()
    bl, cl = start.copy(), c.copy()
    for _ in range(3):
        bp, cp = L.pc_step_arr(bp, cp, dt, 10_000_000, drop_tol=0.0, depolarize_eps=eps)
        bl, cl = L.pc_step_arr(bl, cl, dt, 10_000_000, drop_tol=0.0, depolarize_eps=eps, locality=True)
    mp, ml = as_map(bp, cp), as_map(bl, cl)
    assert set(mp) == set(ml)
    assert max(abs(mp[s] - ml[s]) for s in mp) < 1e-12


def test_max_weight_cap_refuses_and_counts():
    n, dt = 6, 0.1
    L = Lindbladian(n, chain_h(n), [])
    start = _basis_to_codes(["IIZIII"], n)
    c = np.array([1.0])
    bc, cc = start.copy(), c.copy()
    bu, cu = start.copy(), c.copy()
    refused, refused_l1 = 0, 0.0
    for _ in range(3):
        bc, cc, st = L.pc_step_arr(bc, cc, dt, 10_000_000, drop_tol=0.0, max_weight=2, return_stats=True)
        assert st["max_weight"] == 2
        assert st["basis_max_weight"] <= 2
        assert all(weight(s) <= 2 for s in _codes_to_basis(bc))
        refused += st["weight_dropped_terms"]
        refused_l1 += st["weight_dropped_l1"]
        bu, cu, su = L.pc_step_arr(bu, cu, dt, 10_000_000, drop_tol=0.0, return_stats=True)
        assert su["weight_dropped_terms"] == 0 and su["weight_dropped_l1"] == 0.0
    assert refused > 0 and refused_l1 > 0.0
    assert any(weight(s) > 2 for s in _codes_to_basis(bu))
    assert len(bc) < len(bu)
    # the cap applies at admission only: a weight-4 observable already in the
    # basis stays and keeps evolving
    b4 = _basis_to_codes(["ZZZZII"], n)
    b4, c4, st4 = L.pc_step_arr(b4, np.array([1.0]), dt, 10_000_000, drop_tol=0.0, max_weight=1, return_stats=True)
    assert "ZZZZII" in _codes_to_basis(b4)
    assert st4["basis_max_weight"] == 4
    assert all(weight(s) <= 1 or s == "ZZZZII" for s in _codes_to_basis(b4))


def test_weight_cap_filters_before_the_room_cap():
    """With a small admission room, the slots go to strings of weight <= l
    (the filter runs before the room cap), so a capped step still admits."""
    n, dt = 6, 0.1
    L = Lindbladian(n, chain_h(n), [])
    b = _basis_to_codes(["IIZIII", "IZIIII", "IIIZII"], n)
    c = np.array([1.0, 0.5, -0.5])
    for _ in range(2):
        b, c = L.pc_step_arr(b, c, dt, 10_000_000, drop_tol=0.0)
    # now b holds weights 1..3+; cap admission tightly and require weight <= 2
    b2, c2, st = L.pc_step_arr(b, c, dt, len(b) + 4, drop_tol=0.0, max_weight=2, return_stats=True)
    new = set(_codes_to_basis(b2)) - set(_codes_to_basis(b))
    assert st["admitted1"] + st["admitted2"] > 0
    assert all(weight(s) <= 2 for s in new)
    assert st["weight_dropped_terms"] > 0


def test_defaults_are_the_plain_step_bit_for_bit():
    n, dt = 5, 0.05
    L = Lindbladian(n, chain_h(n), [("IIZII", 0.1)])
    b0 = _basis_to_codes(["ZIIII", "IXXII"], n)
    c0 = np.array([1.0, 0.25])
    ba, ca = L.pc_step_arr(b0, c0, dt, 50, drop_tol=1e-12, admit_basis=150)
    bb, cb = L.pc_step_arr(b0, c0, dt, 50, drop_tol=1e-12, admit_basis=150, max_weight=None, depolarize_eps=0.0)
    bc, cc, st = L.pc_step_arr(b0, c0, dt, 50, drop_tol=1e-12, admit_basis=150, return_stats=True)
    assert np.array_equal(ba, bb) and np.array_equal(ba, bc)
    assert ca.tobytes() == cb.tobytes() == cc.tobytes()
    assert st["max_weight"] is None and st["depolarize_eps"] == 0.0
    assert st["weight_dropped_terms"] == 0
    assert st["basis_len"] == len(bc) == 50
    assert st["trim_dropped"] > 0  # 150 admitted, 50 kept
    assert st["trim_dropped_l2sq"] >= 0.0
    (bt, ct), d = L._spec.pc_step_timed(b0, c0, dt, 50, 1e-12, None, None, 150, None, False, None, 0.0)
    assert ct.tobytes() == ca.tobytes()
    assert {"leakage1_us", "trim_dropped", "basis_max_weight"} <= set(d)


def test_block_depolarize_h0_counts_physical_sites_only():
    n, m, eps, dt = 3, 3, 0.29, 0.05
    # H = 0 in every species, no exchange: rows decay as exp(-eps * w * t) with
    # w over the three physical sites (the species digit is not a letter)
    B = BlockLindbladian(n, m, [[] for _ in range(m)], None, None, exchange=None, matchings=None)
    strings = ["ZII", "XYI", "ZZZ", "III"]
    basis = _basis_to_codes(strings, n)
    species = np.array([2, 1, 2, 0], dtype=np.uint8)
    coeffs = np.ones(len(strings))
    for step in range(1, 6):
        basis, species, coeffs, st = B.pc_step_arr(
            basis, species, coeffs, dt, 10_000_000, drop_tol=0.0, depolarize_eps=eps, return_stats=True
        )
        assert st["basis_max_weight"] == 3
        t = step * dt
        got = dict(zip(zip(species.tolist(), _codes_to_basis(basis)), coeffs))
        for sp, s in zip([2, 1, 2, 0], strings):
            want = np.exp(-eps * weight(s) * t)
            assert abs(got[(sp, s)] - want) <= 1e-12 * max(want, 1e-300)


def test_block_max_weight_and_defaults():
    n, m, dt = 4, 2, 0.1
    h = [chain_h(n), chain_h(n, 0.3)]
    ex = [[0.0, 0.4], [0.2, 0.0]]
    B = BlockLindbladian(n, m, h, None, None, exchange=ex, matchings={(0, 1): [1, 0, 2, 3]})
    b0 = _basis_to_codes(["ZIII"], n)
    sp0 = np.array([0], dtype=np.uint8)
    c0 = np.array([1.0])
    b, sp, c = b0.copy(), sp0.copy(), c0.copy()
    for _ in range(3):
        b, sp, c, st = B.pc_step_arr(b, sp, c, dt, 10_000_000, drop_tol=0.0, max_weight=2, return_stats=True)
        assert all(weight(s) <= 2 for s in _codes_to_basis(b))
        assert st["basis_max_weight"] <= 2
    ba, spa, ca = B.pc_step_arr(b0, sp0, c0, dt, 10_000_000, drop_tol=0.0)
    bb, spb, cb = B.pc_step_arr(b0, sp0, c0, dt, 10_000_000, drop_tol=0.0, max_weight=None, depolarize_eps=0.0)
    assert np.array_equal(ba, bb) and np.array_equal(spa, spb) and ca.tobytes() == cb.tobytes()


@pytest.mark.parametrize("eps", [0.0, 0.1])
def test_depolarize_preserves_the_identity(eps):
    n, dt = 4, 0.1
    L = Lindbladian(n, chain_h(n), [("IZII", 0.2)])
    b = _basis_to_codes(["IIII", "ZIII"], n)
    c = np.array([0.5, 1.0])
    for _ in range(3):
        b, c = L.pc_step_arr(b, c, dt, 10_000_000, drop_tol=0.0, depolarize_eps=eps)
    assert as_map(b, c)["IIII"] == 0.5
