# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""In-place dissipator setters: ``Lindbladian.set_kossakowski`` and
``Lindbladian.set_jump_rates`` (mirrors ``test_h_coeffs.py``).

Contract: after ``L.set_kossakowski(K)`` / ``L.set_jump_rates(r)`` every
output of ``L`` is bit-identical (``np.array_equal``) to a freshly
constructed ``Lindbladian`` with ``K`` / ``r``; the Hamiltonian and the
other dissipator block are untouched.
"""

import itertools
import time

import numpy as np
import pytest

from ppvm import Lindbladian, sigma_minus, sigma_plus
from ppvm.lindblad import _string_to_codes


def _local_terms(n: int, max_weight: int) -> list[str]:
    """All Pauli strings on n qubits of weight 1..max_weight (nearest-neighbour for weight 2)."""
    out = []
    for q in range(n):
        for p in "XYZ":
            s = ["I"] * n
            s[q] = p
            out.append("".join(s))
    if max_weight >= 2:
        for q in range(n - 1):
            for a, b in itertools.product("XYZ", repeat=2):
                s = ["I"] * n
                s[q], s[q + 1] = a, b
                out.append("".join(s))
    return out


def _random_psd(rng, m, scale=0.05):
    a = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    return scale * (a @ a.conj().T) / m


def _jump_ops(n):
    """Fixed jump operators: Hermitian single Paulis (fast path), σ⁻, σ⁺, a complex lincomb."""
    ops = []
    for q in range(n):
        s = ["I"] * n
        s[q] = "Z"
        ops.append("".join(s))
    ops.append(sigma_minus(0, n))
    ops.append(sigma_plus(n - 1, n))
    xs = ["I"] * n
    xs[1] = "X"
    ys = ["I"] * n
    ys[2] = "Y"
    ops.append([("".join(xs), 0.7 + 0.1j), ("".join(ys), -0.3j)])
    return ops


def _num_jumps(n):
    return n + 3


def _random_rates(rng, n):
    return rng.uniform(0.01, 0.3, size=_num_jumps(n))


def _build(n, rates, K, h_seed=0):
    """H: random weight-1/2 terms; jumps: ``_jump_ops`` with ``rates``; Kossakowski: σ⁻ ops with ``K``."""
    rng = np.random.default_rng(h_seed)
    words = _local_terms(n, 2)
    h_terms = list(zip(words, [float(c) for c in rng.normal(size=len(words))]))
    jumps = list(zip(_jump_ops(n), [float(r) for r in rates]))
    ops = [sigma_minus(j, n) for j in range(n)]
    return Lindbladian(n, h_terms, jumps, kossakowski=(ops, K))


def _assert_same_tuple(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype
        assert np.array_equal(x, y)


def _start_basis(n):
    words = []
    for q in range(n):
        s = ["I"] * n
        s[q] = "Z"
        words.append("".join(s))
    s = ["I"] * n
    s[0] = "X"
    words.append("".join(s))
    basis = np.stack([_string_to_codes(w, n) for w in words])
    coeffs = np.linspace(1.0, 0.3, len(words))
    return basis, coeffs


def _compare_all(L_set, L_fresh, n, rng):
    assert L_set.num_jump_terms == L_fresh.num_jump_terms
    words_all = _local_terms(n, 2)
    # action on several strings
    for w in rng.choice(words_all, size=8, replace=False):
        p = _string_to_codes(str(w), n)
        _assert_same_tuple(L_set.action_arr(p), L_fresh.action_arr(p))
    # leakage + generator on a random basis
    basis_words = list(rng.choice(words_all, size=min(12, len(words_all)), replace=False))
    basis = np.stack([_string_to_codes(str(w), n) for w in basis_words])
    c = rng.normal(size=len(basis_words))
    _assert_same_tuple(L_set.leakage_arr(basis, c), L_fresh.leakage_arr(basis, c))
    prot = basis[:2]
    _assert_same_tuple(L_set.leakage_arr(basis, c, prot), L_fresh.leakage_arr(basis, c, prot))
    _assert_same_tuple(L_set.generator_arr(basis), L_fresh.generator_arr(basis))
    # several pc_steps with truncation, two truncation modes
    for kwargs in (
        {"max_basis": 24, "drop_tol": 1e-10},
        {"max_basis": 16, "drop_tol": 0.0, "admit_basis": 40},
    ):
        b1, c1 = _start_basis(n)
        b2, c2 = b1.copy(), c1.copy()
        for _ in range(6):
            b1, c1 = L_set.pc_step_arr(b1, c1, 0.05, **kwargs)
            b2, c2 = L_fresh.pc_step_arr(b2, c2, 0.05, **kwargs)
            assert np.array_equal(b1, b2)
            assert np.array_equal(c1, c2)
        assert len(c1) <= kwargs["max_basis"]


@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_set_kossakowski_bit_identical_to_fresh(n, seed):
    rng = np.random.default_rng(1000 * n + seed)
    rates = _random_rates(rng, n)
    K0 = _random_psd(rng, n)
    K1 = _random_psd(rng, n)
    # decouple op 0 (block-diagonal K stays PSD): exact-zero off-diagonal
    # pairs are dropped at compile time in both paths
    K1[0, 1:] = 0.0
    K1[1:, 0] = 0.0
    L = _build(n, rates, K0)
    L.set_kossakowski(K1)
    L_fresh = _build(n, rates, K1)
    _compare_all(L, L_fresh, n, rng)
    # H setter and K setter compose: the compiled object tracks both
    L.set_h_coeffs(L_fresh.h_coeffs)
    _compare_all(L, L_fresh, n, rng)
    # Set again (repeated updates, then back to the original).
    L.set_kossakowski(K0, check_psd=False)
    _compare_all(L, _build(n, rates, K0), n, rng)


@pytest.mark.parametrize("n", [3, 8])
def test_set_kossakowski_rank_deficient_and_zero(n):
    """Rank-1 K (a single collective mode) and K = 0 (no Kossakowski pairs at all)."""
    rng = np.random.default_rng(n)
    rates = _random_rates(rng, n)
    L = _build(n, rates, _random_psd(rng, n))
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    K1 = 0.1 * np.outer(v, v.conj())
    L.set_kossakowski(K1)
    _compare_all(L, _build(n, rates, K1), n, rng)
    L.set_kossakowski(np.zeros((n, n)))
    L_zero = _build(n, rates, np.zeros((n, n)))
    assert L.num_jump_terms == L_zero.num_jump_terms == _num_jumps(n)
    _compare_all(L, L_zero, n, rng)
    # pairs come back
    K2 = _random_psd(rng, n)
    L.set_kossakowski(K2)
    _compare_all(L, _build(n, rates, K2), n, rng)


@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("seed", [0, 1])
def test_set_jump_rates_bit_identical_to_fresh(n, seed):
    rng = np.random.default_rng(2000 * n + seed)
    K = _random_psd(rng, n)
    r0 = _random_rates(rng, n)
    r1 = _random_rates(rng, n)
    r1[rng.integers(len(r1))] = 0.0  # a zero rate is still a term
    L = _build(n, r0, K)
    L.set_jump_rates(r1)
    L_fresh = _build(n, r1, K)
    _compare_all(L, L_fresh, n, rng)
    # both setters together
    K2 = _random_psd(rng, n)
    L.set_kossakowski(K2)
    L.set_jump_rates(r0)
    _compare_all(L, _build(n, r0, K2), n, rng)


def test_getters_round_trip_and_copy():
    n = 3
    rng = np.random.default_rng(7)
    rates = _random_rates(rng, n)
    K = _random_psd(rng, n)
    L = _build(n, rates, K)
    assert L.num_jump_ops == _num_jumps(n)
    assert L.num_kossakowski_ops == n
    assert L.num_jump_terms > L.num_jump_ops  # Kossakowski pairs are compiled entries
    got = L.jump_rates
    assert got.dtype == np.float64 and got.shape == (_num_jumps(n),)
    assert np.array_equal(got, rates)
    got[:] = 99.0  # a copy
    assert np.array_equal(L.jump_rates, rates)
    gk = L.kossakowski
    assert gk.dtype == np.complex128 and gk.shape == (n, n)
    assert np.array_equal(gk, K)
    gk[:] = 0.0
    assert np.array_equal(L.kossakowski, K)
    # list input, real input, non-contiguous views are accepted
    r1 = _random_rates(rng, n)
    L.set_jump_rates(list(r1))
    assert np.array_equal(L.jump_rates, r1)
    big = np.zeros(2 * len(r1))
    big[::2] = r1 * 2
    L.set_jump_rates(big[::2])
    assert np.array_equal(L.jump_rates, 2 * r1)
    Kr = np.real(_random_psd(rng, n))
    Kr = 0.5 * (Kr + Kr.T)
    L.set_kossakowski(Kr)
    assert np.array_equal(L.kossakowski, Kr.astype(np.complex128))
    L.set_kossakowski(np.asfortranarray(K))
    assert np.array_equal(L.kossakowski, K)
    L._spec.set_kossakowski(np.asfortranarray(K))  # raw spec too
    assert np.array_equal(L.kossakowski, K)
    assert L._spec.kossakowski.shape == (n, n)
    # The H coefficients survive the dissipator setters.
    h = L.h_coeffs
    L.set_kossakowski(K)
    L.set_jump_rates(rates)
    assert np.array_equal(L.h_coeffs, h)


def test_without_blocks():
    n = 3
    words = _local_terms(n, 1)
    h_terms = list(zip(words, np.linspace(0.1, 1.0, len(words))))
    L_plain = Lindbladian(n, h_terms, [("ZII", 0.1)])
    assert L_plain.num_kossakowski_ops == 0
    assert L_plain.kossakowski is None
    assert L_plain._spec.kossakowski.shape == (0, 0)
    with pytest.raises(ValueError, match="without a kossakowski block"):
        L_plain.set_kossakowski(np.eye(1))
    with pytest.raises(ValueError, match="without a Kossakowski block"):
        L_plain._spec.set_kossakowski(np.eye(1, dtype=np.complex128))
    L_plain.set_jump_rates([0.4])
    assert np.array_equal(L_plain.jump_rates, [0.4])
    L_none = Lindbladian(n, h_terms)
    assert L_none.num_jump_ops == 0
    assert L_none.jump_rates.shape == (0,)
    with pytest.raises(ValueError, match="without jump_terms"):
        L_none.set_jump_rates([])
    with pytest.raises(ValueError, match="without jump operators"):
        L_none._spec.set_jump_rates(np.zeros(0))
    L_konly = Lindbladian(n, h_terms, kossakowski=([sigma_minus(0, n)], [[0.3]]))
    assert L_konly.num_jump_ops == 0 and L_konly.num_kossakowski_ops == 1
    with pytest.raises(ValueError):
        L_konly.set_jump_rates([0.1])
    L_konly.set_kossakowski([[0.7]])
    assert np.array_equal(L_konly.kossakowski, [[0.7]])


def test_wrong_input_raises_and_leaves_object_unchanged():
    n = 3
    rng = np.random.default_rng(11)
    rates = _random_rates(rng, n)
    K = _random_psd(rng, n)
    L = _build(n, rates, K)
    p = _string_to_codes("ZXI", n)
    ref = L.action_arr(p)
    m = _num_jumps(n)
    for bad in (
        np.zeros(m - 1),
        np.zeros(m + 1),
        np.zeros((m, 1)),
        np.zeros((1, m)),
        np.zeros(0),
        np.float64(1.0),
    ):
        with pytest.raises(ValueError):
            L.set_jump_rates(bad)
        with pytest.raises(ValueError):
            L._spec.set_jump_rates(bad)
    neg = rates.copy()
    neg[2] = -1e-3
    with pytest.raises(ValueError, match="non-negative"):
        L.set_jump_rates(neg)
    with pytest.raises(ValueError, match="non-negative"):
        L._spec.set_jump_rates(neg)
    with pytest.raises(ValueError):  # raw spec: no dtype cast
        L._spec.set_jump_rates(np.zeros(m, dtype=np.int64))
    for bad in (
        np.zeros((n - 1, n - 1)),
        np.zeros((n + 1, n + 1)),
        np.zeros((n, n + 1)),
        np.zeros(n),
        np.zeros((n, n, 1)),
    ):
        with pytest.raises(ValueError):
            L.set_kossakowski(bad)
        with pytest.raises(ValueError):
            L._spec.set_kossakowski(np.asarray(bad, dtype=np.complex128))
    with pytest.raises(ValueError):  # raw spec: no dtype cast
        L._spec.set_kossakowski(np.zeros((n, n)))
    nonherm = K.copy()
    nonherm[0, 1] += 1e-3
    with pytest.raises(ValueError, match="not Hermitian"):
        L.set_kossakowski(nonherm)
    with pytest.raises(ValueError, match="not Hermitian"):
        L._spec.set_kossakowski(nonherm)
    # PSD check: on by default, flag to skip (Hermiticity is still enforced).
    w, V = np.linalg.eigh(K)
    w[0] = -1e-3 * w.max()
    nonpsd = (V * w) @ V.conj().T
    with pytest.raises(ValueError, match="not positive semidefinite"):
        L.set_kossakowski(nonpsd)
    _assert_same_tuple(L.action_arr(p), ref)
    assert np.array_equal(L.kossakowski, K)
    assert np.array_equal(L.jump_rates, rates)
    L.set_kossakowski(nonpsd, check_psd=False)
    assert np.allclose(L.kossakowski, nonpsd)
    with pytest.raises(ValueError, match="not Hermitian"):
        L.set_kossakowski(nonherm, check_psd=False)


def test_setter_timing_8_qubits():
    """The setters cost a small fraction of a fresh construction."""
    n = 8
    rng = np.random.default_rng(5)
    rates = _random_rates(rng, n)
    K = _random_psd(rng, n)
    reps = 200
    Ks = [_random_psd(rng, n) for _ in range(reps)]
    rs = [_random_rates(rng, n) for _ in range(reps)]

    t0 = time.perf_counter()
    for Ki in Ks:
        L = _build(n, rates, Ki)
    t_build = (time.perf_counter() - t0) / reps

    L = _build(n, rates, K)
    t0 = time.perf_counter()
    for Ki in Ks:
        L.set_kossakowski(Ki)
    t_set_k = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for Ki in Ks:
        L.set_kossakowski(Ki, check_psd=False)
    t_set_k_nopsd = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for Ki in Ks:
        L._spec.set_kossakowski(Ki)
    t_set_k_raw = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for ri in rs:
        L.set_jump_rates(ri)
    t_set_r = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for ri in rs:
        L._spec.set_jump_rates(ri)
    t_set_r_raw = (time.perf_counter() - t0) / reps
    print(
        f"\n8 qubits, {len(_local_terms(n, 2))} H terms, {_num_jumps(n)} jumps, 8x8 dense K "
        f"({reps} reps): construction {t_build * 1e6:.0f} us; "
        f"set_kossakowski {t_set_k * 1e6:.1f} us (check_psd=False {t_set_k_nopsd * 1e6:.1f} us, "
        f"_LindbladSpec {t_set_k_raw * 1e6:.1f} us); "
        f"set_jump_rates {t_set_r * 1e6:.2f} us (_LindbladSpec {t_set_r_raw * 1e6:.2f} us)"
    )
    assert np.array_equal(L.kossakowski, Ks[-1])
    assert np.array_equal(L.jump_rates, rs[-1])
    assert t_set_k < t_build / 2  # the 8x8 eigvalsh is most of the wrapper's cost here
    assert t_set_k_raw < t_build / 4
    assert t_set_r < t_build / 20


def _two_local_ops(n):
    """All nearest-and-next-nearest pairs, each a 9-term complex lincomb (dipolar-like)."""
    rng = np.random.default_rng(99)
    ops = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 3)):
            lin = []
            for a, b in itertools.product("XYZ", repeat=2):
                s = ["I"] * n
                s[i], s[j] = a, b
                lin.append(("".join(s), complex(rng.normal(), rng.normal())))
            ops.append(lin)
    return ops


def test_setter_timing_8_qubits_two_local_ops():
    """Dipolar-like operator basis: 13 two-local 9-term ops, dense 13x13 K."""
    n = 8
    rng = np.random.default_rng(6)
    ops = _two_local_ops(n)
    m = len(ops)
    words = _local_terms(n, 2)
    h_terms = list(zip(words, [float(c) for c in rng.normal(size=len(words))]))
    jumps = list(zip(_jump_ops(n), [float(r) for r in _random_rates(rng, n)]))
    reps = 100
    Ks = [_random_psd(rng, m) for _ in range(reps)]

    def build(K):
        return Lindbladian(n, h_terms, jumps, kossakowski=(ops, K))

    t0 = time.perf_counter()
    for Ki in Ks:
        L = build(Ki)
    t_build = (time.perf_counter() - t0) / reps
    L = build(Ks[0])
    t0 = time.perf_counter()
    for Ki in Ks:
        L.set_kossakowski(Ki)
    t_set_k = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for Ki in Ks:
        L._spec.set_kossakowski(Ki)
    t_set_k_raw = (time.perf_counter() - t0) / reps
    print(
        f"\n8 qubits, {m} two-local 9-term ops, {m}x{m} dense K ({reps} reps): "
        f"construction {t_build * 1e6:.0f} us; set_kossakowski {t_set_k * 1e6:.1f} us "
        f"(_LindbladSpec {t_set_k_raw * 1e6:.1f} us)"
    )
    # correctness of the last state
    L_fresh = build(Ks[-1])
    p = _string_to_codes("ZXIIYIIZ", n)
    _assert_same_tuple(L.action_arr(p), L_fresh.action_arr(p))
    assert t_set_k < t_build / 2
