# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""Block-structured chemical exchange: ``BlockLindbladian``.

Reference: the Schrödinger-picture direct-sum master equation

    d rho_c/dt = L_c rho_c + sum_{d != c} ( k_dc Pi_dc rho_d Pi_dc^dag - k_cd rho_c )

built as a dense superoperator on the Pauli-coefficient basis (M * 4^N rows,
one block per species). The Heisenberg generator of the block Lindbladian
must be its transpose (the Pauli basis is orthogonal under the trace inner
product), and an uncapped ``pc_step_arr`` must equal ``expm(dt G)``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.linalg

from ppvm import BlockLindbladian, Lindbladian, sigma_minus
from ppvm.lindblad import _basis_to_codes, _codes_to_basis, _string_to_codes

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
BIG = 10_000_000


def pauli_matrix(s):
    out = np.ones((1, 1), dtype=complex)
    for ch in s:
        out = np.kron(out, PAULI[ch])
    return out


def all_strings(n):
    return ["".join(p) for p in itertools.product("IXZY", repeat=n)]  # ppvm code order


def lincomb_matrix(lincomb, n):
    if isinstance(lincomb, str):
        return pauli_matrix(lincomb)
    return sum(complex(c) * pauli_matrix(s) for s, c in lincomb)


def permutation_matrix(perm, n):
    """Pi with Pi |s> = |sigma(s)>, the bit of qubit p moved to qubit perm[p]
    (qubit 0 = most significant bit)."""
    d = 1 << n
    pi = np.zeros((d, d))
    for s in range(d):
        t = 0
        for p in range(n):
            bit = (s >> (n - 1 - p)) & 1
            t |= bit << (n - 1 - perm[p])
        pi[t, s] = 1.0
    return pi


def complete_matchings(matchings, n, m):
    out = {}
    for (c, d), perm in matchings.items():
        out[(c, d)] = list(perm)
    for (c, d), perm in list(out.items()):
        if (d, c) not in out:
            inv = [0] * n
            for p, q in enumerate(perm):
                inv[q] = p
            out[(d, c)] = inv
    return {(c, d): out.get((c, d), list(range(n))) for c in range(m) for d in range(m) if c != d}


def schrodinger_species(h_terms, jump_terms, kossakowski, n):
    """rho -> L_c rho as a function on 2^n x 2^n matrices."""
    h = sum((c * pauli_matrix(s) for s, c in h_terms), np.zeros((1 << n, 1 << n), dtype=complex))
    jumps = [(lincomb_matrix(op, n), rate) for op, rate in (jump_terms or ())]
    k_ops, K = ([], None)
    if kossakowski is not None:
        ops, K = kossakowski
        k_ops = [lincomb_matrix(op, n) for op in ops]
        K = np.asarray(K, dtype=complex)

    def apply(rho):
        out = -1j * (h @ rho - rho @ h)
        for L, rate in jumps:
            LdL = L.conj().T @ L
            out += rate * (L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL))
        for a, A in enumerate(k_ops):
            for b, B in enumerate(k_ops):
                if K[a, b] == 0:
                    continue
                AdB = A.conj().T @ B
                # Heisenberg D*(O) = sum K_ab (A_a^dag O A_b - 1/2 {A_a^dag A_b, O})
                # => Schrödinger D(rho) = sum K_ab (A_b rho A_a^dag - 1/2 {A_a^dag A_b, rho})
                out += K[a, b] * (B @ rho @ A.conj().T - 0.5 * (AdB @ rho + rho @ AdB))
        return out

    return apply


def dense_heisenberg_generator(n, m, h_terms, jump_terms, kossakowski, exchange, matchings):
    """Transpose of the Schrödinger block superoperator on the Pauli basis
    ordered as (species, string) with strings in ``all_strings`` order."""
    strings = all_strings(n)
    mats = [pauli_matrix(s) for s in strings]
    d = 1 << n
    nb = len(strings)
    species_apply = [schrodinger_species(h_terms[c], (jump_terms or [None] * m)[c],
                                         (kossakowski or [None] * m)[c], n) for c in range(m)]
    k = np.asarray(exchange, dtype=float).copy()
    np.fill_diagonal(k, 0.0)
    perms = complete_matchings(matchings or {}, n, m)
    pis = {key: permutation_matrix(perm, n) for key, perm in perms.items()}
    S = np.zeros((m * nb, m * nb))
    for dsp in range(m):
        for j, Q in enumerate(mats):
            # state rho = Q in species dsp
            blocks = {dsp: species_apply[dsp](Q) - k[dsp].sum() * Q}
            for csp in range(m):
                if csp != dsp and k[dsp, csp] != 0.0:
                    pi = pis[(dsp, csp)]
                    blocks[csp] = blocks.get(csp, 0) + k[dsp, csp] * (pi @ Q @ pi.T)
            col = dsp * nb + j
            for csp, mat in blocks.items():
                for i, P in enumerate(mats):
                    v = np.trace(P @ mat) / d
                    assert abs(v.imag) < 1e-12
                    S[csp * nb + i, col] = v.real
    return S.T, strings


def random_h(rng, n, scale=1.0):
    return [(s, float(rng.normal()) * scale) for s in all_strings(n) if s != "I" * n]


def full_basis(n, m):
    strings = all_strings(n)
    basis = np.concatenate([_basis_to_codes(strings, n)] * m)
    species = np.repeat(np.arange(m, dtype=np.uint8), len(strings))
    return basis, species, strings


def dense_generator_from_block(L_block, n, m):
    basis, species, strings = full_basis(n, m)
    rows, cols, vals = L_block.generator_arr(basis, species)
    G = np.zeros((m * len(strings),) * 2)
    np.add.at(G, (rows.astype(int), cols.astype(int)), vals)
    return G, basis, species


def two_species_model(rng):
    n, m = 2, 2
    h_terms = [random_h(rng, n), random_h(rng, n)]
    jump_terms = [[(sigma_minus(0, n), 0.3)], [("ZI", 0.2), ([("XI", 0.6 + 0.2j), ("IY", -0.4j)], 0.15)]]
    ops = [sigma_minus(0, n), sigma_minus(1, n), [("XX", 0.5), ("YY", 0.5j)]]
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    K = 0.2 * a @ a.conj().T / 3
    kossakowski = [None, (ops, K)]
    exchange = [[0.0, 0.37], [0.91, 0.0]]
    return n, m, h_terms, jump_terms, kossakowski, exchange, {}


def three_species_model(rng):
    n, m = 3, 3
    h_terms = [random_h(rng, n, 0.7) for _ in range(m)]
    jump_terms = [[(sigma_minus(1, n), 0.25)], None, [("IZI", 0.1)]]
    ops = [sigma_minus(0, n), sigma_minus(2, n)]
    a = rng.normal(size=(2, 2))
    kossakowski = [(ops, 0.1 * a @ a.T), None, None]
    exchange = [[0.0, 0.5, 0.0], [0.2, 0.0, 0.8], [0.0, 0.3, 0.0]]
    matchings = {(0, 1): [1, 2, 0], (1, 2): [0, 2, 1]}      # (1, 0) and (2, 1) are the inverses
    return n, m, h_terms, jump_terms, kossakowski, exchange, matchings


@pytest.mark.parametrize("model", [two_species_model, three_species_model])
def test_generator_is_adjoint_of_dense_direct_sum(model):
    rng = np.random.default_rng(11 if model is two_species_model else 23)
    n, m, h, j, k_, ex, mt = model(rng)
    L = BlockLindbladian(n, m, h, j, k_, exchange=ex, matchings=mt)
    G, basis, species = dense_generator_from_block(L, n, m)
    G_ref, _ = dense_heisenberg_generator(n, m, h, j, k_, ex, mt)
    assert np.abs(G - G_ref).max() < 1e-12
    # exchange moves weight backwards along the reaction with the right rate
    rate_out = L.rate_out
    assert np.allclose(rate_out, np.asarray(ex).sum(axis=1))


@pytest.mark.parametrize("model", [two_species_model, three_species_model])
def test_uncapped_pc_step_matches_dense_expm(model):
    """From the full block basis an uncapped step is the exact exponential of
    the dense direct-sum generator (several steps, 1e-12)."""
    rng = np.random.default_rng(5 if model is two_species_model else 7)
    n, m, h, j, k_, ex, mt = model(rng)
    L = BlockLindbladian(n, m, h, j, k_, exchange=ex, matchings=mt)
    G_ref, strings = dense_heisenberg_generator(n, m, h, j, k_, ex, mt)
    basis, species, _ = full_basis(n, m)
    v = rng.normal(size=len(basis))
    coeff = v.copy()
    dt, steps = 0.07, 3
    for _ in range(steps):
        basis, species, coeff = L.pc_step_arr(basis, species, coeff, dt, max_basis=BIG, drop_tol=0.0)
    v = scipy.linalg.expm(dt * steps * G_ref) @ v
    assert len(coeff) == len(v)
    got = np.zeros_like(v)
    nb = len(strings)
    for s, c, val in zip(_codes_to_basis(basis), species, coeff):
        got[int(c) * nb + strings.index(s)] = val
    assert np.abs(got - v).max() < 1e-12


@pytest.mark.parametrize("model", [two_species_model, three_species_model])
def test_growing_basis_step_is_exact_on_the_reached_rows(model):
    """From a few rows the basis grows by two hops per step (into other species
    through the exchange) and each step equals the exponential of the generator
    restricted to the rows reached; once every row is reached the trajectory
    coincides with the dense one."""
    rng = np.random.default_rng(5 if model is two_species_model else 7)
    n, m, h, j, k_, ex, mt = model(rng)
    L = BlockLindbladian(n, m, h, j, k_, exchange=ex, matchings=mt)
    G_ref, strings = dense_heisenberg_generator(n, m, h, j, k_, ex, mt)
    nb = len(strings)
    start = {(0, strings[5]): 1.0, (0, strings[9]): -0.4, (1, strings[2]): 0.25}
    basis = _basis_to_codes([s for _, s in start], n)
    species = np.array([c for c, _ in start], dtype=np.uint8)
    coeff = np.array(list(start.values()))
    dt = 0.07
    reached = set()
    for _ in range(4):
        prev = np.zeros(m * nb)
        for s, c, val in zip(_codes_to_basis(basis), species, coeff):
            prev[int(c) * nb + strings.index(s)] = val
        basis, species, coeff = L.pc_step_arr(basis, species, coeff, dt, max_basis=BIG, drop_tol=0.0)
        idx = np.array([int(c) * nb + strings.index(s) for s, c in zip(_codes_to_basis(basis), species)])
        assert reached <= set(idx.tolist())                          # rows are never lost without a cap
        reached = set(idx.tolist())
        assert len(set(species.tolist())) == m                       # the exchange reached every species
        ref = scipy.linalg.expm(dt * G_ref[np.ix_(idx, idx)]) @ prev[idx]
        assert np.abs(coeff - ref).max() < 1e-12
    assert len(reached) == m * nb


def test_total_identity_and_population_kinetics():
    """H = 0, no dissipation: the row (c, I) is the population of species c;
    its Heisenberg evolution is the transpose kinetic matrix, so (0, I)
    becomes p(0 -> stay) on (0, I) and p(1 -> 0) on (1, I) at time t."""
    n, m = 1, 2
    k01, k10 = 0.6, 0.4
    L = BlockLindbladian(n, m, [[], []], exchange=[[0, k01], [k10, 0]])
    basis = _basis_to_codes(["I"], n)
    species = np.array([0], dtype=np.uint8)
    coeff = np.array([1.0])
    t = 0.5
    basis, species, coeff = L.pc_step_arr(basis, species, coeff, t, max_basis=BIG, drop_tol=0.0)
    out = {int(s): c for s, c in zip(species, coeff)}
    ktot = k01 + k10
    pi0 = k10 / ktot
    assert abs(out[0] - (pi0 + (1 - pi0) * np.exp(-ktot * t))) < 1e-13     # P(0 at t | 0 at 0)
    assert abs(out[1] - (pi0 - pi0 * np.exp(-ktot * t))) < 1e-13           # P(0 at t | 1 at 0)
    # the total identity is stationary
    basis = _basis_to_codes(["I", "I"], n)
    species = np.array([0, 1], dtype=np.uint8)
    basis, species, coeff = L.pc_step_arr(basis, species, np.array([1.0, 1.0]), t, max_basis=BIG, drop_tol=0.0)
    assert np.allclose(coeff, 1.0, atol=1e-14)


def test_matching_direction():
    """With the matching perm (0 -> 1, 1 -> 0) between species 0 and 1 and H = 0,
    the observable X on qubit 0 of species 1 flows into X on qubit 1 of species 0
    (Pi^dag X_perm[p] Pi = X_p)."""
    L = BlockLindbladian(2, 2, [[], []], exchange=[[0, 3.0], [0, 0]], matchings={(0, 1): [1, 0]})
    out = L.action(1, "XI")
    assert out == {(0, "IX"): 3.0}          # k_01 moves weight from species 1 to species 0; nothing leaves 1
    out = L.action(0, "XI")
    assert out == {(0, "XI"): -3.0}         # species 0 only loses (k_10 = 0)


def test_single_species_is_bit_identical_to_lindbladian():
    """M = 1 (no species digits): every truncated step equals Lindbladian's."""
    rng = np.random.default_rng(3)
    n = 4
    h = random_h(rng, 2)
    h = [("".join(s) + "II", c) for s, c in h] + [("IIXY", 0.3), ("IZZI", -0.7), ("ZIIZ", 0.4)]
    jumps = [(sigma_minus(0, n), 0.2), ("IZII", 0.1)]
    ops = [sigma_minus(j, n) for j in range(n)]
    a = rng.normal(size=(n, n))
    K = 0.05 * a @ a.T / n
    L1 = Lindbladian(n, h, jumps, kossakowski=(ops, K))
    Lb = BlockLindbladian(n, 1, [h], [jumps], [(ops, K)])
    assert Lb.n_digits == 0
    words = ["ZIII", "IZII", "XIII", "IIYX"]
    b1 = _basis_to_codes(words, n)
    c1 = np.linspace(1.0, 0.4, len(words))
    bb, sb, cb = b1.copy(), np.zeros(len(words), dtype=np.uint8), c1.copy()
    for kwargs in ({"max_basis": 40, "drop_tol": 1e-10}, {"max_basis": 30, "drop_tol": 0.0, "admit_basis": 90}):
        b1_, c1_ = b1.copy(), c1.copy()
        bb_, sb_, cb_ = bb.copy(), sb.copy(), cb.copy()
        for _ in range(5):
            b1_, c1_ = L1.pc_step_arr(b1_, c1_, 0.05, protected_arr=b1[:1], **kwargs)
            bb_, sb_, cb_ = Lb.pc_step_arr(bb_, sb_, cb_, 0.05, protected_arr=b1[:1],
                                           protected_species=np.zeros(1, dtype=np.uint8), **kwargs)
            assert np.array_equal(b1_, bb_) and np.array_equal(c1_, cb_)
            assert not sb_.any()


def test_set_h_coeffs_matches_fresh_construction():
    rng = np.random.default_rng(42)
    n, m, h, j, k_, ex, mt = two_species_model(rng)
    L = BlockLindbladian(n, m, h, j, k_, exchange=ex, matchings=mt)
    new1 = rng.normal(size=len(h[1]))
    L.set_h_coeffs(1, new1)
    assert np.array_equal(L.h_coeffs(1), new1)
    h_new = [h[0], [(s, float(c)) for (s, _), c in zip(h[1], new1)]]
    L_fresh = BlockLindbladian(n, m, h_new, j, k_, exchange=ex, matchings=mt)
    basis, species, strings = full_basis(n, m)
    coeff = rng.normal(size=len(basis))
    out_a = L.pc_step_arr(basis, species, coeff, 0.1, max_basis=20, drop_tol=0.0, admit_basis=60)
    out_b = L_fresh.pc_step_arr(basis, species, coeff, 0.1, max_basis=20, drop_tol=0.0, admit_basis=60)
    for x, y in zip(out_a, out_b):
        assert np.array_equal(x, y)
    with pytest.raises(ValueError):
        L.set_h_coeffs(0, np.zeros(len(h[0]) + 1))
    with pytest.raises(ValueError):
        L.set_h_coeffs(2, np.zeros(len(h[0])))


def test_truncation_caps_total_rows_and_protects():
    rng = np.random.default_rng(1)
    n, m, h, j, k_, ex, mt = three_species_model(rng)
    L = BlockLindbladian(n, m, h, j, k_, exchange=ex, matchings=mt)
    basis = _basis_to_codes(["XII"], n)
    species = np.array([2], dtype=np.uint8)
    coeff = np.array([1.0])
    for _ in range(6):
        basis, species, coeff = L.pc_step_arr(basis, species, coeff, 0.1, max_basis=25, drop_tol=0.0,
                                              admit_basis=75, protected_arr=_basis_to_codes(["XII"], n),
                                              protected_species=np.array([2], dtype=np.uint8))
        assert len(coeff) <= 25
        rows = set(zip(species.tolist(), _codes_to_basis(basis)))
        assert (2, "XII") in rows
    # the exchange does feed the other species (here below the rank cap: H dominates at this k)
    _, lsp, lco = L.leakage_arr(_basis_to_codes(["XII"], n), np.array([2], dtype=np.uint8), np.array([1.0]))
    assert set(lsp.tolist()) == {1, 2} and lco[lsp == 1].sum() == pytest.approx(ex[1][2])


def test_construction_errors():
    h = [[("ZI", 1.0)], [("IZ", 1.0)]]
    with pytest.raises(ValueError):
        BlockLindbladian(2, 2, h, exchange=[[0, 1.0]])                      # wrong shape
    with pytest.raises(ValueError):
        BlockLindbladian(2, 2, h, exchange=[[0, -1.0], [1.0, 0]])           # negative rate
    with pytest.raises(ValueError):
        BlockLindbladian(2, 2, h, matchings={(0, 1): [0, 0]})               # not a permutation
    with pytest.raises(ValueError):
        BlockLindbladian(2, 2, h, matchings={(0, 2): [0, 1]})               # species out of range
    with pytest.raises(ValueError):
        BlockLindbladian(2, 2, [h[0]])                                      # one entry per species
    with pytest.raises(ValueError):
        BlockLindbladian(2, 2, h, kossakowski=[None, ([sigma_minus(0, 2)], [[-1.0]])])   # not PSD
    L = BlockLindbladian(2, 2, h)
    with pytest.raises(ValueError):
        L.pc_step_arr(_basis_to_codes(["ZI"], 2), np.array([2], dtype=np.uint8), np.array([1.0]), 0.1, 10)
    with pytest.raises(ValueError):
        L.pc_step_arr(_basis_to_codes(["ZI", "ZI"], 2), np.array([0, 0], dtype=np.uint8), np.ones(2), 0.1, 10)
    # five species need two digits; the identity row round-trips in each
    L5 = BlockLindbladian(1, 5, [[] for _ in range(5)])
    assert L5.n_digits == 2
    for s in range(5):
        assert L5.action(s, "I") == {}
