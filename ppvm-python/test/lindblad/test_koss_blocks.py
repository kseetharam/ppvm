# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""Block-sparse Kossakowski input: ``set_kossakowski_blocks`` / ``set_kossakowski_sparse``.

Contract: every way of giving the same ``K`` -- the dense matrix, a list of
``(op_indices, K_block)`` blocks, COO triples in both triangles or in one --
compiles the same dissipator: every output of the ``Lindbladian`` is
bit-identical (``np.array_equal``) to the dense construction, and the dense
``(M, M)`` matrix is never formed on the block / COO paths.
"""

import itertools
import resource
import time

import numpy as np
import pytest

from ppvm import Lindbladian, sigma_minus
from ppvm.lindblad import _string_to_codes

from .test_dissipator_setters import (
    _assert_same_tuple,
    _build,
    _compare_all,
    _jump_ops,
    _local_terms,
    _random_psd,
    _random_rates,
)


def _block_pattern(rng, m, sizes, scale=0.05):
    """Random PSD blocks on a random permutation of ``range(m)``; returns (blocks, dense K)."""
    assert sum(sizes) <= m
    perm = rng.permutation(m)
    blocks, K = [], np.zeros((m, m), dtype=np.complex128)
    start = 0
    for size in sizes:
        idx = np.sort(perm[start : start + size])
        K_b = _random_psd(rng, size, scale)
        K[np.ix_(idx, idx)] = K_b
        blocks.append((idx, K_b))
        start += size
    return blocks, K


def _coo_of(K):
    rows, cols = np.nonzero(K)
    return rows, cols, K[rows, cols]


def _upper_of(K):
    rows, cols = np.nonzero(np.triu(K))
    return rows, cols, K[rows, cols]


def _pair_count(blocks):
    return sum(b.size * (b.size + 1) // 2 for b, _ in blocks)


@pytest.mark.parametrize("n, sizes", [(4, [2, 2]), (5, [3, 2]), (6, [3, 2, 1]), (6, [2, 2])])
@pytest.mark.parametrize("seed", [0, 1])
def test_blocks_and_sparse_match_dense(n, sizes, seed):
    """4-6 sigma_minus ops, random complex PSD blocks: dense vs blocks vs COO, bitwise."""
    rng = np.random.default_rng(100 * n + seed)
    rates = _random_rates(rng, n)
    blocks, K = _block_pattern(rng, n, sizes)
    L_dense = _build(n, rates, K)
    assert L_dense.num_pair_templates == _pair_count(blocks)
    assert L_dense.num_kossakowski_nonzeros == int(np.count_nonzero(K))
    np.testing.assert_array_equal(L_dense.kossakowski, K)

    # blocks on an object that held a dense K with the full pattern
    L = _build(n, rates, _random_psd(rng, n))
    assert L.num_pair_templates == n * (n + 1) // 2
    L.set_kossakowski_blocks(blocks)
    assert L.num_pair_templates == _pair_count(blocks)
    np.testing.assert_array_equal(L.kossakowski, K)
    _compare_all(L, L_dense, n, np.random.default_rng(seed))

    # COO, both triangles, shuffled, with n_ops
    rows, cols, vals = _coo_of(K)
    order = rng.permutation(rows.size)
    L.set_kossakowski(_random_psd(rng, n))
    L.set_kossakowski_sparse(rows[order], cols[order], vals[order], n_ops=n)
    np.testing.assert_array_equal(L.kossakowski, K)
    _compare_all(L, L_dense, n, np.random.default_rng(seed))
    r, c, v = L.kossakowski_coo
    assert r.dtype == np.uint64 and c.dtype == np.uint64 and v.dtype == np.complex128
    np.testing.assert_array_equal(np.asarray(r, dtype=np.int64), rows)
    np.testing.assert_array_equal(np.asarray(c, dtype=np.int64), cols)
    np.testing.assert_array_equal(v, vals)

    # upper triangle only: Hermitian completion
    L2 = _build(n, rates, _random_psd(rng, n))
    L2.set_kossakowski_sparse(*_upper_of(K))
    np.testing.assert_array_equal(L2.kossakowski, K)
    _compare_all(L2, L_dense, n, np.random.default_rng(seed))

    # construction without the dense matrix: blocks and COO forms of the K argument
    words = _local_terms(n, 2)
    h_rng = np.random.default_rng(0)
    h_terms = list(zip(words, [float(c) for c in h_rng.normal(size=len(words))]))
    jumps = list(zip(_jump_ops(n), [float(r) for r in rates]))
    ops = [sigma_minus(j, n) for j in range(n)]
    L3 = Lindbladian(n, h_terms, jumps, kossakowski=(ops, {"blocks": blocks}))
    np.testing.assert_array_equal(L3.kossakowski, K)
    _compare_all(L3, L_dense, n, np.random.default_rng(seed))
    L4 = Lindbladian(
        n, h_terms, jumps, kossakowski=(ops, {"rows": rows, "cols": cols, "vals": vals})
    )
    _compare_all(L4, L_dense, n, np.random.default_rng(seed))

    class _KBlocks:  # duck type of a blocks object (nmr-csim's KBlocks)
        def __init__(self, blocks):
            self.blocks = blocks

        def to_coo(self):
            return _coo_of(K)

    L5 = Lindbladian(n, h_terms, jumps, kossakowski=(ops, _KBlocks(blocks)))
    _compare_all(L5, L_dense, n, np.random.default_rng(seed))
    scipy_sparse = pytest.importorskip("scipy.sparse")
    L6 = Lindbladian(n, h_terms, jumps, kossakowski=(ops, scipy_sparse.csr_matrix(K)))
    _compare_all(L6, L_dense, n, np.random.default_rng(seed))


def _ops_20(n=6):
    """20 operators on 6 qubits: 15 two-local 9-term complex lincombs (all pairs) + 5 sigma_minus."""
    rng = np.random.default_rng(7)
    ops = []
    for i, j in itertools.combinations(range(n), 2):
        lin = []
        for a, b in itertools.product("XYZ", repeat=2):
            s = ["I"] * n
            s[i], s[j] = a, b
            lin.append(("".join(s), complex(rng.normal(), rng.normal())))
        ops.append(lin)
    ops += [sigma_minus(q, n) for q in range(n - 1)]
    assert len(ops) == 20
    return ops


def test_20_ops_four_blocks_of_five():
    """20 operators (two-local 9-term + sigma_minus), 4 blocks of 5: dense vs blocks vs COO."""
    n = 6
    rng = np.random.default_rng(20)
    ops = _ops_20(n)
    m = len(ops)
    words = _local_terms(n, 2)
    h_terms = list(zip(words, [float(c) for c in rng.normal(size=len(words))]))
    jumps = list(zip(_jump_ops(n), [float(r) for r in _random_rates(rng, n)]))
    blocks, K = _block_pattern(rng, m, [5, 5, 5, 5])

    L_dense = Lindbladian(n, h_terms, jumps, kossakowski=(ops, K))
    L_blocks = Lindbladian(n, h_terms, jumps, kossakowski=(ops, {"blocks": blocks}))
    rows, cols, vals = _coo_of(K)
    L_coo = Lindbladian(n, h_terms, jumps, kossakowski=(ops, {"rows": rows, "cols": cols, "vals": vals}))
    assert L_dense.num_pair_templates == L_blocks.num_pair_templates == L_coo.num_pair_templates == 4 * 15
    assert L_dense.num_jump_terms == len(jumps) + 60
    _compare_all(L_blocks, L_dense, n, np.random.default_rng(1))
    _compare_all(L_coo, L_dense, n, np.random.default_rng(2))

    # setters across patterns: the dense object takes blocks of a second pattern and back
    blocks2, K2 = _block_pattern(rng, m, [7, 6, 4, 3])
    L_dense.set_kossakowski_blocks(blocks2)
    L_fresh2 = Lindbladian(n, h_terms, jumps, kossakowski=(ops, K2))
    assert L_dense.num_pair_templates == _pair_count(blocks2)
    _compare_all(L_dense, L_fresh2, n, np.random.default_rng(3))
    L_dense.set_kossakowski_sparse(*_upper_of(K), n_ops=m)
    _compare_all(L_dense, L_blocks, n, np.random.default_rng(4))
    # a rescaled block pattern (what a drive does): same templates, bit-identical to fresh
    scaled = [(idx, 0.3 * K_b) for idx, K_b in blocks]
    L_blocks.set_kossakowski_blocks(scaled)
    L_fresh3 = Lindbladian(n, h_terms, jumps, kossakowski=(ops, 0.3 * K))
    _compare_all(L_blocks, L_fresh3, n, np.random.default_rng(5))


def test_overlapping_blocks_add_and_empty_is_zero():
    n = 4
    rng = np.random.default_rng(3)
    rates = _random_rates(rng, n)
    a, b = _random_psd(rng, 2), _random_psd(rng, 3)
    K = np.zeros((n, n), dtype=np.complex128)
    K[np.ix_([0, 1], [0, 1])] += a
    K[np.ix_([1, 2, 3], [1, 2, 3])] += b
    L = _build(n, rates, _random_psd(rng, n))
    L.set_kossakowski_blocks([([0, 1], a), ([1, 2, 3], b)])
    np.testing.assert_array_equal(L.kossakowski, K)
    _compare_all(L, _build(n, rates, K), n, np.random.default_rng(0))
    L.set_kossakowski_blocks([])
    assert L.num_pair_templates == 0
    assert L.num_kossakowski_nonzeros == 0
    assert L.num_jump_terms == L.num_jump_ops
    _compare_all(L, _build(n, rates, np.zeros((n, n))), n, np.random.default_rng(0))
    # a 1x1 block on one operator, the rest zero rows
    L.set_kossakowski_blocks([(np.array([2]), np.array([[0.7]]))])
    K1 = np.zeros((n, n), dtype=np.complex128)
    K1[2, 2] = 0.7
    _compare_all(L, _build(n, rates, K1), n, np.random.default_rng(0))


def test_rejections_leave_object_unchanged():
    n = 4
    rng = np.random.default_rng(5)
    rates = _random_rates(rng, n)
    K0 = _random_psd(rng, n)
    L = _build(n, rates, K0)
    p = _string_to_codes("XYZI", n)
    before = L.action_arr(p)
    good = _random_psd(rng, 2)
    bad_cases = [
        lambda: L.set_kossakowski_blocks([([0, 4], good)]),  # index out of range
        lambda: L.set_kossakowski_blocks([([1, 1], good)]),  # duplicate index
        lambda: L.set_kossakowski_blocks([([0, 1], _random_psd(rng, 3))]),  # shape
        lambda: L.set_kossakowski_blocks([([0, 1], good + np.array([[0, 1e-3j], [0, 0]]))]),  # not Hermitian
        lambda: L.set_kossakowski_blocks([([0, 1], -good)]),  # not PSD
        lambda: L.set_kossakowski_sparse([0, 1], [1, 0], [1.0, 1.0 + 0.5j]),  # mirror mismatch
        lambda: L.set_kossakowski_sparse([2], [2], [1.0 + 1e-3j]),  # imaginary diagonal
        lambda: L.set_kossakowski_sparse([0, 7], [0, 7], [1.0, 1.0]),  # index out of range
        lambda: L.set_kossakowski_sparse([0, 1], [0], [1.0, 1.0]),  # lengths
        lambda: L.set_kossakowski_sparse([0], [0], [-1.0]),  # not PSD
        lambda: L.set_kossakowski_sparse([0], [0], [1.0], n_ops=5),  # n_ops mismatch
    ]
    for case in bad_cases:
        with pytest.raises(ValueError):
            case()
        np.testing.assert_array_equal(L.kossakowski, K0)
        _assert_same_tuple(L.action_arr(p), before)
    # check_psd=False accepts a Hermitian non-PSD block (the caller vouches), as the dense setter does
    L.set_kossakowski_blocks([([0, 1], -good)], check_psd=False)
    L.set_kossakowski(K0)
    # without a block
    L_plain = Lindbladian(n, [("XXII", 1.0)], [("ZIII", 0.2)])
    with pytest.raises(ValueError):
        L_plain.set_kossakowski_blocks([])
    with pytest.raises(ValueError):
        L_plain.set_kossakowski_sparse([], [], [])
    assert L_plain.kossakowski_coo is None
    # a dense-looking dict is rejected with a clear message
    with pytest.raises(ValueError, match="blocks"):
        Lindbladian(n, [("XXII", 1.0)], kossakowski=([sigma_minus(0, n)], {"K": [[1.0]]}))


def test_coo_psd_check_is_per_component():
    """The COO PSD check eigensolves connected components: a non-PSD block among PSD ones is caught."""
    n = 6
    rng = np.random.default_rng(11)
    rates = _random_rates(rng, n)
    L = _build(n, rates, _random_psd(rng, n))
    blocks, K = _block_pattern(rng, n, [3, 2, 1])
    K_bad = K.copy()
    idx = blocks[1][0]
    K_bad[np.ix_(idx, idx)] = -blocks[1][1]  # one negative block
    with pytest.raises(ValueError, match="positive semidefinite"):
        L.set_kossakowski_sparse(*_coo_of(K_bad))
    L.set_kossakowski_sparse(*_coo_of(K_bad), check_psd=False)
    np.testing.assert_array_equal(L.kossakowski, K_bad)
    L.set_kossakowski_sparse(*_coo_of(K))  # fine
    np.testing.assert_array_equal(L.kossakowski, K)
    # the union-find fallback agrees with scipy on the components
    from ppvm.lindblad import _coo_components

    rows, cols, _ = _coo_of(K)
    labels = _coo_components(rows, cols, n)
    for idx, _ in blocks:
        assert len(set(labels[idx].tolist())) == 1
    assert len(set(labels.tolist())) == len(blocks)


def _random_two_local_ops(rng, n, m, n_terms=3):
    ops = []
    for _ in range(m):
        i, j = rng.choice(n, size=2, replace=False)
        lin = []
        for _ in range(n_terms):
            s = ["I"] * n
            s[i], s[j] = rng.choice(list("XYZ")), rng.choice(list("XYZ"))
            lin.append(("".join(s), complex(rng.normal(), rng.normal())))
        ops.append(lin)
    return ops


def test_timing_600_ops_30_blocks():
    """600 three-term two-local ops on 12 qubits, 30 blocks of 20: block / COO setters vs dense.

    Prints the construction and setter times (``-s``); asserts bit identity of
    the compiled action and that the block / COO forms never go through the
    dense path (templates = 30 x 210 pairs, nonzeros = 30 x 400).
    """
    n, m, n_blocks = 12, 600, 30
    rng = np.random.default_rng(600)
    ops = _random_two_local_ops(rng, n, m)
    words = _local_terms(n, 2)
    h_terms = list(zip(words, [float(c) for c in rng.normal(size=len(words))]))
    jumps = list(zip(_jump_ops(n), [float(r) for r in _random_rates(rng, n)]))
    blocks, K = _block_pattern(rng, m, [m // n_blocks] * n_blocks)
    rows, cols, vals = _coo_of(K)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t0 = time.perf_counter()
    L_dense = Lindbladian(n, h_terms, jumps, kossakowski=(ops, K))
    t_build_dense = time.perf_counter() - t0
    t0 = time.perf_counter()
    L_blocks = Lindbladian(n, h_terms, jumps, kossakowski=(ops, {"blocks": blocks}))
    t_build_blocks = time.perf_counter() - t0
    assert L_dense.num_pair_templates == L_blocks.num_pair_templates == n_blocks * 20 * 21 // 2
    assert L_blocks.num_kossakowski_nonzeros == n_blocks * 400

    scaled = [(idx, 0.5 * K_b) for idx, K_b in blocks]
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        L_dense.set_kossakowski(0.5 * K, check_psd=False)
    t_dense = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        L_blocks.set_kossakowski_blocks(scaled, check_psd=False)
    t_blocks = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        L_blocks.set_kossakowski_blocks(scaled, check_psd=True)
    t_blocks_psd = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        L_blocks.set_kossakowski_sparse(rows, cols, 0.5 * vals, n_ops=m, check_psd=False)
    t_coo = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        L_blocks.set_kossakowski_sparse(rows, cols, 0.5 * vals, n_ops=m, check_psd=True)
    t_coo_psd = (time.perf_counter() - t0) / reps
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"\n{n} qubits, {m} three-term two-local ops, {n_blocks} blocks of 20 "
        f"({L_blocks.num_pair_templates} templates of {m * (m + 1) // 2} pairs): "
        f"construction dense {t_build_dense * 1e3:.0f} ms / blocks {t_build_blocks * 1e3:.0f} ms; "
        f"setter dense {t_dense * 1e3:.1f} ms, blocks {t_blocks * 1e3:.1f} ms "
        f"(+PSD {t_blocks_psd * 1e3:.1f} ms), COO {t_coo * 1e3:.1f} ms (+PSD {t_coo_psd * 1e3:.1f} ms); "
        f"maxrss +{(rss1 - rss0) / 2**20:.0f} MB"
    )
    for s in ("ZXIIYIIZIIII", "IIXYIIIIZIII", "IIIIIIZZIIIX"):
        p = _string_to_codes(s, n)
        _assert_same_tuple(L_blocks.action_arr(p), L_dense.action_arr(p))
    basis = np.stack([_string_to_codes(w, n) for w in words[:40]])
    _assert_same_tuple(L_blocks.generator_arr(basis), L_dense.generator_arr(basis))
