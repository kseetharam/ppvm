# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared Kossakowski pair templates (``koss-templates-blocks``).

Contract: pairs with the same operator-pattern relationship -- the Pauli
structure of ``(A_n, A_m)`` up to a qubit relabelling -- share one compiled
template, so ``num_pair_templates`` is bounded by the number of distinct
patterns while ``num_compiled_pairs`` counts the pairs; every output of the
``Lindbladian`` stays bit-identical (``np.array_equal``) across the dense,
block and COO inputs and across a rescaled pattern.
"""

import numpy as np
import pytest

from ppvm import Lindbladian, sigma_minus
from ppvm.lindblad import _string_to_codes

from .test_dissipator_setters import (
    _assert_same_tuple,
    _compare_all,
    _jump_ops,
    _local_terms,
    _random_psd,
    _random_rates,
)


def _site_ops(n_sites):
    """Three operator classes on ``n_sites`` disjoint spin pairs of a 2 n_sites register: ``¼ZZ``,
    ``¼XX + ¼YY`` and ``σ⁻`` on the first spin -- the same pattern on every site pair."""
    n = 2 * n_sites

    def word(pairs):
        s = ["I"] * n
        for q, c in pairs:
            s[q] = c
        return "".join(s)

    ops = []
    for s in range(n_sites):
        ops.append([(word([(2 * s, "Z"), (2 * s + 1, "Z")]), 0.25 + 0j)])
    for s in range(n_sites):
        ops.append(
            [
                (word([(2 * s, "X"), (2 * s + 1, "X")]), 0.25 + 0j),
                (word([(2 * s, "Y"), (2 * s + 1, "Y")]), 0.25 + 0j),
            ]
        )
    for s in range(n_sites):
        ops.append(sigma_minus(2 * s, n))
    return n, ops


def _class_blocks(rng, n_sites):
    """One PSD block per operator class; returns (blocks, dense K)."""
    m = 3 * n_sites
    K = np.zeros((m, m), dtype=np.complex128)
    blocks = []
    for b in range(3):
        idx = np.arange(b * n_sites, (b + 1) * n_sites)
        K_b = _random_psd(rng, n_sites, 0.05)
        K[np.ix_(idx, idx)] = K_b
        blocks.append((idx, K_b))
    return blocks, K


def _build(n, ops, K, seed=0):
    rng = np.random.default_rng(seed)
    words = _local_terms(n, 2)
    h_terms = list(zip(words, [float(c) for c in rng.normal(size=len(words))]))
    jumps = list(zip(_jump_ops(n), [float(r) for r in _random_rates(rng, n)]))
    return Lindbladian(n, h_terms, jumps, kossakowski=(ops, K))


@pytest.mark.parametrize("n_sites", [3, 4])
def test_templates_shared_across_blocks(n_sites):
    """Three classes x n_sites: the pair count grows with the sites, the template count stays at 6
    (diagonal and off-diagonal pattern of each class); dense / blocks / COO bit-identical."""
    n, ops = _site_ops(n_sites)
    rng = np.random.default_rng(10 + n_sites)
    blocks, K = _class_blocks(rng, n_sites)
    L_dense = _build(n, ops, K)
    L_blocks = _build(n, ops, {"blocks": blocks})
    rows, cols = np.nonzero(K)
    L_coo = _build(n, ops, {"rows": rows, "cols": cols, "vals": K[rows, cols]})
    n_pairs = 3 * n_sites * (n_sites + 1) // 2
    for L in (L_dense, L_blocks, L_coo):
        assert L.num_compiled_pairs == n_pairs
        assert L.num_pair_templates == 6
        assert L.num_jump_terms == L.num_jump_ops + n_pairs
    _compare_all(L_blocks, L_dense, n, np.random.default_rng(1))
    _compare_all(L_coo, L_dense, n, np.random.default_rng(2))
    mem = L_blocks.kossakowski_memory_bytes
    assert set(mem) == {"templates", "pairs", "entries"}
    assert mem["templates"] > 0 and mem["pairs"] > 0
    assert mem["entries"] >= 24 * L_blocks.num_kossakowski_nonzeros
    # a rescaled pattern (a drive): same records and templates, bit-identical to a fresh build
    L_blocks.set_kossakowski_blocks([(idx, 0.3 * K_b) for idx, K_b in blocks])
    assert L_blocks.num_pair_templates == 6
    _compare_all(L_blocks, _build(n, ops, 0.3 * K), n, np.random.default_rng(3))


def test_template_memory_does_not_grow_with_sites():
    """Template bytes are the same for 3 and 6 site pairs; the per-pair bytes grow with the pairs."""
    mems = {}
    for n_sites in (3, 6):
        n, ops = _site_ops(n_sites)
        blocks, _ = _class_blocks(np.random.default_rng(n_sites), n_sites)
        L = _build(n, ops, {"blocks": blocks})
        assert L.num_pair_templates == 6
        mems[n_sites] = (L.num_compiled_pairs, L.kossakowski_memory_bytes)
    assert mems[3][1]["templates"] == mems[6][1]["templates"]
    assert mems[6][1]["pairs"] > mems[3][1]["pairs"]
    assert mems[6][0] == 3 * 21 and mems[3][0] == 3 * 6


def test_cross_class_pairs_add_templates_and_are_pruned():
    """One block over every operator brings cross-class patterns; back to the class blocks (or
    to no block) the unused templates are freed; every step bit-identical to a fresh build."""
    n_sites = 3
    n, ops = _site_ops(n_sites)
    rng = np.random.default_rng(99)
    blocks, K = _class_blocks(rng, n_sites)
    L = _build(n, ops, {"blocks": blocks})
    assert L.num_pair_templates == 6
    m = len(ops)
    K_all = _random_psd(rng, m, 0.05)
    L.set_kossakowski_blocks([(np.arange(m), K_all)])
    assert L.num_compiled_pairs == m * (m + 1) // 2
    assert L.num_pair_templates > 6
    _compare_all(L, _build(n, ops, K_all), n, np.random.default_rng(4))
    L.set_kossakowski_blocks(blocks)
    assert L.num_pair_templates == 6
    _compare_all(L, _build(n, ops, K), n, np.random.default_rng(5))
    L.set_kossakowski_blocks([])
    assert L.num_pair_templates == 0 and L.num_compiled_pairs == 0
    for s in ("XIIIII", "ZZIIII", "IIXYII"):
        p = _string_to_codes(s, n)
        _assert_same_tuple(L.action_arr(p), _build(n, ops, np.zeros((m, m))).action_arr(p))
