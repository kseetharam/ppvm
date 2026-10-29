# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0
"""Factor-form Kossakowski dissipator (``kossakowski_factor``): the action and the
adaptive step equal the dense ``kossakowski`` path on ``K_b = (G G† − G_neg G_neg†) ∘ W
+ P + diag(D)`` built from the same blocks; the flat kfactor/1 layout and the per-block
layout agree; the setter is bit-identical to a fresh build; bad input is refused.
"""

import numpy as np
import pytest

from ppvm import Lindbladian
from ppvm.lindblad import _basis_to_codes, _codes_to_basis

BIG = 10_000_000


class _Csr:
    """A minimal CSR holder (what scipy.sparse.csr_matrix exposes)."""

    def __init__(self, dense):
        dense = np.asarray(dense)
        indptr, indices, data = [0], [], []
        for row in dense:
            nz = np.flatnonzero(row)
            indices.extend(int(j) for j in nz)
            data.extend(row[nz])
            indptr.append(len(indices))
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data)
        self.shape = dense.shape


def _pair_ops(n, pairs, rng):
    """Two-site complex lincombs (4 Pauli terms each): the shape of dipolar tensor components."""
    ops = []
    for i, j in pairs:
        terms = []
        for a, b in (("X", "X"), ("X", "Y"), ("Y", "X"), ("Y", "Y")):
            s = ["I"] * n
            s[i], s[j] = a, b
            terms.append(("".join(s), complex(rng.normal(), rng.normal()) * 0.25))
        ops.append(terms)
    return ops


def _blocks(sizes, ranks, rng, *, masked=False, general=False):
    G = [rng.normal(size=(s, r)) + 1j * rng.normal(size=(s, r)) for s, r in zip(sizes, ranks)]
    D = [np.abs(rng.normal(size=s)) * 0.2 for s in sizes]
    out = {"G": G, "D": D}
    if masked:
        masks = []
        for s in sizes:
            w = (rng.random((s, s)) > 0.4).astype(float) * 0.6
            w = np.maximum(w, w.T)
            np.fill_diagonal(w, 1.0)
            masks.append(_Csr(w))
        out["mask"] = masks
    if general:
        out["G_neg"] = [0.3 * (rng.normal(size=(s, 1)) + 1j * rng.normal(size=(s, 1))) for s in sizes]
        po = []
        for s in sizes:
            p = np.zeros((s, s), dtype=complex)
            for i in range(s - 1):
                v = complex(rng.normal(), rng.normal()) * 0.2
                p[i, i + 1], p[i + 1, i] = v, np.conj(v)
            po.append(_Csr(p))
        out["private_offdiag"] = po
    return out


def _dense_k(blocks):
    parts = []
    for b, (g, d) in enumerate(zip(blocks["G"], blocks["D"])):
        k = g @ g.conj().T
        if "G_neg" in blocks:
            h = blocks["G_neg"][b]
            k = k - h @ h.conj().T
        if "mask" in blocks:
            m = blocks["mask"][b]
            w = np.zeros((len(d), len(d)))
            for i in range(len(d)):
                for jj in range(m.indptr[i], m.indptr[i + 1]):
                    w[i, m.indices[jj]] = m.data[jj]
            k = k * w
        if "private_offdiag" in blocks:
            m = blocks["private_offdiag"][b]
            p = np.zeros((len(d), len(d)), dtype=complex)
            for i in range(len(d)):
                for jj in range(m.indptr[i], m.indptr[i + 1]):
                    p[i, m.indices[jj]] = m.data[jj]
            k = k + p
        parts.append(k + np.diag(d))
    total = sum(len(d) for d in blocks["D"])
    out = np.zeros((total, total), dtype=complex)
    off = 0
    for k in parts:
        s = k.shape[0]
        out[off : off + s, off : off + s] = k
        off += s
    return out


def _flat(blocks):
    """The nmr-csim-kfactor/1 flat layout of per-block arrays (G / D [/ mask / G_neg / private])."""
    sizes = [len(d) for d in blocks["D"]]
    ranks = [g.shape[1] for g in blocks["G"]]
    out = {
        "G_values": np.concatenate([g.reshape(-1) for g in blocks["G"]]),
        "G_offsets": np.concatenate([[0], np.cumsum([s * r for s, r in zip(sizes, ranks)])]),
        "G_ranks": np.asarray(ranks),
        "D_values": np.concatenate(blocks["D"]),
        "D_offsets": np.concatenate([[0], np.cumsum(sizes)]),
    }

    def csr(key, prefix):
        ms = blocks[key]
        out[prefix + "indptr"] = np.concatenate([m.indptr for m in ms])
        out[prefix + "indices"] = np.concatenate([m.indices for m in ms])
        out[prefix + "data"] = np.concatenate([m.data for m in ms])
        out[prefix + "offsets"] = np.concatenate([[0], np.cumsum([s + 1 for s in sizes])])

    if "mask" in blocks:
        csr("mask", "mask_")
    if "G_neg" in blocks:
        out["G_neg_values"] = np.concatenate([g.reshape(-1) for g in blocks["G_neg"]])
        out["G_neg_ranks"] = np.asarray([g.shape[1] for g in blocks["G_neg"]])
    if "private_offdiag" in blocks:
        csr("private_offdiag", "private_offdiag_")
    return out


def _all_words(n, max_weight):
    out = []
    for idx in range(4**n):
        codes, x = [], idx
        for _ in range(n):
            codes.append(x % 4)
            x //= 4
        w = sum(1 for c in codes if c)
        if 0 < w <= max_weight:
            out.append("".join("IXZY"[c] for c in codes))
    return out


def _setup(n, seed, masked, general, with_dense_block=False):
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sizes = [4, 3, 2, 1]
    ranks = [2, 3, 1, 0]
    assert sum(sizes) <= len(pairs)
    ops = _pair_ops(n, [pairs[k % len(pairs)] for k in range(sum(sizes))], rng)
    blocks = _blocks(sizes, ranks, rng, masked=masked, general=general)
    k = _dense_k(blocks)
    h = [("X" + "I" * (n - 1), 0.4), ("I" * (n - 2) + "ZZ", -0.7)]
    jumps = [("Z" + "I" * (n - 1), 0.1)]
    if with_dense_block:
        extra = _pair_ops(n, pairs[:2], rng)
        kx = np.array([[0.8, 0.2 + 0.1j], [0.2 - 0.1j, 0.5]])
        k_all = np.zeros((len(ops) + 2, len(ops) + 2), dtype=complex)
        k_all[: len(ops), : len(ops)] = k
        k_all[len(ops) :, len(ops) :] = kx
        dense = _dense_lindbladian(n, h, jumps, ops + extra, k_all)
        factor = Lindbladian(n, h, jumps, kossakowski=(extra, kx), kossakowski_factor=(ops, blocks))
    else:
        dense = _dense_lindbladian(n, h, jumps, ops, k)
        factor = Lindbladian(n, h, jumps, kossakowski_factor=(ops, blocks))
    return dense, factor, ops, blocks, k


def _dense_lindbladian(n, h, jumps, ops, k):
    """The dense path on K: built on a PSD surrogate and set with ``check_psd=False`` -- a masked / G_neg block is indefinite by
    construction, which the constructor's PSD check (a Python-side eigensolve) refuses; the Rust action itself takes any Hermitian K."""
    lind = Lindbladian(n, h, jumps, kossakowski=(ops, np.eye(len(ops))))
    lind.set_kossakowski(k, check_psd=False)
    return lind


@pytest.mark.parametrize("masked,general,with_dense", [(False, False, False), (True, False, False), (False, True, False), (True, True, True)])
def test_factor_action_equals_dense(masked, general, with_dense):
    n = 5
    dense, factor, ops, blocks, k = _setup(n, 7 + 3 * masked + 5 * general, masked, general, with_dense)
    assert factor.num_kossakowski_factor_ops == len(ops)
    assert factor.num_kossakowski_factor_blocks == len(blocks["D"])
    off = 0
    for b, d in enumerate(blocks["D"]):
        s = len(d)
        kb = factor.kossakowski_factor_dense_block(b)
        assert np.allclose(kb, k[off : off + s, off : off + s], atol=1e-13 * max(float(np.abs(k).max()), 1.0), rtol=0)
        off += s
    scale = max(float(np.abs(k).max()), 1.0)
    for p in _all_words(n, 3):
        a, b = dense.action(p), factor.action(p)
        keys = set(a) | set(b)
        for w in keys:
            assert abs(a.get(w, 0.0) - b.get(w, 0.0)) <= 1e-12 * scale, (p, w, a.get(w), b.get(w))


def test_factor_pc_step_equals_dense():
    n = 5
    dense, factor, ops, blocks, k = _setup(n, 11, True, True)
    obs = {"X" + "I" * (n - 1): 1.0, "I" * (n - 1) + "Y": 0.5}
    strings = list(obs)
    basis = _basis_to_codes(strings, n)
    coeff = np.array([obs[s] for s in strings])
    bd, cd = basis.copy(), coeff.copy()
    bf, cf = basis.copy(), coeff.copy()
    for _ in range(4):
        bd, cd = dense.pc_step_arr(bd, cd, 0.05, max_basis=BIG, drop_tol=0.0)
        bf, cf = factor.pc_step_arr(bf, cf, 0.05, max_basis=BIG, drop_tol=0.0)
    out_d = dict(zip(_codes_to_basis(bd), cd))
    out_f = dict(zip(_codes_to_basis(bf), cf))
    for w in set(out_d) | set(out_f):
        assert abs(out_d.get(w, 0.0) - out_f.get(w, 0.0)) <= 1e-10, (w, out_d.get(w), out_f.get(w))


def test_flat_layout_matches_per_block_layout():
    n = 5
    rng = np.random.default_rng(3)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sizes, ranks = [3, 2], [2, 1]
    ops = _pair_ops(n, pairs[: sum(sizes)], rng)
    blocks = _blocks(sizes, ranks, rng, masked=True, general=True)
    h = [("XX" + "I" * (n - 2), 0.3)]
    a = Lindbladian(n, h, kossakowski_factor=(ops, blocks))
    b = Lindbladian(n, h, kossakowski_factor=(ops, _flat(blocks)))
    for p in _all_words(n, 2):
        x, y = a.action(p), b.action(p)
        assert x.keys() == y.keys()
        for w in x:
            assert x[w] == y[w]


def test_setter_is_bit_identical_to_fresh_build():
    n = 4
    rng = np.random.default_rng(5)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sizes = [3, 2]
    ops = _pair_ops(n, pairs[:5], rng)
    b1 = _blocks(sizes, [2, 1], rng)
    b2 = _blocks(sizes, [1, 2], rng, masked=True)
    h = [("XXII", 0.3)]
    a = Lindbladian(n, h, kossakowski_factor=(ops, b1))
    a.set_kossakowski_factor(b2)
    fresh = Lindbladian(n, h, kossakowski_factor=(ops, b2))
    for p in _all_words(n, 4):
        x, y = a.action(p), fresh.action(p)
        assert x.keys() == y.keys()
        for w in x:
            assert x[w] == y[w]
    assert a.kossakowski_factor_memory_bytes[0] > 0
    assert a.kossakowski_factor_floor > 0.0


def test_bad_input_is_refused():
    n = 4
    rng = np.random.default_rng(9)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    ops = _pair_ops(n, pairs[:3], rng)
    h = [("XXII", 0.3)]
    good = _blocks([3], [2], rng)
    bad_d = {"G": good["G"], "D": [np.array([0.1, -1.0, 0.2])]}
    with pytest.raises(ValueError):
        Lindbladian(n, h, kossakowski_factor=(ops, bad_d))
    short = _blocks([2], [2], rng)
    with pytest.raises(ValueError):
        Lindbladian(n, h, kossakowski_factor=(ops, short))
    plain = Lindbladian(n, h, kossakowski=(ops, _dense_k(good)))
    with pytest.raises(ValueError):
        plain.set_kossakowski_factor(good)
    assert plain.num_kossakowski_factor_ops == 0
    assert plain.kossakowski_factor_memory_bytes == (0, 0)
    # a mask row without its diagonal would drop D_n
    w = np.ones((3, 3))
    w[0, 0] = 0.0
    no_diag = {"G": good["G"], "D": good["D"], "mask": [_Csr(w)]}
    with pytest.raises(ValueError):
        Lindbladian(n, h, kossakowski_factor=(ops, no_diag))
