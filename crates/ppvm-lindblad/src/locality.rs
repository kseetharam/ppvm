// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! Locality-aware predictor-corrector step (`PcStepConfig::locality`).
//!
//! The legacy step runs three full dissipator action passes over the admitted
//! working set — one per `leakage_with_prune` / `expm_step` phase — and each
//! pass visits, for every string `p`, every compiled pair whose support meets
//! `supp(p)` and materialises every output word (tens of thousands per string
//! on a 72-spin Redfield dissipator), although
//!
//! 1. the second leakage pass runs with `room = admit − |basis| = 0` once the
//!    first admission filled the working set, and its result is discarded;
//! 2. the two generator builds (`expm_step`) keep only the output words that
//!    lie *in the basis* — a few per string — plus the column 1-norm `raw`
//!    and the diagonal, which do not depend on the basis;
//! 3. the per-thread output vectors of a leakage pass are merged into one map
//!    sequentially.
//!
//! This module keeps the action itself (`compute_action_terms`) untouched
//! and reorganises the step around those three facts:
//!
//! * a **transfer index** of the compiled dissipator, built once per pair
//!   set ([`TransferIndex`]): every table entry of every pair maps its input
//!   word `p` to `p ⊕ Δ` for a *fixed* difference word `Δ` (the product of
//!   the template's left and right words, or the anticommutator word), so
//!   the index lists, per `Δ`, the entries `(pair, kind, position)` that can
//!   produce it, in the order the full action visits them;
//! * a **basis-local generator build** ([`mf_cache_local`]): for a column
//!   `p` the in-basis outputs are the basis words `q` with `|supp(p ⊕ q)| ≤
//!   max |Δ|`; for each such `q` only the index entries of `Δ = p ⊕ q`
//!   belonging to candidate pairs (support meets `supp(p)`) are evaluated,
//!   in the full action's order, so the matrix entry is the same
//!   floating-point sum as the full action's deduplicated coefficient;
//! * a **raw / diag cache** keyed by word: the column 1-norm `Σ|L*(p)_s|`
//!   and the diagonal need the full output and are taken from the last
//!   full evaluation of `p` (the first leakage pass of the step evaluates
//!   the retained strings; admitted strings are evaluated once and cached
//!   across steps). Both are what `build_mf_cols` computes from the same
//!   deduplicated output list;
//! * a **sharded parallel merge** of the leakage outputs ([`leakage_locality`]):
//!   each thread buckets its output words by a hash shard, each shard is
//!   merged by one thread in basis order, so every merged value is the same
//!   sum in the same order as the sequential merge; the room cap is applied
//!   with the same `k`-th-largest cutoff; the result is returned sorted by
//!   word (the legacy order is the hash map's);
//! * the second leakage pass returns early when `room == 0`, and the
//!   corrector reuses the predictor's generator cache when the second pass
//!   admitted nothing.
//!
//! What is bit-identical: every action coefficient (the action is not
//! touched), every generator matrix entry, every leakage value, the
//! admitted set up to ties at the room cutoff. What is not: the *order* of
//! the admitted strings in the basis (sorted here, hash-map order before),
//! which changes the association order of the CSC matvec reduction — a
//! round-off-level difference in the propagated coefficients, of the same
//! nature as changing the thread count on the legacy path.

use crate::mf_expm::MfCache;
use crate::{
    CHUNK_BITS, Chunk, JumpKind, LindbladSpec, W_CHUNKS, Word, build_basis_index, comm_product,
    comm_product_local, pauli_mul, pauli_mul_local, phase_factor, word_hash,
};
use fxhash::{FxBuildHasher, FxHashMap};
use num::Complex;
use ppvm_traits::PauliWordTrait;
use rayon::prelude::*;

/// Sandwich entry of a Kossakowski pair (`pos` = flat index over the grouped table).
const KIND_SAND: u64 = 0;
/// Anticommutator entry of a Kossakowski pair (`pos` = index into `adag_b`).
const KIND_ADAG: u64 = 1;
/// Sandwich entry of a `General` jump (`pos` = `a · n_terms + b`).
const KIND_GEN_SAND: u64 = 2;
/// `L†L` entry of a `General` jump (`pos` = index into `dagger_dagger`).
const KIND_GEN_DD: u64 = 3;

#[inline(always)]
fn pack(jk: u32, kind: u64, pos: u32) -> u64 {
    debug_assert!(pos < (1 << 30));
    ((jk as u64) << 32) | (kind << 30) | (pos as u64)
}

#[inline(always)]
fn unpack(e: u64) -> (u32, u64, u32) {
    ((e >> 32) as u32, (e >> 30) & 3, (e & 0x3fff_ffff) as u32)
}

#[inline(always)]
fn support_mask(w: &Word) -> [Chunk; W_CHUNKS] {
    let mut m = [0 as Chunk; W_CHUNKS];
    for i in 0..W_CHUNKS {
        m[i] = w.xbits.data[i] | w.zbits.data[i];
    }
    m
}

#[inline(always)]
fn intersects(a: &[Chunk; W_CHUNKS], b: &[Chunk; W_CHUNKS]) -> bool {
    (0..W_CHUNKS).any(|i| a[i] & b[i] != 0)
}

/// A local-frame word placed on the register qubits `qs` (zero elsewhere).
fn embed_zero(w: &Word, qs: &[u8], n: usize) -> Word {
    let mut out = Word::new(n);
    for (li, &q) in qs.iter().enumerate() {
        let q = q as usize;
        let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
        let (lc, lb) = (li / CHUNK_BITS, li % CHUNK_BITS);
        out.xbits.data[ci] |= ((w.xbits.data[lc] >> lb) & 1) << bi;
        out.zbits.data[ci] |= ((w.zbits.data[lc] >> lb) & 1) << bi;
    }
    out.rehash();
    out
}

/// `a ⊕ b` as a Pauli word (the product up to phase).
fn xor_word(a: &Word, b: &Word) -> Word {
    let mut out = Word::new(a.n_qubits());
    for i in 0..W_CHUNKS {
        out.xbits.data[i] = a.xbits.data[i] ^ b.xbits.data[i];
        out.zbits.data[i] = a.zbits.data[i] ^ b.zbits.data[i];
    }
    out.rehash();
    out
}

fn weight(w: &Word) -> u32 {
    (0..W_CHUNKS)
        .map(|i| (w.xbits.data[i] | w.zbits.data[i]).count_ones())
        .sum()
}

/// The difference-word index of a compiled dissipator (see the module docs).
pub struct TransferIndex {
    /// Sorted distinct hashes of the difference words `Δ`.
    keys: Vec<u64>,
    /// `entries[offsets[i]..offsets[i + 1]]` belong to `keys[i]`, sorted by
    /// `(j_kind index, kind, position)` — the full action's visiting order.
    offsets: Vec<u32>,
    /// Packed `(j_kind, kind, pos)` ([`pack`]).
    entries: Vec<u64>,
    /// Register-frame support mask of every `j_kinds` entry (the candidate test).
    jk_mask: Vec<[Chunk; W_CHUNKS]>,
    /// Per template: flat sandwich position → `(group, item)`.
    sand_flat: Vec<Vec<(u32, u32)>>,
    /// Hamiltonian terms by the hash of their word (ascending indices).
    h_by_word: FxHashMap<u64, Vec<u32>>,
    /// Largest support of any difference word (the neighbour filter).
    max_delta: u32,
    /// Heap bytes of the index.
    pub bytes: usize,
}

impl TransferIndex {
    pub(crate) fn build(spec: &LindbladSpec) -> Self {
        let n = spec.n_qubits;
        let sand_flat: Vec<Vec<(u32, u32)>> = spec
            .k_templates
            .iter()
            .map(|t| {
                let mut v = Vec::new();
                for (ga, (_, rights)) in t.sand.iter().enumerate() {
                    for gb in 0..rights.len() {
                        v.push((ga as u32, gb as u32));
                    }
                }
                v
            })
            .collect();
        // One pass per j_kind: its support mask and its (Δ hash, packed entry) rows.
        let per_kind: Vec<([Chunk; W_CHUNKS], Vec<(u64, u64)>, u32)> = (0..spec.j_kinds.len())
            .into_par_iter()
            .map(|jk| {
                let mut rows: Vec<(u64, u64)> = Vec::new();
                let mut maxw = 0u32;
                let mut push = |d: &Word, kind: u64, pos: usize| {
                    maxw = maxw.max(weight(d));
                    rows.push((word_hash(d), pack(jk as u32, kind, pos as u32)));
                };
                let mask = match &spec.j_kinds[jk] {
                    JumpKind::HermitianPauli { word, .. } => support_mask(word),
                    JumpKind::General {
                        terms,
                        dagger_dagger,
                        ..
                    } => {
                        let mut mask = [0 as Chunk; W_CHUNKS];
                        for t in terms {
                            let m = support_mask(&t.word);
                            for i in 0..W_CHUNKS {
                                mask[i] |= m[i];
                            }
                        }
                        let nt = terms.len();
                        for (ai, a) in terms.iter().enumerate() {
                            for (bi, b) in terms.iter().enumerate() {
                                push(&xor_word(&a.word, &b.word), KIND_GEN_SAND, ai * nt + bi);
                            }
                        }
                        for (ci, c) in dagger_dagger.iter().enumerate() {
                            push(&c.word, KIND_GEN_DD, ci);
                        }
                        mask
                    }
                    JumpKind::KossakowskiPair {
                        template, q_off, ..
                    } => {
                        let t = &spec.k_templates[*template as usize];
                        let nl = t.n_local as usize;
                        let start = *q_off as usize;
                        let qs = &spec.k_pair_qubits[start..start + nl];
                        let mut mask = [0 as Chunk; W_CHUNKS];
                        for &q in qs {
                            let q = q as usize;
                            mask[q / CHUNK_BITS] |= (1 as Chunk) << (q % CHUNK_BITS);
                        }
                        let mut pos = 0usize;
                        for (wa, rights) in &t.sand {
                            for (wb, _) in rights {
                                let d = xor_word(wa, wb);
                                push(&embed_zero(&d, qs, n), KIND_SAND, pos);
                                pos += 1;
                            }
                        }
                        for (ti, term) in t.adag_b.iter().enumerate() {
                            push(&embed_zero(&term.word, qs, n), KIND_ADAG, ti);
                        }
                        mask
                    }
                };
                (mask, rows, maxw)
            })
            .collect();
        let total: usize = per_kind.iter().map(|(_, r, _)| r.len()).sum();
        assert!(total < u32::MAX as usize, "transfer index: more than 2^32 entries");
        let mut jk_mask = Vec::with_capacity(per_kind.len());
        let mut rows: Vec<(u64, u64)> = Vec::with_capacity(total);
        let mut max_delta = 0u32;
        for (mask, r, maxw) in per_kind {
            jk_mask.push(mask);
            rows.extend(r);
            max_delta = max_delta.max(maxw);
        }
        rows.par_sort_unstable();
        let mut keys: Vec<u64> = Vec::new();
        let mut offsets: Vec<u32> = Vec::new();
        let mut entries: Vec<u64> = Vec::with_capacity(rows.len());
        for (h, e) in rows {
            if keys.last() != Some(&h) {
                keys.push(h);
                offsets.push(entries.len() as u32);
            }
            entries.push(e);
        }
        offsets.push(entries.len() as u32);
        let mut h_by_word: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
        for (i, h) in spec.h_terms.iter().enumerate() {
            max_delta = max_delta.max(weight(&h.word));
            h_by_word.entry(word_hash(&h.word)).or_default().push(i as u32);
        }
        let bytes = keys.capacity() * 8
            + offsets.capacity() * 4
            + entries.capacity() * 8
            + jk_mask.capacity() * std::mem::size_of::<[Chunk; W_CHUNKS]>()
            + sand_flat.iter().map(|v| v.capacity() * 8).sum::<usize>()
            + h_by_word.len() * 32;
        Self {
            keys,
            offsets,
            entries,
            jk_mask,
            sand_flat,
            h_by_word,
            max_delta,
            bytes,
        }
    }

    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    pub fn num_keys(&self) -> usize {
        self.keys.len()
    }

    pub fn max_delta(&self) -> u32 {
        self.max_delta
    }

    #[inline]
    fn entries_of(&self, h: u64) -> &[u64] {
        match self.keys.binary_search(&h) {
            Ok(i) => &self.entries[self.offsets[i] as usize..self.offsets[i + 1] as usize],
            Err(_) => &[],
        }
    }
}

/// The contribution of one indexed table entry to `L*(p)`: the output word
/// and the coefficient exactly as `compute_action_terms` forms it (same
/// operands, same order, same drop rules), or `None` where the full action
/// adds nothing (commuting words, dropped off-diagonal parts, an entry kind
/// the pair's path does not evaluate for this `p`).
#[inline]
fn eval_entry(
    spec: &LindbladSpec,
    idx: &TransferIndex,
    jk: u32,
    kind: u64,
    pos: u32,
    p: &Word,
) -> Option<(Word, Complex<f64>)> {
    let pos = pos as usize;
    match &spec.j_kinds[jk as usize] {
        JumpKind::HermitianPauli { .. } => None, // diagonal only (taken from the cache)
        JumpKind::General {
            terms,
            dagger_dagger,
            rate,
        } => {
            let rate_c = Complex::new(*rate, 0.0);
            if kind == KIND_GEN_SAND {
                let nt = terms.len();
                let a = &terms[pos / nt];
                let b = &terms[pos % nt];
                let (r_ap, phi1) = pauli_mul(&a.word, p);
                let (s, phi2) = pauli_mul(&r_ap, &b.word);
                let coeff = a.coeff.conj() * b.coeff * phase_factor(phi1 + phi2) * rate_c;
                Some((s, coeff))
            } else {
                let c_term = &dagger_dagger[pos];
                let (r, phase) = pauli_mul(&c_term.word, p);
                if phase & 1 == 0 {
                    let sign = if phase == 0 { 1.0 } else { -1.0 };
                    let coeff = -c_term.coeff * rate_c * Complex::new(sign, 0.0);
                    Some((r, coeff))
                } else {
                    None
                }
            }
        }
        JumpKind::KossakowskiPair {
            template,
            q_off,
            off_diag,
            k_nm,
        } => {
            let t = &spec.k_templates[*template as usize];
            let nl = t.n_local as usize;
            let start = *q_off as usize;
            let qs = &spec.k_pair_qubits[start..start + nl];
            let mut p_loc = Word::new(nl);
            let mut sup = [0 as Chunk; W_CHUNKS];
            for (li, &q) in qs.iter().enumerate() {
                let q = q as usize;
                let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
                let (lc, lb) = (li / CHUNK_BITS, li % CHUNK_BITS);
                sup[ci] |= (1 as Chunk) << bi;
                p_loc.xbits.data[lc] |= ((p.xbits.data[ci] >> bi) & 1) << lb;
                p_loc.zbits.data[lc] |= ((p.zbits.data[ci] >> bi) & 1) << lb;
            }
            let embed = |w: &Word| -> Word {
                let mut out = Word::new(p.n_qubits());
                for i in 0..W_CHUNKS {
                    out.xbits.data[i] = p.xbits.data[i] & !sup[i];
                    out.zbits.data[i] = p.zbits.data[i] & !sup[i];
                }
                for (li, &q) in qs.iter().enumerate() {
                    let q = q as usize;
                    let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
                    let (lc, lb) = (li / CHUNK_BITS, li % CHUNK_BITS);
                    out.xbits.data[ci] |= ((w.xbits.data[lc] >> lb) & 1) << bi;
                    out.zbits.data[ci] |= ((w.zbits.data[lc] >> lb) & 1) << bi;
                }
                out.rehash();
                out
            };
            let hits = |mask: &[Chunk; W_CHUNKS]| {
                (0..W_CHUNKS).any(|i| mask[i] & (p_loc.xbits.data[i] | p_loc.zbits.data[i]) != 0)
            };
            let (hit_l, hit_r) = (hits(&t.left_mask), hits(&t.right_mask));
            if !(hit_l && hit_r) {
                // one-sided path: only the anticommutator table is evaluated
                if kind != KIND_ADAG {
                    return None;
                }
                let half_i = if *off_diag {
                    if hit_r {
                        Complex::new(0.0, -0.5)
                    } else {
                        Complex::new(0.0, 0.5)
                    }
                } else if hit_r {
                    Complex::new(0.0, 0.5)
                } else {
                    Complex::new(0.0, -0.5)
                };
                let term = &t.adag_b[pos];
                let c = *k_nm * term.coeff;
                let coeff = if *off_diag {
                    if c.im.abs() > 1e-14 {
                        Complex::new(0.0, -2.0 * c.im)
                    } else {
                        return None;
                    }
                } else {
                    c
                };
                let (out, eps) = comm_product_local(&term.word, &p_loc);
                if eps != 0.0 {
                    Some((embed(&out), coeff * half_i * eps))
                } else {
                    None
                }
            } else if kind == KIND_SAND {
                let (ga, gb) = idx.sand_flat[*template as usize][pos];
                let (wa, rights) = &t.sand[ga as usize];
                let (wb, c0) = &rights[gb as usize];
                let (r_ap, phi1) = pauli_mul_local(wa, &p_loc);
                let (s, phi2) = pauli_mul_local(&r_ap, wb);
                let v = (c0 * k_nm) * phase_factor(phi1 + phi2);
                let contrib = if *off_diag {
                    Complex::new(2.0 * v.re, 0.0)
                } else {
                    v
                };
                Some((embed(&s), contrib))
            } else {
                let term = &t.adag_b[pos];
                let c = *k_nm * term.coeff;
                let coeff = if *off_diag {
                    if c.re.abs() > 1e-14 {
                        Complex::new(2.0 * c.re, 0.0)
                    } else {
                        return None;
                    }
                } else {
                    c
                };
                let (r, phase) = pauli_mul_local(&term.word, &p_loc);
                if phase & 1 == 0 {
                    let sign = if phase == 0 { 1.0 } else { -1.0 };
                    // the full action does `local -= coeff · sign`; `a − b == a + (−b)` in IEEE
                    Some((embed(&r), -(coeff * Complex::new(sign, 0.0))))
                } else {
                    None
                }
            }
        }
    }
}

/// One in-basis generator column from the transfer index: `(row, coeff)` for
/// every basis word `q ≠ p` the action maps `p` to, plus the cached diagonal.
fn local_column(
    spec: &LindbladSpec,
    idx: &TransferIndex,
    basis: &[Word],
    c: usize,
    raw: f64,
    diag: f64,
) -> (Vec<(u32, f64)>, (f64, f64)) {
    let p = &basis[c];
    let psup = support_mask(p);
    let zero = Complex::new(0.0, 0.0);
    let mut out: Vec<(u32, f64)> = Vec::new();
    for (c2, q) in basis.iter().enumerate() {
        if c2 == c {
            continue;
        }
        let mut pop = 0u32;
        for i in 0..W_CHUNKS {
            pop += ((p.xbits.data[i] ^ q.xbits.data[i]) | (p.zbits.data[i] ^ q.zbits.data[i]))
                .count_ones();
        }
        if pop == 0 || pop > idx.max_delta {
            continue;
        }
        let delta = xor_word(p, q);
        let h = word_hash(&delta);
        let mut acc = zero;
        // i [H, p]: the terms whose word is Δ, if their support meets supp(p)
        if let Some(list) = idx.h_by_word.get(&h) {
            for &hi in list {
                let ht = &spec.h_terms[hi as usize];
                if ht.word != delta || !intersects(&support_mask(&ht.word), &psup) {
                    continue;
                }
                let (r, eps) = comm_product(&ht.word, p);
                if eps != 0.0 {
                    debug_assert!(r == *q);
                    acc += Complex::new(ht.coeff * eps, 0.0);
                }
            }
        }
        // the dissipator entries producing Δ, candidate pairs only, in visiting order
        for &e in idx.entries_of(h) {
            let (jk, kind, pos) = unpack(e);
            if !intersects(&idx.jk_mask[jk as usize], &psup) {
                continue;
            }
            if let Some((w, v)) = eval_entry(spec, idx, jk, kind, pos, p) {
                if w == *q {
                    acc += v;
                }
            }
        }
        debug_assert!(acc.im.abs() < 1e-9, "non-real generator entry {acc}");
        if acc.re != 0.0 {
            out.push((c2 as u32, acc.re));
        }
    }
    if diag != 0.0 {
        out.push((c as u32, diag));
    }
    (out, (raw, diag))
}

/// The legacy column (`build_mf_cols`' body) for a string whose raw / diag
/// are not cached: the full action, filtered to the basis.
fn full_column(
    spec: &LindbladSpec,
    p: &Word,
    index: &FxHashMap<Word, u32>,
    s1: &mut Vec<u32>,
    s2: &mut Vec<u32>,
    lm: &mut FxHashMap<Word, Complex<f64>>,
) -> (Vec<(u32, f64)>, (f64, f64)) {
    let terms = spec.compute_action_terms(p, s1, s2, lm);
    let mut out = Vec::with_capacity(terms.len());
    let mut raw = 0.0;
    let mut diag = 0.0;
    for (w, c) in terms.iter() {
        raw += c.abs();
        if w == p {
            diag = *c;
        }
        if let Some(&row) = index.get(w) {
            out.push((row, *c));
        }
    }
    (out, (raw, diag))
}

/// `(raw, diag)` of a full action output list, as `build_mf_cols` sums them.
#[inline]
pub(crate) fn raw_diag(p: &Word, terms: &[(Word, f64)]) -> (f64, f64) {
    let mut raw = 0.0;
    let mut diag = 0.0;
    for (w, c) in terms {
        raw += c.abs();
        if w == p {
            diag = *c;
        }
    }
    (raw, diag)
}

/// Upper bound on the raw / diag cache before it is cleared (≈ 64 B per entry).
const CACHE_MAX_ENTRIES: usize = 4_000_000;

impl LindbladSpec {
    /// The transfer index, built on first use after a pair-set change.
    pub fn transfer_index(&self) -> &TransferIndex {
        self.transfer.get_or_init(|| TransferIndex::build(self))
    }

    /// `(built, bytes, entries, keys, max_delta, cache_len)` of the locality structures.
    pub fn locality_stats(&self) -> (bool, usize, usize, usize, u32, usize) {
        let cache_len = self.rd_cache.lock().map(|c| c.len()).unwrap_or(0);
        match self.transfer.get() {
            Some(t) => (true, t.bytes, t.num_entries(), t.num_keys(), t.max_delta, cache_len),
            None => (false, 0, 0, 0, 0, cache_len),
        }
    }

    /// Drop the raw / diag cache (any change of H coefficients, rates or `K`).
    pub(crate) fn clear_locality_cache(&mut self) {
        if let Ok(mut c) = self.rd_cache.lock() {
            c.clear();
        }
    }

    /// Drop the transfer index (a change of the pair set or the jump list).
    pub(crate) fn clear_transfer_index(&mut self) {
        self.transfer.take();
    }

    /// Drop the raw / diag cache through a shared reference (tests: force the
    /// cold-cache column path).
    #[cfg(test)]
    pub(crate) fn clear_locality_cache_for_test(&self) {
        if let Ok(mut c) = self.rd_cache.lock() {
            c.clear();
        }
    }

    fn cache_store(&self, items: impl IntoIterator<Item = (Word, (f64, f64))>) {
        if let Ok(mut c) = self.rd_cache.lock() {
            if c.len() > CACHE_MAX_ENTRIES {
                c.clear();
            }
            for (w, v) in items {
                c.insert(w, v);
            }
        }
    }

    /// Evaluate the full action of `words` once and cache their raw / diag
    /// (tests and callers that want the locality build to hit on every column).
    pub fn prime_locality_cache(&self, words: &[Word]) {
        let items: Vec<(Word, (f64, f64))> = words
            .par_iter()
            .map_init(
                || {
                    (
                        Vec::<u32>::with_capacity(self.n_qubits),
                        Vec::<u32>::with_capacity(128),
                        FxHashMap::<Word, Complex<f64>>::with_capacity_and_hasher(
                            128,
                            FxBuildHasher::default(),
                        ),
                    )
                },
                |(s1, s2, lm), p| {
                    let terms = self.compute_action_terms(p, s1, s2, lm);
                    (*p, raw_diag(p, &terms))
                },
            )
            .collect();
        self.cache_store(items);
    }

    /// The generator cache of `basis` from the transfer index (columns of
    /// cached strings) and the full action (the others, which are then
    /// cached). Value-identical to `mf_expm::mf_cache_full`.
    pub(crate) fn mf_cache_local(&self, basis: &[Word]) -> MfCache {
        let idx = self.transfer_index();
        let index = build_basis_index(basis);
        let cached: Vec<Option<(f64, f64)>> = match self.rd_cache.lock() {
            Ok(c) => basis.iter().map(|w| c.get(w).copied()).collect(),
            Err(_) => vec![None; basis.len()],
        };
        let results: Vec<(Vec<(u32, f64)>, (f64, f64))> = (0..basis.len())
            .into_par_iter()
            .map_init(
                || {
                    (
                        Vec::<u32>::with_capacity(self.n_qubits),
                        Vec::<u32>::with_capacity(128),
                        FxHashMap::<Word, Complex<f64>>::with_capacity_and_hasher(
                            128,
                            FxBuildHasher::default(),
                        ),
                    )
                },
                |(s1, s2, lm), c| match cached[c] {
                    Some((raw, diag)) => local_column(self, idx, basis, c, raw, diag),
                    None => full_column(self, &basis[c], &index, s1, s2, lm),
                },
            )
            .collect();
        let misses: Vec<(Word, (f64, f64))> = results
            .iter()
            .enumerate()
            .filter(|(c, _)| cached[*c].is_none())
            .map(|(c, (_, rd))| (basis[c], *rd))
            .collect();
        if !misses.is_empty() {
            self.cache_store(misses);
        }
        let (cols, per_col): (Vec<Vec<(u32, f64)>>, Vec<(f64, f64)>) = results.into_iter().unzip();
        MfCache { cols, per_col }
    }

    /// [`Self::leakage_with_prune`] with the sharded parallel merge, the
    /// `room == 0` early return and the raw / diag cache fill; values are
    /// those of the legacy merge, the result is sorted by word.
    pub(crate) fn leakage_locality(
        &self,
        basis: &[Word],
        coeffs: &[f64],
        protected: &[Word],
        max_basis: usize,
        tau_add: f64,
    ) -> Vec<(Word, f64)> {
        debug_assert_eq!(basis.len(), coeffs.len());
        let room = max_basis.saturating_sub(basis.len());
        if room == 0 || basis.is_empty() {
            return Vec::new();
        }
        let in_basis: FxHashMap<u64, ()> = basis.iter().map(|w| (word_hash(w), ())).collect();
        let protected_set: FxHashMap<u64, ()> =
            protected.iter().map(|w| (word_hash(w), ())).collect();
        let mut order: Vec<usize> = (0..basis.len()).collect();
        order.sort_by(|&a, &b| {
            coeffs[b]
                .abs()
                .partial_cmp(&coeffs[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        const CHUNK_SIZE: usize = 4096;
        let shards = rayon::current_num_threads().clamp(1, 64);
        let shard_of = |h: u64| ((h >> 24) as usize) % shards;
        let mut merged: Vec<FxHashMap<Word, f64>> = (0..shards).map(|_| FxHashMap::default()).collect();
        for chunk_indices in order.chunks(CHUNK_SIZE) {
            // the action of every string of the chunk, its outputs bucketed by shard
            let local: Vec<(Vec<Vec<(Word, f64)>>, (f64, f64))> = chunk_indices
                .par_iter()
                .map_init(
                    || {
                        (
                            Vec::<u32>::with_capacity(self.n_qubits),
                            Vec::<u32>::with_capacity(128),
                            FxHashMap::<Word, Complex<f64>>::with_capacity_and_hasher(
                                128,
                                FxBuildHasher::default(),
                            ),
                        )
                    },
                    |(s1, s2, lm), &i| {
                        let p = &basis[i];
                        let c = coeffs[i];
                        let terms = self.compute_action_terms(p, s1, s2, lm);
                        let rd = raw_diag(p, &terms);
                        let mut buckets: Vec<Vec<(Word, f64)>> = (0..shards).map(|_| Vec::new()).collect();
                        for (w, v) in terms.iter() {
                            let h = word_hash(w);
                            if !in_basis.contains_key(&h) && !protected_set.contains_key(&h) {
                                buckets[shard_of(h)].push((*w, c * *v));
                            }
                        }
                        (buckets, rd)
                    },
                )
                .collect();
            self.cache_store(chunk_indices.iter().zip(local.iter()).map(|(&i, (_, rd))| (basis[i], *rd)));
            // per shard, merge in chunk (= descending |c|) order: the legacy accumulation order per word
            merged.par_iter_mut().enumerate().for_each(|(s, map)| {
                let extra: usize = local.iter().map(|(b, _)| b[s].len()).sum();
                map.reserve(extra);
                for (buckets, _) in &local {
                    for (w, v) in &buckets[s] {
                        *map.entry(*w).or_insert(0.0) += *v;
                    }
                }
            });
            // room cap: the legacy `k`-th largest cutoff over the whole merged map
            let total: usize = merged.iter().map(|m| m.len()).sum();
            if total > room {
                let mut mags: Vec<f64> = Vec::with_capacity(total);
                for m in &merged {
                    mags.extend(m.values().map(|v| v.abs()));
                }
                let k = room.min(mags.len() - 1);
                mags.select_nth_unstable_by(k, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let cutoff = mags[k];
                merged.par_iter_mut().for_each(|m| m.retain(|_, &mut v| v.abs() >= cutoff));
            }
        }
        let mut out: Vec<(Word, f64)> = merged
            .into_iter()
            .flat_map(|m| m.into_iter().filter(|(_, c)| c.abs() > tau_add))
            .collect();
        out.sort_unstable_by(|a, b| (a.0.xbits.data, a.0.zbits.data).cmp(&(b.0.xbits.data, b.0.zbits.data)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mf_expm::{apply_cache, mf_cache_full};
    use crate::{JumpInput, PcStepConfig, parse_pauli_string};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() * n as f64) as usize % n
        }
    }

    fn word_str(n: usize, pairs: &[(usize, char)]) -> String {
        let mut s = vec!['I'; n];
        for &(q, c) in pairs {
            s[q] = c;
        }
        s.into_iter().collect()
    }

    fn sigma_minus(site: usize, n: usize) -> Vec<(String, Complex<f64>)> {
        vec![
            (word_str(n, &[(site, 'X')]), Complex::new(0.5, 0.0)),
            (word_str(n, &[(site, 'Y')]), Complex::new(0.0, -0.5)),
        ]
    }

    fn psd(m: usize, rng: &mut Rng) -> Vec<Vec<Complex<f64>>> {
        let a: Vec<Vec<Complex<f64>>> = (0..m)
            .map(|_| (0..m).map(|_| Complex::new(rng.next() - 0.5, rng.next() - 0.5)).collect())
            .collect();
        (0..m)
            .map(|i| (0..m).map(|j| (0..m).map(|l| a[i][l] * a[j][l].conj()).sum()).collect())
            .collect()
    }

    /// 6 qubits: an XY + Z Hamiltonian, one general jump (σ⁻ on qubit 1), one
    /// Hermitian-Pauli jump (Z on qubit 4), and a dense PSD Kossakowski form
    /// over three operator classes on the three site pairs (9 operators),
    /// so every `JumpKind` and both pair paths are exercised.
    fn fixture(seed: u64) -> LindbladSpec {
        let n = 6;
        let mut rng = Rng(seed);
        let mut h: Vec<(String, f64)> = Vec::new();
        for j in 0..n - 1 {
            h.push((word_str(n, &[(j, 'X'), (j + 1, 'X')]), 0.7 + 0.1 * j as f64));
            h.push((word_str(n, &[(j, 'Y'), (j + 1, 'Y')]), 0.4 - 0.05 * j as f64));
        }
        for j in 0..n {
            h.push((word_str(n, &[(j, 'Z')]), 1.3 + 0.2 * j as f64));
        }
        let jumps = vec![
            JumpInput { lincomb: sigma_minus(1, n), rate: 0.3 },
            JumpInput { lincomb: vec![(word_str(n, &[(4, 'Z')]), Complex::new(1.0, 0.0))], rate: 0.2 },
        ];
        let mut spec = LindbladSpec::new(n, &h, &jumps).unwrap();
        let mut ops = Vec::new();
        for s in 0..3 {
            ops.push(vec![(word_str(n, &[(2 * s, 'Z'), (2 * s + 1, 'Z')]), Complex::new(0.25, 0.0))]);
        }
        for s in 0..3 {
            ops.push(vec![
                (word_str(n, &[(2 * s, 'X'), (2 * s + 1, 'X')]), Complex::new(0.25, 0.0)),
                (word_str(n, &[(2 * s, 'Y'), (2 * s + 1, 'Y')]), Complex::new(0.25, 0.0)),
            ]);
        }
        for s in 0..3 {
            ops.push(sigma_minus(2 * s, n));
        }
        let k = psd(ops.len(), &mut rng);
        spec.add_kossakowski(&ops, &k).unwrap();
        spec
    }

    fn random_basis(n: usize, count: usize, rng: &mut Rng) -> Vec<Word> {
        let letters = ['X', 'Y', 'Z'];
        let mut seen: FxHashMap<Word, ()> = FxHashMap::default();
        let mut out = Vec::new();
        while out.len() < count {
            let w = 1 + rng.below(4);
            let mut pairs = Vec::new();
            for _ in 0..w {
                pairs.push((rng.below(n), letters[rng.below(3)]));
            }
            let (word, _) = parse_pauli_string(&word_str(n, &pairs), n).unwrap();
            if seen.insert(word, ()).is_none() {
                out.push(word);
            }
        }
        out
    }

    fn sorted_col(col: &[(u32, f64)]) -> Vec<(u32, u64)> {
        let mut v: Vec<(u32, u64)> = col.iter().map(|(r, c)| (*r, c.to_bits())).collect();
        v.sort();
        v
    }

    #[test]
    fn local_generator_matches_full_bitwise() {
        let spec = fixture(11);
        let mut rng = Rng(5);
        let basis = random_basis(6, 70, &mut rng);
        spec.prime_locality_cache(&basis);
        let full = mf_cache_full(&spec, &basis);
        let local = spec.mf_cache_local(&basis);
        assert_eq!(full.cols.len(), local.cols.len());
        let mut nnz = 0;
        for c in 0..basis.len() {
            assert_eq!(sorted_col(&full.cols[c]), sorted_col(&local.cols[c]), "column {c}");
            assert_eq!(full.per_col[c].0.to_bits(), local.per_col[c].0.to_bits(), "raw {c}");
            assert_eq!(full.per_col[c].1.to_bits(), local.per_col[c].1.to_bits(), "diag {c}");
            nnz += full.cols[c].len();
        }
        assert!(nnz > basis.len(), "the fixture's generator is not diagonal");
        let (built, bytes, entries, keys, max_delta, cache_len) = spec.locality_stats();
        assert!(built && bytes > 0 && entries > 0 && keys > 0 && max_delta >= 2);
        assert_eq!(cache_len, basis.len());
        // a cold cache (misses) gives the same bits through the full-action column
        spec.clear_locality_cache_for_test();
        let cold = spec.mf_cache_local(&basis);
        for c in 0..basis.len() {
            assert_eq!(sorted_col(&full.cols[c]), sorted_col(&cold.cols[c]));
            assert_eq!(full.per_col[c].0.to_bits(), cold.per_col[c].0.to_bits());
        }
        assert_eq!(spec.locality_stats().5, basis.len());
        // and the exponential from either cache is the same vector
        let coeffs: Vec<f64> = (0..basis.len()).map(|_| rng.next() - 0.5).collect();
        let a = apply_cache(&full, 0.01, &coeffs, 0.0);
        let b = apply_cache(&local, 0.01, &coeffs, 0.0);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    fn sorted_leak(mut v: Vec<(Word, f64)>) -> Vec<([Chunk; W_CHUNKS], [Chunk; W_CHUNKS], u64)> {
        v.sort_by(|a, b| (a.0.xbits.data, a.0.zbits.data).cmp(&(b.0.xbits.data, b.0.zbits.data)));
        v.into_iter().map(|(w, c)| (w.xbits.data, w.zbits.data, c.to_bits())).collect()
    }

    #[test]
    fn leakage_locality_matches_legacy_values() {
        let spec = fixture(3);
        let mut rng = Rng(9);
        let basis = random_basis(6, 50, &mut rng);
        let coeffs: Vec<f64> = (0..basis.len()).map(|_| rng.next() - 0.5).collect();
        let protected = vec![basis[0], random_basis(6, 3, &mut Rng(77))[2]];
        for (max_basis, tau_add) in [
            (usize::MAX, 0.0),
            (basis.len() + 400, 0.0),
            (basis.len() + 17, 0.0),
            (basis.len() + 60, 1e-3),
        ] {
            let legacy = spec
                .leakage_with_prune(&basis, &coeffs, &protected, max_basis, tau_add)
                .unwrap();
            let local = spec.leakage_locality(&basis, &coeffs, &protected, max_basis, tau_add);
            assert_eq!(sorted_leak(legacy), sorted_leak(local), "max_basis {max_basis} tau_add {tau_add}");
            // the locality result is sorted by word
            let again = spec.leakage_locality(&basis, &coeffs, &protected, max_basis, tau_add);
            assert!(again.windows(2).all(|w| {
                (w[0].0.xbits.data, w[0].0.zbits.data) < (w[1].0.xbits.data, w[1].0.zbits.data)
            }));
        }
        // room == 0: both empty, the locality one without evaluating anything
        assert!(spec.leakage_with_prune(&basis, &coeffs, &protected, basis.len(), 0.0).unwrap().is_empty());
        assert!(spec.leakage_locality(&basis, &coeffs, &protected, basis.len(), 0.0).is_empty());
        // the pass fills the raw / diag cache for every basis word
        assert_eq!(spec.locality_stats().5, basis.len());
    }

    fn as_map(basis: &[Word], coeffs: &[f64]) -> FxHashMap<Word, f64> {
        basis.iter().copied().zip(coeffs.iter().copied()).collect()
    }

    #[test]
    fn locality_step_tracks_legacy_within_roundoff() {
        let spec = fixture(21);
        let (z0, _) = parse_pauli_string("ZIIIII", 6).unwrap();
        let (x3, _) = parse_pauli_string("IIIXII", 6).unwrap();
        let mut cfg = PcStepConfig {
            max_basis: 40,
            admit_basis: Some(120),
            drop_tol: 0.0,
            tau_add: None,
            num_threads: Some(1),
            locality: false,
        };
        let (mut b1, mut c1) = (vec![z0, x3], vec![0.8, 0.3]);
        let (mut b2, mut c2) = (b1.clone(), c1.clone());
        for step in 0..6 {
            cfg.locality = false;
            spec.pc_step(&mut b1, &mut c1, 0.02, &[z0], &cfg).unwrap();
            cfg.locality = true;
            spec.pc_step(&mut b2, &mut c2, 0.02, &[z0], &cfg).unwrap();
            let (m1, m2) = (as_map(&b1, &c1), as_map(&b2, &c2));
            assert_eq!(m1.len(), m2.len(), "step {step}: basis sizes");
            let scale = c1.iter().fold(0.0_f64, |a, c| a.max(c.abs()));
            let mut worst = 0.0_f64;
            for (w, v) in &m1 {
                let u = m2
                    .get(w)
                    .unwrap_or_else(|| panic!("step {step}: word missing on the locality path"));
                worst = worst.max((u - v).abs());
            }
            assert!(worst <= 1e-13 * scale, "step {step}: max |Δc| = {worst:e} (scale {scale:e})");
        }
        assert!(b1.len() <= 40 && b1.len() > 2);
    }

    #[test]
    fn transfer_index_survives_rescale_and_resets_on_pattern_change() {
        let mut spec = fixture(4);
        let basis = random_basis(6, 20, &mut Rng(2));
        let _ = spec.mf_cache_local(&basis);
        let (built, bytes, entries, ..) = spec.locality_stats();
        assert!(built);
        // a rescaled K: same pattern, the index stays, the cache is dropped
        let k: Vec<Vec<Complex<f64>>> = spec
            .kossakowski()
            .iter()
            .map(|r| r.iter().map(|v| v * 2.0).collect())
            .collect();
        spec.set_kossakowski(&k).unwrap();
        let s = spec.locality_stats();
        assert!(s.0 && s.1 == bytes && s.2 == entries && s.5 == 0, "{s:?}");
        // a changed pattern (one class zeroed): the index is rebuilt on the next use
        let mut k2 = k.clone();
        for i in 0..3 {
            for j in 0..k2.len() {
                k2[i][j] = Complex::new(0.0, 0.0);
                k2[j][i] = Complex::new(0.0, 0.0);
            }
        }
        spec.set_kossakowski(&k2).unwrap();
        assert!(!spec.locality_stats().0);
        let full = mf_cache_full(&spec, &basis);
        let local = spec.mf_cache_local(&basis);
        for c in 0..basis.len() {
            assert_eq!(sorted_col(&full.cols[c]), sorted_col(&local.cols[c]));
        }
        assert!(spec.locality_stats().2 < entries);
        // set_h_coeffs keeps the index and drops the cache
        let hc: Vec<f64> = spec.h_coeffs().iter().map(|c| c * 0.5).collect();
        spec.set_h_coeffs(&hc).unwrap();
        let s = spec.locality_stats();
        assert!(s.0 && s.5 == 0);
        let full = mf_cache_full(&spec, &basis);
        let local = spec.mf_cache_local(&basis);
        for c in 0..basis.len() {
            assert_eq!(sorted_col(&full.cols[c]), sorted_col(&local.cols[c]));
            assert_eq!(full.per_col[c].0.to_bits(), local.per_col[c].0.to_bits());
        }
    }
}
