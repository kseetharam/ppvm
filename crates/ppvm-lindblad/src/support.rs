// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! The support-grouped action of the locality step (R11.17: "an action
//! indexed by the string's support").
//!
//! `compute_action_terms` visits, for a string `p`, every compiled pair whose
//! support meets `supp(p)` (the `j_support` candidate index) in ascending pair
//! order, and accumulates every table entry's contribution into one hash map
//! keyed by the full output word — one `embed` + `rehash` + hash-map insert
//! per entry, some 25–30 entries per distinct output word on a dipolar
//! Redfield dissipator, where the pairs of the same operator quadruple all
//! write into the same 4-qubit Pauli algebra.
//!
//! This module keeps every floating-point operation and the per-word
//! accumulation order of the legacy action and changes only where the
//! partial sums live:
//!
//! * every compiled pair of ≤ 4 local qubits is assigned its qubit **set**
//!   (the sorted register qubits); a difference word `Δ = p ⊕ w` whose
//!   support is the whole set can only be produced by the pairs on that set
//!   when no other compiled object (a wider pair, a Hamiltonian term, a jump)
//!   can produce a difference word with exactly that support — such sets are
//!   **exclusive**, decided once per compiled dissipator from the templates'
//!   tables (p-independent);
//! * the candidates of `p` are visited grouped by set (within a group in
//!   ascending pair order, the legacy order); a group's full-support
//!   contributions go into a dense 256-entry accumulator indexed by the
//!   canonical local difference word — no hashing, no `embed` — and are
//!   emitted when the group ends (every contributor of such a word is in the
//!   group, visited in legacy order, so the sum is the legacy sum bit for
//!   bit);
//! * everything else (sub-support outputs, pairs on non-exclusive sets,
//!   Hamiltonian terms, jump operators, templates wider than 4 qubits) is
//!   recorded as `(word, legacy sequence number, value)` and summed per word
//!   in sequence order at the end — the legacy accumulation order again;
//! * the template tables are evaluated once per `(template, local input
//!   pattern)` and cached on the thread: the geometric part of every entry
//!   (output difference, phase / `eps` / sign) is `p`-independent given the
//!   restriction of `p` to the pair's qubits; the `K_nm` products are formed
//!   per pair exactly as the legacy loop forms them.
//!
//! Identity: `action_terms_locality(p)` and `compute_action_terms(p)` hold
//! the same `(word, coefficient)` set with every coefficient bit-identical
//! (Rust test `support_action_matches_legacy_bitwise`); the list order
//! differs (the legacy order is the hash map's drain order, which the legacy
//! path does not control across runs either).

use crate::{
    CHUNK_BITS, Chunk, JumpKind, LindbladSpec, W_CHUNKS, Word, anti_commutes, candidate_terms,
    comm_product, comm_product_local, pauli_mul, pauli_mul_local, phase_factor, word_support,
};
use fxhash::FxHashMap;
use num::Complex;
use ppvm_traits::PauliWordTrait;

const NO_TUPLE: u32 = u32::MAX;
const NO_GROUP: u32 = u32::MAX;
const MAX_LOCAL: usize = 4;
const SEQ_DISS: u64 = 1 << 56;

/// Local 8-bit word layout: `x` bits in 0..4, `z` bits in 4..8.
#[inline(always)]
fn local8(w: &Word) -> u8 {
    ((w.xbits.data[0] & 0xF) as u8) | (((w.zbits.data[0] & 0xF) as u8) << 4)
}

#[inline(always)]
fn support_of_local8(d: u8) -> u8 {
    (d & 0xF) | (d >> 4)
}

fn local_mask(w: &Word) -> u8 {
    ((w.xbits.data[0] | w.zbits.data[0]) & 0xF) as u8
}

/// The support-set tables of a compiled dissipator (built with the transfer index).
#[allow(dead_code)]
pub struct SupportTables {
    /// Per `j_kinds` entry: its qubit tuple (`NO_TUPLE` for jumps and wide templates).
    pair_tuple: Vec<u32>,
    tuple_nl: Vec<u8>,
    tuple_qubits: Vec<[u8; MAX_LOCAL]>,
    tuple_set: Vec<u32>,
    /// Per tuple: local difference word → the same word in the set's canonical (sorted-qubit) layout.
    tuple_canon: Vec<[u8; 256]>,
    /// Per tuple: local support mask (bit per local position) → the set id of that sub-support.
    tuple_subset_set: Vec<[u32; 16]>,
    /// Per tuple: local difference word → its canonical layout over its OWN support (the sub-support's sorted qubits).
    tuple_sub_canon: Vec<[u8; 256]>,
    set_size: Vec<u8>,
    set_qubits: Vec<[u8; MAX_LOCAL]>,
    set_exclusive: Vec<bool>,
    /// `(size, sorted qubits)` → set id (the generic path's lookup).
    set_index: FxHashMap<(u8, [u8; MAX_LOCAL]), u32>,
    /// Per qubit: `(group << 32) | k` for every `k` of `j_support[q]`, sorted; `group` = the set
    /// of a dense-eligible pair on an exclusive set, else `NO_GROUP`.
    by_group: Vec<Vec<u64>>,
    pub n_tuples: usize,
    pub n_sets: usize,
    pub n_exclusive: usize,
    pub bytes: usize,
}

struct SetRegistry {
    index: FxHashMap<(u8, [u8; MAX_LOCAL]), u32>,
    size: Vec<u8>,
    qubits: Vec<[u8; MAX_LOCAL]>,
    marked: Vec<bool>,
}

impl SetRegistry {
    fn get(&mut self, qs: &[u8]) -> u32 {
        let mut sorted = [0u8; MAX_LOCAL];
        sorted[..qs.len()].copy_from_slice(qs);
        sorted[..qs.len()].sort_unstable();
        let key = (qs.len() as u8, sorted);
        if let Some(&i) = self.index.get(&key) {
            return i;
        }
        let i = self.size.len() as u32;
        self.index.insert(key, i);
        self.size.push(qs.len() as u8);
        self.qubits.push(sorted);
        self.marked.push(false);
        i
    }
    fn mark(&mut self, qs: &[u8]) {
        if qs.len() <= MAX_LOCAL {
            let i = self.get(qs) as usize;
            self.marked[i] = true;
        }
    }
}

fn register_support(w: &Word, out: &mut Vec<u8>) {
    out.clear();
    for q in 0..w.n_qubits() {
        if w.xbits[q] || w.zbits[q] {
            out.push(q as u8);
        }
    }
}

impl SupportTables {
    pub(crate) fn build(spec: &LindbladSpec) -> Self {
        let mut reg = SetRegistry {
            index: FxHashMap::default(),
            size: Vec::new(),
            qubits: Vec::new(),
            marked: Vec::new(),
        };
        // the local difference-word supports every template can produce (p-independent)
        let template_masks: Vec<Vec<u8>> = spec
            .k_templates
            .iter()
            .map(|t| {
                let mut masks: Vec<u8> = Vec::new();
                for term in &t.adag_b {
                    masks.push(local_mask(&term.word));
                }
                for (wa, rights) in &t.sand {
                    for (wb, _) in rights {
                        let mut m = 0u8;
                        for i in 0..W_CHUNKS {
                            m |= (((wa.xbits.data[i] ^ wb.xbits.data[i]) | (wa.zbits.data[i] ^ wb.zbits.data[i])) & 0xF)
                                as u8;
                        }
                        masks.push(m);
                    }
                }
                masks.sort_unstable();
                masks.dedup();
                masks
            })
            .collect();
        let mut tuple_index: FxHashMap<(u8, [u8; MAX_LOCAL]), u32> = FxHashMap::default();
        let mut tuple_nl: Vec<u8> = Vec::new();
        let mut tuple_qubits: Vec<[u8; MAX_LOCAL]> = Vec::new();
        let mut tuple_template: Vec<u32> = Vec::new();
        let mut pair_tuple: Vec<u32> = vec![NO_TUPLE; spec.j_kinds.len()];
        let mut sup: Vec<u8> = Vec::new();
        for (k, jk) in spec.j_kinds.iter().enumerate() {
            match jk {
                JumpKind::HermitianPauli { .. } => reg.mark(&[]),
                JumpKind::General {
                    terms,
                    dagger_dagger,
                    ..
                } => {
                    for a in terms {
                        for b in terms {
                            let mut d = Word::new(a.word.n_qubits());
                            for i in 0..W_CHUNKS {
                                d.xbits.data[i] = a.word.xbits.data[i] ^ b.word.xbits.data[i];
                                d.zbits.data[i] = a.word.zbits.data[i] ^ b.word.zbits.data[i];
                            }
                            register_support(&d, &mut sup);
                            reg.mark(&sup);
                        }
                    }
                    for c in dagger_dagger {
                        register_support(&c.word, &mut sup);
                        reg.mark(&sup);
                    }
                }
                JumpKind::KossakowskiPair {
                    template, q_off, ..
                } => {
                    let t = &spec.k_templates[*template as usize];
                    let nl = t.n_local as usize;
                    let start = *q_off as usize;
                    let qs = &spec.k_pair_qubits[start..start + nl];
                    let full = ((1u16 << nl) - 1) as u8;
                    // every produced sub-support (and, for a wide template, every produced support ≤ 4) is marked
                    for &m in &template_masks[*template as usize] {
                        if nl > MAX_LOCAL || m != full {
                            sup.clear();
                            for (li, &q) in qs.iter().enumerate() {
                                if m >> li & 1 == 1 {
                                    sup.push(q);
                                }
                            }
                            reg.mark(&sup);
                        }
                    }
                    if nl <= MAX_LOCAL {
                        let mut key = [0u8; MAX_LOCAL];
                        key[..nl].copy_from_slice(qs);
                        let id = *tuple_index.entry((nl as u8, key)).or_insert_with(|| {
                            tuple_nl.push(nl as u8);
                            tuple_qubits.push(key);
                            tuple_template.push(*template);
                            (tuple_nl.len() - 1) as u32
                        });
                        pair_tuple[k] = id;
                    }
                }
            }
        }
        for h in &spec.h_terms {
            register_support(&h.word, &mut sup);
            reg.mark(&sup);
        }
        // the tuples' own sets (registered after the marking so a set that is also some object's
        // sub-support is non-exclusive)
        let tuple_set: Vec<u32> = (0..tuple_nl.len())
            .map(|ti| reg.get(&tuple_qubits[ti][..tuple_nl[ti] as usize]))
            .collect();
        // every sub-support of every tuple is a registered set (the deferred emissions are keyed by it)
        let tuple_subset_set: Vec<[u32; 16]> = (0..tuple_nl.len())
            .map(|ti| {
                let nl = tuple_nl[ti] as usize;
                let qs = &tuple_qubits[ti][..nl];
                let mut row = [u32::MAX; 16];
                for m in 0..(1usize << nl) {
                    sup.clear();
                    for (li, &q) in qs.iter().enumerate() {
                        if m >> li & 1 == 1 {
                            sup.push(q);
                        }
                    }
                    row[m] = reg.get(&sup);
                }
                row
            })
            .collect();
        let set_exclusive: Vec<bool> = reg.marked.iter().map(|m| !m).collect();
        // canonical layout per tuple: local position t → rank of its qubit in the sorted set
        let tuple_canon: Vec<[u8; 256]> = (0..tuple_nl.len())
            .map(|ti| {
                let nl = tuple_nl[ti] as usize;
                let qs = &tuple_qubits[ti][..nl];
                let mut rank = [0u8; MAX_LOCAL];
                for t in 0..nl {
                    rank[t] = qs.iter().filter(|&&q| q < qs[t]).count() as u8;
                }
                let mut tab = [0u8; 256];
                for d in 0..256usize {
                    let (x, z) = ((d & 0xF) as u8, (d >> 4) as u8);
                    let (mut cx, mut cz) = (0u8, 0u8);
                    for t in 0..nl {
                        cx |= ((x >> t) & 1) << rank[t];
                        cz |= ((z >> t) & 1) << rank[t];
                    }
                    tab[d] = cx | (cz << 4);
                }
                tab
            })
            .collect();
        // canonical layout of a local word over its own support: bits in ascending order of the support's qubits
        let tuple_sub_canon: Vec<[u8; 256]> = (0..tuple_nl.len())
            .map(|ti| {
                let nl = tuple_nl[ti] as usize;
                let qs = &tuple_qubits[ti][..nl];
                let mut tab = [0u8; 256];
                for d in 0..256usize {
                    let (x, z) = ((d & 0xF) as u8, (d >> 4) as u8);
                    let m = (x | z) as usize;
                    if m >> nl != 0 {
                        continue;
                    }
                    let mut pos: Vec<usize> = (0..nl).filter(|&t| m >> t & 1 == 1).collect();
                    pos.sort_by_key(|&t| qs[t]);
                    let (mut cx, mut cz) = (0u8, 0u8);
                    for (i, &t) in pos.iter().enumerate() {
                        cx |= ((x >> t) & 1) << i;
                        cz |= ((z >> t) & 1) << i;
                    }
                    tab[d] = cx | (cz << 4);
                }
                tab
            })
            .collect();
        let by_group: Vec<Vec<u64>> = spec
            .j_support
            .iter()
            .map(|list| {
                let mut v: Vec<u64> = list
                    .iter()
                    .map(|&k| {
                        let t = pair_tuple[k as usize];
                        let group = if t != NO_TUPLE && set_exclusive[tuple_set[t as usize] as usize] {
                            tuple_set[t as usize]
                        } else {
                            NO_GROUP
                        };
                        ((group as u64) << 32) | k as u64
                    })
                    .collect();
                v.sort_unstable();
                v
            })
            .collect();
        let bytes = pair_tuple.len() * 4
            + tuple_nl.len() * (1 + MAX_LOCAL + 4 + 256 + 64 + 256)
            + reg.size.len() * (1 + MAX_LOCAL + 1)
            + by_group.iter().map(|v| v.len() * 8).sum::<usize>();
        let n_exclusive = set_exclusive.iter().filter(|&&e| e).count();
        let n_tuples = tuple_nl.len();
        let n_sets = reg.size.len();
        Self {
            pair_tuple,
            tuple_nl,
            tuple_qubits,
            tuple_set,
            tuple_canon,
            tuple_subset_set,
            tuple_sub_canon,
            set_size: reg.size,
            set_qubits: reg.qubits,
            set_exclusive,
            set_index: reg.index,
            by_group,
            n_tuples,
            n_sets,
            n_exclusive,
            bytes,
        }
    }
}

/// One cached table entry of a `(template, local input pattern)`: the local
/// output difference `d`, whether it has full local support, its legacy
/// position in the pair's visiting order and the geometric factor.
#[derive(Clone, Copy)]
struct TpEntry {
    d: u8,
    full: bool,
    /// 0 = one-sided anticommutator (`geo` = eps), 1 = sandwich (`ph` = phase), 2 = both-sided anticommutator (`geo` = sign).
    kind: u8,
    ph: u8,
    pos: u16,
    a: u16,
    b: u16,
    geo: f64,
}

#[derive(Clone, Copy)]
struct TpList {
    start: u32,
    len: u32,
    hit_r: bool,
}

/// Per-thread scratch of the support-grouped action.
pub struct ActScratch {
    sup: Vec<u32>,
    cands: Vec<u32>,
    merged: Vec<u64>,
    tp: FxHashMap<u32, TpList>,
    tp_entries: Vec<TpEntry>,
    /// Deferred contributions keyed by `(set << 8) | canonical word`: slot per key, `(seq, value)` per contribution.
    dmap: FxHashMap<u32, u32>,
    dslots: Vec<(u32, Vec<(u64, Complex<f64>)>)>,
    n_dslots: usize,
    /// Deferred contributions of words whose difference support is wider than 4 qubits (jumps only).
    wmap: FxHashMap<Word, u32>,
    wslots: Vec<(Word, Vec<(u64, Complex<f64>)>)>,
    n_wslots: usize,
    acc: Vec<Complex<f64>>,
    seen: Vec<bool>,
    touched: Vec<u8>,
    sup8: Vec<u8>,
    /// Counters of the last call: `[dense groups, pairs in groups, dense emissions, deferred emissions from
    /// groups, generic candidates, generic emissions, Hamiltonian emissions, distinct deferred words]`.
    pub stats: [u64; 8],
}

impl Default for ActScratch {
    fn default() -> Self {
        Self {
            sup: Vec::new(),
            cands: Vec::new(),
            merged: Vec::new(),
            tp: FxHashMap::default(),
            tp_entries: Vec::new(),
            dmap: FxHashMap::default(),
            dslots: Vec::new(),
            n_dslots: 0,
            wmap: FxHashMap::default(),
            wslots: Vec::new(),
            n_wslots: 0,
            acc: vec![Complex::new(0.0, 0.0); 256],
            seen: vec![false; 256],
            touched: Vec::new(),
            sup8: Vec::new(),
            stats: [0; 8],
        }
    }
}

const TP_CAP: usize = 1 << 24;

impl ActScratch {
    #[inline(always)]
    fn defer_key(&mut self, key: u32, seq: u64, v: Complex<f64>) {
        let slot = match self.dmap.get(&key) {
            Some(&s) => s as usize,
            None => {
                let s = self.n_dslots;
                if s == self.dslots.len() {
                    self.dslots.push((key, Vec::new()));
                }
                self.dslots[s].0 = key;
                self.dslots[s].1.clear();
                self.n_dslots += 1;
                self.dmap.insert(key, s as u32);
                s
            }
        };
        self.dslots[slot].1.push((seq, v));
    }

    #[inline(always)]
    fn defer_word(&mut self, w: Word, seq: u64, v: Complex<f64>) {
        let slot = match self.wmap.get(&w) {
            Some(&s) => s as usize,
            None => {
                let s = self.n_wslots;
                if s == self.wslots.len() {
                    self.wslots.push((w, Vec::new()));
                }
                self.wslots[s].0 = w;
                self.wslots[s].1.clear();
                self.n_wslots += 1;
                self.wmap.insert(w, s as u32);
                s
            }
        };
        self.wslots[slot].1.push((seq, v));
    }
}

/// Route a full-word deferred emission: by `(set, canonical word)` when the difference support has ≤ 4
/// qubits (the key space the dense groups' sub-support emissions use), else by the word itself.
#[inline]
fn defer_full_word(st: &SupportTables, sc: &mut ActScratch, p: &Word, w: &Word, seq: u64, v: Complex<f64>) {
    let mut qs = [0u8; MAX_LOCAL];
    let mut n = 0usize;
    let mut wide = false;
    for i in 0..W_CHUNKS {
        let mut diff = (w.xbits.data[i] ^ p.xbits.data[i]) | (w.zbits.data[i] ^ p.zbits.data[i]);
        while diff != 0 {
            let b = diff.trailing_zeros() as usize;
            diff &= diff - 1;
            if n == MAX_LOCAL {
                wide = true;
                break;
            }
            qs[n] = (i * CHUNK_BITS + b) as u8;
            n += 1;
        }
        if wide {
            break;
        }
    }
    if !wide {
        if let Some(&set) = st.set_index.get(&(n as u8, qs)) {
            // canonical word over the sorted support (qs is ascending: chunks then bits)
            let (mut cx, mut cz) = (0u8, 0u8);
            for (i, &q) in qs[..n].iter().enumerate() {
                let q = q as usize;
                let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
                cx |= ((((w.xbits.data[ci] ^ p.xbits.data[ci]) >> bi) & 1) as u8) << i;
                cz |= ((((w.zbits.data[ci] ^ p.zbits.data[ci]) >> bi) & 1) as u8) << i;
            }
            sc.defer_key((set << 8) | cx as u32 | ((cz as u32) << 4), seq, v);
            return;
        }
    }
    sc.defer_word(*w, seq, v);
}

/// The template tables on the local pattern `(x8, z8)`: entries in the legacy visiting order.
fn build_tp(spec: &LindbladSpec, template: u32, x8: u8, z8: u8, out: &mut Vec<TpEntry>) -> TpList {
    let t = &spec.k_templates[template as usize];
    let nl = t.n_local as usize;
    let full = ((1u16 << nl) - 1) as u8;
    let mut p_loc = Word::new(nl);
    p_loc.xbits.data[0] = x8 as Chunk;
    p_loc.zbits.data[0] = z8 as Chunk;
    let p8 = x8 | (z8 << 4);
    let hits = |mask: &[Chunk; W_CHUNKS]| (0..W_CHUNKS).any(|i| mask[i] & (p_loc.xbits.data[i] | p_loc.zbits.data[i]) != 0);
    let (hit_l, hit_r) = (hits(&t.left_mask), hits(&t.right_mask));
    let start = out.len() as u32;
    if !(hit_l && hit_r) {
        for (ti, term) in t.adag_b.iter().enumerate() {
            let (o, eps) = comm_product_local(&term.word, &p_loc);
            if eps != 0.0 {
                let d = local8(&o) ^ p8;
                out.push(TpEntry { d, full: support_of_local8(d) == full, kind: 0, ph: 0, pos: ti as u16, a: ti as u16, b: 0, geo: eps });
            }
        }
    } else {
        let mut pos = 0u16;
        for (ga, (wa, rights)) in t.sand.iter().enumerate() {
            let (r_ap, phi1) = pauli_mul_local(wa, &p_loc);
            for (gb, (wb, _)) in rights.iter().enumerate() {
                let (s, phi2) = pauli_mul_local(&r_ap, wb);
                let d = local8(&s) ^ p8;
                out.push(TpEntry { d, full: support_of_local8(d) == full, kind: 1, ph: phi1 + phi2, pos, a: ga as u16, b: gb as u16, geo: 0.0 });
                pos += 1;
            }
        }
        for (ti, term) in t.adag_b.iter().enumerate() {
            let (r, phase) = pauli_mul_local(&term.word, &p_loc);
            if phase & 1 == 0 {
                let sign = if phase == 0 { 1.0 } else { -1.0 };
                let d = local8(&r) ^ p8;
                out.push(TpEntry { d, full: support_of_local8(d) == full, kind: 2, ph: 0, pos: pos + ti as u16, a: ti as u16, b: 0, geo: sign });
            }
        }
    }
    TpList { start, len: out.len() as u32 - start, hit_r }
}

/// `p ⊕ Δ` with the local difference `d` (local layout) placed on `qs`.
#[inline(always)]
fn xor_local(p: &Word, d: u8, qs: &[u8]) -> Word {
    let mut out = *p;
    for (li, &q) in qs.iter().enumerate() {
        let q = q as usize;
        let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
        out.xbits.data[ci] ^= ((((d >> li) & 1) as Chunk) << bi) as Chunk;
        out.zbits.data[ci] ^= ((((d >> (4 + li)) & 1) as Chunk) << bi) as Chunk;
    }
    out.rehash();
    out
}

/// Merge the sorted `(group, k)` lists of the support qubits, deduplicated.
fn merge_candidates(lists: &[&[u64]], out: &mut Vec<u64>) {
    out.clear();
    if lists.len() > 8 {
        for l in lists {
            out.extend_from_slice(l);
        }
        out.sort_unstable();
        out.dedup();
        return;
    }
    let mut pos = vec![0usize; lists.len()];
    loop {
        let mut best: Option<u64> = None;
        for (i, l) in lists.iter().enumerate() {
            if pos[i] < l.len() {
                let v = l[pos[i]];
                best = Some(best.map_or(v, |b| b.min(v)));
            }
        }
        let Some(v) = best else { break };
        for (i, l) in lists.iter().enumerate() {
            if pos[i] < l.len() && l[pos[i]] == v {
                pos[i] += 1;
            }
        }
        out.push(v);
    }
}

impl LindbladSpec {
    /// The support-grouped action: `L*(p)` as deduplicated `(word, coefficient)`
    /// pairs, value-identical to [`Self::compute_action_terms`] (the list order differs).
    pub(crate) fn action_terms_locality(&self, p: &Word, sc: &mut ActScratch) -> Vec<(Word, f64)> {
        let st = &self.transfer_index().support;
        let zero = Complex::new(0.0, 0.0);
        word_support(p, &mut sc.sup);
        sc.dmap.clear();
        sc.n_dslots = 0;
        sc.wmap.clear();
        sc.n_wslots = 0;
        sc.stats = [0; 8];
        let mut out: Vec<(Word, f64)> = Vec::new();

        // ── i [H, p] (legacy: first, in term order) ──
        candidate_terms(&sc.sup, &self.h_support, &mut sc.cands);
        for ci in 0..sc.cands.len() {
            let i = sc.cands[ci];
            let h = &self.h_terms[i as usize];
            let (r, eps) = comm_product(&h.word, p);
            if eps != 0.0 {
                defer_full_word(st, sc, p, &r, i as u64, Complex::new(h.coeff * eps, 0.0));
                sc.stats[6] += 1;
            }
        }

        // ── dissipator candidates grouped by set ──
        {
            let lists: Vec<&[u64]> = sc.sup.iter().map(|&q| st.by_group[q as usize].as_slice()).collect();
            merge_candidates(&lists, &mut sc.merged);
        }
        if sc.tp_entries.len() > TP_CAP {
            sc.tp.clear();
            sc.tp_entries.clear();
        }
        let n_merged = sc.merged.len();
        let mut i = 0usize;
        while i < n_merged {
            let group = (sc.merged[i] >> 32) as u32;
            if group == NO_GROUP {
                break;
            }
            // one dense group: the set `group`, its pairs in ascending k
            let set = group as usize;
            let nl_set = st.set_size[set] as usize;
            let set_qs = &st.set_qubits[set][..nl_set];
            sc.stats[0] += 1;
            let mut j = i;
            while j < n_merged && (sc.merged[j] >> 32) as u32 == group {
                let k = (sc.merged[j] & 0xffff_ffff) as u32;
                j += 1;
                let JumpKind::KossakowskiPair { template, q_off, off_diag, k_nm } = &self.j_kinds[k as usize] else {
                    unreachable!("dense group holds Kossakowski pairs only")
                };
                let t = &self.k_templates[*template as usize];
                let nl = t.n_local as usize;
                let start = *q_off as usize;
                let qs = &self.k_pair_qubits[start..start + nl];
                let tuple = st.pair_tuple[k as usize] as usize;
                let canon = &st.tuple_canon[tuple];
                let sub_set = &st.tuple_subset_set[tuple];
                let sub_canon = &st.tuple_sub_canon[tuple];
                sc.stats[1] += 1;
                let (mut x8, mut z8) = (0u8, 0u8);
                for (li, &q) in qs.iter().enumerate() {
                    let q = q as usize;
                    let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
                    x8 |= (((p.xbits.data[ci] >> bi) & 1) as u8) << li;
                    z8 |= (((p.zbits.data[ci] >> bi) & 1) as u8) << li;
                }
                let key = (*template << 8) | (x8 as u32) | ((z8 as u32) << 4);
                let list = match sc.tp.get(&key) {
                    Some(l) => *l,
                    None => {
                        let l = build_tp(self, *template, x8, z8, &mut sc.tp_entries);
                        sc.tp.insert(key, l);
                        l
                    }
                };
                let half_i = if *off_diag {
                    if list.hit_r { Complex::new(0.0, -0.5) } else { Complex::new(0.0, 0.5) }
                } else if list.hit_r {
                    Complex::new(0.0, 0.5)
                } else {
                    Complex::new(0.0, -0.5)
                };
                let seq_k = SEQ_DISS | ((k as u64) << 16);
                for ei in list.start as usize..(list.start + list.len) as usize {
                    let e = sc.tp_entries[ei];
                    let v = match e.kind {
                        0 => {
                            let term = &t.adag_b[e.a as usize];
                            let c = *k_nm * term.coeff;
                            let coeff = if *off_diag {
                                if c.im.abs() > 1e-14 { Complex::new(0.0, -2.0 * c.im) } else { continue }
                            } else {
                                c
                            };
                            coeff * half_i * e.geo
                        }
                        1 => {
                            let (_, rights) = &t.sand[e.a as usize];
                            let (_, c0) = &rights[e.b as usize];
                            let v = (c0 * k_nm) * phase_factor(e.ph);
                            if *off_diag { Complex::new(2.0 * v.re, 0.0) } else { v }
                        }
                        _ => {
                            let term = &t.adag_b[e.a as usize];
                            let c = *k_nm * term.coeff;
                            let coeff = if *off_diag {
                                if c.re.abs() > 1e-14 { Complex::new(2.0 * c.re, 0.0) } else { continue }
                            } else {
                                c
                            };
                            // the legacy action does `local -= coeff · sign`; `a − b == a + (−b)` in IEEE
                            -(coeff * Complex::new(e.geo, 0.0))
                        }
                    };
                    if e.full {
                        let idx = canon[e.d as usize] as usize;
                        sc.acc[idx] += v;
                        if !sc.seen[idx] {
                            sc.seen[idx] = true;
                            sc.touched.push(idx as u8);
                        }
                        sc.stats[2] += 1;
                    } else {
                        let set = sub_set[support_of_local8(e.d) as usize];
                        sc.defer_key((set << 8) | sub_canon[e.d as usize] as u32, seq_k | e.pos as u64, v);
                        sc.stats[3] += 1;
                    }
                }
            }
            // flush the group: the touched full-support words of the set, canonical layout
            for ti in 0..sc.touched.len() {
                let idx = sc.touched[ti] as usize;
                let c = sc.acc[idx];
                debug_assert!(c.im.abs() < 1e-9, "L*(p) produced non-real coefficient {c}; bug in dissipator");
                if c.re != 0.0 {
                    out.push((xor_local(p, idx as u8, set_qs), c.re));
                }
                sc.acc[idx] = zero;
                sc.seen[idx] = false;
            }
            sc.touched.clear();
            i = j;
        }
        // ── the generic candidates (ascending k), recorded for the ordered sum ──
        while i < n_merged {
            let k = (sc.merged[i] & 0xffff_ffff) as u32;
            i += 1;
            sc.stats[4] += 1;
            let seq_k = SEQ_DISS | ((k as u64) << 16);
            match &self.j_kinds[k as usize] {
                JumpKind::HermitianPauli { word, rate, .. } => {
                    if anti_commutes(word, p) {
                        defer_full_word(st, sc, p, p, seq_k, Complex::new(-2.0 * *rate, 0.0));
                    }
                }
                JumpKind::General { terms, dagger_dagger, rate } => {
                    let rate_c = Complex::new(*rate, 0.0);
                    let nt = terms.len();
                    for (ai, a) in terms.iter().enumerate() {
                        let (r_ap, phi1) = pauli_mul(&a.word, p);
                        for (bi, b) in terms.iter().enumerate() {
                            let (s, phi2) = pauli_mul(&r_ap, &b.word);
                            let coeff = a.coeff.conj() * b.coeff * phase_factor(phi1 + phi2) * rate_c;
                            defer_full_word(st, sc, p, &s, seq_k | (ai * nt + bi) as u64, coeff);
                        }
                    }
                    for (ci, c_term) in dagger_dagger.iter().enumerate() {
                        let (r, phase) = pauli_mul(&c_term.word, p);
                        if phase & 1 == 0 {
                            let sign = if phase == 0 { 1.0 } else { -1.0 };
                            let coeff = -c_term.coeff * rate_c * Complex::new(sign, 0.0);
                            defer_full_word(st, sc, p, &r, seq_k | (nt * nt + ci) as u64, coeff);
                        }
                    }
                }
                JumpKind::KossakowskiPair { template, q_off, off_diag, k_nm } => {
                    // a pair on a non-exclusive set or a wide template: the legacy evaluation, recorded
                    let t = &self.k_templates[*template as usize];
                    let nl = t.n_local as usize;
                    let start = *q_off as usize;
                    let qs = &self.k_pair_qubits[start..start + nl];
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
                        let mut o = Word::new(p.n_qubits());
                        for i in 0..W_CHUNKS {
                            o.xbits.data[i] = p.xbits.data[i] & !sup[i];
                            o.zbits.data[i] = p.zbits.data[i] & !sup[i];
                        }
                        for (li, &q) in qs.iter().enumerate() {
                            let q = q as usize;
                            let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
                            let (lc, lb) = (li / CHUNK_BITS, li % CHUNK_BITS);
                            o.xbits.data[ci] |= ((w.xbits.data[lc] >> lb) & 1) << bi;
                            o.zbits.data[ci] |= ((w.zbits.data[lc] >> lb) & 1) << bi;
                        }
                        o.rehash();
                        o
                    };
                    let hits = |mask: &[Chunk; W_CHUNKS]| (0..W_CHUNKS).any(|i| mask[i] & (p_loc.xbits.data[i] | p_loc.zbits.data[i]) != 0);
                    let (hit_l, hit_r) = (hits(&t.left_mask), hits(&t.right_mask));
                    if !(hit_l && hit_r) {
                        let half_i = if *off_diag {
                            if hit_r { Complex::new(0.0, -0.5) } else { Complex::new(0.0, 0.5) }
                        } else if hit_r {
                            Complex::new(0.0, 0.5)
                        } else {
                            Complex::new(0.0, -0.5)
                        };
                        for (ti, term) in t.adag_b.iter().enumerate() {
                            let c = *k_nm * term.coeff;
                            let coeff = if *off_diag {
                                if c.im.abs() > 1e-14 { Complex::new(0.0, -2.0 * c.im) } else { continue }
                            } else {
                                c
                            };
                            let (o, eps) = comm_product_local(&term.word, &p_loc);
                            if eps != 0.0 {
                                defer_full_word(st, sc, p, &embed(&o), seq_k | ti as u64, coeff * half_i * eps);
                            }
                        }
                        continue;
                    }
                    let mut pos = 0u64;
                    for (wa, rights) in &t.sand {
                        let (r_ap, phi1) = pauli_mul_local(wa, &p_loc);
                        for (wb, c0) in rights {
                            let (s, phi2) = pauli_mul_local(&r_ap, wb);
                            let v = (c0 * k_nm) * phase_factor(phi1 + phi2);
                            let contrib = if *off_diag { Complex::new(2.0 * v.re, 0.0) } else { v };
                            defer_full_word(st, sc, p, &embed(&s), seq_k | pos, contrib);
                            pos += 1;
                        }
                    }
                    for (ti, term) in t.adag_b.iter().enumerate() {
                        let c = *k_nm * term.coeff;
                        let coeff = if *off_diag {
                            if c.re.abs() > 1e-14 { Complex::new(2.0 * c.re, 0.0) } else { continue }
                        } else {
                            c
                        };
                        let (r, phase) = pauli_mul_local(&term.word, &p_loc);
                        if phase & 1 == 0 {
                            let sign = if phase == 0 { 1.0 } else { -1.0 };
                            defer_full_word(st, sc, p, &embed(&r), seq_k | (pos + ti as u64), -(coeff * Complex::new(sign, 0.0)));
                        }
                    }
                }
            }
        }
        // ── the deferred words: per word, the contributions in legacy (sequence) order ──
        for si in 0..sc.n_dslots {
            let (key, list) = &mut sc.dslots[si];
            if list.len() > 1 {
                list.sort_unstable_by_key(|(seq, _)| *seq);
            }
            let mut acc = zero;
            for (_, v) in list.iter() {
                acc += *v;
            }
            debug_assert!(acc.im.abs() < 1e-9, "L*(p) produced non-real coefficient {acc}; bug in dissipator");
            sc.stats[7] += 1;
            if acc.re != 0.0 {
                let set = (*key >> 8) as usize;
                let d = (*key & 0xff) as u8;
                let qs = &st.set_qubits[set][..st.set_size[set] as usize];
                out.push((xor_local(p, d, qs), acc.re));
            }
        }
        for si in 0..sc.n_wslots {
            let (w, list) = &mut sc.wslots[si];
            if list.len() > 1 {
                list.sort_unstable_by_key(|(seq, _)| *seq);
            }
            let mut acc = zero;
            for (_, v) in list.iter() {
                acc += *v;
            }
            debug_assert!(acc.im.abs() < 1e-9, "L*(p) produced non-real coefficient {acc}; bug in dissipator");
            sc.stats[7] += 1;
            if acc.re != 0.0 {
                out.push((*w, acc.re));
            }
        }
        let _ = &mut sc.sup8;
        out
    }

    /// [`Self::action_terms_locality`] for callers outside the crate (the Python binding's `action_locality`).
    pub fn action_terms_locality_pub(&self, p: &Word, sc: &mut ActScratch) -> Vec<(Word, f64)> {
        self.action_terms_locality(p, sc)
    }

    /// `(tuples, sets, exclusive sets, bytes)` of the support tables.
    pub fn support_stats(&self) -> (usize, usize, usize, usize) {
        let st = &self.transfer_index().support;
        (st.n_tuples, st.n_sets, st.n_exclusive, st.bytes)
    }
}
