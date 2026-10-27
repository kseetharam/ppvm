// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! Direct Heisenberg-picture Lindbladian evolution on an adaptive
//! Pauli-string basis.
//!
//! For a Hermitian Pauli Hamiltonian `H = Σ c_i P_i` and jump operators
//! `L_k = Σ_a λ_{k,a} P_{k,a}` (each a Hermitian-Pauli linear combination
//! with possibly complex coefficients) with rates `γ_k ≥ 0`, the adjoint
//! Lindbladian acts on a single Pauli string `p` as
//!
//! ```text
//! L*(p) = i [H, p] + Σ_k γ_k ( L_k† p L_k − 1/2 {L_k† L_k, p} ).
//! ```
//!
//! Two jump shapes are supported with separate code paths:
//!
//! - **Hermitian Pauli** (`L = P`, `λ ∈ ℝ`): the dissipator collapses to a
//!   diagonal `-2γ` on Pauli strings that anti-commute with `P`. Same fast
//!   path used by every dephasing-style model.
//!
//! - **General** (complex `λ_a`, e.g. `σ± = (X ± iY)/2`): the dissipator
//!   becomes a double sum `Σ_{a,b} λ_a* λ_b P_a p P_b` plus a Pauli-
//!   linear-combination anti-commutator with `L†L`, which is precomputed
//!   once at construction. Intermediate coefficients are complex; the
//!   result is real because `L*` preserves Hermiticity, so we cast back
//!   to `f64` at the boundary (with a debug-only check that `|Im|` is at
//!   FP noise).
//!
//! Pauli strings are stored as [`ppvm_pauli_word::word::PauliWord`] backed by
//! two 64-bit chunks (≤128 qubits; four 32-bit chunks on 32-bit targets)
//! with cached hashes for fast HashMap lookup. The hot-path commutator/
//! product loops bypass the higher-level word API and operate directly on
//! the raw chunks for speed.

use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use num::Complex;
use ppvm_pauli_word::word::PauliWord;
use ppvm_traits::PauliWordTrait;
use rayon::prelude::*;
use std::time::Instant;

pub mod block;
pub mod config;
pub mod error;
pub(crate) mod expm;

/// Matrix-free / quspin-expm-backed `exp(dt·L*)·b` engine. See module docs.
pub(crate) mod mf_expm;

/// Per-step orbit-rep evolution under translation symmetry, with a
/// phase-aware complex action. See module docs.
pub mod orbit_rep;

/// Locality-aware predictor-corrector step (`PcStepConfig::locality`): the
/// transfer index of the compiled dissipator, the basis-local generator
/// build, the raw / diag cache and the sharded leakage merge. See module docs.
pub mod locality;
pub mod support;

/// Pauli-word storage chunk: `u64` on 64-bit targets, `u32` elsewhere
/// (`bitvec` implements `BitStore` for `u64` only on 64-bit targets, so
/// e.g. wasm32 builds use four 32-bit chunks instead of two 64-bit ones).
#[cfg(target_pointer_width = "64")]
type Chunk = u64;
#[cfg(not(target_pointer_width = "64"))]
type Chunk = u32;

/// Chunks per word; words pack up to 128 qubits on every target.
#[cfg(target_pointer_width = "64")]
const W_CHUNKS: usize = 2;
#[cfg(not(target_pointer_width = "64"))]
const W_CHUNKS: usize = 4;

/// Maximum number of qubits supported by [`Word`].
pub const MAX_QUBITS: usize = 128;

/// The Pauli-word storage type used throughout this crate.
///
/// `[Chunk; W_CHUNKS]` covers up to 128 qubits; the `FxBuildHasher`
/// matches the hash used by the `FxHashMap` keys we wrap with;
/// `REHASH=true` means `set()` keeps the cached hash in sync.
pub type Word = PauliWord<[Chunk; W_CHUNKS], FxBuildHasher, true>;

pub use block::BlockLindbladSpec;
pub use config::PcStepConfig;
pub use error::Error;

/// Per-phase timing breakdown (microseconds) returned by
/// [`LindbladSpec::pc_step_timed`].
#[derive(Default, Clone, Copy, Debug)]
pub struct PcStepTimings {
    pub leakage1_us: u64,
    pub expand1_us: u64,
    pub expm1_us: u64,
    pub leakage2_us: u64,
    pub expand2_us: u64,
    pub expm2_us: u64,
    /// The generator-build part of `expm1_us` / `expm2_us` (the action pass
    /// or the locality build; the remainder is the Taylor matvecs).
    pub build1_us: u64,
    pub build2_us: u64,
    /// `prune_basis` + `cap_basis` at the end of the step.
    pub prune_cap_us: u64,
    /// The locality step's inner phases (zero on the legacy path).
    pub locality: locality::LocalityProfile,
}

impl PcStepTimings {
    pub fn total_us(&self) -> u64 {
        self.leakage1_us
            + self.expand1_us
            + self.expm1_us
            + self.leakage2_us
            + self.expand2_us
            + self.expm2_us
    }
}

// ────────────────── codec helpers ──────────────────

/// Build a [`Word`] from a length-`n_qubits` slice of Pauli codes
/// (`0=I, 1=X, 2=Z, 3=Y`). Sets all bits and rehashes once.
pub fn word_from_codes(codes: &[u8]) -> Result<Word, Error> {
    let n_qubits = codes.len();
    if n_qubits > MAX_QUBITS {
        return Err(Error::TooManyQubits { got: n_qubits });
    }
    let mut w = Word::new(n_qubits);
    for (q, &b) in codes.iter().enumerate() {
        if b > 3 {
            return Err(Error::InvalidPauliCode { code: b });
        }
        if b & 1 != 0 {
            w.xbits.set(q, true);
        }
        if b & 2 != 0 {
            w.zbits.set(q, true);
        }
    }
    w.rehash();
    Ok(w)
}

/// Inverse of [`word_from_codes`]: write `n_qubits` Pauli codes into `out`.
pub fn codes_from_word(w: &Word, out: &mut [u8]) {
    debug_assert_eq!(out.len(), w.n_qubits());
    for (q, slot) in out.iter_mut().enumerate() {
        let xb = w.xbits[q] as u8;
        let zb = w.zbits[q] as u8;
        *slot = xb | (zb << 1);
    }
}

/// Parse a `"IXYZ..."` string into a [`Word`] together with the list of
/// qubits where the Pauli is non-identity (the term's support).
pub fn parse_pauli_string(s: &str, n_qubits: usize) -> Result<(Word, Vec<u32>), Error> {
    if n_qubits > MAX_QUBITS {
        return Err(Error::TooManyQubits { got: n_qubits });
    }
    let chars: Vec<char> = s.chars().filter(|c| *c != '_').collect();
    if chars.len() != n_qubits {
        return Err(Error::WrongLength {
            expected: n_qubits,
            got: chars.len(),
        });
    }
    let mut w = Word::new(n_qubits);
    let mut support = Vec::new();
    for (q, c) in chars.into_iter().enumerate() {
        match c {
            'I' => {}
            'X' => {
                w.xbits.set(q, true);
                support.push(q as u32);
            }
            'Z' => {
                w.zbits.set(q, true);
                support.push(q as u32);
            }
            'Y' => {
                w.xbits.set(q, true);
                w.zbits.set(q, true);
                support.push(q as u32);
            }
            other => return Err(Error::InvalidPauliChar { c: other }),
        }
    }
    w.rehash();
    Ok((w, support))
}

/// Compute the support (non-identity qubits) of `w`.
pub(crate) fn word_support(w: &Word, out: &mut Vec<u32>) {
    out.clear();
    for q in 0..w.n_qubits() {
        if w.xbits[q] || w.zbits[q] {
            out.push(q as u32);
        }
    }
}

// ────────────────── Pauli algebra ──────────────────
//
// Phase encoding: `Pauli product P·Q = ω · R` where `ω = i^phase` and
// `phase ∈ {0,1,2,3}` ↔ `ω ∈ {1, i, -1, -i}`. The per-byte XOR/AND
// formulas are the same ones used by `PhasedPauliWord::mul_assign` in
// `ppvm-runtime`.

#[inline(always)]
pub(crate) fn phase_factor(phase: u8) -> Complex<f64> {
    match phase & 3 {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

/// `true` if Pauli words `a` and `b` anti-commute.
///
/// Two Pauli strings anti-commute iff
/// `popcount(a.x & b.z) + popcount(a.z & b.x)` is odd.
#[inline(always)]
pub(crate) fn anti_commutes(a: &Word, b: &Word) -> bool {
    let mut bits: u32 = 0;
    for i in 0..W_CHUNKS {
        bits += (a.xbits.data[i] & b.zbits.data[i]).count_ones();
        bits += (a.zbits.data[i] & b.xbits.data[i]).count_ones();
    }
    bits & 1 == 1
}

/// Commutator product `h · p`: returns `(out, eps)` where `out = h ⊕ p` and
///
/// - `eps =  0` if `h` and `p` commute (caller should skip — `[h,p] = 0`),
/// - `eps = -2.0` if `h·p` has phase `+i` (so `i·[h,p] = -2·out`),
/// - `eps = +2.0` if `h·p` has phase `-i` (so `i·[h,p] = +2·out`).
#[inline(always)]
pub(crate) fn comm_product(h: &Word, p: &Word) -> (Word, f64) {
    let (out, phase) = pauli_mul(h, p);
    let eps = match phase {
        1 => -2.0,
        3 => 2.0,
        _ => 0.0,
    };
    (out, eps)
}

/// Full Pauli product `p · q`: returns `(out, phase)` where the product
/// is `ω · out` with `ω = i^phase`.
#[inline(always)]
pub(crate) fn pauli_mul(p: &Word, q: &Word) -> (Word, u8) {
    let mut out = Word::new(p.n_qubits());
    let mut sign_count: u32 = 0;
    let mut imag_count: u32 = 0;
    for i in 0..W_CHUNKS {
        let a = p.xbits.data[i];
        let b = p.zbits.data[i];
        let c = q.xbits.data[i];
        let d = q.zbits.data[i];
        let sign = (a & b & c & !d) | (a & !b & !c & d) | (!a & b & c & d);
        let imag = (a & !b & d) | (a & !c & d) | (!a & b & c) | (b & c & !d);
        sign_count += sign.count_ones();
        imag_count += imag.count_ones();
        out.xbits.data[i] = a ^ c;
        out.zbits.data[i] = b ^ d;
    }
    out.rehash();
    (out, ((2 * sign_count + imag_count) & 3) as u8)
}

/// Bits per storage chunk of a [`Word`] (`bitvec` `Lsb0`: bit `q` lives at
/// `1 << (q % CHUNK_BITS)` of chunk `q / CHUNK_BITS`; asserted by the test
/// `raw_bit_access_matches_bitvec`).
const CHUNK_BITS: usize = 8 * std::mem::size_of::<Chunk>();

/// [`pauli_mul`] for words in a Kossakowski pair's local frame: the same
/// bits and phase, without the hash (the result is embedded into the
/// register and hashed there).
#[inline(always)]
pub(crate) fn pauli_mul_local(p: &Word, q: &Word) -> (Word, u8) {
    let mut out = Word::new(p.n_qubits());
    let mut sign_count: u32 = 0;
    let mut imag_count: u32 = 0;
    for i in 0..W_CHUNKS {
        let a = p.xbits.data[i];
        let b = p.zbits.data[i];
        let c = q.xbits.data[i];
        let d = q.zbits.data[i];
        let sign = (a & b & c & !d) | (a & !b & !c & d) | (!a & b & c & d);
        let imag = (a & !b & d) | (a & !c & d) | (!a & b & c) | (b & c & !d);
        sign_count += sign.count_ones();
        imag_count += imag.count_ones();
        out.xbits.data[i] = a ^ c;
        out.zbits.data[i] = b ^ d;
    }
    (out, ((2 * sign_count + imag_count) & 3) as u8)
}

/// [`comm_product`] on local-frame words ([`pauli_mul_local`]).
#[inline(always)]
pub(crate) fn comm_product_local(h: &Word, p: &Word) -> (Word, f64) {
    let (out, phase) = pauli_mul_local(h, p);
    let eps = match phase {
        1 => -2.0,
        3 => 2.0,
        _ => 0.0,
    };
    (out, eps)
}

// ────────────────── Spec types ──────────────────

/// Parsed Hamiltonian term.
#[derive(Clone)]
struct HTerm {
    word: Word,
    coeff: f64,
}

/// One Pauli term in a complex linear combination (a single summand of
/// `L = Σ_a λ_a P_a` or of the precomputed `L†L`).
#[derive(Clone)]
struct PauliTerm {
    word: Word,
    coeff: Complex<f64>,
}

/// Sandwich table of a Kossakowski pair, grouped by the left word: one
/// `(P_a, [(P_b, coeff), …])` group per distinct `P_a`, so `P_a · p` is
/// computed once per group and reused across its `P_b` partners.
type SandwichGroups = Vec<(Word, Vec<(Word, Complex<f64>)>)>;

/// One Pauli term of a Kossakowski operator in the local frame of a pair,
/// as part of a template key: the `x` / `z` bits over the local qubits and
/// the exact coefficient bits.
#[derive(Clone, PartialEq, Eq, Hash)]
struct LocalTerm {
    x: [Chunk; W_CHUNKS],
    z: [Chunk; W_CHUNKS],
    re: u64,
    im: u64,
}

/// The operator-pattern relationship of a pair `(A_n, A_m)`: both
/// operators' terms, in order, relabelled into the pair's local frame
/// (qubits numbered by first appearance over the terms of `A_n`, then
/// `A_m`; [`LindbladSpec::pair_local_frame`]). Two pairs with the same key
/// have the same Pauli products `A_n†A_m`, sandwich tables and support
/// masks up to that relabelling — a qubit permutation changes no
/// Pauli-product phase — so they share one [`PairTemplate`]. The
/// tensor-pattern operators of a Redfield dissipator (the components
/// `T^(l)_k` of every dipolar pair / CSA interaction) repeat the same few
/// patterns across blocks, so the number of templates is bounded by the
/// number of distinct patterns, not by the number of compiled pairs.
#[derive(Clone, PartialEq, Eq, Hash)]
struct TemplateKey {
    left: Vec<LocalTerm>,
    right: Vec<LocalTerm>,
}

/// The `K`-independent compilation of one operator-pattern relationship,
/// in the pair's local frame (qubits `0..n_local`): the Pauli products of
/// `A_n†A_m`, the sandwich word table with coefficients `λ_a*·μ_b` (no
/// `K_nm` yet) and the support masks. Shared by every compiled pair with
/// the same [`TemplateKey`]; the pair's own record ([`PairRef`] and its
/// `JumpKind::KossakowskiPair` entry) holds `K_nm` and the qubit map that
/// embeds the local frame into the register. Built once per distinct key
/// ([`LindbladSpec::pair_ref`]); the action folds `K_nm` in on the fly
/// (one complex multiply per table coefficient, the same products the
/// fully precompiled per-pair tables held).
#[derive(Clone)]
struct PairTemplate {
    /// Number of qubits of the local frame (the union support of the pair).
    n_local: u8,
    /// `A_n†A_m = Σ_w γ_w W` (FP-noise zeros dropped), local words.
    adag_b: Vec<PauliTerm>,
    /// `(P_a, [(P_b, λ_a*·μ_b), …])` grouped by the left word, local words.
    sand: SandwichGroups,
    /// Support bit masks (`xbits | zbits` union) of `A_n` and `A_m`, local frame.
    left_mask: [Chunk; W_CHUNKS],
    right_mask: [Chunk; W_CHUNKS],
}

/// One compiled pair `(n ≤ m)`: its template and the start of its qubit
/// map (local index → register qubit, `n_local` bytes) in
/// `LindbladSpec::k_pair_qubits`.
#[derive(Clone, Copy)]
struct PairRef {
    n: u32,
    m: u32,
    template: u32,
    q_off: u32,
}

/// One jump operator `L_k` with rate `γ_k`. The `HermitianPauli` variant
/// is a fast path; `General` handles arbitrary complex Pauli sums.
#[derive(Clone)]
enum JumpKind {
    HermitianPauli {
        word: Word,
        /// Effective rate `γ_k · c²` for `L_k = c · P` (what the action reads).
        rate: f64,
        /// The real coefficient `c`, kept so [`LindbladSpec::set_jump_rates`]
        /// can recompute `rate` exactly as [`LindbladSpec::new`] did.
        coeff_re: f64,
    },
    General {
        terms: Vec<PauliTerm>,         // L = Σ_a λ_a P_a
        dagger_dagger: Vec<PauliTerm>, // L†L = Σ_c μ_c P_c  (μ_c ∈ ℝ)
        rate: f64,
    },
    /// One `(n, m)` pair of a Kossakowski-form dissipator
    /// `D*(O) = Σ_{n,m} K_nm (A_n† O A_m − ½ {A_n† A_m, O})`: the shared
    /// `K`-independent template (index into `LindbladSpec::k_templates`),
    /// the qubit map that embeds the template's local frame into the
    /// register (`n_local` bytes of `LindbladSpec::k_pair_qubits` from
    /// `q_off`) and `K_nm`. The action restricts the input word to the
    /// pair's qubits, runs the template's sandwich table — grouped by the
    /// left word `P_a`, so `P_a · p` is computed once per group and reused
    /// across its `P_b` partners (a win when `A_n` has several Pauli terms,
    /// e.g. the 2-local rank-2 tensors of dipolar relaxation) — and its
    /// anticommutator table `−K_nm · A_n†A_m` (the ½ cancels the
    /// anticommutator's 2) in that frame, folding `λ_a*·μ_b·K_nm` in on
    /// the fly, and embeds the output words back. Coefficients are complex
    /// — not Hermitian per pair; Hermiticity of the total action is
    /// restored by folding the conjugate `(m, n)` pair.
    KossakowskiPair {
        template: u32,
        q_off: u32,
        /// `true` for a folded off-diagonal pair (`n < m`): the both-sided
        /// sandwich doubles and takes the real part, the anticommutator
        /// uses the Hermitian sum `2·Re(K_nm·A_n†A_m)` and the one-sided
        /// path the anti-Hermitian difference `−2i·Im(K_nm·A_n†A_m)`.
        /// `false` on the diagonal (`K_nn · A_n†A_n`, original single-pair
        /// handling).
        off_diag: bool,
        k_nm: Complex<f64>,
    },
}

/// Expand `A†B = (Σ_a λ_a P_a)† (Σ_b μ_b P_b) = Σ_{a,b} λ_a* μ_b P_a P_b`
/// as a Pauli linear combination, dropping FP-noise zeros. For `A = B`
/// (the jump-operator `L†L`) the coefficients are real; in general they
/// are complex.
fn precompute_adag_b(a_terms: &[PauliTerm], b_terms: &[PauliTerm]) -> Vec<PauliTerm> {
    let zero = Complex::new(0.0, 0.0);
    let mut acc: FxHashMap<Word, Complex<f64>> = FxHashMap::default();
    for a in a_terms {
        for b in b_terms {
            let (word, phase) = pauli_mul(&a.word, &b.word);
            let coeff = a.coeff.conj() * b.coeff * phase_factor(phase);
            *acc.entry(word).or_insert(zero) += coeff;
        }
    }
    acc.into_iter()
        .filter(|(_, c)| c.norm() > 1e-14)
        .map(|(word, coeff)| PauliTerm { word, coeff })
        .collect()
}

/// Validate a Kossakowski pair matrix: `m_ops × m_ops` and Hermitian to
/// `1e-10 · max(max|K|, 1)`. A non-Hermitian K is not a valid GKSL pair
/// matrix and would produce a non-Hermiticity-preserving action.
fn check_k_matrix(
    k: &[Vec<Complex<f64>>],
    m_ops: usize,
    what_rows: &'static str,
    what_cols: &'static str,
) -> Result<(), Error> {
    if k.len() != m_ops {
        return Err(Error::LengthMismatch {
            what: what_rows,
            a: m_ops,
            b: k.len(),
        });
    }
    for row in k.iter() {
        if row.len() != m_ops {
            return Err(Error::LengthMismatch {
                what: what_cols,
                a: row.len(),
                b: m_ops,
            });
        }
    }
    let max_abs = k
        .iter()
        .flat_map(|row| row.iter().map(|c| c.norm()))
        .fold(0.0_f64, f64::max);
    let herm_tol = 1e-10 * max_abs.max(1.0);
    for (n, row_n) in k.iter().enumerate() {
        for (m, k_nm) in row_n.iter().enumerate().skip(n) {
            if (k_nm - k[m][n].conj()).norm() > herm_tol {
                return Err(Error::KMatrixNotHermitian { n, m });
            }
        }
    }
    Ok(())
}

/// One stored entry `(row, col, K[row][col])` of the Kossakowski matrix.
pub type KEntry = (u32, u32, Complex<f64>);

/// Bring `(row, col, value)` triples over an `m_ops`-operator table into the
/// stored form: indices checked, sorted by `(row, col)`, duplicates summed
/// (the COO convention), exact zeros dropped. No Hermiticity check here
/// ([`hermitian_completion`] does it on the merged list).
fn ingest_k_entries(
    entries: impl IntoIterator<Item = (usize, usize, Complex<f64>)>,
    m_ops: usize,
    what: &'static str,
) -> Result<Vec<KEntry>, Error> {
    let mut out: Vec<KEntry> = Vec::new();
    for (r, c, v) in entries {
        for &i in &[r, c] {
            if i >= m_ops {
                return Err(Error::KIndexOutOfRange {
                    what,
                    index: i,
                    m_ops,
                });
            }
        }
        out.push((r as u32, c as u32, v));
    }
    out.sort_by_key(|&(r, c, _)| (r, c));
    let mut merged: Vec<KEntry> = Vec::with_capacity(out.len());
    for e in out {
        if let Some(last) = merged.last_mut() {
            if last.0 == e.0 && last.1 == e.1 {
                last.2 += e.2;
                continue;
            }
        }
        merged.push(e);
    }
    merged.retain(|e| e.2.re != 0.0 || e.2.im != 0.0);
    Ok(merged)
}

/// The nonzero entries of a dense (already Hermitian-checked) `K`, offset
/// by `base` on both indices, in row-major order — the stored form of a
/// dense input.
fn dense_k_entries(k: &[Vec<Complex<f64>>], base: usize) -> impl Iterator<Item = KEntry> + '_ {
    k.iter().enumerate().flat_map(move |(n, row)| {
        row.iter().enumerate().filter_map(move |(m, &v)| {
            (v.re != 0.0 || v.im != 0.0).then_some(((base + n) as u32, (base + m) as u32, v))
        })
    })
}

/// Check sorted, deduplicated entries for Hermiticity to
/// `1e-10·max(max|K|, 1)` — the tolerance of [`check_k_matrix`] — and return
/// the Hermitian completion: every entry as stored plus the conjugate mirror
/// of each off-diagonal entry whose mirror was not given (a one-sided COO
/// input: "only one triangle" means "the other is its conjugate"). A
/// diagonal entry must be real to the tolerance; an off-diagonal entry
/// present in both triangles must match its mirror's conjugate. Sorted by
/// `(row, col)`.
fn hermitian_completion(entries: &[KEntry]) -> Result<Vec<KEntry>, Error> {
    let max_abs = entries
        .iter()
        .map(|e| e.2.norm())
        .fold(0.0_f64, f64::max);
    let tol = 1e-10 * max_abs.max(1.0);
    let mut out: Vec<KEntry> = Vec::with_capacity(entries.len());
    for &(r, c, v) in entries {
        if r == c {
            if (v - v.conj()).norm() > tol {
                return Err(Error::KMatrixNotHermitian {
                    n: r as usize,
                    m: c as usize,
                });
            }
            out.push((r, c, v));
            continue;
        }
        match entries.binary_search_by_key(&(c, r), |e| (e.0, e.1)) {
            Ok(j) => {
                if (v - entries[j].2.conj()).norm() > tol {
                    return Err(Error::KMatrixNotHermitian {
                        n: r.min(c) as usize,
                        m: r.max(c) as usize,
                    });
                }
                out.push((r, c, v));
            }
            Err(_) => {
                out.push((r, c, v));
                out.push((c, r, v.conj()));
            }
        }
    }
    out.sort_by_key(|&(r, c, _)| (r, c));
    Ok(out)
}

/// Precompiled Lindbladian. Constructed once from string-form Hamiltonian
/// terms + jump operators; reused across many calls to [`Self::action`],
/// [`Self::leakage`], [`Self::generator`]. `L*(p)` is recomputed on every
/// call rather than cached: for sparse-local Hamiltonians a per-word cache
/// costs more than the recompute (hash lookup ≳ recompute) and its several
/// KB per cached word dominate memory at large basis sizes.
pub struct LindbladSpec {
    n_qubits: usize,
    h_terms: Vec<HTerm>,
    j_kinds: Vec<JumpKind>,
    /// `h_support[q]` = indices of Hamiltonian terms acting on qubit `q`.
    h_support: Vec<Vec<u32>>,
    /// `j_support[q]` = indices of `j_kinds` entries whose support
    /// contains qubit `q` (jump operators and Kossakowski pairs alike).
    j_support: Vec<Vec<u32>>,
    /// Kossakowski operator table: `k_ops[i]` holds the Pauli linear
    /// combination of operator `A_i`, shared by every pair that
    /// references it.
    k_ops: Vec<Vec<PauliTerm>>,
    /// Per-qubit support of each `k_ops` entry (sorted qubit indices).
    k_op_support: Vec<Vec<u32>>,
    /// The compiled pairs `n ≤ m` of `k_ops` that are non-negligible in the
    /// current `K` (`|K_nm| > 1e-14·max|K|`), in `(n, m)` lexicographic
    /// order — the order the dense constructor always compiled them in.
    /// Pairs that are zero in `K` (the off-block entries of a
    /// block-diagonal `K`) get no record, so the memory is `O(nonzero
    /// pairs)`, never `O(M²)`; a record is a few words (template index,
    /// qubit map offset) — the Pauli tables live in the shared
    /// `k_templates`. Records survive a `set_kossakowski*` call while their
    /// pair stays non-negligible ([`Self::recompile_kossakowski`]).
    k_pairs: Vec<PairRef>,
    /// `K_nm` of each `k_pairs` entry (parallel to `k_pairs`).
    k_vals: Vec<Complex<f64>>,
    /// The distinct pair templates in use: one per operator-pattern
    /// relationship ([`TemplateKey`]) among the compiled pairs; `k_pairs`
    /// and the `KossakowskiPair` entries index into it.
    k_templates: Vec<PairTemplate>,
    /// Key → index into `k_templates`.
    k_template_index: FxHashMap<TemplateKey, u32>,
    /// Qubit maps of the compiled pairs, concatenated: `k_pairs[i]` owns
    /// `k_pair_qubits[q_off..q_off + n_local]` (local index → qubit).
    k_pair_qubits: Vec<u8>,
    /// Current Kossakowski matrix over the whole `k_ops` table in sparse
    /// form: every nonzero `(row, col, K[row][col])`, both triangles,
    /// Hermitian-completed, sorted by `(row, col)`. Source of truth for
    /// [`Self::kossakowski`] / [`Self::kossakowski_entries`]; the dense
    /// matrix is never stored.
    k_entries: Vec<KEntry>,
    /// User-facing GKSL rates `γ_k` of the jump operators given to
    /// [`Self::new`], in construction order. `j_kinds[..j_rates.len()]`
    /// are those jumps; every later entry is a `KossakowskiPair`.
    j_rates: Vec<f64>,
    /// The difference-word index of the locality-aware step
    /// ([`locality::TransferIndex`]), built on first use and dropped when
    /// the jump list or the compiled pair set changes.
    transfer: std::sync::OnceLock<locality::TransferIndex>,
    /// Per-word `(raw, diag)` of the full action — the column 1-norm and the
    /// diagonal the generator build needs — filled by the locality step's
    /// leakage pass and generator build, dropped on any coefficient change.
    rd_cache: std::sync::Mutex<FxHashMap<Word, (f64, f64)>>,
}

/// User-facing description of one jump operator: a complex Pauli linear
/// combination together with its rate.
#[derive(Clone, Debug)]
pub struct JumpInput {
    /// `(pauli_string, λ)` pairs forming `L_k = Σ_a λ_a P_a`.
    pub lincomb: Vec<(String, Complex<f64>)>,
    /// Non-negative GKSL rate `γ_k`.
    pub rate: f64,
}

impl LindbladSpec {
    /// Construct a Lindbladian spec from Hamiltonian terms and jump operators.
    ///
    /// `h_terms` are `(pauli_string, coefficient)` pairs forming the Hermitian
    /// Hamiltonian. Each jump operator is a complex Pauli linear combination;
    /// a length-1 jump with imaginary part `0` is routed to the Hermitian-Pauli
    /// fast path (with rate scaled by the squared real coefficient).
    pub fn new(
        n_qubits: usize,
        h_terms: &[(String, f64)],
        jumps: &[JumpInput],
    ) -> Result<Self, Error> {
        if n_qubits > MAX_QUBITS {
            return Err(Error::TooManyQubits { got: n_qubits });
        }

        let mut h_parsed: Vec<HTerm> = Vec::with_capacity(h_terms.len());
        let mut h_support_idx: Vec<Vec<u32>> = vec![Vec::new(); n_qubits];
        for (i, (s, c)) in h_terms.iter().enumerate() {
            let (word, support) = parse_pauli_string(s, n_qubits)?;
            for q in support {
                h_support_idx[q as usize].push(i as u32);
            }
            h_parsed.push(HTerm { word, coeff: *c });
        }

        let mut j_kinds: Vec<JumpKind> = Vec::with_capacity(jumps.len());
        let mut j_support_idx: Vec<Vec<u32>> = vec![Vec::new(); n_qubits];
        let mut j_rates: Vec<f64> = Vec::with_capacity(jumps.len());
        for (k, jump) in jumps.iter().enumerate() {
            if jump.rate < 0.0 {
                return Err(Error::NegativeRate {
                    index: k,
                    rate: jump.rate,
                });
            }
            if jump.lincomb.is_empty() {
                return Err(Error::EmptyLincomb { index: k });
            }
            j_rates.push(jump.rate);

            // Fast path: single-term, purely real → Hermitian Pauli.
            if jump.lincomb.len() == 1 && jump.lincomb[0].1.im == 0.0 {
                let (s, c) = &jump.lincomb[0];
                let (word, support) = parse_pauli_string(s, n_qubits)?;
                for q in support {
                    j_support_idx[q as usize].push(k as u32);
                }
                j_kinds.push(JumpKind::HermitianPauli {
                    word,
                    rate: jump.rate * c.re * c.re,
                    coeff_re: c.re,
                });
                continue;
            }

            // General path: parse all terms, precompute L†L, record union support.
            let mut terms: Vec<PauliTerm> = Vec::with_capacity(jump.lincomb.len());
            let mut union_support: std::collections::BTreeSet<u32> =
                std::collections::BTreeSet::new();
            for (s, c) in &jump.lincomb {
                let (word, support) = parse_pauli_string(s, n_qubits)?;
                for q in &support {
                    union_support.insert(*q);
                }
                terms.push(PauliTerm { word, coeff: *c });
            }
            for q in union_support {
                j_support_idx[q as usize].push(k as u32);
            }
            let dagger_dagger = precompute_adag_b(&terms, &terms);
            j_kinds.push(JumpKind::General {
                terms,
                dagger_dagger,
                rate: jump.rate,
            });
        }

        Ok(Self {
            n_qubits,
            h_terms: h_parsed,
            j_kinds,
            h_support: h_support_idx,
            j_support: j_support_idx,
            k_ops: Vec::new(),
            k_op_support: Vec::new(),
            k_pairs: Vec::new(),
            k_vals: Vec::new(),
            k_templates: Vec::new(),
            k_template_index: FxHashMap::default(),
            k_pair_qubits: Vec::new(),
            k_entries: Vec::new(),
            j_rates,
            transfer: std::sync::OnceLock::new(),
            rd_cache: std::sync::Mutex::new(FxHashMap::default()),
        })
    }

    /// Add a Kossakowski-form dissipator
    /// `D*(O) = Σ_{n,m} K_nm ( A_n† O A_m − ½ {A_n† A_m, O} )`
    /// specified by an operator list and a Hermitian pair matrix.
    ///
    /// `ops[i]` is the Pauli linear combination of `A_i` (e.g. `σ⁻_i`);
    /// `k` is the `M × M` pair matrix, row-major (`k[n][m] = K_nm`).
    /// Mathematically identical to eigendecomposing `K = V diag(γ) V†`
    /// and adding the jumps `L_ν = √γ_ν Σ_j V*_jν A_j`, but the action
    /// costs one term-pair product per nonzero `K_nm` instead of one per
    /// `(jump, term-pair)` — an `M`-fold saving for dense `K`.
    ///
    /// Validates Hermiticity of `k` (positive semi-definiteness is the
    /// caller's responsibility — checked in the Python layer, where an
    /// eigensolver is available). Entries with
    /// `|K_nm| ≤ 1e-14 · max|K|` are dropped at compile time.
    pub fn add_kossakowski(
        &mut self,
        ops: &[Vec<(String, Complex<f64>)>],
        k: &[Vec<Complex<f64>>],
    ) -> Result<(), Error> {
        let m_ops = ops.len();
        check_k_matrix(k, m_ops, "kossakowski ops and K rows", "kossakowski K row and ops")?;
        let base = self.push_k_ops(ops)?;
        let mut entries = self.k_entries.clone();
        entries.extend(dense_k_entries(k, base));
        self.recompile_kossakowski(entries)
    }

    /// [`Self::add_kossakowski`] with the pair matrix given as COO triples
    /// `(row, col, K[row][col])` over the indices `0..ops.len()` of this
    /// block (both triangles, or one triangle to be Hermitian-completed —
    /// see [`Self::set_kossakowski_sparse`]), so a block-sparse `K` over
    /// thousands of operators is never materialised.
    pub fn add_kossakowski_sparse(
        &mut self,
        ops: &[Vec<(String, Complex<f64>)>],
        entries: &[(usize, usize, Complex<f64>)],
    ) -> Result<(), Error> {
        let m_ops = ops.len();
        let block = ingest_k_entries(
            entries.iter().copied(),
            m_ops,
            "add_kossakowski_sparse",
        )?;
        hermitian_completion(&block)?;
        let base = self.push_k_ops(ops)?;
        let mut all = self.k_entries.clone();
        all.extend(
            block
                .into_iter()
                .map(|(r, c, v)| (r + base as u32, c + base as u32, v)),
        );
        self.recompile_kossakowski(all)
    }

    /// Parse an operator table into `k_ops` / `k_op_support` (appended;
    /// nothing is pushed when a string fails to parse). Returns the index
    /// of the first new operator.
    fn push_k_ops(&mut self, ops: &[Vec<(String, Complex<f64>)>]) -> Result<usize, Error> {
        let mut parsed: Vec<Vec<PauliTerm>> = Vec::with_capacity(ops.len());
        let mut supports: Vec<Vec<u32>> = Vec::with_capacity(ops.len());
        for (i, op) in ops.iter().enumerate() {
            if op.is_empty() {
                return Err(Error::EmptyLincomb { index: i });
            }
            let mut terms: Vec<PauliTerm> = Vec::with_capacity(op.len());
            let mut union_support: std::collections::BTreeSet<u32> =
                std::collections::BTreeSet::new();
            for (s, c) in op {
                let (word, support) = parse_pauli_string(s, self.n_qubits)?;
                for q in &support {
                    union_support.insert(*q);
                }
                terms.push(PauliTerm { word, coeff: *c });
            }
            supports.push(union_support.into_iter().collect());
            parsed.push(terms);
        }
        let base = self.k_ops.len();
        self.k_ops.extend(parsed);
        self.k_op_support.extend(supports);
        Ok(base)
    }

    /// The local frame of the pair `(n, m)`: the qubit map (local index →
    /// register qubit; qubits numbered by first appearance over the terms
    /// of `A_n`, then `A_m`, each term's qubits in increasing order) and
    /// the two operators relabelled into it. The relabelling is a qubit
    /// permutation, so every Pauli product and phase of the pair is the
    /// same in the local frame as in the register.
    fn pair_local_frame(&self, n: usize, m: usize) -> (Vec<u8>, Vec<PauliTerm>, Vec<PauliTerm>) {
        let mut qubits: Vec<u8> = Vec::new();
        let mut local_of = [u8::MAX; MAX_QUBITS];
        let n_qubits = self.n_qubits;
        let mut relabel = |terms: &[PauliTerm], support: &[u32]| -> Vec<PauliTerm> {
            terms
                .iter()
                .map(|t| {
                    let mut w = Word::new(n_qubits);
                    for &q in support {
                        let q = q as usize;
                        let (x, z) = (t.word.xbits[q], t.word.zbits[q]);
                        if x || z {
                            if local_of[q] == u8::MAX {
                                local_of[q] = qubits.len() as u8;
                                qubits.push(q as u8);
                            }
                            let li = local_of[q] as usize;
                            w.xbits.set(li, x);
                            w.zbits.set(li, z);
                        }
                    }
                    w.rehash();
                    PauliTerm { word: w, coeff: t.coeff }
                })
                .collect()
        };
        let left = relabel(&self.k_ops[n], &self.k_op_support[n]);
        let right = relabel(&self.k_ops[m], &self.k_op_support[m]);
        (qubits, left, right)
    }

    /// The template key of a pair given in its local frame: the terms of
    /// both operators, in order, with their exact coefficient bits.
    fn template_key(left: &[PauliTerm], right: &[PauliTerm]) -> TemplateKey {
        let local = |terms: &[PauliTerm]| -> Vec<LocalTerm> {
            terms
                .iter()
                .map(|t| LocalTerm {
                    x: t.word.xbits.data,
                    z: t.word.zbits.data,
                    re: t.coeff.re.to_bits(),
                    im: t.coeff.im.to_bits(),
                })
                .collect()
        };
        TemplateKey {
            left: local(left),
            right: local(right),
        }
    }

    /// The `K`-independent template of a pair given in its local frame:
    /// `A_n†A_m`, the sandwich table `λ_a*·μ_b` and the support masks — the
    /// per-pair precompilation of the dense constructor, now done once per
    /// distinct [`TemplateKey`].
    fn build_template(left: &[PauliTerm], right: &[PauliTerm], n_local: usize) -> PairTemplate {
        let support_mask = |terms: &[PauliTerm]| {
            let mut mask = [0 as Chunk; W_CHUNKS];
            for t in terms {
                for (i, slot) in mask.iter_mut().enumerate() {
                    *slot |= t.word.xbits.data[i] | t.word.zbits.data[i];
                }
            }
            mask
        };
        let adag_b = precompute_adag_b(left, right);
        let mut sand = Vec::with_capacity(left.len());
        for a in left {
            let rights = right
                .iter()
                .map(|b| (b.word, a.coeff.conj() * b.coeff))
                .collect();
            sand.push((a.word, rights));
        }
        PairTemplate {
            n_local: n_local as u8,
            adag_b,
            sand,
            left_mask: support_mask(left),
            right_mask: support_mask(right),
        }
    }

    /// The record of the pair `(n, m)`: its template (looked up by key,
    /// built when the key is new) and its qubit map, appended to
    /// `k_pair_qubits`.
    fn pair_ref(&mut self, n: usize, m: usize) -> PairRef {
        let (qubits, left, right) = self.pair_local_frame(n, m);
        let key = Self::template_key(&left, &right);
        let template = match self.k_template_index.get(&key) {
            Some(&t) => t,
            None => {
                let t = self.k_templates.len() as u32;
                self.k_templates
                    .push(Self::build_template(&left, &right, qubits.len()));
                self.k_template_index.insert(key, t);
                t
            }
        };
        let q_off = self.k_pair_qubits.len() as u32;
        self.k_pair_qubits.extend_from_slice(&qubits);
        PairRef {
            n: n as u32,
            m: m as u32,
            template,
            q_off,
        }
    }

    /// Drop the templates no compiled pair uses any more (a pattern that
    /// left `K`), renumbering the survivors, so that
    /// [`Self::num_pair_templates`] counts the distinct patterns of the
    /// current pairs and the template memory is bounded by them.
    fn prune_templates(&mut self) {
        let mut used = vec![false; self.k_templates.len()];
        for r in &self.k_pairs {
            used[r.template as usize] = true;
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new_index = vec![u32::MAX; self.k_templates.len()];
        let mut kept = Vec::with_capacity(used.iter().filter(|&&u| u).count());
        for (t, tpl) in std::mem::take(&mut self.k_templates).into_iter().enumerate() {
            if used[t] {
                new_index[t] = kept.len() as u32;
                kept.push(tpl);
            }
        }
        self.k_templates = kept;
        self.k_template_index.retain(|_, t| {
            let i = new_index[*t as usize];
            *t = i;
            i != u32::MAX
        });
        for r in &mut self.k_pairs {
            r.template = new_index[r.template as usize];
        }
    }

    /// Install a new Kossakowski matrix given as ingested entries
    /// ([`ingest_k_entries`] form, over the whole operator table): check
    /// Hermiticity and complete the mirrors, select the non-negligible
    /// upper-triangle pairs (`|K_nm| > 1e-14·max|K|`, `n ≤ m`) in `(n, m)`
    /// order, reuse the template of every pair that already has one and
    /// build the templates of the new pairs, then rebuild the
    /// `KossakowskiPair` entries. Nothing is modified when the check fails.
    ///
    /// The pair list, its order and every compiled coefficient are exactly
    /// what the dense constructor produced for the same `K`, so the action
    /// is bit-identical however `K` was given (dense, COO or blocks).
    fn recompile_kossakowski(&mut self, entries: Vec<KEntry>) -> Result<(), Error> {
        let full = hermitian_completion(&entries)?;
        let max_abs = full.iter().map(|e| e.2.norm()).fold(0.0_f64, f64::max);
        let pair_tol = 1e-14 * max_abs;
        let kept: Vec<KEntry> = full
            .iter()
            .filter(|e| e.0 <= e.1 && e.2.norm() > pair_tol)
            .copied()
            .collect();

        // Merge-walk the old records (sorted by (n, m)) against the kept
        // pairs (sorted the same way): reuse the record — its template and
        // qubit map, the map copied into the fresh arena — else build one.
        let old_pattern: Vec<(u32, u32, u32)> =
            self.k_pairs.iter().map(|r| (r.n, r.m, r.template)).collect();
        let old_qubits = std::mem::take(&mut self.k_pair_qubits);
        let mut old = std::mem::take(&mut self.k_pairs).into_iter().peekable();
        let mut pairs: Vec<PairRef> = Vec::with_capacity(kept.len());
        let mut vals: Vec<Complex<f64>> = Vec::with_capacity(kept.len());
        for &(n, m, v) in &kept {
            let key = (n, m);
            while old.peek().is_some_and(|r| (r.n, r.m) < key) {
                old.next();
            }
            let rec = if old.peek().is_some_and(|r| (r.n, r.m) == key) {
                let r = old.next().expect("peeked");
                let nl = self.k_templates[r.template as usize].n_local as usize;
                let q_off = self.k_pair_qubits.len() as u32;
                let start = r.q_off as usize;
                self.k_pair_qubits
                    .extend_from_slice(&old_qubits[start..start + nl]);
                PairRef { q_off, ..r }
            } else {
                self.pair_ref(n as usize, m as usize)
            };
            pairs.push(rec);
            vals.push(v);
        }
        self.k_pairs = pairs;
        self.k_vals = vals;
        self.k_entries = full;
        self.prune_templates();
        self.rebuild_kossakowski_pairs();
        // the raw / diag cache depends on `K`; the transfer index only on the pair
        // set and its templates (a rescaled pattern keeps it)
        self.clear_locality_cache();
        let same_pattern = old_pattern.len() == self.k_pairs.len()
            && old_pattern
                .iter()
                .zip(self.k_pairs.iter())
                .all(|(o, r)| *o == (r.n, r.m, r.template));
        if !same_pattern {
            self.clear_transfer_index();
        }
        Ok(())
    }

    /// Replace the Kossakowski matrix in place, keeping the operator basis
    /// `A_n` given to [`Self::add_kossakowski`].
    ///
    /// `k` is the new `M × M` Hermitian pair matrix over the whole operator
    /// table (`M = `[`Self::num_kossakowski_ops`]); positive
    /// semi-definiteness is the caller's responsibility, as at
    /// construction. `K` is folded into every precompiled
    /// `KossakowskiPair` table (`sand`, `dd`, `dd_anti`) and decides which
    /// pairs exist at all (entries below `1e-14 · max|K|` are dropped), so
    /// the setter rebuilds the pair list from the stored operator table
    /// with the very same loop the constructor runs: the result is
    /// bit-identical to a spec freshly built with `k`. The Pauli products
    /// of `A_n†A_m` and the sandwich word tables are `K`-independent and
    /// cached per non-negligible pair ([`PairTemplate`]), so while the
    /// zero pattern of `K` is unchanged (a drive rescaling a block-diagonal
    /// `K`) the rebuild is one complex multiply per stored coefficient
    /// (`O(Σ_{nonzero n≤m} |A_n|·|A_m|)` multiplies, no Pauli arithmetic,
    /// no hashing); pairs new to the pattern are templated on the spot and
    /// pairs that left it are dropped. The Hamiltonian and plain jump data
    /// are untouched. For a large block-sparse `K` prefer
    /// [`Self::set_kossakowski_sparse`] / [`Self::set_kossakowski_blocks`],
    /// which never see the `M × M` matrix.
    pub fn set_kossakowski(&mut self, k: &[Vec<Complex<f64>>]) -> Result<(), Error> {
        let m_ops = self.k_ops.len();
        if m_ops == 0 {
            return Err(Error::NoKossakowski);
        }
        check_k_matrix(
            k,
            m_ops,
            "set_kossakowski: num_kossakowski_ops vs K rows",
            "set_kossakowski: K row and num_kossakowski_ops",
        )?;
        let entries: Vec<KEntry> = dense_k_entries(k, 0).collect();
        self.recompile_kossakowski(entries)
    }

    /// [`Self::set_kossakowski`] with `K` given as COO triples
    /// `(row, col, K[row][col])` over the operator index
    /// `0..num_kossakowski_ops`, in any order; duplicate `(row, col)`
    /// triples are summed (the COO convention), exact zeros ignored.
    /// Hermiticity (to `1e-10·max(max|K|, 1)`) is checked on the merged
    /// entries: a diagonal entry must be real, an off-diagonal entry given
    /// in both triangles must equal its mirror's conjugate, and an
    /// off-diagonal entry given in one triangle only is Hermitian-completed
    /// (its mirror is taken to be the conjugate), so a single triangle is a
    /// valid input. The compiled dissipator is bit-identical to
    /// [`Self::set_kossakowski`] with the dense matrix these triples
    /// describe; the dense matrix is never formed (`O(nnz)` memory and
    /// `O(nnz log nnz)` time plus the templates of pairs new to the
    /// pattern).
    pub fn set_kossakowski_sparse(
        &mut self,
        entries: &[(usize, usize, Complex<f64>)],
    ) -> Result<(), Error> {
        let m_ops = self.k_ops.len();
        if m_ops == 0 {
            return Err(Error::NoKossakowski);
        }
        let ingested = ingest_k_entries(entries.iter().copied(), m_ops, "set_kossakowski_sparse")?;
        self.recompile_kossakowski(ingested)
    }

    /// [`Self::set_kossakowski`] with `K` given as blocks: `blocks[b] =
    /// (indices, K_b)` places the Hermitian `|indices| × |indices|` matrix
    /// `K_b` at the operator indices `indices` (row-major, `K[indices[i]]
    /// [indices[j]] += K_b[i][j]`). Blocks normally partition the operator
    /// table (a block-diagonal `K`); blocks sharing operators add up (`K =
    /// Σ_b P_b K_b P_bᵀ`, PSD when every `K_b` is). An index listed twice
    /// in one block, an index outside the table or a non-Hermitian `K_b`
    /// is an error and leaves the spec unchanged. Operators in no block
    /// have zero rows. Bit-identical to the dense setter with the assembled
    /// matrix, which is never formed.
    pub fn set_kossakowski_blocks(
        &mut self,
        blocks: &[(Vec<usize>, Vec<Vec<Complex<f64>>>)],
    ) -> Result<(), Error> {
        let m_ops = self.k_ops.len();
        if m_ops == 0 {
            return Err(Error::NoKossakowski);
        }
        let mut triples: Vec<(usize, usize, Complex<f64>)> = Vec::new();
        for (b, (idx, kb)) in blocks.iter().enumerate() {
            check_k_matrix(
                kb,
                idx.len(),
                "set_kossakowski_blocks: block indices vs K_block rows",
                "set_kossakowski_blocks: K_block row and block indices",
            )?;
            let mut seen = vec![false; m_ops];
            for &i in idx {
                if i >= m_ops {
                    return Err(Error::KIndexOutOfRange {
                        what: "set_kossakowski_blocks",
                        index: i,
                        m_ops,
                    });
                }
                if seen[i] {
                    return Err(Error::KBlockDuplicateIndex { block: b, index: i });
                }
                seen[i] = true;
            }
            for (i, &r) in idx.iter().enumerate() {
                for (j, &c) in idx.iter().enumerate() {
                    let v = kb[i][j];
                    if v.re != 0.0 || v.im != 0.0 {
                        triples.push((r, c, v));
                    }
                }
            }
        }
        let ingested = ingest_k_entries(triples, m_ops, "set_kossakowski_blocks")?;
        self.recompile_kossakowski(ingested)
    }

    /// Current Kossakowski matrix over the whole operator table, row-major
    /// (`M × M`, empty when no block was added). Materialises the dense
    /// matrix from the stored entries: `O(M²)` memory — use
    /// [`Self::kossakowski_entries`] for a large sparse `K`.
    pub fn kossakowski(&self) -> Vec<Vec<Complex<f64>>> {
        let m = self.k_ops.len();
        let mut out = vec![vec![Complex::new(0.0, 0.0); m]; m];
        for &(r, c, v) in &self.k_entries {
            out[r as usize][c as usize] = v;
        }
        out
    }

    /// Current Kossakowski matrix as sorted COO triples
    /// `(row, col, K[row][col])`: every nonzero entry, both triangles
    /// (Hermitian-completed), the stored form.
    pub fn kossakowski_entries(&self) -> &[KEntry] {
        &self.k_entries
    }

    /// Number of stored nonzero entries of the Kossakowski matrix (both
    /// triangles).
    pub fn num_kossakowski_nonzeros(&self) -> usize {
        self.k_entries.len()
    }

    /// Number of distinct pair templates in use: the operator-pattern
    /// relationships ([`TemplateKey`]: the Pauli structure of `(A_n, A_m)`
    /// up to a qubit relabelling) among the compiled pairs. Bounded by the
    /// number of distinct patterns, not by the number of pairs
    /// ([`Self::num_compiled_pairs`]); the compiled memory scales with it.
    pub fn num_pair_templates(&self) -> usize {
        self.k_templates.len()
    }

    /// Number of compiled pairs `n ≤ m` with `|K_nm| > 1e-14·max|K|`, one
    /// `KossakowskiPair` entry each — what `num_pair_templates` counted
    /// before the templates were shared across pairs.
    pub fn num_compiled_pairs(&self) -> usize {
        self.k_pairs.len()
    }

    /// Heap bytes of the Kossakowski compilation as `(templates, pairs,
    /// entries)`: the shared templates (Pauli tables, keys and the key
    /// index), the per-pair records (`k_pairs`, `k_vals`, the qubit maps,
    /// the `KossakowskiPair` entries and their `j_support` slots) and the
    /// stored `K` entries. Vector capacities, i.e. what was asked of the
    /// allocator; the operator table `k_ops` is not counted.
    pub fn kossakowski_memory_bytes(&self) -> (usize, usize, usize) {
        use std::mem::size_of;
        let mut templates = self.k_templates.capacity() * size_of::<PairTemplate>();
        for t in &self.k_templates {
            templates += t.adag_b.capacity() * size_of::<PauliTerm>();
            templates += t.sand.capacity() * size_of::<(Word, Vec<(Word, Complex<f64>)>)>();
            for (_, rights) in &t.sand {
                templates += rights.capacity() * size_of::<(Word, Complex<f64>)>();
            }
        }
        templates += self.k_template_index.capacity() * (size_of::<TemplateKey>() + size_of::<u32>());
        for key in self.k_template_index.keys() {
            templates += (key.left.capacity() + key.right.capacity()) * size_of::<LocalTerm>();
        }
        let start = self.j_rates.len();
        let mut pairs = self.k_pairs.capacity() * size_of::<PairRef>()
            + self.k_vals.capacity() * size_of::<Complex<f64>>()
            + self.k_pair_qubits.capacity();
        pairs += (self.j_kinds.capacity() - start) * size_of::<JumpKind>();
        pairs += self
            .j_support
            .iter()
            .map(|l| l.iter().filter(|&&i| (i as usize) >= start).count())
            .sum::<usize>()
            * size_of::<u32>();
        let entries = self.k_entries.capacity() * size_of::<KEntry>();
        (templates, pairs, entries)
    }

    /// Number `M` of Kossakowski operators `A_n` (0 without a block).
    pub fn num_kossakowski_ops(&self) -> usize {
        self.k_ops.len()
    }

    /// Drop every `KossakowskiPair` entry and rebuild them from the pair
    /// templates `k_pairs` and their values `k_vals` (the non-negligible
    /// upper-triangle pairs of the current `K`, in `(n, m)` order). The
    /// plain jumps of [`Self::new`] occupy `j_kinds[..j_rates.len()]` and
    /// are kept; `j_support` is pruned of the old pair indices
    /// (`candidate_terms` sorts and dedups, so the order of a support list
    /// is immaterial) and refilled.
    fn rebuild_kossakowski_pairs(&mut self) {
        let start = self.j_rates.len();
        self.j_kinds.truncate(start);
        for list in self.j_support.iter_mut() {
            list.retain(|&i| (i as usize) < start);
        }

        // One JumpKind entry per non-negligible pair, indexed by the union
        // support of A_n and A_m so the per-string candidate lookup only
        // visits pairs that can act nontrivially.
        // Each Hermitian-conjugate pair `(n,m)`+`(m,n)` is folded into one
        // upper-triangle entry (`n ≤ m`). Both sandwiches produce the same
        // output words with conjugate phase, and the final action keeps only
        // the real part, so `(n,m)` alone suffices if it doubles-and-takes-Re
        // — halving the pair count. The diagonal `(n,n)` is a single pair.
        // The `K`-dependent tables (`λ_a*·μ_b·K_nm`, `2·Re(K_nm γ_w)`,
        // `−2i·Im(K_nm γ_w)`) are no longer materialised per pair: the
        // action forms the same products from the shared template and
        // `K_nm` when it visits the pair.
        for (r, &k_nm) in self.k_pairs.iter().zip(self.k_vals.iter()) {
            let nl = self.k_templates[r.template as usize].n_local as usize;
            let idx = self.j_kinds.len() as u32;
            self.j_kinds.push(JumpKind::KossakowskiPair {
                template: r.template,
                q_off: r.q_off,
                off_diag: r.n != r.m,
                k_nm,
            });
            let start = r.q_off as usize;
            for &q in &self.k_pair_qubits[start..start + nl] {
                self.j_support[q as usize].push(idx);
            }
        }
    }

    /// Current GKSL rates `γ_k` of the jump operators given to
    /// [`Self::new`], in construction order (Kossakowski pairs excluded).
    pub fn jump_rates(&self) -> Vec<f64> {
        self.j_rates.clone()
    }

    /// Number of jump operators given to [`Self::new`] (the length of
    /// [`Self::jump_rates`]); unlike [`Self::num_jump_terms`] this does not
    /// count Kossakowski pairs.
    pub fn num_jump_ops(&self) -> usize {
        self.j_rates.len()
    }

    /// Overwrite the jump rates in place, keeping the jump operators, the
    /// support index and the Kossakowski block.
    ///
    /// `rates[k]` is the new non-negative `γ_k` of the `k`-th jump of
    /// [`Self::new`]. The rate is read directly by the action for a
    /// general jump, and as `γ_k · c²` for the Hermitian-Pauli fast path
    /// (recomputed here exactly as the constructor does), so this is an
    /// `O(num_jump_ops)` update and every later `action` / `leakage` /
    /// `generator` / `pc_step` is bit-identical to a spec freshly built
    /// with `rates`. A zero rate keeps its term, as the constructor does.
    pub fn set_jump_rates(&mut self, rates: &[f64]) -> Result<(), Error> {
        if self.j_rates.is_empty() {
            return Err(Error::NoJumpOps);
        }
        if rates.len() != self.j_rates.len() {
            return Err(Error::LengthMismatch {
                what: "set_jump_rates: num_jump_ops vs rates",
                a: self.j_rates.len(),
                b: rates.len(),
            });
        }
        for (k, &r) in rates.iter().enumerate() {
            if r < 0.0 {
                return Err(Error::NegativeRate { index: k, rate: r });
            }
        }
        for (k, &r) in rates.iter().enumerate() {
            self.j_rates[k] = r;
            match &mut self.j_kinds[k] {
                JumpKind::HermitianPauli { rate, coeff_re, .. } => {
                    *rate = r * *coeff_re * *coeff_re;
                }
                JumpKind::General { rate, .. } => {
                    *rate = r;
                }
                JumpKind::KossakowskiPair { .. } => {
                    unreachable!("Kossakowski pairs follow the plain jumps in j_kinds")
                }
            }
        }
        self.clear_locality_cache();
        Ok(())
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn num_h_terms(&self) -> usize {
        self.h_terms.len()
    }

    /// Current Hamiltonian coefficients `c_i` of `H = Σ_i c_i P_i`, in
    /// construction order.
    pub fn h_coeffs(&self) -> Vec<f64> {
        self.h_terms.iter().map(|h| h.coeff).collect()
    }

    /// Overwrite the Hamiltonian coefficients in place, keeping the Pauli
    /// words, the per-qubit support index and all jump / Kossakowski data.
    ///
    /// The coefficients are read directly by the commutator loop of the
    /// action (no table folds them in), so this is an `O(num_h_terms)` copy
    /// and every subsequent `action` / `leakage` / `generator` / `pc_step`
    /// is bit-identical to a spec freshly built with `coeffs`. The support
    /// index does not depend on coefficient values (zero coefficients are
    /// kept as terms by [`Self::new`] too), so it needs no rebuild.
    pub fn set_h_coeffs(&mut self, coeffs: &[f64]) -> Result<(), Error> {
        if coeffs.len() != self.h_terms.len() {
            return Err(Error::LengthMismatch {
                what: "set_h_coeffs: num_h_terms vs coeffs",
                a: self.h_terms.len(),
                b: coeffs.len(),
            });
        }
        for (h, &c) in self.h_terms.iter_mut().zip(coeffs) {
            h.coeff = c;
        }
        self.clear_locality_cache(); // raw / diag depend on the coefficients; the index does not
        Ok(())
    }

    pub fn num_jump_terms(&self) -> usize {
        self.j_kinds.len()
    }

    /// Apply `L*` to a single Pauli string `p`. Returns the output Pauli
    /// strings and their real coefficients (zero entries omitted).
    pub fn action(&self, p: &Word) -> Vec<(Word, f64)> {
        let mut out: FxHashMap<Word, f64> = FxHashMap::default();
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        self.accumulate_action(p, 1.0, &mut out, &mut s1, &mut s2);
        out.into_iter().filter(|(_, c)| *c != 0.0).collect()
    }

    /// Off-basis component of `L*( Σ_j coeffs[j] · basis[j] )`. Output
    /// strings that lie in `basis` or in `protected` are dropped.
    pub fn leakage(
        &self,
        basis: &[Word],
        coeffs: &[f64],
        protected: &[Word],
    ) -> Result<Vec<(Word, f64)>, Error> {
        self.leakage_with_prune(basis, coeffs, protected, usize::MAX, 0.0)
    }

    /// Like [`Self::leakage`], but caps the live off-basis leakage map to
    /// the *available room* `room = max_basis − basis.len()` — only the
    /// strings we could actually add to the basis are worth keeping. The
    /// cap is applied during accumulation (after each chunk), keeping the
    /// `room` largest-magnitude entries.
    ///
    /// Basis indices are processed in descending-`|c|` order so the
    /// running cap keeps the entries that are most likely to be the true
    /// largest contributors. When `max_basis` is large enough that
    /// `room ≥ all candidates`, nothing is dropped — the near-exact case.
    pub fn leakage_with_prune(
        &self,
        basis: &[Word],
        coeffs: &[f64],
        protected: &[Word],
        max_basis: usize,
        tau_add: f64,
    ) -> Result<Vec<(Word, f64)>, Error> {
        if basis.len() != coeffs.len() {
            return Err(Error::LengthMismatch {
                what: "basis and coeffs",
                a: basis.len(),
                b: coeffs.len(),
            });
        }
        // Hash-only membership tables: storing 8-byte `u64` keys instead
        // of 48-byte Words shrinks the in-basis structure ~6×, keeping it
        // in L3 (and often L2) at basis sizes where the full-Word version
        // would spill to DRAM.
        let in_basis: FxHashMap<u64, ()> = basis.iter().map(|w| (word_hash(w), ())).collect();
        let protected_set: FxHashMap<u64, ()> =
            protected.iter().map(|w| (word_hash(w), ())).collect();

        // Descending sort by |c|: process largest-magnitude contributors
        // first so the running room-cap keeps the right entries.
        let mut order: Vec<usize> = (0..basis.len()).collect();
        order.sort_by(|&a, &b| {
            coeffs[b]
                .abs()
                .partial_cmp(&coeffs[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        const CHUNK_SIZE: usize = 4096;
        let room = max_basis.saturating_sub(basis.len());
        let mut merged: FxHashMap<Word, f64> = FxHashMap::default();
        for chunk_indices in order.chunks(CHUNK_SIZE) {
            let local: Vec<Vec<(Word, f64)>> = chunk_indices
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
                        let mut out = Vec::with_capacity(terms.len());
                        for (w, v) in terms.iter() {
                            let h = word_hash(w);
                            if !in_basis.contains_key(&h) && !protected_set.contains_key(&h) {
                                out.push((*w, c * *v));
                            }
                        }
                        out
                    },
                )
                .collect();
            for v in local {
                for (k, val) in v {
                    *merged.entry(k).or_insert(0.0) += val;
                }
            }

            // Room-cap: keep only the `room` largest-magnitude entries.
            if merged.len() > room {
                if room == 0 {
                    merged.clear();
                } else {
                    let mut mags: Vec<f64> = merged.values().map(|v| v.abs()).collect();
                    let k = room.min(mags.len() - 1);
                    mags.select_nth_unstable_by(k, |a, b| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let cutoff = mags[k];
                    merged.retain(|_, &mut v| v.abs() >= cutoff);
                }
            }
        }
        // Rate-based admission: keep only candidates whose leakage rate
        // exceeds `tau_add`. `tau_add = 0` admits everything except exact
        // zeros.
        Ok(merged
            .into_iter()
            .filter(|(_, c)| c.abs() > tau_add)
            .collect())
    }

    /// Sparse generator matrix in COO form: returns `(row, col, val)`
    /// triplets. Row = output Pauli's position in `basis`; col = input
    /// Pauli's position. Output Paulis not in `basis` are silently dropped.
    ///
    /// Precondition: `basis` must not contain duplicate Pauli words
    /// (asserted in debug builds).
    pub fn generator(&self, basis: &[Word]) -> Vec<(usize, usize, f64)> {
        let index = build_basis_index(basis);

        // `compute_action_terms` returns a deduplicated `Vec<(Word, f64)>`,
        // so it can be scattered directly into COO triplets.
        let local: Vec<Vec<(usize, usize, f64)>> = basis
            .par_iter()
            .enumerate()
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
                |(s1, s2, lm), (col, p)| {
                    let terms = self.compute_action_terms(p, s1, s2, lm);
                    let mut out = Vec::with_capacity(terms.len());
                    for (w, v) in terms.iter() {
                        if let Some(&row) = index.get(w) {
                            out.push((row as usize, col, *v));
                        }
                    }
                    out
                },
            )
            .collect();

        // Pre-allocate the flat output to avoid sequential push reallocation.
        let total: usize = local.iter().map(|v| v.len()).sum();
        let mut flat = Vec::with_capacity(total);
        for v in local {
            flat.extend(v);
        }
        flat
    }

    /// Complex-coefficient variant of [`Self::leakage`]: off-basis
    /// component of `L*( Σ_j coeffs[j] · basis[j] )` with complex `coeffs`.
    pub fn leakage_complex(
        &self,
        basis: &[Word],
        coeffs: &[Complex<f64>],
        protected: &[Word],
    ) -> Result<Vec<(Word, Complex<f64>)>, Error> {
        if basis.len() != coeffs.len() {
            return Err(Error::LengthMismatch {
                what: "basis and coeffs",
                a: basis.len(),
                b: coeffs.len(),
            });
        }
        let in_basis: FxHashMap<u64, ()> = basis.iter().map(|w| (word_hash(w), ())).collect();
        let protected_set: FxHashMap<u64, ()> =
            protected.iter().map(|w| (word_hash(w), ())).collect();

        const CHUNK_SIZE: usize = 4096;
        let mut merged: FxHashMap<Word, Complex<f64>> = FxHashMap::default();
        for chunk_start in (0..basis.len()).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(basis.len());
            let chunk_basis = &basis[chunk_start..chunk_end];
            let chunk_coeffs = &coeffs[chunk_start..chunk_end];
            let local: Vec<Vec<(Word, Complex<f64>)>> = chunk_basis
                .par_iter()
                .zip(chunk_coeffs.par_iter())
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
                    |(s1, s2, lm), (p, &c)| {
                        let terms = self.compute_action_terms(p, s1, s2, lm);
                        let mut out = Vec::with_capacity(terms.len());
                        for (w, v) in terms.iter() {
                            let h = word_hash(w);
                            if !in_basis.contains_key(&h) && !protected_set.contains_key(&h) {
                                out.push((*w, c * *v));
                            }
                        }
                        out
                    },
                )
                .collect();
            for v in local {
                for (k, val) in v {
                    *merged.entry(k).or_insert(Complex::new(0.0, 0.0)) += val;
                }
            }
        }
        Ok(merged.into_iter().filter(|(_, c)| c.norm() > 0.0).collect())
    }

    /// One predictor-corrector step `O ← exp(dt·L*) O` in the adaptive
    /// real-coefficient Pauli basis: first-hop leakage admission, predictor
    /// exponential, second-hop admission from the predicted state, corrector
    /// exponential from the saved pre-step state, then truncation (prune +
    /// rank cap) per [`PcStepConfig`]. Exact in `dt` within the working
    /// basis — the only error is basis truncation.
    ///
    /// `protected` words are never dropped. All tuning knobs live in `cfg`.
    pub fn pc_step(
        &self,
        basis: &mut Vec<Word>,
        coeffs: &mut Vec<f64>,
        dt: f64,
        protected: &[Word],
        cfg: &PcStepConfig,
    ) -> Result<(), Error> {
        if let Some(n) = cfg.num_threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| Error::Internal(format!("rayon pool build: {e}")))?;
            pool.install(|| self.pc_step_inner(basis, coeffs, dt, protected, cfg))
        } else {
            self.pc_step_inner(basis, coeffs, dt, protected, cfg)
        }
    }

    /// Same as [`Self::pc_step`] but also returns a per-phase timing
    /// breakdown (microseconds), for profiling parallel scaling and hot
    /// spots.
    pub fn pc_step_timed(
        &self,
        basis: &mut Vec<Word>,
        coeffs: &mut Vec<f64>,
        dt: f64,
        protected: &[Word],
        cfg: &PcStepConfig,
    ) -> Result<PcStepTimings, Error> {
        if let Some(n) = cfg.num_threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| Error::Internal(format!("rayon pool build: {e}")))?;
            pool.install(|| self.pc_step_inner_timed(basis, coeffs, dt, protected, cfg))
        } else {
            self.pc_step_inner_timed(basis, coeffs, dt, protected, cfg)
        }
    }

    fn pc_step_inner_timed(
        &self,
        basis: &mut Vec<Word>,
        coeffs: &mut Vec<f64>,
        dt: f64,
        protected: &[Word],
        cfg: &PcStepConfig,
    ) -> Result<PcStepTimings, Error> {
        let PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            ..
        } = *cfg;
        let admit = admit_basis.unwrap_or(max_basis).max(max_basis);
        let tau_add = tau_add.unwrap_or(0.0);
        let mut t = PcStepTimings::default();
        if cfg.locality {
            self.pc_step_inner_locality(basis, coeffs, dt, protected, admit, tau_add, max_basis, drop_tol, Some(&mut t));
            return Ok(t);
        }

        let t0 = Instant::now();
        let leak = self.leakage_with_prune(basis, coeffs, protected, admit, tau_add)?;
        t.leakage1_us = t0.elapsed().as_micros() as u64;

        let t0 = Instant::now();
        add_leakage_capped(basis, coeffs, leak, admit);
        t.expand1_us = t0.elapsed().as_micros() as u64;

        // `coeffs` is read-only in the predictor, so it also serves as the
        // pre-step input for the corrector below — no clone needed.
        let t0 = Instant::now();
        let cache = mf_expm::mf_cache_full(self, basis);
        t.build1_us = t0.elapsed().as_micros() as u64;
        let coeffs_predict = mf_expm::apply_cache(&cache, dt, coeffs, drop_tol);
        drop(cache);
        t.expm1_us = t0.elapsed().as_micros() as u64;

        let t0 = Instant::now();
        let leak2 = self.leakage_with_prune(basis, &coeffs_predict, protected, admit, tau_add)?;
        t.leakage2_us = t0.elapsed().as_micros() as u64;
        drop(coeffs_predict);

        let t0 = Instant::now();
        add_leakage_capped(basis, coeffs, leak2, admit);
        t.expand2_us = t0.elapsed().as_micros() as u64;

        let t0 = Instant::now();
        let cache = mf_expm::mf_cache_full(self, basis);
        t.build2_us = t0.elapsed().as_micros() as u64;
        *coeffs = mf_expm::apply_cache(&cache, dt, coeffs, drop_tol);
        drop(cache);
        t.expm2_us = t0.elapsed().as_micros() as u64;

        let t0 = Instant::now();
        prune_basis(basis, coeffs, drop_tol, protected);
        cap_basis(basis, coeffs, max_basis, protected);
        t.prune_cap_us = t0.elapsed().as_micros() as u64;

        Ok(t)
    }

    fn pc_step_inner(
        &self,
        basis: &mut Vec<Word>,
        coeffs: &mut Vec<f64>,
        dt: f64,
        protected: &[Word],
        cfg: &PcStepConfig,
    ) -> Result<(), Error> {
        let PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            ..
        } = *cfg;
        // Admission bound: enrichment may grow the live basis to `admit`
        // >= `max_basis`; the final `cap_basis` then keeps the top-
        // `max_basis` strings by evolved |coeff| over the whole union
        // (retained + admitted) — rank displacement. With `admit_basis =
        // None` admission is bounded by `max_basis` itself, `cap_basis` is
        // a no-op, and membership turnover requires `drop_tol > 0`.
        let admit = admit_basis.unwrap_or(max_basis).max(max_basis);
        let tau_add = tau_add.unwrap_or(0.0);
        if cfg.locality {
            self.pc_step_inner_locality(basis, coeffs, dt, protected, admit, tau_add, max_basis, drop_tol, None);
            return Ok(());
        }
        // 1. First-hop expansion. After this, `coeffs` contains the pre-step
        // coefficients followed by zeros for the newly-added leakage strings.
        // We rely on `coeffs` itself as the pre-step buffer for the corrector
        // — no `.clone()` is needed because `expm_step` only borrows it.
        let leak = self.leakage_with_prune(basis, coeffs, protected, admit, tau_add)?;
        add_leakage_capped(basis, coeffs, leak, admit);
        // 2. Predictor: `expm_step` reads `coeffs` immutably and returns a
        // new owned vector with the predicted state.
        let coeffs_predict = self.expm_step(basis, dt, coeffs, drop_tol);
        // 3. Second-hop expansion from the predicted state. After leakage2
        // we no longer need `coeffs_predict`. Extend `coeffs` with zeros for
        // any newly-added second-hop strings so it remains a valid input
        // (pre-step state) for the corrector.
        let leak2 = self.leakage_with_prune(basis, &coeffs_predict, protected, admit, tau_add)?;
        drop(coeffs_predict);
        add_leakage_capped(basis, coeffs, leak2, admit);
        // 4. Corrector: redo from pre-step state on the doubly-enlarged basis.
        *coeffs = self.expm_step(basis, dt, coeffs, drop_tol);
        // 5. Prune basis entries below `drop_tol` (protected words never dropped).
        prune_basis(basis, coeffs, drop_tol, protected);
        cap_basis(basis, coeffs, max_basis, protected);
        Ok(())
    }

    /// The locality-aware step ([`locality`]): the same five phases as
    /// [`Self::pc_step_inner`] with the sharded leakage merge and the
    /// `room == 0` early return in the leakage passes, the transfer-index
    /// generator build in the exponentials and the predictor's generator
    /// reused by the corrector when the second pass admitted nothing.
    #[allow(clippy::too_many_arguments)]
    fn pc_step_inner_locality(
        &self,
        basis: &mut Vec<Word>,
        coeffs: &mut Vec<f64>,
        dt: f64,
        protected: &[Word],
        admit: usize,
        tau_add: f64,
        max_basis: usize,
        drop_tol: f64,
        mut timings: Option<&mut PcStepTimings>,
    ) {
        let mut prof = locality::LocalityProfile::default();
        let want = timings.is_some();
        let mut lap = |slot: fn(&mut PcStepTimings) -> &mut u64, t0: Instant| {
            if let Some(t) = timings.as_deref_mut() {
                *slot(t) = t0.elapsed().as_micros() as u64;
            }
        };
        let t0 = Instant::now();
        let leak = self.leakage_locality(basis, coeffs, protected, admit, tau_add, if want { Some(&mut prof) } else { None });
        lap(|t| &mut t.leakage1_us, t0);
        let t0 = Instant::now();
        add_leakage_capped(basis, coeffs, leak, admit);
        lap(|t| &mut t.expand1_us, t0);
        let t0 = Instant::now();
        let cache = self.mf_cache_local(basis, if want { Some(&mut prof) } else { None });
        lap(|t| &mut t.build1_us, t0);
        let coeffs_predict = mf_expm::apply_cache(&cache, dt, coeffs, drop_tol);
        lap(|t| &mut t.expm1_us, t0);
        let t0 = Instant::now();
        let n1 = basis.len();
        let leak2 = self.leakage_locality(basis, &coeffs_predict, protected, admit, tau_add, None);
        drop(coeffs_predict);
        lap(|t| &mut t.leakage2_us, t0);
        let t0 = Instant::now();
        add_leakage_capped(basis, coeffs, leak2, admit);
        lap(|t| &mut t.expand2_us, t0);
        let t0 = Instant::now();
        let cache = if basis.len() == n1 { cache } else { self.mf_cache_local(basis, None) };
        lap(|t| &mut t.build2_us, t0);
        *coeffs = mf_expm::apply_cache(&cache, dt, coeffs, drop_tol);
        lap(|t| &mut t.expm2_us, t0);
        let t0 = Instant::now();
        prune_basis(basis, coeffs, drop_tol, protected);
        cap_basis(basis, coeffs, max_basis, protected);
        lap(|t| &mut t.prune_cap_us, t0);
        if let Some(t) = timings.as_deref_mut() {
            t.locality = prof;
        }
    }

    /// Compute `exp(dt · M) · b` for the in-basis-restricted generator
    /// `M`, matrix-free, via `quspin-expm` (see [`mf_expm`]).
    fn expm_step(&self, basis: &[Word], dt: f64, b: &[f64], drop_tol: f64) -> Vec<f64> {
        mf_expm::expm_apply_mf(self, basis, dt, b, drop_tol)
    }

    /// Compute the unscaled list of `(output, coefficient)` pairs that
    /// `L*(p)` contributes (without the input coefficient).
    pub(crate) fn compute_action_terms(
        &self,
        p: &Word,
        scratch_support: &mut Vec<u32>,
        scratch_cands: &mut Vec<u32>,
        scratch_local: &mut FxHashMap<Word, Complex<f64>>,
    ) -> Vec<(Word, f64)> {
        word_support(p, scratch_support);
        let zero = Complex::new(0.0, 0.0);
        scratch_local.clear();
        let local = scratch_local;

        // ── i [H, p] ─────────────────────────────────────────────────
        candidate_terms(scratch_support, &self.h_support, scratch_cands);
        for &i in scratch_cands.iter() {
            let h = &self.h_terms[i as usize];
            let (r, eps) = comm_product(&h.word, p);
            if eps != 0.0 {
                *local.entry(r).or_insert(zero) += Complex::new(h.coeff * eps, 0.0);
            }
        }

        // ── dissipator ───────────────────────────────────────────────
        candidate_terms(scratch_support, &self.j_support, scratch_cands);
        for &k in scratch_cands.iter() {
            match &self.j_kinds[k as usize] {
                JumpKind::HermitianPauli { word, rate, .. } => {
                    if anti_commutes(word, p) {
                        *local.entry(*p).or_insert(zero) += Complex::new(-2.0 * *rate, 0.0);
                    }
                }
                JumpKind::General {
                    terms,
                    dagger_dagger,
                    rate,
                } => {
                    let rate_c = Complex::new(*rate, 0.0);
                    // Sandwich: γ Σ_{a,b} λ_a* λ_b P_a p P_b.
                    for a in terms {
                        let (r_ap, phi1) = pauli_mul(&a.word, p);
                        for b in terms {
                            let (s, phi2) = pauli_mul(&r_ap, &b.word);
                            let coeff =
                                a.coeff.conj() * b.coeff * phase_factor(phi1 + phi2) * rate_c;
                            *local.entry(s).or_insert(zero) += coeff;
                        }
                    }
                    // -1/2 γ {L†L, p}. For Pauli words P_c and p,
                    // {P_c, p} = 2·sign·R if they commute (P_c·p = sign·R),
                    //         = 0          if they anti-commute.
                    for c_term in dagger_dagger {
                        let (r, phase) = pauli_mul(&c_term.word, p);
                        if phase & 1 == 0 {
                            let sign = if phase == 0 { 1.0 } else { -1.0 };
                            let coeff = -c_term.coeff * rate_c * Complex::new(sign, 0.0);
                            *local.entry(r).or_insert(zero) += coeff;
                        }
                    }
                }
                JumpKind::KossakowskiPair {
                    template,
                    q_off,
                    off_diag,
                    k_nm,
                } => {
                    // The pair's tables live in a shared template over the
                    // pair's local frame: restrict `p` to the pair's qubits,
                    // run the tables there and embed the outputs back. The
                    // template's words are the identity outside those qubits,
                    // so every product and phase below is what the register-
                    // frame product would give, and `p`'s other bits pass
                    // through unchanged. `K_nm` is folded in exactly as the
                    // per-pair precompilation did (`c0 * K_nm`, `K_nm * γ_w`
                    // with the same drop thresholds), so the result is
                    // bit-identical to it.
                    let t = &self.k_templates[*template as usize];
                    let nl = t.n_local as usize;
                    let start = *q_off as usize;
                    let qs = &self.k_pair_qubits[start..start + nl];
                    // `p` on the pair's qubits in the local frame, and the
                    // register mask of those qubits (raw chunk ops: the
                    // storage is Lsb0, bit `q` of chunk `q / CHUNK_BITS`).
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
                    // One-sided fast path: when `p` is disjoint from one of
                    // the two operators, the pair contribution collapses to a
                    // commutator. For a diagonal pair with `D = K·A_n†A_m`:
                    //   p disjoint from A_n (left):  C = −½ [D, p]
                    //   p disjoint from A_m (right): C = +½ [D, p]
                    // For a folded off-diagonal pair the two conjugate
                    // one-sided contributions combine into `±½ [F, p]` with
                    // the anti-Hermitian `F = −2i·Im(K_nm A_n†A_m)` and the
                    // opposite sign:
                    //   p hits A_m only (right): C = +½ [F, p]
                    //   p hits A_n only (left):  C = −½ [F, p]
                    // With `[P_c, p] = −i·eps·out` from `comm_product`, the
                    // term coefficient is `∓ t_c · (i/2) · eps`.
                    let hits = |mask: &[Chunk; W_CHUNKS]| {
                        (0..W_CHUNKS)
                            .any(|i| mask[i] & (p_loc.xbits.data[i] | p_loc.zbits.data[i]) != 0)
                    };
                    let (hit_l, hit_r) = (hits(&t.left_mask), hits(&t.right_mask));
                    if !(hit_l && hit_r) {
                        // exactly one side hit (pair is a candidate, so at
                        // least one side overlaps). `off_diag` flips the sign
                        // and uses the anti-Hermitian part instead of `dd`.
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
                        for term in &t.adag_b {
                            let c = *k_nm * term.coeff;
                            // the term's coefficient in `dd_anti` (off-diagonal:
                            // −2i·Im, dropped below 1e-14) or `dd` (diagonal: K_nn γ_w)
                            let coeff = if *off_diag {
                                if c.im.abs() > 1e-14 {
                                    Complex::new(0.0, -2.0 * c.im)
                                } else {
                                    continue;
                                }
                            } else {
                                c
                            };
                            let (out, eps) = comm_product_local(&term.word, &p_loc);
                            if eps != 0.0 {
                                *local.entry(embed(&out)).or_insert(zero) += coeff * half_i * eps;
                            }
                        }
                        continue;
                    }
                    // Both sides hit: full sandwich + anticommutator. Group
                    // by the left word so `P_a · p` is computed once per
                    // distinct `P_a` and reused across all its `P_b` partners.
                    // For a folded off-diagonal pair the sandwich is doubled
                    // and its real part taken (the conjugate `(m,n)` pair
                    // supplies the other half); the anticommutator uses the
                    // Hermitian `dd = 2·Re(K_nm A_n†A_m)`.
                    for (wa, rights) in &t.sand {
                        let (r_ap, phi1) = pauli_mul_local(wa, &p_loc);
                        for (wb, c0) in rights {
                            let (s, phi2) = pauli_mul_local(&r_ap, wb);
                            let v = (c0 * k_nm) * phase_factor(phi1 + phi2);
                            let contrib = if *off_diag {
                                Complex::new(2.0 * v.re, 0.0)
                            } else {
                                v
                            };
                            *local.entry(embed(&s)).or_insert(zero) += contrib;
                        }
                    }
                    for term in &t.adag_b {
                        let c = *k_nm * term.coeff;
                        // the term's coefficient in `dd` (off-diagonal: 2·Re,
                        // dropped below 1e-14; diagonal: K_nn γ_w)
                        let coeff = if *off_diag {
                            if c.re.abs() > 1e-14 {
                                Complex::new(2.0 * c.re, 0.0)
                            } else {
                                continue;
                            }
                        } else {
                            c
                        };
                        let (r, phase) = pauli_mul_local(&term.word, &p_loc);
                        if phase & 1 == 0 {
                            let sign = if phase == 0 { 1.0 } else { -1.0 };
                            *local.entry(embed(&r)).or_insert(zero) -=
                                coeff * Complex::new(sign, 0.0);
                        }
                    }
                }
            }
        }

        // L* preserves Hermiticity; imaginary parts must cancel to FP noise.
        // `drain()` empties `scratch_local` so its allocation can be reused
        // by the next call on the same thread (`Vec` keeps capacity).
        local
            .drain()
            .filter_map(|(w, c)| {
                debug_assert!(
                    c.im.abs() < 1e-9,
                    "L*(p) produced non-real coefficient {c}; bug in dissipator"
                );
                if c.re == 0.0 { None } else { Some((w, c.re)) }
            })
            .collect()
    }

    /// Accumulate `scale · L*(p)` into `out`.
    fn accumulate_action(
        &self,
        p: &Word,
        scale: f64,
        out: &mut FxHashMap<Word, f64>,
        scratch_support: &mut Vec<u32>,
        scratch_cands: &mut Vec<u32>,
    ) {
        let mut scratch_local = FxHashMap::default();
        let terms =
            self.compute_action_terms(p, scratch_support, scratch_cands, &mut scratch_local);
        for (w, c) in terms.iter() {
            *out.entry(*w).or_insert(0.0) += scale * c;
        }
    }
}

/// Compact 64-bit hash of a [`Word`], used as the key in cache-friendly
/// membership tables: an `FxHashMap<u64, ()>` over the basis has a working
/// set ~6× smaller than `FxHashMap<Word, ()>`. The hash mixes the word's
/// cached hash once through `FxHasher` and never touches the 32-byte
/// payload.
#[inline(always)]
pub(crate) fn word_hash(w: &Word) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = fxhash::FxHasher::default();
    w.hash(&mut h);
    h.finish()
}

/// Compact `basis` / `coeffs` in place: drop entries whose absolute
/// coefficient is below `drop_tol` unless the word appears in `protected`.
/// No-op when `drop_tol ≤ 0`.
pub(crate) fn prune_basis(basis: &mut Vec<Word>, coeffs: &mut Vec<f64>, drop_tol: f64, protected: &[Word]) {
    if drop_tol <= 0.0 {
        return;
    }
    debug_assert_eq!(basis.len(), coeffs.len());
    let protected_set: FxHashSet<&Word> = protected.iter().collect();
    let mut write = 0;
    for read in 0..basis.len() {
        if coeffs[read].abs() >= drop_tol || protected_set.contains(&basis[read]) {
            if write != read {
                basis.swap(write, read);
                coeffs.swap(write, read);
            }
            write += 1;
        }
    }
    basis.truncate(write);
    coeffs.truncate(write);
}

/// Global max-basis cap (PauliStrings.jl-style top-M trim): keep only the
/// `max_basis` largest-|coeff| terms (protected strings always kept),
/// dropping the rest. Rank-based total-basis bound; dual of `drop_tol`.
/// A `max_basis` large enough to cover the whole basis is a no-op.
pub(crate) fn cap_basis(basis: &mut Vec<Word>, coeffs: &mut Vec<f64>, max_basis: usize, protected: &[Word]) {
    if basis.len() <= max_basis {
        return;
    }
    let protected_set: FxHashSet<&Word> = protected.iter().collect();
    let n_prot = basis.iter().filter(|w| protected_set.contains(w)).count();
    let slots = max_basis.saturating_sub(n_prot);
    let mut mags: Vec<f64> = basis
        .iter()
        .zip(coeffs.iter())
        .filter(|(w, _)| !protected_set.contains(w))
        .map(|(_, c)| c.abs())
        .collect();
    let cutoff = if slots == 0 {
        f64::INFINITY
    } else if slots >= mags.len() {
        return;
    } else {
        let k = slots - 1;
        mags.select_nth_unstable_by(k, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        mags[k]
    };
    let mut write = 0;
    for read in 0..basis.len() {
        if protected_set.contains(&basis[read]) || coeffs[read].abs() >= cutoff {
            if write != read {
                basis.swap(write, read);
                coeffs.swap(write, read);
            }
            write += 1;
        }
    }
    basis.truncate(write);
    coeffs.truncate(write);
}

/// Add the largest leakage strings to the basis, up to the available room
/// `room = max_basis − basis.len()` — so the in-step basis (hence the
/// expm/leakage peak memory) never exceeds `max_basis`. New strings get
/// coefficient 0; the surrounding expm fills them. No magnitude filter: the
/// top-`room` by `|leakage|` are added (a large `max_basis` adds them all).
pub(crate) fn add_leakage_capped(
    basis: &mut Vec<Word>,
    coeffs: &mut Vec<f64>,
    mut leak: Vec<(Word, f64)>,
    max_basis: usize,
) {
    let room = max_basis.saturating_sub(basis.len());
    if leak.len() > room {
        if room > 0 {
            leak.select_nth_unstable_by(room - 1, |a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        leak.truncate(room);
    }
    for (w, _) in leak {
        basis.push(w);
        coeffs.push(0.0);
    }
}

/// Build a `word → row` map for a basis assumed to contain unique Pauli
/// words; debug-asserts the uniqueness invariant.
pub fn build_basis_index(basis: &[Word]) -> FxHashMap<Word, u32> {
    let mut index: FxHashMap<Word, u32> = FxHashMap::default();
    for (i, w) in basis.iter().enumerate() {
        let prev = index.insert(*w, i as u32);
        debug_assert!(
            prev.is_none(),
            "basis contains duplicate Pauli word at positions {} and {}",
            prev.unwrap(),
            i,
        );
    }
    index
}

/// Union of `index[q]` for each `q ∈ p_support`, deduped.
#[inline]
pub(crate) fn candidate_terms(p_support: &[u32], index: &[Vec<u32>], scratch: &mut Vec<u32>) {
    scratch.clear();
    for &q in p_support {
        scratch.extend_from_slice(&index[q as usize]);
    }
    scratch.sort_unstable();
    scratch.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only full-space complex predictor-corrector step, UNTRUNCATED:
    /// adds every nonzero leakage string (two hops) and applies the exact
    /// in-basis exponential to the complex coefficient vector. Reference
    /// bridge between the real `pc_step` and the orbit-rep path.
    fn pc_step_complex_full(
        spec: &LindbladSpec,
        basis: &mut Vec<Word>,
        coeffs: &mut Vec<Complex<f64>>,
        dt: f64,
    ) {
        let protected: Vec<Word> = Vec::new();
        let leak = spec.leakage_complex(basis, coeffs, &protected).unwrap();
        for (w, v) in leak {
            if v.norm() > 0.0 {
                basis.push(w);
                coeffs.push(Complex::new(0.0, 0.0));
            }
        }
        let predict = mf_expm::expm_apply_mf_cxvec(spec, basis, dt, coeffs, 0.0);
        let leak2 = spec.leakage_complex(basis, &predict, &protected).unwrap();
        drop(predict);
        for (w, v) in leak2 {
            if v.norm() > 0.0 {
                basis.push(w);
                coeffs.push(Complex::new(0.0, 0.0));
            }
        }
        *coeffs = mf_expm::expm_apply_mf_cxvec(spec, basis, dt, coeffs, 0.0);
    }

    fn sigma_minus_lincomb(site: usize, n: usize) -> Vec<(String, Complex<f64>)> {
        let mut x = vec!['I'; n];
        x[site] = 'X';
        let mut y = vec!['I'; n];
        y[site] = 'Y';
        vec![
            (x.into_iter().collect(), Complex::new(0.5, 0.0)),
            (y.into_iter().collect(), Complex::new(0.0, -0.5)),
        ]
    }

    /// Collect `spec.action` into a map for comparison.
    fn action_map(spec: &LindbladSpec, p: &Word) -> FxHashMap<Word, f64> {
        spec.action(p).into_iter().collect()
    }

    fn assert_actions_match(a: &FxHashMap<Word, f64>, b: &FxHashMap<Word, f64>, tol: f64) {
        for (w, va) in a {
            let vb = b.get(w).copied().unwrap_or(0.0);
            assert!(
                (va - vb).abs() < tol,
                "coeff mismatch at {w:?}: {va} vs {vb}"
            );
        }
        for (w, vb) in b {
            assert!(
                a.contains_key(w) || vb.abs() < tol,
                "extra word {w:?} with coeff {vb}"
            );
        }
    }

    /// A single-operator Kossakowski block `K = [[γ]]`, `A = σ⁻`, equals
    /// the ordinary jump `L = σ⁻` with rate `γ`.
    #[test]
    fn kossakowski_single_op_matches_jump() {
        let n = 2;
        let gamma = 0.7;
        let jump = JumpInput {
            lincomb: sigma_minus_lincomb(0, n),
            rate: gamma,
        };
        let spec_jump = LindbladSpec::new(n, &[], &[jump]).unwrap();
        let mut spec_k = LindbladSpec::new(n, &[], &[]).unwrap();
        spec_k
            .add_kossakowski(
                &[sigma_minus_lincomb(0, n)],
                &[vec![Complex::new(gamma, 0.0)]],
            )
            .unwrap();
        for s in ["ZI", "XI", "YI", "ZZ", "XY"] {
            let (p, _) = parse_pauli_string(s, n).unwrap();
            assert_actions_match(&action_map(&spec_jump, &p), &action_map(&spec_k, &p), 1e-12);
        }
    }

    /// Two-site collective decay: the Kossakowski form with
    /// `K = [[1.0, 0.6], [0.6, 1.0]]` equals the eigenmode jumps
    /// `L_± = √γ_± (σ⁻_0 ± σ⁻_1)/√2` with `γ_± = 1 ± 0.6`.
    #[test]
    fn kossakowski_pair_matches_eigenmode_jumps() {
        let n = 2;
        let (g_plus, g_minus) = (1.6, 0.4);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let mk_jump = |sign: f64, rate: f64| {
            let mut lin = Vec::new();
            for (site, s) in [(0usize, 1.0), (1usize, sign)] {
                for (p, c) in sigma_minus_lincomb(site, n) {
                    lin.push((p, c * Complex::new(s * inv_sqrt2, 0.0)));
                }
            }
            JumpInput { lincomb: lin, rate }
        };
        let spec_jump =
            LindbladSpec::new(n, &[], &[mk_jump(1.0, g_plus), mk_jump(-1.0, g_minus)]).unwrap();
        let mut spec_k = LindbladSpec::new(n, &[], &[]).unwrap();
        let k = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(0.6, 0.0)],
            vec![Complex::new(0.6, 0.0), Complex::new(1.0, 0.0)],
        ];
        spec_k
            .add_kossakowski(&[sigma_minus_lincomb(0, n), sigma_minus_lincomb(1, n)], &k)
            .unwrap();
        for s in ["ZI", "IZ", "XX", "YY", "XY", "ZZ", "XI"] {
            let (p, _) = parse_pauli_string(s, n).unwrap();
            assert_actions_match(&action_map(&spec_jump, &p), &action_map(&spec_k, &p), 1e-12);
        }
    }

    /// Non-Hermitian K is rejected.
    #[test]
    fn kossakowski_rejects_non_hermitian_k() {
        let n = 2;
        let mut spec = LindbladSpec::new(n, &[], &[]).unwrap();
        let k = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(0.6, 0.1)],
            vec![Complex::new(0.6, 0.1), Complex::new(1.0, 0.0)], // should be conj
        ];
        let err = spec.add_kossakowski(&[sigma_minus_lincomb(0, n), sigma_minus_lincomb(1, n)], &k);
        assert!(matches!(err, Err(Error::KMatrixNotHermitian { .. })));
    }

    /// Exact bit-level equality of two action maps (same words, same f64s).
    fn assert_actions_identical(a: &FxHashMap<Word, f64>, b: &FxHashMap<Word, f64>) {
        assert_eq!(a.len(), b.len(), "different numbers of output words");
        for (w, va) in a {
            let vb = b.get(w).expect("word missing from the other action");
            assert!(va.to_bits() == vb.to_bits(), "{va} != {vb} (bitwise)");
        }
    }

    /// Deterministic complex Hermitian PSD `M × M` matrix `A A†` with a
    /// seed-dependent `A` (no RNG dependency in the crate).
    fn psd_matrix(m: usize, seed: u64) -> Vec<Vec<Complex<f64>>> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let a: Vec<Vec<Complex<f64>>> = (0..m)
            .map(|_| (0..m).map(|_| Complex::new(next(), next())).collect())
            .collect();
        (0..m)
            .map(|i| {
                (0..m)
                    .map(|j| (0..m).map(|l| a[i][l] * a[j][l].conj()).sum::<Complex<f64>>())
                    .collect()
            })
            .collect()
    }

    fn setter_fixture(
        n: usize,
        rates: &[f64],
        k: &[Vec<Complex<f64>>],
    ) -> LindbladSpec {
        let h: Vec<(String, f64)> = (0..n)
            .map(|q| {
                let mut s = vec!['Z'; n];
                s[q] = 'X';
                (s.into_iter().collect(), 0.3 + 0.1 * q as f64)
            })
            .collect();
        let jumps = vec![
            // Hermitian-Pauli fast path with a non-unit coefficient (rate · c²).
            JumpInput {
                lincomb: vec![(format!("Z{}", "I".repeat(n - 1)), Complex::new(0.7, 0.0))],
                rate: rates[0],
            },
            JumpInput {
                lincomb: sigma_minus_lincomb(1, n),
                rate: rates[1],
            },
            jump_hpauli(&format!("{}Z", "I".repeat(n - 1)), rates[2]),
        ];
        let mut spec = LindbladSpec::new(n, &h, &jumps).unwrap();
        let ops: Vec<_> = (0..n).map(|q| sigma_minus_lincomb(q, n)).collect();
        spec.add_kossakowski(&ops, k).unwrap();
        spec
    }

    /// `set_kossakowski` / `set_jump_rates` followed by `action` are
    /// bit-identical to a freshly built spec, including a K with an exact
    /// zero entry (a pair that is dropped) and a zero rate.
    #[test]
    fn setters_match_fresh_construction_bitwise() {
        let n = 3;
        let rates0 = [0.2, 0.5, 0.1];
        let rates1 = [0.9, 0.0, 0.35];
        let k0 = psd_matrix(n, 1);
        let mut k1 = psd_matrix(n, 2);
        k1[0][2] = Complex::new(0.0, 0.0);
        k1[2][0] = Complex::new(0.0, 0.0);

        let mut spec = setter_fixture(n, &rates0, &k0);
        assert_eq!(spec.num_jump_ops(), 3);
        assert_eq!(spec.num_kossakowski_ops(), n);
        assert_eq!(spec.jump_rates(), rates0.to_vec());
        assert_eq!(spec.kossakowski(), k0);

        spec.set_kossakowski(&k1).unwrap();
        assert_eq!(spec.kossakowski(), k1);
        let fresh = setter_fixture(n, &rates0, &k1);
        assert_eq!(spec.num_jump_terms(), fresh.num_jump_terms());
        for s in ["ZII", "IZI", "IIZ", "XII", "IYI", "XXI", "ZYZ", "XYZ", "IXX"] {
            let (p, _) = parse_pauli_string(s, n).unwrap();
            assert_actions_identical(&action_map(&spec, &p), &action_map(&fresh, &p));
        }

        spec.set_jump_rates(&rates1).unwrap();
        assert_eq!(spec.jump_rates(), rates1.to_vec());
        let fresh = setter_fixture(n, &rates1, &k1);
        for s in ["ZII", "IZI", "IIZ", "XII", "IYI", "XXI", "ZYZ", "XYZ", "IXX"] {
            let (p, _) = parse_pauli_string(s, n).unwrap();
            assert_actions_identical(&action_map(&spec, &p), &action_map(&fresh, &p));
        }

        // Back to the original K: identical to the original construction.
        spec.set_kossakowski(&k0).unwrap();
        let fresh = setter_fixture(n, &rates1, &k0);
        for s in ["ZII", "XXI", "ZYZ"] {
            let (p, _) = parse_pauli_string(s, n).unwrap();
            assert_actions_identical(&action_map(&spec, &p), &action_map(&fresh, &p));
        }
    }

    #[test]
    fn setters_reject_bad_input() {
        let n = 2;
        let k0 = psd_matrix(n, 3);
        let mut spec = setter_fixture(n, &[0.1, 0.2, 0.3], &k0);
        // Wrong shapes.
        assert!(matches!(
            spec.set_kossakowski(&psd_matrix(3, 4)),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            spec.set_kossakowski(&[vec![Complex::new(1.0, 0.0)], vec![Complex::new(1.0, 0.0)]]),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(spec.set_jump_rates(&[0.1, 0.2]), Err(Error::LengthMismatch { .. })));
        assert!(matches!(
            spec.set_jump_rates(&[0.1, -0.2, 0.3]),
            Err(Error::NegativeRate { index: 1, .. })
        ));
        // Non-Hermitian K.
        let mut bad = k0.clone();
        bad[0][1] += Complex::new(0.0, 1e-3);
        assert!(matches!(spec.set_kossakowski(&bad), Err(Error::KMatrixNotHermitian { .. })));
        // Nothing changed after the failures.
        assert_eq!(spec.kossakowski(), k0);
        assert_eq!(spec.jump_rates(), vec![0.1, 0.2, 0.3]);
        // Specs without the corresponding block.
        let mut plain = LindbladSpec::new(n, &[], &[jump_hpauli("ZI", 0.1)]).unwrap();
        assert!(matches!(plain.set_kossakowski(&k0), Err(Error::NoKossakowski)));
        let mut none = LindbladSpec::new(n, &[("XX".to_string(), 1.0)], &[]).unwrap();
        assert!(matches!(none.set_jump_rates(&[]), Err(Error::NoJumpOps)));
        assert_eq!(none.jump_rates(), Vec::<f64>::new());
        assert!(none.kossakowski().is_empty());
    }

    /// Dense, block and COO inputs compile the same dissipator: bitwise
    /// identical actions for every way of giving one block-diagonal `K`
    /// (dense setter, blocks, COO in both triangles with shuffled and split
    /// entries, COO upper triangle only, fresh `add_kossakowski_sparse`);
    /// the templates follow the zero pattern and the stored entries round
    /// trip through `kossakowski()`.
    #[test]
    fn block_and_sparse_setters_match_dense() {
        let n = 4;
        let rates = [0.2, 0.5, 0.1];
        let zero = Complex::new(0.0, 0.0);
        // blocks {0, 2} and {1, 3} of a complex PSD matrix
        let mut block_k = psd_matrix(n, 11);
        for i in 0..n {
            for j in 0..n {
                if (i % 2) != (j % 2) {
                    block_k[i][j] = zero;
                }
            }
        }
        let words = [
            "ZIII", "IZII", "XIII", "IYII", "XXII", "ZYZI", "XYZZ", "IXXI", "IIZX", "YIIY",
        ];
        let same_actions = |a: &LindbladSpec, b: &LindbladSpec| {
            assert_eq!(a.num_jump_terms(), b.num_jump_terms());
            for s in words {
                let (p, _) = parse_pauli_string(s, n).unwrap();
                assert_actions_identical(&action_map(a, &p), &action_map(b, &p));
            }
        };

        let dense = setter_fixture(n, &rates, &block_k);
        assert_eq!(dense.num_compiled_pairs(), 6); // 3 upper-triangle pairs per block
        assert_eq!(dense.num_pair_templates(), 2); // σ⁻σ⁻ on one site, σ⁻σ⁻ on two sites
        assert_eq!(dense.num_kossakowski_nonzeros(), 8);
        assert_eq!(dense.kossakowski(), block_k);
        assert_eq!(dense.num_jump_terms(), rates.len() + 6);

        // blocks, on a spec that held a dense K with another pattern
        let sub = |idx: &[usize]| -> Vec<Vec<Complex<f64>>> {
            idx.iter()
                .map(|&i| idx.iter().map(|&j| block_k[i][j]).collect())
                .collect()
        };
        let mut spec = setter_fixture(n, &rates, &psd_matrix(n, 3));
        assert_eq!(spec.num_compiled_pairs(), 10);
        assert_eq!(spec.num_pair_templates(), 2);
        spec.set_kossakowski_blocks(&[(vec![0, 2], sub(&[0, 2])), (vec![1, 3], sub(&[1, 3]))])
            .unwrap();
        assert_eq!(spec.num_compiled_pairs(), 6);
        assert_eq!(spec.num_pair_templates(), 2);
        assert_eq!(spec.kossakowski(), block_k);
        same_actions(&spec, &dense);

        // the same blocks in another order and with a block's rows permuted
        let perm = |idx: &[usize]| -> Vec<Vec<Complex<f64>>> { sub(idx) };
        spec.set_kossakowski_blocks(&[(vec![3, 1], perm(&[3, 1])), (vec![0, 2], sub(&[0, 2]))])
            .unwrap();
        assert_eq!(spec.kossakowski(), block_k);
        same_actions(&spec, &dense);

        // COO, both triangles, reversed order, the (0, 0) entry split in two
        let mut triples: Vec<(usize, usize, Complex<f64>)> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if block_k[i][j] != zero {
                    triples.push((i, j, block_k[i][j]));
                }
            }
        }
        triples.reverse();
        let k00 = block_k[0][0];
        triples.retain(|t| !(t.0 == 0 && t.1 == 0));
        triples.push((0, 0, k00 * 0.5));
        triples.insert(0, (0, 0, k00 * 0.5));
        spec.set_kossakowski_sparse(&triples).unwrap();
        assert_eq!(spec.kossakowski(), block_k);
        same_actions(&spec, &dense);

        // the upper triangle alone: Hermitian completion fills the mirrors
        let upper: Vec<(usize, usize, Complex<f64>)> = (0..n)
            .flat_map(|i| (i..n).map(move |j| (i, j)))
            .filter(|&(i, j)| block_k[i][j] != zero)
            .map(|(i, j)| (i, j, block_k[i][j]))
            .collect();
        let mut spec2 = setter_fixture(n, &rates, &psd_matrix(n, 5));
        spec2.set_kossakowski_sparse(&upper).unwrap();
        assert_eq!(spec2.kossakowski(), block_k);
        assert_eq!(spec2.num_kossakowski_nonzeros(), 8);
        same_actions(&spec2, &dense);

        // the lower triangle alone, likewise
        let lower: Vec<(usize, usize, Complex<f64>)> =
            upper.iter().map(|&(i, j, v)| (j, i, v.conj())).collect();
        spec2.set_kossakowski_sparse(&lower).unwrap();
        assert_eq!(spec2.kossakowski(), block_k);
        same_actions(&spec2, &dense);

        // a fresh spec built through add_kossakowski_sparse
        let h = vec![("XZZZ".to_string(), 0.3), ("ZXZZ".to_string(), 0.4)];
        let ops: Vec<_> = (0..n).map(|q| sigma_minus_lincomb(q, n)).collect();
        let mut fresh_dense = LindbladSpec::new(n, &h, &[]).unwrap();
        fresh_dense.add_kossakowski(&ops, &block_k).unwrap();
        let mut fresh_sparse = LindbladSpec::new(n, &h, &[]).unwrap();
        fresh_sparse.add_kossakowski_sparse(&ops, &upper).unwrap();
        assert_eq!(fresh_sparse.kossakowski(), block_k);
        same_actions(&fresh_sparse, &fresh_dense);

        // templates are reused when the pattern is unchanged and rebuilt
        // when it grows: a dense K after the blocks has all 10 pairs
        spec.set_kossakowski(&psd_matrix(n, 3)).unwrap();
        assert_eq!(spec.num_compiled_pairs(), 10);
        let fresh = setter_fixture(n, &rates, &psd_matrix(n, 3));
        same_actions(&spec, &fresh);
    }

    /// Overlapping blocks add up (`K = Σ_b P_b K_b P_bᵀ`), an empty block
    /// list is the zero dissipator, and bad input leaves the spec unchanged.
    #[test]
    fn block_and_sparse_setters_overlap_and_reject_bad_input() {
        let n = 3;
        let rates = [0.2, 0.5, 0.1];
        let k0 = psd_matrix(n, 2);
        let mut spec = setter_fixture(n, &rates, &k0);
        let (p, _) = parse_pauli_string("XYZ", n).unwrap();
        let before = action_map(&spec, &p);

        // overlap: blocks {0, 1} and {1, 2} sum on the shared operator 1
        let a = psd_matrix(2, 4);
        let b = psd_matrix(2, 6);
        let zero = Complex::new(0.0, 0.0);
        let mut expect = vec![vec![zero; n]; n];
        for i in 0..2 {
            for j in 0..2 {
                expect[i][j] += a[i][j];
                expect[i + 1][j + 1] += b[i][j];
            }
        }
        spec.set_kossakowski_blocks(&[(vec![0, 1], a.clone()), (vec![1, 2], b.clone())])
            .unwrap();
        assert_eq!(spec.kossakowski(), expect);
        let dense = setter_fixture(n, &rates, &expect);
        assert_actions_identical(&action_map(&spec, &p), &action_map(&dense, &p));

        // no blocks: K = 0, no Kossakowski pairs
        spec.set_kossakowski_blocks(&[]).unwrap();
        assert_eq!(spec.num_compiled_pairs(), 0);
        assert_eq!(spec.num_pair_templates(), 0); // unused templates are pruned
        assert_eq!(spec.num_jump_terms(), rates.len());
        spec.set_kossakowski_sparse(&[]).unwrap();
        assert_eq!(spec.num_kossakowski_nonzeros(), 0);

        // back to k0, then every rejection leaves it there
        spec.set_kossakowski(&k0).unwrap();
        assert_actions_identical(&action_map(&spec, &p), &before);
        let one = Complex::new(1.0, 0.0);
        assert!(matches!(
            spec.set_kossakowski_blocks(&[(vec![0, 3], psd_matrix(2, 1))]),
            Err(Error::KIndexOutOfRange { index: 3, m_ops: 3, .. })
        ));
        assert!(matches!(
            spec.set_kossakowski_blocks(&[(vec![1, 1], psd_matrix(2, 1))]),
            Err(Error::KBlockDuplicateIndex { block: 0, index: 1 })
        ));
        assert!(matches!(
            spec.set_kossakowski_blocks(&[(vec![0, 1], psd_matrix(3, 1))]),
            Err(Error::LengthMismatch { .. })
        ));
        let mut bad = psd_matrix(2, 1);
        bad[0][1] += Complex::new(0.0, 1e-3);
        assert!(matches!(
            spec.set_kossakowski_blocks(&[(vec![0, 1], bad)]),
            Err(Error::KMatrixNotHermitian { .. })
        ));
        assert!(matches!(
            spec.set_kossakowski_sparse(&[(0, 5, one)]),
            Err(Error::KIndexOutOfRange { index: 5, .. })
        ));
        // mirror mismatch and an imaginary diagonal
        assert!(matches!(
            spec.set_kossakowski_sparse(&[(0, 1, one), (1, 0, Complex::new(1.0, 0.5))]),
            Err(Error::KMatrixNotHermitian { n: 0, m: 1 })
        ));
        assert!(matches!(
            spec.set_kossakowski_sparse(&[(2, 2, Complex::new(1.0, 1e-3))]),
            Err(Error::KMatrixNotHermitian { n: 2, m: 2 })
        ));
        assert_eq!(spec.kossakowski(), k0);
        assert_actions_identical(&action_map(&spec, &p), &before);
        // without a block
        let mut plain = LindbladSpec::new(n, &[], &[jump_hpauli("ZII", 0.1)]).unwrap();
        assert!(matches!(plain.set_kossakowski_blocks(&[]), Err(Error::NoKossakowski)));
        assert!(matches!(plain.set_kossakowski_sparse(&[]), Err(Error::NoKossakowski)));
    }

    /// Pairs with the same operator-pattern relationship on other qubits
    /// share one template: four site pairs carrying the same three
    /// operators (`¼ZZ`; `¼XX + ¼YY`; `σ⁻` on the first spin), `K`
    /// block-diagonal by operator class — 30 compiled pairs but 6 templates
    /// (diagonal and off-diagonal pattern of each class), the same 6 with
    /// eight site pairs. Dense and block inputs are bit-identical; the
    /// action matches the jump form `L_l = Σ_m conj(A_ml) A_m` of
    /// `K = A A†` to 1e-12 (an independent reference for the local-frame
    /// action); a pattern that leaves `K` frees its templates.
    #[test]
    fn pair_templates_shared_across_blocks() {
        let build = |n_sites: usize| {
            let n = 2 * n_sites;
            let word = |pairs: &[(usize, char)]| -> String {
                let mut s = vec!['I'; n];
                for &(q, c) in pairs {
                    s[q] = c;
                }
                s.into_iter().collect()
            };
            let sites: Vec<(usize, usize)> = (0..n_sites).map(|s| (2 * s, 2 * s + 1)).collect();
            let mut ops: Vec<Vec<(String, Complex<f64>)>> = Vec::new();
            for &(i, j) in &sites {
                ops.push(vec![(word(&[(i, 'Z'), (j, 'Z')]), Complex::new(0.25, 0.0))]);
            }
            for &(i, j) in &sites {
                ops.push(vec![
                    (word(&[(i, 'X'), (j, 'X')]), Complex::new(0.25, 0.0)),
                    (word(&[(i, 'Y'), (j, 'Y')]), Complex::new(0.25, 0.0)),
                ]);
            }
            for &(i, _) in &sites {
                ops.push(sigma_minus_lincomb(i, n));
            }
            // K = A A† per class block (deterministic complex A), so the
            // jump form is L_l = Σ_m conj(A[m][l]) A_m with unit rate.
            let m = ops.len();
            let zero = Complex::new(0.0, 0.0);
            let mut k = vec![vec![zero; m]; m];
            let mut blocks: Vec<(Vec<usize>, Vec<Vec<Complex<f64>>>)> = Vec::new();
            let mut jumps: Vec<JumpInput> = Vec::new();
            for b in 0..3 {
                let idx: Vec<usize> = (b * n_sites..(b + 1) * n_sites).collect();
                let a: Vec<Vec<Complex<f64>>> = (0..n_sites)
                    .map(|i| {
                        (0..n_sites)
                            .map(|l| {
                                let t = (1 + i + 3 * l + 7 * b) as f64;
                                Complex::new(0.3 * t.sin(), 0.2 * (0.7 * t).cos())
                            })
                            .collect()
                    })
                    .collect();
                let kb: Vec<Vec<Complex<f64>>> = (0..n_sites)
                    .map(|i| {
                        (0..n_sites)
                            .map(|j| (0..n_sites).map(|l| a[i][l] * a[j][l].conj()).sum())
                            .collect()
                    })
                    .collect();
                for i in 0..n_sites {
                    for j in 0..n_sites {
                        k[idx[i]][idx[j]] = kb[i][j];
                    }
                }
                for l in 0..n_sites {
                    let mut lincomb = Vec::new();
                    for (i, &op) in idx.iter().enumerate() {
                        for (s, c) in &ops[op] {
                            lincomb.push((s.clone(), a[i][l].conj() * c));
                        }
                    }
                    jumps.push(JumpInput { lincomb, rate: 1.0 });
                }
                blocks.push((idx, kb));
            }
            let h = vec![
                (word(&[(0, 'X'), (1, 'Z')]), 0.3),
                (word(&[(3, 'Z')]), 0.7),
                (word(&[(2, 'Y'), (n - 1, 'Y')]), 0.2),
            ];
            let mut dense = LindbladSpec::new(n, &h, &[]).unwrap();
            dense.add_kossakowski(&ops, &k).unwrap();
            let mut blk = LindbladSpec::new(n, &h, &[]).unwrap();
            blk.add_kossakowski(&ops, &vec![vec![zero; m]; m]).unwrap();
            blk.set_kossakowski_blocks(&blocks).unwrap();
            let jump_form = LindbladSpec::new(n, &h, &jumps).unwrap();
            (n, dense, blk, jump_form)
        };

        let (n, dense, blk, jump_form) = build(4);
        assert_eq!(dense.num_compiled_pairs(), 3 * 10);
        assert_eq!(dense.num_pair_templates(), 6);
        assert_eq!(blk.num_compiled_pairs(), 30);
        assert_eq!(blk.num_pair_templates(), 6);
        let (t_bytes, p_bytes, e_bytes) = dense.kossakowski_memory_bytes();
        assert!(t_bytes > 0 && p_bytes > 0 && e_bytes >= 48 * std::mem::size_of::<KEntry>());
        for s in [
            "XIIIIIII", "ZZIIIIII", "XYZIIIII", "IIIIIIZX", "XIXIXIXI", "YZYZYZYZ", "IXIIIYII",
            "IIZZIIII",
        ] {
            let (p, _) = parse_pauli_string(s, n).unwrap();
            let a = action_map(&dense, &p);
            assert_actions_identical(&a, &action_map(&blk, &p));
            assert_actions_match(&a, &action_map(&jump_form, &p), 1e-12);
        }

        // twice the sites: the pair count grows, the template count does not
        let (_, dense8, _, _) = build(8);
        assert_eq!(dense8.num_compiled_pairs(), 3 * 36);
        assert_eq!(dense8.num_pair_templates(), 6);
        assert_eq!(dense8.kossakowski_memory_bytes().0, t_bytes);

        // one block over every operator: cross-class pairs bring new
        // patterns; back to the class blocks they are pruned again
        let mut spec = dense;
        let idx: Vec<usize> = (0..12).collect();
        let kb: Vec<Vec<Complex<f64>>> = psd_matrix(12, 5);
        spec.set_kossakowski_blocks(&[(idx, kb.clone())]).unwrap();
        assert_eq!(spec.num_compiled_pairs(), 78);
        assert!(spec.num_pair_templates() > 6);
        let fresh = {
            let mut f = LindbladSpec::new(n, &[("XZIIIIII".to_string(), 0.3), ("IIIZIIII".to_string(), 0.7), ("IIYIIIIY".to_string(), 0.2)], &[]).unwrap();
            f.add_kossakowski(&blk_ops(n), &kb).unwrap();
            f
        };
        let (p, _) = parse_pauli_string("XYZIIIII", n).unwrap();
        assert_actions_identical(&action_map(&spec, &p), &action_map(&fresh, &p));
        spec.set_kossakowski_blocks(&[]).unwrap();
        assert_eq!(spec.num_pair_templates(), 0);
    }

    /// The raw chunk access the Kossakowski action uses to move bits between
    /// the register and a pair's local frame agrees with the storage's own
    /// indexing on every qubit of a word spanning both chunks.
    #[test]
    fn raw_bit_access_matches_bitvec() {
        let n = MAX_QUBITS;
        let mut s = String::new();
        for q in 0..n {
            s.push(['I', 'X', 'Y', 'Z'][(q * 7 + q / 5) % 4]);
        }
        let (w, _) = parse_pauli_string(&s, n).unwrap();
        for q in 0..n {
            let (ci, bi) = (q / CHUNK_BITS, q % CHUNK_BITS);
            assert_eq!((w.xbits.data[ci] >> bi) & 1 == 1, w.xbits[q], "x bit {q}");
            assert_eq!((w.zbits.data[ci] >> bi) & 1 == 1, w.zbits[q], "z bit {q}");
        }
    }

    /// The operator table of `pair_templates_shared_across_blocks` for `n` qubits.
    fn blk_ops(n: usize) -> Vec<Vec<(String, Complex<f64>)>> {
        let n_sites = n / 2;
        let word = |pairs: &[(usize, char)]| -> String {
            let mut s = vec!['I'; n];
            for &(q, c) in pairs {
                s[q] = c;
            }
            s.into_iter().collect()
        };
        let mut ops = Vec::new();
        for s in 0..n_sites {
            ops.push(vec![(word(&[(2 * s, 'Z'), (2 * s + 1, 'Z')]), Complex::new(0.25, 0.0))]);
        }
        for s in 0..n_sites {
            ops.push(vec![
                (word(&[(2 * s, 'X'), (2 * s + 1, 'X')]), Complex::new(0.25, 0.0)),
                (word(&[(2 * s, 'Y'), (2 * s + 1, 'Y')]), Complex::new(0.25, 0.0)),
            ]);
        }
        for s in 0..n_sites {
            ops.push(sigma_minus_lincomb(2 * s, n));
        }
        ops
    }

    fn jump_hpauli(s: &str, rate: f64) -> JumpInput {
        JumpInput {
            lincomb: vec![(s.to_string(), Complex::new(1.0, 0.0))],
            rate,
        }
    }

    #[test]
    fn z_dephasing_action_on_x() {
        // L = Z on a single qubit; L*(X) = γ(ZXZ - X) = γ(-X - X) = -2γ X.
        let spec = LindbladSpec::new(
            1,
            &[("X".to_string(), 0.0)], // no Hamiltonian
            &[jump_hpauli("Z", 0.5)],
        )
        .unwrap();
        let (x, _) = parse_pauli_string("X", 1).unwrap();
        let terms = spec.action(&x);
        assert_eq!(terms.len(), 1);
        assert!((terms[0].1 - (-1.0)).abs() < 1e-12); // -2·0.5 = -1
    }

    #[test]
    fn amplitude_damping_action_on_z() {
        // Single-qubit σ⁻ jump: L*(Z) = -γ(I + Z). With γ=1 we expect
        // I coefficient = -1, Z coefficient = -1.
        let sigma_minus = JumpInput {
            lincomb: vec![
                ("X".to_string(), Complex::new(0.5, 0.0)),
                ("Y".to_string(), Complex::new(0.0, -0.5)),
            ],
            rate: 1.0,
        };
        let spec = LindbladSpec::new(1, &[], &[sigma_minus]).unwrap();
        let (z, _) = parse_pauli_string("Z", 1).unwrap();
        let terms = spec.action(&z);
        let (i_word, _) = parse_pauli_string("I", 1).unwrap();
        let mut i_coeff = 0.0;
        let mut z_coeff = 0.0;
        for (w, c) in &terms {
            if w == &i_word {
                i_coeff = *c;
            } else if w == &z {
                z_coeff = *c;
            }
        }
        assert!((i_coeff - (-1.0)).abs() < 1e-10, "I coeff = {i_coeff}");
        assert!((z_coeff - (-1.0)).abs() < 1e-10, "Z coeff = {z_coeff}");
    }

    #[test]
    fn word_codec_roundtrip() {
        let codes = [0u8, 1, 2, 3, 1, 0, 3, 2];
        let w = word_from_codes(&codes).unwrap();
        let mut out = vec![0u8; codes.len()];
        codes_from_word(&w, &mut out);
        assert_eq!(out.as_slice(), &codes);
    }

    /// Per-step orbit-rep evolution gives the SAME final orbit-rep
    /// state as full-basis complex evolution followed by a single
    /// projection at the end. Validates that the phase-aware complex
    /// action machinery is consistent with the full-basis reference.
    #[test]
    fn pc_step_orbit_rep_matches_full_basis_projection() {
        use ppvm_pauli_sum::symmetry::canonicalize_pauli_sum_complex;
        use std::f64::consts::PI;
        let n = 4usize;
        let dt = 0.01f64;
        let n_steps = 3usize;
        let mut h_terms: Vec<(String, f64)> = Vec::new();
        for j in 0..n {
            let nxt = (j + 1) % n;
            for op in ["X", "Y"] {
                let mut s = vec!['I'; n];
                s[j] = op.chars().next().unwrap();
                s[nxt] = op.chars().next().unwrap();
                h_terms.push((s.into_iter().collect(), 1.0));
            }
        }
        let spec = LindbladSpec::new(n, &h_terms, &[]).unwrap();
        let group = ppvm_pauli_sum::symmetry::TranslationGroup::chain_1d(n);
        let k_mode: i32 = 1;
        let k = vec![k_mode];

        // Build the k=1 eigenstate in FULL basis form.
        let basis_full: Vec<Word> = (0..n)
            .map(|j| {
                let mut s = vec!['I'; n];
                s[j] = 'Z';
                let (w, _) = parse_pauli_string(&s.into_iter().collect::<String>(), n).unwrap();
                w
            })
            .collect();
        let coeffs_full: Vec<Complex<f64>> = (0..n as i32)
            .map(|a| {
                Complex::from_polar(1.0, -2.0 * PI * (k_mode as f64) * (a as f64) / (n as f64))
            })
            .collect();

        // ----- Full-basis path -----
        let mut bf = basis_full.clone();
        let mut cf = coeffs_full.clone();
        let protected: Vec<Word> = Vec::new();
        for _ in 0..n_steps {
            // Full enrichment (tau_add = 0.0 adds every leakage string):
            // for a momentum eigenstate the leakage is pure-sector, so the
            // full-basis and orbit-rep paths build corresponding bases and
            // the projection theorem gives an exact match. The orbit-rep
            // side uses a large max_basis so its rank cap never binds.
            pc_step_complex_full(&spec, &mut bf, &mut cf, dt);
        }
        // Project at the end.
        canonicalize_pauli_sum_complex(&mut bf, &mut cf, &group, &k);

        // ----- Orbit-rep path -----
        // Initial orbit-rep form: project the full-basis input.
        let mut br = basis_full.clone();
        let mut cr = coeffs_full.clone();
        canonicalize_pauli_sum_complex(&mut br, &mut cr, &group, &k);
        // Evolve in orbit-rep form (max_basis large ⇒ full enrichment).
        for _ in 0..n_steps {
            orbit_rep::pc_step_orbit_rep(
                &spec,
                &mut br,
                &mut cr,
                dt,
                &protected,
                &group,
                &k,
                &PcStepConfig {
                    max_basis: 10_000_000,
                    ..Default::default()
                },
            )
            .unwrap();
        }

        // Compare.
        let mf: FxHashMap<Word, Complex<f64>> = bf.into_iter().zip(cf).collect();
        let mr: FxHashMap<Word, Complex<f64>> = br.into_iter().zip(cr).collect();
        assert_eq!(
            mf.len(),
            mr.len(),
            "orbit-rep ({}) and full-basis-projected ({}) basis sizes differ",
            mr.len(),
            mf.len()
        );
        let mut max_diff = 0.0_f64;
        for (w, cm) in &mr {
            let cf_val = mf
                .get(w)
                .copied()
                .unwrap_or_else(|| panic!("rep {:?} in orbit-rep but not in full-basis", w));
            max_diff = max_diff.max((cm - cf_val).norm());
        }
        assert!(
            max_diff < 1e-9,
            "orbit-rep diverged from full-basis: max |Δc| = {max_diff:e}"
        );
    }

    /// The full-space complex step at momentum k=0 must reproduce the real
    /// pc_step on the same trajectory exactly.
    #[test]
    fn complex_full_matches_real_at_kzero() {
        let n = 4usize;
        let dt = 0.01f64;
        let n_steps = 5usize;
        let mut h_terms: Vec<(String, f64)> = Vec::new();
        for j in 0..n {
            let nxt = (j + 1) % n;
            for op in ["X", "Y"] {
                let mut s = vec!['I'; n];
                s[j] = op.chars().next().unwrap();
                s[nxt] = op.chars().next().unwrap();
                h_terms.push((s.into_iter().collect(), 1.0));
            }
        }
        let spec = LindbladSpec::new(n, &h_terms, &[]).unwrap();

        let mut basis_r: Vec<Word> = (0..n)
            .map(|j| {
                let mut s = vec!['I'; n];
                s[j] = 'Z';
                let st: String = s.into_iter().collect();
                let (w, _) = parse_pauli_string(&st, n).unwrap();
                w
            })
            .collect();
        let mut coeffs_r: Vec<f64> = vec![1.0; n];

        let mut basis_c = basis_r.clone();
        let mut coeffs_c: Vec<Complex<f64>> =
            coeffs_r.iter().map(|&v| Complex::new(v, 0.0)).collect();

        let protected: Vec<Word> = Vec::new();
        for _ in 0..n_steps {
            // Large max_basis: rank cap never binds, so the real path
            // enriches fully (adds every leakage string). Match the
            // complex path by setting its tau_add=0.0 (also full
            // enrichment) so the two stay in lock-step at k=0.
            spec.pc_step(
                &mut basis_r,
                &mut coeffs_r,
                dt,
                &protected,
                &PcStepConfig {
                    max_basis: 10_000_000,
                    ..Default::default()
                },
            )
            .unwrap();
            pc_step_complex_full(&spec, &mut basis_c, &mut coeffs_c, dt);
        }
        // Match as (word → coeff) maps.
        let map_r: FxHashMap<Word, f64> = basis_r.into_iter().zip(coeffs_r).collect();
        let map_c: FxHashMap<Word, Complex<f64>> = basis_c.into_iter().zip(coeffs_c).collect();
        assert_eq!(
            map_r.len(),
            map_c.len(),
            "real and complex pc_step produced different basis sizes ({} vs {})",
            map_r.len(),
            map_c.len()
        );
        let mut max_diff = 0.0_f64;
        for (w, cr) in &map_r {
            let cc = map_c
                .get(w)
                .copied()
                .unwrap_or_else(|| panic!("word {:?} in real but not complex", w));
            assert!(cc.im.abs() < 1e-10, "expected zero imag at k=0, got {cc:?}");
            max_diff = max_diff.max((cr - cc.re).abs());
        }
        assert!(
            max_diff < 1e-10,
            "real vs complex pc_step diverged: max |Δc| = {max_diff:e}"
        );
    }

    /// Small-system end-to-end check that orbit-rep merging gives the
    /// same physics as standard evolution, when no truncation is applied.
    ///
    /// Setup: n=4 qubit chain, PBC, translation-invariant XY Hamiltonian
    /// `H = Σ_j (X_j X_{j+1} + Y_j Y_{j+1})`, no dissipation. Initial
    /// operator `O(0) = Σ_j Z_j` is translation-invariant (k=0 sector).
    ///
    /// Run 10 pc_step iterations with `drop_tol = 0` (no truncation):
    /// once without merging, once applying `canonicalize_pauli_sum`
    /// after each step. Canonicalize the un-merged final state once at
    /// the end. The two orbit-rep representations should be
    /// bit-identical up to FP noise.
    #[test]
    fn pc_step_matches_symmetry_merged_on_small_chain() {
        use ppvm_pauli_sum::symmetry::{TranslationGroup, canonicalize_pauli_sum};

        let n = 4usize;
        let dt = 0.05f64;
        let n_steps = 10usize;

        // Build XY-chain Hamiltonian with PBC. 8 terms (4 bonds × {XX, YY}).
        let mut h_terms: Vec<(String, f64)> = Vec::new();
        for j in 0..n {
            let nxt = (j + 1) % n;
            for op in ["X", "Y"] {
                let mut s = vec!['I'; n];
                s[j] = op.chars().next().unwrap();
                s[nxt] = op.chars().next().unwrap();
                h_terms.push((s.into_iter().collect(), 1.0));
            }
        }
        // No dissipation.
        let spec = LindbladSpec::new(n, &h_terms, &[]).unwrap();
        let group = TranslationGroup::chain_1d(n);

        // Initial: O(0) = Σ_j Z_j (translation-invariant).
        let mut basis_u: Vec<Word> = (0..n)
            .map(|j| {
                let mut s = vec!['I'; n];
                s[j] = 'Z';
                let st: String = s.into_iter().collect();
                let (w, _) = parse_pauli_string(&st, n).unwrap();
                w
            })
            .collect();
        let mut coeffs_u: Vec<f64> = vec![1.0; n];

        // Mirror state for the "with merging" run.
        let mut basis_m = basis_u.clone();
        let mut coeffs_m = coeffs_u.clone();

        let protected: Vec<Word> = Vec::new();
        for _ in 0..n_steps {
            // max_basis == current basis size → room = 0: no leakage
            // enrichment, only the expm step (the regime where merging
            // commutes with evolution). drop_tol = 0 → no truncation.
            let cfg_u = PcStepConfig {
                max_basis: basis_u.len(),
                ..Default::default()
            };
            spec.pc_step(&mut basis_u, &mut coeffs_u, dt, &protected, &cfg_u)
                .unwrap();

            let cfg_m = PcStepConfig {
                max_basis: basis_m.len(),
                ..Default::default()
            };
            spec.pc_step(&mut basis_m, &mut coeffs_m, dt, &protected, &cfg_m)
                .unwrap();
            // Apply symmetry merging on the "with merging" run only.
            canonicalize_pauli_sum(&mut basis_m, &mut coeffs_m, &group);
        }

        // Canonicalize the un-merged final state once.
        canonicalize_pauli_sum(&mut basis_u, &mut coeffs_u, &group);

        // Both representations should now be in orbit-rep form; compare
        // as (word → coeff) maps with FP tolerance.
        let map_u: FxHashMap<Word, f64> = basis_u.into_iter().zip(coeffs_u).collect();
        let map_m: FxHashMap<Word, f64> = basis_m.into_iter().zip(coeffs_m).collect();
        assert_eq!(
            map_u.len(),
            map_m.len(),
            "merged basis size {} != post-merged-unmerged basis size {}",
            map_m.len(),
            map_u.len()
        );
        let mut max_diff = 0.0f64;
        for (w, c_u) in &map_u {
            let c_m = map_m.get(w).copied().unwrap_or_else(|| {
                panic!(
                    "rep {:?} present in un-merged-then-canonicalized but not in merged",
                    w
                );
            });
            max_diff = max_diff.max((c_u - c_m).abs());
        }
        assert!(
            max_diff < 1e-9,
            "with-merging vs without-merging diverged: max |Δc| = {max_diff:e}"
        );
    }
}
