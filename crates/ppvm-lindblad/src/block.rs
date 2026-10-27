// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! Block-structured Lindbladian for chemical exchange between `M` species
//! (conformers) of the same `N` qubits.
//!
//! Schrödinger picture, species `c` with Lindbladian `L_c`, first-order
//! rates `k_cd` (`c → d`) and spin matchings `Π_cd` (permutations of qubit
//! positions):
//!
//! ```text
//! d ρ_c / dt = L_c ρ_c + Σ_{d≠c} ( k_dc Π_dc ρ_d Π_dc† − k_cd ρ_c ).
//! ```
//!
//! The observable is a direct sum `O = ⊕_c O_c` with `⟨O⟩ = Σ_c Tr[O_c ρ_c]`,
//! and its Heisenberg-picture equation of motion (the adjoint of the above,
//! using `Tr[O_c Π ρ_d Π†] = Tr[(Π† O_c Π) ρ_d]`) is
//!
//! ```text
//! d O_c / dt = L_c* O_c + Σ_{d≠c} k_cd ( Π_cd† O_d Π_cd − O_c ).
//! ```
//!
//! On a Pauli string `P` of species `d` the exchange generator therefore
//! produces `+k_cd · (P ∘ π_cd)` in every species `c` with `k_cd > 0` (the
//! string's letters permuted by the matching, no phase) and
//! `−(Σ_c k_dc) · P` in species `d` itself.
//!
//! **Reset masks (unequal species padded to a common qubit set).** A
//! transition `c → d` may carry a set of qubit positions of `c` whose state is
//! *not* carried into `d` but replaced by the unit state (spins that exist
//! only in `c` are traced out; spins that exist only in `d` arrive
//! unpolarised): Schrödinger `ρ_d += k_cd Π_cd [ Tr_R(ρ_c) ⊗ 1_R / 2^|R| ]`.
//! On Pauli strings this map is the projector onto strings that are the
//! identity on `R`, followed by the permutation; it is self-adjoint, so in the
//! Heisenberg picture the inflow term `+k_cd (P ∘ π_cd)` is simply dropped
//! whenever `P ∘ π_cd` has a non-identity letter on a masked position of `c`.
//! The drain `−k_cd P` is unchanged (the whole reactant block leaves at the
//! rate). Without masks the block is the ordinary permutation exchange.
//!
//! **Representation.** A block row `(species, P)` is stored as one ordinary
//! [`Word`] of `N + a` positions: the first `N` hold `P`, the trailing
//! `a = ⌈log₄ M⌉` positions hold the species index as base-4 digits (each
//! Pauli slot is two bits). Every basis utility of the crate — the hash
//! index, leakage merging, pruning, the rank cap and the matrix-free CSC
//! exponential — works on the rows unchanged; only the action knows about
//! the digits: it strips them, applies the species' own [`LindbladSpec`],
//! re-attaches them and adds the exchange terms. The cost of a step is that
//! of one species' Lindbladian per row, so `M ×` the strings of one species
//! rather than `4^(N+a)` strings of an ancilla encoding.

use fxhash::{FxBuildHasher, FxHashMap};
use num::Complex;
use ppvm_traits::PauliWordTrait;
use rayon::prelude::*;

use crate::mf_expm::{self, ActionTerms};
use crate::{
    Error, LindbladSpec, MAX_QUBITS, PcStepConfig, Word, add_leakage_capped, build_basis_index,
    cap_basis, prune_basis, word_hash,
};

/// Heisenberg-picture inflow into one species from the rows of another:
/// `(target species c, rate k_cd, matching π_cd)`.
struct Inflow {
    target: usize,
    rate: f64,
    perm: Vec<u8>,
    /// Reset mask of the transition `target → d` as an `n_qubits` word with
    /// the x bit set on every masked position of `target`; `None` = no mask.
    reset: Option<Word>,
}

/// `M` per-species Lindbladians coupled by first-order exchange; see the
/// module docs for the equation of motion and the row encoding.
pub struct BlockLindbladSpec {
    n_qubits: usize,
    n_species: usize,
    /// Species digits appended to every word (`⌈log₄ M⌉`, 0 for `M = 1`).
    n_digits: usize,
    species: Vec<LindbladSpec>,
    /// `exchange[c][d] = k_cd`, the rate `c → d` (diagonal zero).
    exchange: Vec<Vec<f64>>,
    /// `rate_out[d] = Σ_c k_dc`: the Heisenberg diagonal of species `d`.
    rate_out: Vec<f64>,
    /// `inflow[d]`: the species `c` whose observables receive weight from
    /// the rows of species `d` (`k_cd > 0`), with the matching `π_cd`.
    inflow: Vec<Vec<Inflow>>,
    /// Reset masks as given: `(c, d, positions of c reset on c → d)`.
    resets: Vec<(usize, usize, Vec<usize>)>,
}

impl BlockLindbladSpec {
    /// Couple `species` (one `LindbladSpec` per species, all on `n_qubits`)
    /// by the rate matrix `exchange` (`exchange[c][d] = k_cd ≥ 0`, diagonal
    /// ignored) and the `matchings` `(c, d, perm)` with `perm[p]` = position
    /// in `d` of the qubit at position `p` in `c`. A missing reverse
    /// matching `(d, c)` is the inverse permutation; a missing matching is
    /// the identity.
    pub fn new(
        n_qubits: usize,
        species: Vec<LindbladSpec>,
        exchange: &[Vec<f64>],
        matchings: &[(usize, usize, Vec<usize>)],
    ) -> Result<Self, Error> {
        Self::with_resets(n_qubits, species, exchange, matchings, &[])
    }

    /// [`Self::new`] with reset masks: `resets` lists `(c, d, positions)` with
    /// `positions` the qubit positions **of species `c`** whose state is
    /// replaced by the unit state on the transition `c → d` (module docs). A
    /// pair may appear once; a pair without a transition (`k_cd = 0`) is an
    /// error, as is a position at or beyond `n_qubits`. An empty list is
    /// [`Self::new`] exactly.
    pub fn with_resets(
        n_qubits: usize,
        species: Vec<LindbladSpec>,
        exchange: &[Vec<f64>],
        matchings: &[(usize, usize, Vec<usize>)],
        resets: &[(usize, usize, Vec<usize>)],
    ) -> Result<Self, Error> {
        let m = species.len();
        if m == 0 {
            return Err(Error::Internal("BlockLindbladSpec needs at least one species".into()));
        }
        for (s, spec) in species.iter().enumerate() {
            if spec.n_qubits() != n_qubits {
                return Err(Error::LengthMismatch {
                    what: "species Lindbladian n_qubits vs n_qubits",
                    a: spec.n_qubits(),
                    b: n_qubits,
                });
            }
            let _ = s;
        }
        let mut n_digits = 0usize;
        while (1usize << (2 * n_digits)) < m {
            n_digits += 1;
        }
        if n_qubits + n_digits > MAX_QUBITS {
            return Err(Error::TooManyQubits { got: n_qubits + n_digits });
        }
        if exchange.len() != m || exchange.iter().any(|row| row.len() != m) {
            return Err(Error::LengthMismatch {
                what: "exchange matrix shape vs n_species",
                a: exchange.len(),
                b: m,
            });
        }
        let mut k = vec![vec![0.0; m]; m];
        for c in 0..m {
            for d in 0..m {
                if c == d {
                    continue;
                }
                let v = exchange[c][d];
                if !(v >= 0.0) {
                    return Err(Error::NegativeRate { index: c * m + d, rate: v });
                }
                k[c][d] = v;
            }
        }
        // Matchings: validate, store, complete with inverses.
        let identity: Vec<u8> = (0..n_qubits).map(|p| p as u8).collect();
        let mut perms: FxHashMap<(usize, usize), Vec<u8>> = FxHashMap::default();
        for (c, d, perm) in matchings {
            if *c >= m || *d >= m || c == d {
                return Err(Error::SpeciesOutOfRange { got: (*c).max(*d), n_species: m });
            }
            let mut seen = vec![false; n_qubits];
            if perm.len() != n_qubits {
                return Err(Error::InvalidMatching { c: *c, d: *d });
            }
            for &q in perm {
                if q >= n_qubits || seen[q] {
                    return Err(Error::InvalidMatching { c: *c, d: *d });
                }
                seen[q] = true;
            }
            perms.insert((*c, *d), perm.iter().map(|&q| q as u8).collect());
        }
        let keys: Vec<(usize, usize)> = perms.keys().copied().collect();
        for (c, d) in keys {
            if !perms.contains_key(&(d, c)) {
                let perm = &perms[&(c, d)];
                let mut inv = vec![0u8; n_qubits];
                for (p, &q) in perm.iter().enumerate() {
                    inv[q as usize] = p as u8;
                }
                perms.insert((d, c), inv);
            }
        }
        // Reset masks: one word per (c, d) with the x bit set on the masked positions of c.
        let mut masks: FxHashMap<(usize, usize), Word> = FxHashMap::default();
        for (c, d, positions) in resets {
            if *c >= m || *d >= m || c == d || k[*c][*d] == 0.0 || masks.contains_key(&(*c, *d)) {
                return Err(Error::InvalidReset { c: *c, d: *d });
            }
            let mut w = Word::new(n_qubits);
            for &q in positions {
                if q >= n_qubits {
                    return Err(Error::InvalidReset { c: *c, d: *d });
                }
                w.xbits.set(q, true);
            }
            w.rehash();
            if !positions.is_empty() {
                masks.insert((*c, *d), w);
            }
        }
        let mut rate_out = vec![0.0; m];
        let mut inflow: Vec<Vec<Inflow>> = (0..m).map(|_| Vec::new()).collect();
        for c in 0..m {
            for d in 0..m {
                if c == d || k[c][d] == 0.0 {
                    continue;
                }
                rate_out[c] += k[c][d];
                inflow[d].push(Inflow {
                    target: c,
                    rate: k[c][d],
                    perm: perms.get(&(c, d)).cloned().unwrap_or_else(|| identity.clone()),
                    reset: masks.get(&(c, d)).cloned(),
                });
            }
        }
        Ok(Self {
            n_qubits,
            n_species: m,
            n_digits,
            species,
            exchange: k,
            rate_out,
            inflow,
            resets: resets.to_vec(),
        })
    }

    /// The reset masks as given to [`Self::with_resets`] (empty for [`Self::new`]).
    pub fn resets(&self) -> &[(usize, usize, Vec<usize>)] {
        &self.resets
    }

    /// Whether `w` (an `n_qubits` word) has a non-identity letter on a
    /// position of `mask` (x bits).
    #[inline]
    fn hits_mask(w: &Word, mask: &Word) -> bool {
        w.xbits
            .data
            .iter()
            .zip(w.zbits.data.iter())
            .zip(mask.xbits.data.iter())
            .any(|((x, z), m)| (*x | *z) & *m != 0)
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn n_species(&self) -> usize {
        self.n_species
    }

    /// Number of species digits appended to each row word.
    pub fn n_digits(&self) -> usize {
        self.n_digits
    }

    /// Word length of a block row (`n_qubits + n_digits`).
    pub fn row_len(&self) -> usize {
        self.n_qubits + self.n_digits
    }

    /// `exchange[c][d] = k_cd` (diagonal zero).
    pub fn exchange(&self) -> &[Vec<f64>] {
        &self.exchange
    }

    /// `Σ_d k_cd` per species.
    pub fn rate_out(&self) -> &[f64] {
        &self.rate_out
    }

    /// The Lindbladian of one species.
    pub fn species(&self, s: usize) -> Result<&LindbladSpec, Error> {
        self.species.get(s).ok_or(Error::SpeciesOutOfRange {
            got: s,
            n_species: self.n_species,
        })
    }

    /// Overwrite the Hamiltonian coefficients of species `s` in place
    /// ([`LindbladSpec::set_h_coeffs`]).
    pub fn set_h_coeffs(&mut self, s: usize, coeffs: &[f64]) -> Result<(), Error> {
        let m = self.n_species;
        self.species
            .get_mut(s)
            .ok_or(Error::SpeciesOutOfRange { got: s, n_species: m })?
            .set_h_coeffs(coeffs)
    }

    // ────────────────── row encoding ──────────────────

    /// Block row from `n_qubits` Pauli codes and a species index.
    pub fn encode_row(&self, codes: &[u8], species: usize) -> Result<Word, Error> {
        if codes.len() != self.n_qubits {
            return Err(Error::WrongLength {
                expected: self.n_qubits,
                got: codes.len(),
            });
        }
        if species >= self.n_species {
            return Err(Error::SpeciesOutOfRange {
                got: species,
                n_species: self.n_species,
            });
        }
        let mut w = Word::new(self.row_len());
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
        self.write_digits(&mut w, species);
        w.rehash();
        Ok(w)
    }

    /// Inverse of [`Self::encode_row`]: the physical codes into `codes`
    /// (length `n_qubits`) and the species index returned.
    pub fn decode_row(&self, w: &Word, codes: &mut [u8]) -> usize {
        debug_assert_eq!(codes.len(), self.n_qubits);
        for (q, slot) in codes.iter_mut().enumerate() {
            *slot = (w.xbits[q] as u8) | ((w.zbits[q] as u8) << 1);
        }
        self.read_digits(w)
    }

    /// Species index of a row word.
    pub fn species_of(&self, w: &Word) -> usize {
        self.read_digits(w)
    }

    #[inline]
    fn write_digits(&self, w: &mut Word, species: usize) {
        for j in 0..self.n_digits {
            let code = (species >> (2 * j)) & 3;
            let q = self.n_qubits + j;
            w.xbits.set(q, code & 1 != 0);
            w.zbits.set(q, code & 2 != 0);
        }
    }

    #[inline]
    fn read_digits(&self, w: &Word) -> usize {
        let mut s = 0usize;
        for j in 0..self.n_digits {
            let q = self.n_qubits + j;
            let code = (w.xbits[q] as usize) | ((w.zbits[q] as usize) << 1);
            s |= code << (2 * j);
        }
        s
    }

    /// `(species, physical word)` of a row: the digits cleared and the word
    /// length set to `n_qubits`, so the species spec's support loops see a
    /// plain `N`-qubit string.
    #[inline]
    fn detach(&self, p: &Word) -> (usize, Word) {
        let s = self.read_digits(p);
        let mut phys = Word::new(self.n_qubits);
        phys.xbits.data = p.xbits.data;
        phys.zbits.data = p.zbits.data;
        for j in 0..self.n_digits {
            let q = self.n_qubits + j;
            phys.xbits.set(q, false);
            phys.zbits.set(q, false);
        }
        phys.rehash();
        (s, phys)
    }

    /// Physical `N`-qubit word → row word of species `s`.
    #[inline]
    fn attach(&self, w: &Word, s: usize) -> Word {
        let mut row = Word::new(self.row_len());
        row.xbits.data = w.xbits.data;
        row.zbits.data = w.zbits.data;
        self.write_digits(&mut row, s);
        row.rehash();
        row
    }

    /// `Π† P Π` for the matching `perm`: `out[p] = w[perm[p]]` (no phase).
    #[inline]
    fn permute(&self, w: &Word, perm: &[u8]) -> Word {
        let mut out = Word::new(self.n_qubits);
        for (p, &q) in perm.iter().enumerate() {
            let q = q as usize;
            if w.xbits[q] {
                out.xbits.set(p, true);
            }
            if w.zbits[q] {
                out.zbits.set(p, true);
            }
        }
        out.rehash();
        out
    }

    // ────────────────── action ──────────────────

    /// `(output row, coefficient)` pairs of `L*_block(p)` for a row word `p`,
    /// without the input coefficient; deduplicated.
    pub(crate) fn compute_action_terms(
        &self,
        p: &Word,
        scratch_support: &mut Vec<u32>,
        scratch_cands: &mut Vec<u32>,
        scratch_local: &mut FxHashMap<Word, Complex<f64>>,
    ) -> Vec<(Word, f64)> {
        let (d, phys) = self.detach(p);
        let terms =
            self.species[d].compute_action_terms(&phys, scratch_support, scratch_cands, scratch_local);
        let inflow = &self.inflow[d];
        let r = self.rate_out[d];
        let mut out = Vec::with_capacity(terms.len() + inflow.len() + 1);
        let mut diag_seen = false;
        for (w, c) in terms {
            let mut c = c;
            if r != 0.0 && w == phys {
                c -= r;
                diag_seen = true;
            }
            if c != 0.0 {
                out.push((self.attach(&w, d), c));
            }
        }
        if r != 0.0 && !diag_seen {
            out.push((*p, -r));
        }
        for f in inflow {
            let w = self.permute(&phys, &f.perm);
            if let Some(mask) = &f.reset {
                if Self::hits_mask(&w, mask) {
                    continue; // traced / fresh spin: the string has no expectation on the unit state
                }
            }
            out.push((self.attach(&w, f.target), f.rate));
        }
        out
    }

    fn scratch(&self) -> (Vec<u32>, Vec<u32>, FxHashMap<Word, Complex<f64>>) {
        (
            Vec::with_capacity(self.n_qubits),
            Vec::with_capacity(128),
            FxHashMap::with_capacity_and_hasher(128, FxBuildHasher::default()),
        )
    }

    /// Apply `L*_block` to a single row word.
    pub fn action(&self, p: &Word) -> Vec<(Word, f64)> {
        let (mut s1, mut s2, mut lm) = self.scratch();
        self.compute_action_terms(p, &mut s1, &mut s2, &mut lm)
    }

    /// Off-basis component of `L*_block( Σ_j coeffs[j] · basis[j] )`, capped
    /// to the room `max_basis − basis.len()` and filtered by `tau_add`
    /// exactly as [`LindbladSpec::leakage_with_prune`].
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
        let room = max_basis.saturating_sub(basis.len());
        let mut merged: FxHashMap<Word, f64> = FxHashMap::default();
        for chunk_indices in order.chunks(CHUNK_SIZE) {
            let local: Vec<Vec<(Word, f64)>> = chunk_indices
                .par_iter()
                .map_init(
                    || self.scratch(),
                    |(s1, s2, lm), &i| {
                        let terms = self.compute_action_terms(&basis[i], s1, s2, lm);
                        let c = coeffs[i];
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
        Ok(merged
            .into_iter()
            .filter(|(_, c)| c.abs() > tau_add)
            .collect())
    }

    /// Uncapped leakage ([`Self::leakage_with_prune`] with no room limit).
    pub fn leakage(
        &self,
        basis: &[Word],
        coeffs: &[f64],
        protected: &[Word],
    ) -> Result<Vec<(Word, f64)>, Error> {
        self.leakage_with_prune(basis, coeffs, protected, usize::MAX, 0.0)
    }

    /// Sparse generator in COO form `(row, col, val)` on `basis` (rows that
    /// leave the basis are dropped), as [`LindbladSpec::generator`].
    pub fn generator(&self, basis: &[Word]) -> Vec<(usize, usize, f64)> {
        let index = build_basis_index(basis);
        let local: Vec<Vec<(usize, usize, f64)>> = basis
            .par_iter()
            .enumerate()
            .map_init(
                || self.scratch(),
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
        let total: usize = local.iter().map(|v| v.len()).sum();
        let mut flat = Vec::with_capacity(total);
        for v in local {
            flat.extend(v);
        }
        flat
    }

    /// One predictor-corrector step `O ← exp(dt·L*_block) O` on row words,
    /// with the truncation semantics of [`LindbladSpec::pc_step`]
    /// (`max_basis` counts rows over all species).
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
        let admit = admit_basis.unwrap_or(max_basis).max(max_basis);
        let tau_add = tau_add.unwrap_or(0.0);
        let leak = self.leakage_with_prune(basis, coeffs, protected, admit, tau_add)?;
        add_leakage_capped(basis, coeffs, leak, admit);
        let coeffs_predict = mf_expm::expm_apply_mf(self, basis, dt, coeffs, drop_tol);
        let leak2 = self.leakage_with_prune(basis, &coeffs_predict, protected, admit, tau_add)?;
        drop(coeffs_predict);
        add_leakage_capped(basis, coeffs, leak2, admit);
        *coeffs = mf_expm::expm_apply_mf(self, basis, dt, coeffs, drop_tol);
        prune_basis(basis, coeffs, drop_tol, protected);
        cap_basis(basis, coeffs, max_basis, protected);
        Ok(())
    }
}

impl ActionTerms for BlockLindbladSpec {
    fn scratch_qubits(&self) -> usize {
        self.n_qubits
    }

    fn action_terms(
        &self,
        p: &Word,
        s1: &mut Vec<u32>,
        s2: &mut Vec<u32>,
        lm: &mut FxHashMap<Word, Complex<f64>>,
    ) -> Vec<(Word, f64)> {
        self.compute_action_terms(p, s1, s2, lm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_pauli_string;

    fn spec(n: usize, h: &[(&str, f64)]) -> LindbladSpec {
        let terms: Vec<(String, f64)> = h.iter().map(|(s, c)| (s.to_string(), *c)).collect();
        LindbladSpec::new(n, &terms, &[]).unwrap()
    }

    fn row(block: &BlockLindbladSpec, s: &str, species: usize) -> Word {
        let (w, _) = parse_pauli_string(s, block.n_qubits()).unwrap();
        let mut codes = vec![0u8; block.n_qubits()];
        crate::codes_from_word(&w, &mut codes);
        block.encode_row(&codes, species).unwrap()
    }

    fn as_map(block: &BlockLindbladSpec, terms: Vec<(Word, f64)>) -> FxHashMap<(usize, String), f64> {
        let mut out = FxHashMap::default();
        for (w, c) in terms {
            let mut codes = vec![0u8; block.n_qubits()];
            let s = block.decode_row(&w, &mut codes);
            let st: String = codes
                .iter()
                .map(|&b| ['I', 'X', 'Z', 'Y'][b as usize])
                .collect();
            *out.entry((s, st)).or_insert(0.0) += c;
        }
        out
    }

    /// Rows round-trip through the digit encoding for every species.
    #[test]
    fn row_codec_roundtrip() {
        let specs: Vec<LindbladSpec> = (0..5).map(|_| spec(3, &[("ZII", 1.0)])).collect();
        let k = vec![vec![0.0; 5]; 5];
        let block = BlockLindbladSpec::new(3, specs, &k, &[]).unwrap();
        assert_eq!(block.n_digits(), 2);
        for s in 0..5 {
            let codes = [1u8, 3, 2];
            let w = block.encode_row(&codes, s).unwrap();
            let mut back = [0u8; 3];
            assert_eq!(block.decode_row(&w, &mut back), s);
            assert_eq!(back, codes);
        }
        assert!(block.encode_row(&[0, 0, 0], 5).is_err());
    }

    /// One qubit, two species: the species Lindbladian acts on its own
    /// block and the exchange moves weight backwards along the reaction.
    #[test]
    fn two_species_action_is_the_adjoint_generator() {
        let (a, b, k01, k10) = (0.4, 1.1, 0.3, 0.7);
        let specs = vec![spec(1, &[("Z", a)]), spec(1, &[("X", b)])];
        let k = vec![vec![0.0, k01], vec![k10, 0.0]];
        let block = BlockLindbladSpec::new(1, specs, &k, &[]).unwrap();
        // Row (1, Y): L_1*(Y) = i b [X, Y] = -2 b Z; inflow to species 0 at
        // k_01 (reaction 0 -> 1 moves the observable from 1 to 0); diagonal
        // -k_10 (species 1 leaves at k_10).
        let out = as_map(&block, block.action(&row(&block, "Y", 1)));
        assert!((out[&(1, "Z".into())] + 2.0 * b).abs() < 1e-14);
        assert!((out[&(0, "Y".into())] - k01).abs() < 1e-14);
        assert!((out[&(1, "Y".into())] + k10).abs() < 1e-14);
        assert_eq!(out.len(), 3);
        // Row (0, X): L_0*(X) = i a [Z, X] = -2 a Y; inflow to species 1 at k_10; diagonal -k_01.
        let out = as_map(&block, block.action(&row(&block, "X", 0)));
        assert!((out[&(0, "Y".into())] + 2.0 * a).abs() < 1e-14);
        assert!((out[&(1, "X".into())] - k10).abs() < 1e-14);
        assert!((out[&(0, "X".into())] + k01).abs() < 1e-14);
    }

    /// The total identity `⊕_c 1` (coefficient 1 on every species' identity)
    /// is stationary in the Heisenberg picture: the generator on that basis
    /// has zero row sums (its Schrödinger transpose has zero column sums,
    /// trace preservation).
    #[test]
    fn total_identity_is_conserved() {
        let specs: Vec<LindbladSpec> = (0..3).map(|_| spec(2, &[("ZZ", 0.5)])).collect();
        let k = vec![vec![0.0, 1.0, 2.0], vec![0.5, 0.0, 0.0], vec![3.0, 4.0, 0.0]];
        let block = BlockLindbladSpec::new(2, specs, &k, &[]).unwrap();
        let basis: Vec<Word> = (0..3).map(|s| row(&block, "II", s)).collect();
        let coo = block.generator(&basis);
        let mut rowsum = [0.0; 3];
        for (r, _, v) in coo {
            rowsum[r] += v;
        }
        for v in rowsum {
            assert!(v.abs() < 1e-14);
        }
        // every species' own identity has zero action apart from the exchange
        let out = as_map(&block, block.action(&basis[1]));
        assert!((out[&(1, "II".into())] + 0.5).abs() < 1e-14);  // -rate_out[1]
        assert!((out[&(0, "II".into())] - 1.0).abs() < 1e-14);  // k_01
        assert!((out[&(2, "II".into())] - 4.0).abs() < 1e-14);  // k_21
    }

    /// A matching permutes the letters: with perm (0 -> 1, 1 -> 0, 2 -> 2)
    /// between species 0 and 1, the row (1, "XZI") feeds (0, "ZXI"); the
    /// inverse is filled in for the reverse transition.
    #[test]
    fn matching_permutes_letters_and_inverse_is_completed() {
        let specs: Vec<LindbladSpec> = (0..2).map(|_| spec(3, &[])).collect();
        let k = vec![vec![0.0, 2.0], vec![5.0, 0.0]];
        let perm = vec![1usize, 2, 0]; // qubit 0 of c sits at 1 in d, 1 at 2, 2 at 0
        let block = BlockLindbladSpec::new(3, specs, &k, &[(0, 1, perm)]).unwrap();
        // row of species 1: "XZI"; into species 0 at k_01 = 2 with out[p] = P[perm[p]] -> "ZIX"
        let out = as_map(&block, block.action(&row(&block, "XZI", 1)));
        assert!((out[&(0, "ZIX".into())] - 2.0).abs() < 1e-14);
        assert!((out[&(1, "XZI".into())] + 5.0).abs() < 1e-14);
        // row of species 0: "ZIX"; into species 1 at k_10 = 5 with the inverse perm -> "XZI"
        let out = as_map(&block, block.action(&row(&block, "ZIX", 0)));
        assert!((out[&(1, "XZI".into())] - 5.0).abs() < 1e-14);
        assert!((out[&(0, "ZIX".into())] + 2.0).abs() < 1e-14);
        assert!(BlockLindbladSpec::new(3, vec![spec(3, &[]), spec(3, &[])], &k, &[(0, 1, vec![0, 0, 1])]).is_err());
    }

    /// A reset mask drops the inflow of strings with a non-identity letter on
    /// a masked position of the target species (after the permutation) and
    /// leaves everything else, the drain included, unchanged; the total
    /// identity stays conserved (trace preservation).
    #[test]
    fn reset_mask_drops_masked_inflow_only() {
        let specs: Vec<LindbladSpec> = (0..2).map(|_| spec(2, &[])).collect();
        let k = vec![vec![0.0, 2.0], vec![5.0, 0.0]];
        // species 0 = {F, pad}, species 1 = {F, P}: on 0 -> 1 the pad (position 1 of 0) is
        // replaced by the unit state, on 1 -> 0 the P (position 1 of 1) is traced out.
        let resets = vec![(0usize, 1usize, vec![1usize]), (1, 0, vec![1])];
        let block = BlockLindbladSpec::with_resets(2, specs, &k, &[], &resets).unwrap();
        assert_eq!(block.resets().len(), 2);
        // row (1, "ZI"): identity on the masked position -> feeds (0, "ZI") at k_01 = 2
        let out = as_map(&block, block.action(&row(&block, "ZI", 1)));
        assert!((out[&(0, "ZI".into())] - 2.0).abs() < 1e-14);
        assert!((out[&(1, "ZI".into())] + 5.0).abs() < 1e-14);
        assert_eq!(out.len(), 2);
        // row (1, "IZ"): P letter -> no inflow into species 0, the drain stays
        let out = as_map(&block, block.action(&row(&block, "IZ", 1)));
        assert!((out[&(1, "IZ".into())] + 5.0).abs() < 1e-14);
        assert_eq!(out.len(), 1);
        // row (0, "ZX"): pad letter -> no inflow into species 1
        let out = as_map(&block, block.action(&row(&block, "ZX", 0)));
        assert!((out[&(0, "ZX".into())] + 2.0).abs() < 1e-14);
        assert_eq!(out.len(), 1);
        // total identity conserved
        let basis: Vec<Word> = (0..2).map(|s| row(&block, "II", s)).collect();
        let mut rowsum = [0.0; 2];
        for (r, _, v) in block.generator(&basis) {
            rowsum[r] += v;
        }
        assert!(rowsum.iter().all(|v| v.abs() < 1e-14));
        // errors: no transition, bad qubit, duplicate pair
        let mk = || (0..2).map(|_| spec(2, &[])).collect::<Vec<_>>();
        let k0 = vec![vec![0.0, 2.0], vec![0.0, 0.0]];
        assert!(BlockLindbladSpec::with_resets(2, mk(), &k0, &[], &[(1, 0, vec![1])]).is_err());
        assert!(BlockLindbladSpec::with_resets(2, mk(), &k, &[], &[(0, 1, vec![2])]).is_err());
        assert!(BlockLindbladSpec::with_resets(2, mk(), &k, &[], &[(0, 1, vec![0]), (0, 1, vec![1])]).is_err());
        // no masks: bit-identical to `new`
        let a = BlockLindbladSpec::with_resets(2, mk(), &k, &[], &[]).unwrap();
        let b = BlockLindbladSpec::new(2, mk(), &k, &[]).unwrap();
        let ra = as_map(&a, a.action(&row(&a, "XZ", 1)));
        let rb = as_map(&b, b.action(&row(&b, "XZ", 1)));
        assert_eq!(ra, rb);
    }

    /// `pc_step` without truncation equals the Taylor series of the
    /// generator on the closed basis.
    #[test]
    fn pc_step_matches_generator_exponential() {
        let specs = vec![spec(1, &[("Z", 0.9)]), spec(1, &[("X", 0.3), ("Z", 0.2)])];
        let k = vec![vec![0.0, 0.8], vec![0.5, 0.0]];
        let block = BlockLindbladSpec::new(1, specs, &k, &[]).unwrap();
        // full basis: 2 species x 4 strings
        let basis: Vec<Word> = (0..2)
            .flat_map(|s| ["I", "X", "Z", "Y"].into_iter().map(move |p| (s, p)))
            .map(|(s, p)| row(&block, p, s))
            .collect();
        let n = basis.len();
        let mut g = vec![vec![0.0; n]; n];
        for (r, c, v) in block.generator(&basis) {
            g[r][c] += v;
        }
        let v0: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        let dt = 0.3;
        // Taylor series of exp(dt g) v0
        let mut result = v0.clone();
        let mut term = v0.clone();
        for kk in 1..60 {
            let mut next = vec![0.0; n];
            for r in 0..n {
                for c in 0..n {
                    next[r] += g[r][c] * term[c];
                }
            }
            for x in next.iter_mut() {
                *x *= dt / kk as f64;
            }
            for i in 0..n {
                result[i] += next[i];
            }
            term = next;
        }
        let mut b = basis.clone();
        let mut c = v0.clone();
        block
            .pc_step(&mut b, &mut c, dt, &[], &PcStepConfig::default())
            .unwrap();
        assert_eq!(b.len(), n);
        let pos = build_basis_index(&basis);
        for (w, val) in b.iter().zip(c.iter()) {
            let i = pos[w] as usize;
            assert!((val - result[i]).abs() < 1e-12, "{val} vs {}", result[i]);
        }
    }
}
