// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! Factor-form Kossakowski dissipator applied **without a per-pair table**:
//!
//! ```text
//! K_b = (G_b G_b† − G_neg,b G_neg,b†) ∘ W_b + P_b + diag(D_b)
//! ```
//!
//! per block — nmr-csim's `nmr-csim-kfactor/1` form in full (`G`: the PSD
//! part, `G_neg`: the clipped negative part of an unclipped hard-mask form,
//! `W`: the real Schur mask of a kernel cutoff, `P`: the sparse Hermitian
//! private off-diagonal part, `D`: the non-negative diagonal). The request's
//! "K = G G† + diag(D)" is the case `G_neg = ∅`, `W = 1`, `P = 0`; the same
//! loop serves every case (a block without mask and private part iterates
//! all its partners; the mask row and the private row are the pattern
//! otherwise).
//!
//! The dense Kossakowski path of this crate ([`crate::LindbladSpec::add_kossakowski`])
//! precompiles one [`crate::PairTemplate`] per operator pair `n ≤ m` with
//! `|K_nm|` above the compile floor — memory proportional to the number of
//! pairs, `Σ_b s_b²/2` for a block whose `K_b` is a Gram matrix. The factor
//! path stores `G_b` (`s_b × r_b`) and `D_b` instead and recomputes `K_nm`
//! when a pair is visited (`r_b` complex multiply-adds), so memory is
//! `Σ_b s_b r_b` while the Pauli work per live word stays that of the dense
//! path: with `p` the live word and
//!
//! ```text
//! D*(p) = ½ Σ_nm K_nm ( [A_n†, p] A_m + A_n† [p, A_m] )
//! ```
//!
//! only pairs where `p` anticommutes with a term of `A_n` or `A_m` contribute,
//! so the double sum collapses to `Σ_{n hit} Σ_m`. For every hit `n` the
//! commutator `c_n = [A_n†, p]` is formed once (≤ |A_n| words), and for every
//! partner `m` of its block (or of the mask's row `n`) the pair's words are
//! `c_n · A_m` (≤ |A_n|·|A_m| Pauli products) plus `A_n† · [p, A_m]` when `m` is
//! hit too. A folded pair `(n, m) + (m, n)`, `n ≠ m`, sums to `X + X†`, so the
//! **real part** of every coefficient of `X = K_nm(…)` is accumulated (the
//! imaginary parts are the anti-Hermitian half the conjugate pair cancels
//! exactly); the diagonal `n = m` carries the factor ½ and is Hermitian on its
//! own. Each unordered pair with at least one hit side is visited exactly once
//! (a both-hit pair from its lower index). Pairs with
//! `|K_nm| ≤ 1e-14 · max_n K_nn` are skipped — the dense path's compile floor,
//! applied on the fly (for a PSD block `max |K_nm| = max_n K_nn`).
//!
//! The module touches no `[Chunk; W_CHUNKS]` bit masks: hits are found through
//! the per-qubit operator index and a binary search in the sorted hit list, so
//! it depends on the word type only through [`crate::Word`] and the Pauli
//! primitives of `lib.rs`.

use fxhash::FxHashMap;
use num::Complex;

use crate::{
    Error, LindbladSpec, PauliTerm, Word, candidate_terms, comm_product, parse_pauli_string,
    pauli_mul, phase_factor,
};

/// Relative compile floor shared with the dense path: pairs with
/// `|K_nm| ≤ FLOOR_REL · max K` are skipped.
pub const FLOOR_REL: f64 = 1e-14;

/// Compressed sparse rows of a real block mask `W_b` (block-local indices).
#[derive(Clone, Debug)]
pub struct Csr {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

/// Compressed sparse rows of a complex block addend `P_b` (block-local indices).
#[derive(Clone, Debug)]
pub struct CsrC {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<Complex<f64>>,
}

/// One block of the factor form as the caller gives it.
#[derive(Clone, Debug)]
pub struct FactorBlockInput {
    /// `s_b`: the block's operators are the next `size` entries of `ops`.
    pub size: usize,
    /// `r_b`: columns of `G_b` (0 allowed: `K_b = diag(D_b)`).
    pub rank: usize,
    /// `G_b` row-major, `size × rank`.
    pub g: Vec<Complex<f64>>,
    /// `D_b`, `size` non-negative reals.
    pub d: Vec<f64>,
    /// Optional mask `W_b` (CSR on the block's operators; must carry the
    /// diagonal, which is where `D_b` lives). `None` = all ones.
    pub mask: Option<Csr>,
    /// `r_neg`: columns of `G_neg,b` (0: no negative part).
    pub rank_neg: usize,
    /// `G_neg,b` row-major, `size × rank_neg` (`K` gets `− G_neg G_neg†`).
    pub g_neg: Vec<Complex<f64>>,
    /// Optional Hermitian addend `P_b` (CSR, both triangles written, no
    /// diagonal needed — a diagonal entry is accepted and added).
    pub private: Option<CsrC>,
}

impl FactorBlockInput {
    /// The request's form `K_b = G_b G_b† + diag(D_b)` (no mask, no negative
    /// part, no private part).
    pub fn plain(size: usize, rank: usize, g: Vec<Complex<f64>>, d: Vec<f64>) -> Self {
        FactorBlockInput { size, rank, g, d, mask: None, rank_neg: 0, g_neg: Vec::new(), private: None }
    }
}

/// One compiled block.
#[derive(Clone, Debug)]
pub(crate) struct FactorBlock {
    /// First operator of the block in `kf_ops`.
    start: usize,
    size: usize,
    rank: usize,
    g: Vec<Complex<f64>>,
    d: Vec<f64>,
    mask: Option<Csr>,
    rank_neg: usize,
    g_neg: Vec<Complex<f64>>,
    private: Option<CsrC>,
    /// The floor's scale: `max_n (W_nn (‖G_n‖² + ‖G_neg,n‖²) + |P_nn| + D_n)`
    /// — equal to `max_n K_nn = max |K_nm|` for a PSD block (G_neg, P empty),
    /// an upper bound on `max |K_nm|` otherwise (Cauchy–Schwarz on the rows).
    max_diag: f64,
}

impl FactorBlock {
    #[inline(always)]
    fn g_row(&self, local: usize) -> &[Complex<f64>] {
        &self.g[local * self.rank..(local + 1) * self.rank]
    }

    /// `Σ_r G_nr conj(G_mr) − Σ_r G_neg,nr conj(G_neg,mr)` for block-local rows `n`, `m`.
    #[inline(always)]
    fn gram(&self, n: usize, m: usize) -> Complex<f64> {
        let (gn, gm) = (self.g_row(n), self.g_row(m));
        let mut acc = Complex::new(0.0, 0.0);
        for r in 0..self.rank {
            acc += gn[r] * gm[r].conj();
        }
        if self.rank_neg > 0 {
            let (hn, hm) = (
                &self.g_neg[n * self.rank_neg..(n + 1) * self.rank_neg],
                &self.g_neg[m * self.rank_neg..(m + 1) * self.rank_neg],
            );
            for r in 0..self.rank_neg {
                acc -= hn[r] * hm[r].conj();
            }
        }
        acc
    }

    /// `P_nm` (0 without a private part); rows are sorted at compile time.
    #[inline(always)]
    fn private_entry(&self, n: usize, m: usize) -> Complex<f64> {
        match &self.private {
            None => Complex::new(0.0, 0.0),
            Some(p) => {
                let row = &p.indices[p.indptr[n]..p.indptr[n + 1]];
                match row.binary_search(&m) {
                    Ok(j) => p.data[p.indptr[n] + j],
                    Err(_) => Complex::new(0.0, 0.0),
                }
            }
        }
    }

    fn validate(input: &FactorBlockInput, b: usize) -> Result<(), Error> {
        let s = input.size;
        if input.g.len() != s * input.rank {
            return Err(Error::KossakowskiFactor(format!(
                "block {b}: G has {} entries, expected size x rank = {} x {}",
                input.g.len(),
                s,
                input.rank
            )));
        }
        if input.d.len() != s {
            return Err(Error::KossakowskiFactor(format!(
                "block {b}: D has {} entries, expected size {s}",
                input.d.len()
            )));
        }
        for (i, &v) in input.d.iter().enumerate() {
            if !(v >= 0.0) || !v.is_finite() {
                return Err(Error::KossakowskiFactor(format!(
                    "block {b}: D[{i}] = {v} is not a non-negative finite rate"
                )));
            }
        }
        if input.g.iter().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
            return Err(Error::KossakowskiFactor(format!("block {b}: G has a non-finite entry")));
        }
        if let Some(m) = &input.mask {
            if m.indptr.len() != s + 1 {
                return Err(Error::KossakowskiFactor(format!(
                    "block {b}: mask indptr has {} entries, expected size + 1 = {}",
                    m.indptr.len(),
                    s + 1
                )));
            }
            if m.indices.len() != m.data.len() || m.indptr[s] != m.indices.len() || m.indptr[0] != 0 {
                return Err(Error::KossakowskiFactor(format!(
                    "block {b}: mask indices / data / indptr are inconsistent"
                )));
            }
            for n in 0..s {
                let (lo, hi) = (m.indptr[n], m.indptr[n + 1]);
                if lo > hi {
                    return Err(Error::KossakowskiFactor(format!("block {b}: mask indptr not monotone at row {n}")));
                }
                let mut has_diag = false;
                for j in lo..hi {
                    if m.indices[j] >= s {
                        return Err(Error::KossakowskiFactor(format!(
                            "block {b}: mask column {} out of range (size {s})",
                            m.indices[j]
                        )));
                    }
                    if !m.data[j].is_finite() {
                        return Err(Error::KossakowskiFactor(format!("block {b}: mask weight not finite")));
                    }
                    has_diag |= m.indices[j] == n;
                }
                if !has_diag {
                    return Err(Error::KossakowskiFactor(format!(
                        "block {b}: mask row {n} has no diagonal entry (D_n would be dropped)"
                    )));
                }
            }
        }
        if input.g_neg.len() != s * input.rank_neg {
            return Err(Error::KossakowskiFactor(format!(
                "block {b}: G_neg has {} entries, expected size x rank_neg = {} x {}",
                input.g_neg.len(),
                s,
                input.rank_neg
            )));
        }
        if input.g_neg.iter().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
            return Err(Error::KossakowskiFactor(format!("block {b}: G_neg has a non-finite entry")));
        }
        if let Some(p) = &input.private {
            if p.indptr.len() != s + 1 || p.indices.len() != p.data.len() || p.indptr[s] != p.indices.len() || p.indptr[0] != 0 {
                return Err(Error::KossakowskiFactor(format!("block {b}: private part CSR is inconsistent")));
            }
            for n in 0..s {
                if p.indptr[n] > p.indptr[n + 1] {
                    return Err(Error::KossakowskiFactor(format!("block {b}: private indptr not monotone at row {n}")));
                }
                for j in p.indptr[n]..p.indptr[n + 1] {
                    if p.indices[j] >= s {
                        return Err(Error::KossakowskiFactor(format!(
                            "block {b}: private column {} out of range (size {s})",
                            p.indices[j]
                        )));
                    }
                    if !p.data[j].re.is_finite() || !p.data[j].im.is_finite() {
                        return Err(Error::KossakowskiFactor(format!("block {b}: private entry not finite")));
                    }
                }
            }
        }
        Ok(())
    }

    /// Sort every CSR row by column (binary-search lookups; duplicate columns summed); returns the new indptr.
    fn sort_rows<T: Copy + std::ops::AddAssign>(indptr: &[usize], indices: &mut Vec<usize>, data: &mut Vec<T>) -> Vec<usize> {
        let mut new_idx = Vec::with_capacity(indices.len());
        let mut new_dat = Vec::with_capacity(data.len());
        let mut new_ptr = vec![0usize];
        for n in 0..indptr.len() - 1 {
            let mut row: Vec<(usize, T)> = (indptr[n]..indptr[n + 1]).map(|j| (indices[j], data[j])).collect();
            row.sort_by_key(|e| e.0);
            let row_start = new_idx.len();
            for (c, v) in row {
                if new_idx.len() > row_start && *new_idx.last().unwrap() == c {
                    *new_dat.last_mut().unwrap() += v;
                } else {
                    new_idx.push(c);
                    new_dat.push(v);
                }
            }
            new_ptr.push(new_idx.len());
        }
        *indices = new_idx;
        *data = new_dat;
        new_ptr
    }

    fn compile(input: &FactorBlockInput, start: usize) -> FactorBlock {
        let mut mask = input.mask.clone();
        if let Some(m) = mask.as_mut() {
            let ptr = m.indptr.clone();
            m.indptr = Self::sort_rows(&ptr, &mut m.indices, &mut m.data);
        }
        let mut private = input.private.clone();
        if let Some(p) = private.as_mut() {
            let ptr = p.indptr.clone();
            p.indptr = Self::sort_rows(&ptr, &mut p.indices, &mut p.data);
        }
        let mut blk = FactorBlock {
            start,
            size: input.size,
            rank: input.rank,
            g: input.g.clone(),
            d: input.d.clone(),
            mask,
            rank_neg: input.rank_neg,
            g_neg: input.g_neg.clone(),
            private,
            max_diag: 0.0,
        };
        let mut scale = 0.0_f64;
        for n in 0..blk.size {
            let w = blk.mask_diag(n).abs();
            let gg: f64 = blk.g_row(n).iter().map(|c| c.norm_sqr()).sum();
            let hh: f64 = if blk.rank_neg > 0 {
                blk.g_neg[n * blk.rank_neg..(n + 1) * blk.rank_neg].iter().map(|c| c.norm_sqr()).sum()
            } else {
                0.0
            };
            scale = scale.max(w * (gg + hh) + blk.private_entry(n, n).norm() + blk.d[n]);
        }
        blk.max_diag = scale;
        blk
    }

    /// The mask's diagonal weight of row `n` (1 without a mask).
    fn mask_diag(&self, n: usize) -> f64 {
        match &self.mask {
            None => 1.0,
            Some(m) => {
                for j in m.indptr[n]..m.indptr[n + 1] {
                    if m.indices[j] == n {
                        return m.data[j];
                    }
                }
                0.0
            }
        }
    }

    /// `K_nm` for block-local `n`, `m` with mask weight `w` already looked up.
    #[inline(always)]
    fn k_entry(&self, n: usize, m: usize, w: f64) -> Complex<f64> {
        let mut k = if w != 0.0 { self.gram(n, m) * w } else { Complex::new(0.0, 0.0) };
        if self.private.is_some() {
            k += self.private_entry(n, m);
        }
        if n == m {
            k.re += self.d[n];
        }
        k
    }

    /// Visit every partner `m` of row `n` with its mask weight: all of them
    /// (w = 1) without a mask; with a mask its row, plus the private part's
    /// columns outside the mask row (w = 0: `K_nm = P_nm`).
    #[inline(always)]
    fn for_each_partner(&self, n: usize, mut visit: impl FnMut(usize, f64)) {
        match &self.mask {
            None => {
                for m in 0..self.size {
                    visit(m, 1.0);
                }
            }
            Some(mask) => {
                let row = &mask.indices[mask.indptr[n]..mask.indptr[n + 1]];
                for (j, &m) in row.iter().enumerate() {
                    visit(m, mask.data[mask.indptr[n] + j]);
                }
                if let Some(p) = &self.private {
                    for &m in &p.indices[p.indptr[n]..p.indptr[n + 1]] {
                        if row.binary_search(&m).is_err() {
                            visit(m, 0.0);
                        }
                    }
                }
            }
        }
    }

    fn bytes(&self) -> usize {
        (self.g.len() + self.g_neg.len()) * 16
            + self.d.len() * 8
            + self
                .mask
                .as_ref()
                .map(|m| (m.indptr.len() + m.indices.len()) * 8 + m.data.len() * 8)
                .unwrap_or(0)
            + self
                .private
                .as_ref()
                .map(|p| (p.indptr.len() + p.indices.len()) * 8 + p.data.len() * 16)
                .unwrap_or(0)
    }
}

impl LindbladSpec {
    /// Add a factor-form Kossakowski dissipator (module docstring).
    ///
    /// `ops[i]` is the Pauli lincomb of `A_i`; the blocks partition `ops` in
    /// order (`Σ_b size_b = ops.len()`). `D*(O) = Σ_nm K_nm (A_n† O A_m −
    /// ½{A_n†A_m, O})` with `K_b = (G_b G_b†) ∘ W_b + diag(D_b)`; blocks do not
    /// couple. Positive semi-definiteness is the caller's responsibility, as on
    /// the dense path (a Gram form plus a non-negative diagonal is PSD; a masked
    /// block is PSD when the caller loaded its diagonal, which nmr-csim's
    /// kernel forms record). Coexists with a dense `add_kossakowski` block
    /// and with the jump operators of `new`. One factor set per spec: call
    /// [`Self::set_kossakowski_factor`] to replace it.
    pub fn add_kossakowski_factor(
        &mut self,
        ops: &[Vec<(String, Complex<f64>)>],
        blocks: &[FactorBlockInput],
    ) -> Result<(), Error> {
        if !self.kf_blocks.is_empty() {
            return Err(Error::KossakowskiFactor(
                "this spec already carries a factor block; use set_kossakowski_factor".into(),
            ));
        }
        let total: usize = blocks.iter().map(|b| b.size).sum();
        if total != ops.len() {
            return Err(Error::LengthMismatch {
                what: "kossakowski factor: block sizes vs ops",
                a: total,
                b: ops.len(),
            });
        }
        for (b, blk) in blocks.iter().enumerate() {
            FactorBlock::validate(blk, b)?;
        }
        // parse the operator table; per-qubit index
        let mut kf_ops: Vec<Vec<PauliTerm>> = Vec::with_capacity(ops.len());
        let mut kf_support: Vec<Vec<u32>> = vec![Vec::new(); self.n_qubits];
        for (i, op) in ops.iter().enumerate() {
            if op.is_empty() {
                return Err(Error::EmptyLincomb { index: i });
            }
            let mut terms: Vec<PauliTerm> = Vec::with_capacity(op.len());
            let mut union: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
            for (s, c) in op {
                let (word, support) = parse_pauli_string(s, self.n_qubits)?;
                union.extend(support.iter().copied());
                terms.push(PauliTerm { word, coeff: *c });
            }
            for q in union {
                kf_support[q as usize].push(i as u32);
            }
            kf_ops.push(terms);
        }
        let mut compiled = Vec::with_capacity(blocks.len());
        let mut block_of = Vec::with_capacity(ops.len());
        let mut start = 0;
        for (b, blk) in blocks.iter().enumerate() {
            compiled.push(FactorBlock::compile(blk, start));
            block_of.extend(std::iter::repeat_n(b as u32, blk.size));
            start += blk.size;
        }
        self.kf_ops = kf_ops;
        self.kf_support = kf_support;
        self.kf_block_of = block_of;
        self.kf_blocks = compiled;
        self.kf_floor = FLOOR_REL * self.kf_blocks.iter().map(|b| b.max_diag).fold(0.0, f64::max);
        Ok(())
    }

    /// Replace `G`, `D` and the masks in place, keeping the operator table of
    /// [`Self::add_kossakowski_factor`] (same number of blocks, same sizes).
    pub fn set_kossakowski_factor(&mut self, blocks: &[FactorBlockInput]) -> Result<(), Error> {
        if self.kf_blocks.is_empty() {
            return Err(Error::KossakowskiFactor(
                "set_kossakowski_factor: this spec was built without a factor block".into(),
            ));
        }
        if blocks.len() != self.kf_blocks.len() {
            return Err(Error::LengthMismatch {
                what: "set_kossakowski_factor: blocks",
                a: blocks.len(),
                b: self.kf_blocks.len(),
            });
        }
        for (b, (new, old)) in blocks.iter().zip(self.kf_blocks.iter()).enumerate() {
            if new.size != old.size {
                return Err(Error::KossakowskiFactor(format!(
                    "set_kossakowski_factor: block {b} has size {}, the spec's has {}",
                    new.size, old.size
                )));
            }
            FactorBlock::validate(new, b)?;
        }
        let starts: Vec<usize> = self.kf_blocks.iter().map(|b| b.start).collect();
        self.kf_blocks = blocks
            .iter()
            .zip(starts)
            .map(|(blk, start)| FactorBlock::compile(blk, start))
            .collect();
        self.kf_floor = FLOOR_REL * self.kf_blocks.iter().map(|b| b.max_diag).fold(0.0, f64::max);
        Ok(())
    }

    /// Number of operators of the factor dissipator (0 without one).
    pub fn num_kossakowski_factor_ops(&self) -> usize {
        self.kf_ops.len()
    }

    /// Number of factor blocks.
    pub fn num_kossakowski_factor_blocks(&self) -> usize {
        self.kf_blocks.len()
    }

    /// Bytes held by the factor data (`G`, `D`, masks) and by the parsed
    /// operator table, as `(factor_bytes, ops_bytes)`.
    pub fn kossakowski_factor_memory_bytes(&self) -> (usize, usize) {
        let factor: usize = self.kf_blocks.iter().map(|b| b.bytes()).sum();
        let ops: usize = self
            .kf_ops
            .iter()
            .map(|t| t.len() * std::mem::size_of::<PauliTerm>())
            .sum::<usize>()
            + self.kf_support.iter().map(|v| v.len() * 4).sum::<usize>();
        (factor, ops)
    }

    /// The absolute compile floor in force (`FLOOR_REL · max_n K_nn`).
    pub fn kossakowski_factor_floor(&self) -> f64 {
        self.kf_floor
    }

    /// Block `b` as the dense `K_b = (G G†) ∘ W + diag(D)` (inspection, tests).
    pub fn kossakowski_factor_dense_block(&self, b: usize) -> Result<Vec<Vec<Complex<f64>>>, Error> {
        let blk = self.kf_blocks.get(b).ok_or_else(|| {
            Error::KossakowskiFactor(format!("block {b} out of range ({} blocks)", self.kf_blocks.len()))
        })?;
        let s = blk.size;
        let mut out = vec![vec![Complex::new(0.0, 0.0); s]; s];
        for n in 0..s {
            blk.for_each_partner(n, |m, w| out[n][m] = blk.k_entry(n, m, w));
        }
        Ok(out)
    }

    /// Accumulate the factor dissipator's action on `p` into `local`
    /// (module docstring). `p_support` is `supp(p)`; `scratch` is reused for
    /// the candidate list.
    pub(crate) fn factor_action(
        &self,
        p: &Word,
        p_support: &[u32],
        scratch: &mut Vec<u32>,
        local: &mut FxHashMap<Word, Complex<f64>>,
    ) {
        candidate_terms(p_support, &self.kf_support, scratch);
        if scratch.is_empty() {
            return;
        }
        // hits: candidate operators with a non-zero commutator with p; per hit the two
        // commutators  c_n = [A_n†, p] = Σ_a conj(λ_a) [P_a, p]  and  [p, A_n] = −Σ_a λ_a [P_a, p],
        // with [P, p] = −i·eps·out from `comm_product`.
        let mut hits: Vec<u32> = Vec::with_capacity(scratch.len());
        let mut comm_left: Vec<Vec<(Word, Complex<f64>)>> = Vec::with_capacity(scratch.len());
        let mut comm_right: Vec<Vec<(Word, Complex<f64>)>> = Vec::with_capacity(scratch.len());
        for &n in scratch.iter() {
            let terms = &self.kf_ops[n as usize];
            let mut cl = Vec::with_capacity(terms.len());
            let mut cr = Vec::with_capacity(terms.len());
            for t in terms {
                let (out, eps) = comm_product(&t.word, p);
                if eps != 0.0 {
                    let c = Complex::new(0.0, -eps); // [P, p] = −i·eps·out
                    cl.push((out, t.coeff.conj() * c));
                    cr.push((out, -(t.coeff * c)));
                }
            }
            if !cl.is_empty() {
                hits.push(n);
                comm_left.push(cl);
                comm_right.push(cr);
            }
        }
        if hits.is_empty() {
            return;
        }
        let zero = Complex::new(0.0, 0.0);
        let floor = self.kf_floor;
        for (hi, &n_glob) in hits.iter().enumerate() {
            let n_glob = n_glob as usize;
            let blk = &self.kf_blocks[self.kf_block_of[n_glob] as usize];
            let n = n_glob - blk.start;
            let cl = &comm_left[hi];
            let a_n = &self.kf_ops[n_glob];
            // visit every partner m of n's block (mask row n when masked)
            let mut visit = |m: usize, w: f64| {
                let m_glob = blk.start + m;
                // a both-hit pair is taken from its lower index
                let hm = hits.binary_search(&(m_glob as u32)).ok();
                if hm.is_some() && m_glob < n_glob {
                    return;
                }
                let k_nm = blk.k_entry(n, m, w);
                if k_nm.norm() <= floor {
                    return;
                }
                let kf = if m == n { k_nm * 0.5 } else { k_nm };
                let a_m = &self.kf_ops[m_glob];
                // c_n · A_m
                for (wa, ca) in cl {
                    let cak = ca * kf;
                    for t in a_m {
                        let (s, ph) = pauli_mul(wa, &t.word);
                        let v = cak * t.coeff * phase_factor(ph);
                        if v.re != 0.0 {
                            local.entry(s).or_insert(zero).re += v.re;
                        }
                    }
                }
                // A_n† · [p, A_m] when m is hit too
                if let Some(hm) = hm {
                    let cr = &comm_right[hm];
                    for t in a_n {
                        let ck = t.coeff.conj() * kf;
                        for (wb, cb) in cr {
                            let (s, ph) = pauli_mul(&t.word, wb);
                            let v = ck * cb * phase_factor(ph);
                            if v.re != 0.0 {
                                local.entry(s).or_insert(zero).re += v.re;
                            }
                        }
                    }
                }
            };
            blk.for_each_partner(n, &mut visit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JumpInput, word_from_codes};

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    fn pauli_string(codes: &[u8]) -> String {
        codes.iter().map(|&c| match c { 0 => 'I', 1 => 'X', 2 => 'Z', _ => 'Y' }).collect()
    }

    /// Two-site lowering/raising-like lincombs on random site pairs (4 Pauli terms each, complex coefficients),
    /// the shape of nmr-csim's dipolar tensor components.
    fn random_ops(n: usize, count: usize, seed: &mut u64) -> Vec<Vec<(String, Complex<f64>)>> {
        let mut ops = Vec::new();
        for _ in 0..count {
            let i = ((lcg(seed) + 0.5) * n as f64) as usize % n;
            let mut j = ((lcg(seed) + 0.5) * n as f64) as usize % n;
            if j == i {
                j = (i + 1) % n;
            }
            let mut terms = Vec::new();
            for &(a, b) in &[(1u8, 1u8), (1, 3), (3, 1), (3, 3)] {
                let mut codes = vec![0u8; n];
                codes[i] = a;
                codes[j] = b;
                terms.push((pauli_string(&codes), Complex::new(lcg(seed), lcg(seed))));
            }
            ops.push(terms);
        }
        ops
    }

    fn random_factor(sizes: &[usize], ranks: &[usize], seed: &mut u64, masked: bool) -> Vec<FactorBlockInput> {
        sizes
            .iter()
            .zip(ranks)
            .map(|(&s, &r)| {
                let g: Vec<Complex<f64>> = (0..s * r).map(|_| Complex::new(lcg(seed), lcg(seed))).collect();
                let d: Vec<f64> = (0..s).map(|_| lcg(seed).abs() * 0.3).collect();
                let mask = if masked {
                    // symmetric 0/1 pattern with the diagonal: keep (n, m) when (n + m) % 3 != 1 or n == m
                    let mut indptr = vec![0usize];
                    let mut indices = Vec::new();
                    let mut data = Vec::new();
                    for n in 0..s {
                        for m in 0..s {
                            if n == m || (n + m) % 3 != 1 {
                                indices.push(m);
                                data.push(if n == m { 1.0 } else { 0.7 });
                            }
                        }
                        indptr.push(indices.len());
                    }
                    Some(Csr { indptr, indices, data })
                } else {
                    None
                };
                FactorBlockInput { size: s, rank: r, g, d, mask, rank_neg: 0, g_neg: Vec::new(), private: None }
            })
            .collect()
    }

    /// The general form: a negative part of rank 1 and a Hermitian private part on the pairs (n, n+1).
    fn generalise(blocks: &mut [FactorBlockInput], seed: &mut u64) {
        for b in blocks.iter_mut() {
            let s = b.size;
            b.rank_neg = 1;
            b.g_neg = (0..s).map(|_| Complex::new(0.3 * lcg(seed), 0.3 * lcg(seed))).collect();
            let mut indptr = vec![0usize];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            let v: Vec<Complex<f64>> = (0..s).map(|_| Complex::new(lcg(seed), lcg(seed))).collect();
            for n in 0..s {
                // P_{n,n+1} = v_n, P_{n+1,n} = conj(v_n); rows written in descending column order on purpose
                if n + 1 < s {
                    indices.push(n + 1);
                    data.push(v[n]);
                }
                if n > 0 {
                    indices.push(n - 1);
                    data.push(v[n - 1].conj());
                }
                indptr.push(indices.len());
            }
            b.private = Some(CsrC { indptr, indices, data });
        }
    }

    fn dense_of(blocks: &[FactorBlockInput]) -> Vec<Vec<Complex<f64>>> {
        let total: usize = blocks.iter().map(|b| b.size).sum();
        let mut k = vec![vec![Complex::new(0.0, 0.0); total]; total];
        let mut off = 0;
        for b in blocks {
            for n in 0..b.size {
                for m in 0..b.size {
                    let w = match &b.mask {
                        None => 1.0,
                        Some(mk) => {
                            let mut w = 0.0;
                            for j in mk.indptr[n]..mk.indptr[n + 1] {
                                if mk.indices[j] == m {
                                    w = mk.data[j];
                                }
                            }
                            w
                        }
                    };
                    let mut acc = Complex::new(0.0, 0.0);
                    for r in 0..b.rank {
                        acc += b.g[n * b.rank + r] * b.g[m * b.rank + r].conj();
                    }
                    for r in 0..b.rank_neg {
                        acc -= b.g_neg[n * b.rank_neg + r] * b.g_neg[m * b.rank_neg + r].conj();
                    }
                    let mut v = acc * w;
                    if let Some(p) = &b.private {
                        for j in p.indptr[n]..p.indptr[n + 1] {
                            if p.indices[j] == m {
                                v += p.data[j];
                            }
                        }
                    }
                    if n == m {
                        v.re += b.d[n];
                    }
                    k[off + n][off + m] = v;
                }
            }
            off += b.size;
        }
        k
    }

    fn action_map(spec: &LindbladSpec, p: &Word) -> FxHashMap<Word, f64> {
        spec.action(p).into_iter().collect()
    }

    fn all_words(n: usize, max_weight: usize) -> Vec<Word> {
        let mut out = Vec::new();
        let total = 4usize.pow(n as u32);
        for idx in 0..total {
            let mut codes = vec![0u8; n];
            let mut x = idx;
            for q in 0..n {
                codes[q] = (x % 4) as u8;
                x /= 4;
            }
            let weight = codes.iter().filter(|&&c| c != 0).count();
            if weight > 0 && weight <= max_weight {
                out.push(word_from_codes(&codes).unwrap());
            }
        }
        out
    }

    fn assert_match(a: &FxHashMap<Word, f64>, b: &FxHashMap<Word, f64>, tol: f64, scale: f64) {
        let mut keys: Vec<&Word> = a.keys().chain(b.keys()).collect();
        keys.sort_by_key(|w| crate::word_hash(w));
        keys.dedup();
        for w in keys {
            let (x, y) = (a.get(w).copied().unwrap_or(0.0), b.get(w).copied().unwrap_or(0.0));
            assert!((x - y).abs() <= tol * scale, "mismatch {x} vs {y} (scale {scale})");
        }
    }

    fn check_factor_equals_dense(masked: bool, with_dense_block: bool, general: bool, seed0: u64) {
        let n = 5;
        let mut seed = seed0;
        let sizes = [4usize, 3, 1, 5];
        let ranks = [2usize, 3, 1, 0];
        let ops = random_ops(n, sizes.iter().sum(), &mut seed);
        let mut blocks = random_factor(&sizes, &ranks, &mut seed, masked);
        if general {
            generalise(&mut blocks, &mut seed);
        }
        let k = dense_of(&blocks);
        let h = vec![("XXIII".to_string(), 0.7), ("IZZII".to_string(), -0.4), ("IIIYX".to_string(), 0.2)];
        let jumps = vec![JumpInput { lincomb: vec![("ZIIII".to_string(), Complex::new(1.0, 0.0))], rate: 0.3 }];
        let mut dense = LindbladSpec::new(n, &h, &jumps).unwrap();
        let mut factor = LindbladSpec::new(n, &h, &jumps).unwrap();
        if with_dense_block {
            // an extra dense block on both (the t1_t2 block of nmr-csim): ops + K appended block-diagonally
            let extra_ops = random_ops(n, 2, &mut seed);
            let kx = vec![
                vec![Complex::new(0.9, 0.0), Complex::new(0.2, 0.1)],
                vec![Complex::new(0.2, -0.1), Complex::new(0.5, 0.0)],
            ];
            let mut all_ops = ops.clone();
            all_ops.extend(extra_ops.iter().cloned());
            let total = ops.len() + 2;
            let mut kk = vec![vec![Complex::new(0.0, 0.0); total]; total];
            for (i, row) in k.iter().enumerate() {
                kk[i][..ops.len()].copy_from_slice(row);
            }
            for i in 0..2 {
                for j in 0..2 {
                    kk[ops.len() + i][ops.len() + j] = kx[i][j];
                }
            }
            dense.add_kossakowski(&all_ops, &kk).unwrap();
            factor.add_kossakowski(&extra_ops, &kx).unwrap();
        } else {
            dense.add_kossakowski(&ops, &k).unwrap();
        }
        factor.add_kossakowski_factor(&ops, &blocks).unwrap();
        // the reconstructed blocks equal the input K
        let mut off = 0;
        for (b, blk) in blocks.iter().enumerate() {
            let kb = factor.kossakowski_factor_dense_block(b).unwrap();
            for i in 0..blk.size {
                for j in 0..blk.size {
                    assert!((kb[i][j] - k[off + i][off + j]).norm() < 1e-15);
                }
            }
            off += blk.size;
        }
        let scale = k.iter().flat_map(|r| r.iter().map(|c| c.norm())).fold(0.0, f64::max);
        for p in all_words(n, 3) {
            let a = action_map(&dense, &p);
            let b = action_map(&factor, &p);
            assert_match(&a, &b, 1e-12, scale.max(1.0));
        }
    }

    #[test]
    fn factor_action_matches_dense_unmasked() {
        check_factor_equals_dense(false, false, false, 11);
    }

    #[test]
    fn factor_action_matches_dense_masked() {
        check_factor_equals_dense(true, false, false, 23);
    }

    #[test]
    fn factor_coexists_with_dense_block() {
        check_factor_equals_dense(false, true, false, 37);
    }

    #[test]
    fn factor_action_matches_dense_general_form_unmasked() {
        // G_neg (indefinite block) + private off-diagonal part, no mask
        check_factor_equals_dense(false, false, true, 41);
    }

    #[test]
    fn factor_action_matches_dense_general_form_masked() {
        // mask + G_neg + private part whose pairs may lie outside the mask row
        check_factor_equals_dense(true, true, true, 43);
    }

    #[test]
    fn setter_matches_fresh_construction() {
        let n = 4;
        let mut seed = 5u64;
        let sizes = [3usize, 2];
        let ranks = [2usize, 1];
        let ops = random_ops(n, 5, &mut seed);
        let b1 = random_factor(&sizes, &ranks, &mut seed, false);
        let b2 = random_factor(&sizes, &[1, 2], &mut seed, true);
        let h = vec![("XXII".to_string(), 0.3)];
        let mut a = LindbladSpec::new(n, &h, &[]).unwrap();
        a.add_kossakowski_factor(&ops, &b1).unwrap();
        a.set_kossakowski_factor(&b2).unwrap();
        let mut fresh = LindbladSpec::new(n, &h, &[]).unwrap();
        fresh.add_kossakowski_factor(&ops, &b2).unwrap();
        for p in all_words(n, 4) {
            let x = action_map(&a, &p);
            let y = action_map(&fresh, &p);
            assert_eq!(x.len(), y.len());
            for (w, c) in &x {
                assert_eq!(y[w].to_bits(), c.to_bits(), "setter must be bit-identical to a fresh build");
            }
        }
        assert!(a.set_kossakowski_factor(&b1[..1]).is_err());
        assert!(fresh.add_kossakowski_factor(&ops, &b1).is_err());
    }

    #[test]
    fn rejects_bad_input() {
        let n = 3;
        let mut seed = 9u64;
        let ops = random_ops(n, 3, &mut seed);
        let h = vec![("XXI".to_string(), 0.3)];
        let mut spec = LindbladSpec::new(n, &h, &[]).unwrap();
        let mut bad = random_factor(&[3], &[2], &mut seed, false);
        bad[0].d[1] = -1.0;
        assert!(spec.add_kossakowski_factor(&ops, &bad).is_err());
        let short = random_factor(&[2], &[2], &mut seed, false);
        assert!(spec.add_kossakowski_factor(&ops, &short).is_err());
        let mut no_diag = random_factor(&[3], &[1], &mut seed, true);
        let mk = no_diag[0].mask.as_mut().unwrap();
        // drop the diagonal of row 0
        let keep: Vec<usize> = (mk.indptr[0]..mk.indptr[1]).filter(|&j| mk.indices[j] != 0).collect();
        let shift = (mk.indptr[1] - mk.indptr[0]) - keep.len();
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for &j in &keep {
            indices.push(mk.indices[j]);
            data.push(mk.data[j]);
        }
        indices.extend_from_slice(&mk.indices[mk.indptr[1]..]);
        data.extend_from_slice(&mk.data[mk.indptr[1]..]);
        for v in mk.indptr.iter_mut().skip(1) {
            *v -= shift;
        }
        mk.indices = indices;
        mk.data = data;
        assert!(spec.add_kossakowski_factor(&ops, &no_diag).is_err());
        assert!(spec.set_kossakowski_factor(&no_diag).is_err());
    }

    #[test]
    fn dense_path_untouched_without_factor() {
        // a spec without a factor block reports no factor data and the compile floor 0
        let spec = LindbladSpec::new(2, &[("XX".to_string(), 1.0)], &[]).unwrap();
        assert_eq!(spec.num_kossakowski_factor_ops(), 0);
        assert_eq!(spec.kossakowski_factor_memory_bytes(), (0, 0));
        assert_eq!(spec.kossakowski_factor_floor(), 0.0);
    }
}
