// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! PyO3 wrapper around [`ppvm_lindblad::LindbladSpec`].
//!
//! All algorithmic work — Pauli arithmetic, active-site iteration, and the
//! dissipator branches (Hermitian Pauli fast path vs general complex Pauli
//! sum) — lives in the [`ppvm_lindblad`] crate. This module is responsible
//! only for the Python boundary: decoding the `(N, n_qubits)` numpy uint8
//! arrays into [`ppvm_lindblad::Word`] vectors, re-encoding outputs back
//! into numpy, and **dispatching on the Pauli-word width**.
//!
//! # Word-width dispatch
//!
//! `ppvm_lindblad` is generic over the chunk count `W` of its packed Pauli
//! word (`Word<W>` holds up to `W · CHUNK_BITS` qubits, 64 bits per chunk on
//! 64-bit targets). This extension compiles the widths [`WORD_WIDTHS`] =
//! {2, 4, 8, 16, 17} (128 / 256 / 512 / 1 024 / 1 088 qubits) and picks, at
//! construction, the smallest one that holds `n_qubits` — as the tableau
//! classes are instantiated per width range in `interface_tableau.rs`.
//! `W = 2` is the historical 128-qubit word and its code path, bit for bit.
//! A caller may force a wider instantiation with `word_width=` (tests,
//! benchmarks). `n_qubits` above the largest compiled width is refused at
//! construction with a message naming the knob ([`pick_width`]);
//! [`lindblad_max_qubits`] / [`lindblad_word_widths`] expose the compiled
//! maximum to Python (`ppvm.lindblad.max_qubits()`).

use std::collections::HashMap;

use num::Complex;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use ppvm_lindblad::{
    CHUNK_BITS, Csr, CsrC, FactorBlockInput, JumpInput, LindbladSpec as CoreSpec, Word,
    codes_from_word, max_qubits_for, word_from_codes,
};
use pyo3::{exceptions::PyValueError, prelude::*};

type PyPauliMap<'py> = (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<f64>>);
type PyPauliMapComplex<'py> = (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<Complex64>>);
type PyCoo<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<f64>>,
);

/// The Pauli-word widths (chunks of `CHUNK_BITS` bits) compiled into this
/// extension, ascending: 128 / 256 / 512 / 1 024 / 1 088 qubits on a 64-bit
/// target. Adding a width means one more entry here, one more variant in
/// `AnySpec` / `AnyBlock` and one more arm in `with_width!` / `dispatch!`.
pub(crate) const WORD_WIDTHS: [usize; 5] = [2, 4, 8, 16, 17];

/// The largest `n_qubits` any compiled width holds.
pub(crate) fn max_qubits() -> usize {
    max_qubits_for(WORD_WIDTHS[WORD_WIDTHS.len() - 1])
}

/// The width to instantiate for `n_qubits`: the smallest compiled one that
/// holds it, or `requested` (validated). The refusal names the knob.
pub(crate) fn pick_width(n_qubits: usize, requested: Option<usize>, what: &str) -> PyResult<usize> {
    let widths: Vec<String> = WORD_WIDTHS
        .iter()
        .map(|&w| format!("{w} ({} qubits)", max_qubits_for(w)))
        .collect();
    match requested {
        Some(w) => {
            if !WORD_WIDTHS.contains(&w) {
                return Err(PyValueError::new_err(format!(
                    "{what}: word_width {w} is not compiled into this ppvm build; the compiled \
                     Pauli-word widths (chunks of {CHUNK_BITS} bits) are {}",
                    widths.join(", ")
                )));
            }
            if n_qubits > max_qubits_for(w) {
                return Err(PyValueError::new_err(format!(
                    "{what}: word_width {w} holds at most {} qubits; got n_qubits = {n_qubits} \
                     (leave word_width unset to dispatch automatically)",
                    max_qubits_for(w)
                )));
            }
            Ok(w)
        }
        None => WORD_WIDTHS
            .iter()
            .copied()
            .find(|&w| n_qubits <= max_qubits_for(w))
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{what} supports n_qubits ≤ {}; got {n_qubits}. The limit is the largest \
                     Pauli-word width compiled into this ppvm build (W = {} chunks of \
                     {CHUNK_BITS} bits; compiled widths: {}); a larger system needs a build with \
                     a wider entry in WORD_WIDTHS (crates/ppvm-python-native/src/lindblad.rs). \
                     The installed maximum is ppvm.lindblad.max_qubits()",
                    max_qubits(),
                    WORD_WIDTHS[WORD_WIDTHS.len() - 1],
                    widths.join(", ")
                ))
            }),
    }
}

/// Run `$body` with `$W` bound to the const width `$w` (one arm per entry
/// of [`WORD_WIDTHS`]).
macro_rules! with_width {
    ($w:expr, $W:ident => $body:block) => {
        match $w {
            2 => {
                const $W: usize = 2;
                $body
            }
            4 => {
                const $W: usize = 4;
                $body
            }
            8 => {
                const $W: usize = 8;
                $body
            }
            16 => {
                const $W: usize = 16;
                $body
            }
            17 => {
                const $W: usize = 17;
                $body
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "internal: no Pauli-word width {other} compiled in"
                )));
            }
        }
    };
}
pub(crate) use with_width;

/// A `LindbladSpec` at one of the compiled widths.
pub(crate) enum AnySpec {
    W2(CoreSpec<2>),
    W4(CoreSpec<4>),
    W8(CoreSpec<8>),
    W16(CoreSpec<16>),
    W17(CoreSpec<17>),
}

macro_rules! impl_from_spec {
    ($($v:ident = $w:literal),*) => {
        $(impl From<CoreSpec<$w>> for AnySpec {
            fn from(s: CoreSpec<$w>) -> Self {
                AnySpec::$v(s)
            }
        })*
    };
}
impl_from_spec!(W2 = 2, W4 = 4, W8 = 8, W16 = 16, W17 = 17);

/// `$body` with `$s` bound to the inner `&CoreSpec<W>` (or `&mut`) of an
/// [`AnySpec`], whichever width it holds.
macro_rules! dispatch {
    ($e:expr, $s:ident => $body:expr) => {
        match $e {
            AnySpec::W2($s) => $body,
            AnySpec::W4($s) => $body,
            AnySpec::W8($s) => $body,
            AnySpec::W16($s) => $body,
            AnySpec::W17($s) => $body,
        }
    };
}

impl AnySpec {
    pub(crate) fn word_width(&self) -> usize {
        dispatch!(self, s => word_width_of(s))
    }
}

fn word_width_of<const W: usize>(_: &CoreSpec<W>) -> usize {
    W
}

fn word_bytes_of<const W: usize>(_: &CoreSpec<W>) -> usize {
    std::mem::size_of::<Word<W>>()
}

pub(crate) fn map_err(e: ppvm_lindblad::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `(pauli_string, re, im)` triples per operator into complex lincombs.
fn lincombs_from_triples(ops: Vec<Vec<(String, f64, f64)>>) -> Vec<Vec<(String, Complex<f64>)>> {
    ops.into_iter()
        .map(|op| {
            op.into_iter()
                .map(|(s, re, im)| (s, Complex::new(re, im)))
                .collect()
        })
        .collect()
}

/// Validate COO arrays (equal lengths, non-negative indices) and turn them
/// into the core's `(row, col, value)` triples.
fn coo_entries(
    rows: &PyReadonlyArray1<'_, i64>,
    cols: &PyReadonlyArray1<'_, i64>,
    vals: &PyReadonlyArray1<'_, Complex64>,
    what: &str,
) -> PyResult<Vec<(usize, usize, Complex<f64>)>> {
    let (r, c, v) = (rows.as_array(), cols.as_array(), vals.as_array());
    if r.len() != c.len() || r.len() != v.len() {
        return Err(PyValueError::new_err(format!(
            "{what}: rows, cols and vals must have the same length; got {}, {} and {}",
            r.len(),
            c.len(),
            v.len()
        )));
    }
    let mut out = Vec::with_capacity(r.len());
    for ((&ri, &ci), &vi) in r.iter().zip(c.iter()).zip(v.iter()) {
        if ri < 0 || ci < 0 {
            return Err(PyValueError::new_err(format!(
                "{what}: operator indices must be non-negative; got ({ri}, {ci})"
            )));
        }
        out.push((ri as usize, ci as usize, Complex::new(vi.re, vi.im)));
    }
    Ok(out)
}

/// Reject a basis that contains the same Pauli word at two distinct rows.
/// Duplicate rows would silently overwrite each other in the generator's
/// row-index map and produce an incorrect sparse matrix.
fn assert_basis_unique<const W: usize>(basis: &[Word<W>]) -> PyResult<()> {
    let mut seen: HashMap<&Word<W>, usize> = HashMap::with_capacity(basis.len());
    for (i, w) in basis.iter().enumerate() {
        if let Some(prev) = seen.insert(w, i) {
            return Err(PyValueError::new_err(format!(
                "basis contains duplicate Pauli word at row {prev} and row {i}"
            )));
        }
    }
    Ok(())
}

/// Decode a `(N, n_qubits)` uint8 ndarray view into `N` packed [`Word`]s.
pub(crate) fn decode_basis<const W: usize>(
    view: &numpy::ndarray::ArrayView2<u8>,
    n_qubits: usize,
) -> PyResult<Vec<Word<W>>> {
    let n_basis = view.shape()[0];
    let n_cols = view.shape()[1];
    if n_cols != n_qubits {
        return Err(PyValueError::new_err(format!(
            "basis has {n_cols} columns but spec.n_qubits = {n_qubits}"
        )));
    }
    let mut out = Vec::with_capacity(n_basis);
    let mut row_buf = vec![0u8; n_qubits];
    for i in 0..n_basis {
        let row = view.row(i);
        for (q, slot) in row_buf.iter_mut().enumerate() {
            *slot = row[q];
        }
        out.push(word_from_codes::<W>(&row_buf).map_err(map_err)?);
    }
    Ok(out)
}

/// Pack `Vec<(Word, f64)>` into the standard PyO3 return shape.
pub(crate) fn pack_pauli_map<'py, const W: usize>(
    py: Python<'py>,
    pairs: Vec<(Word<W>, f64)>,
    n_qubits: usize,
) -> PyResult<PyPauliMap<'py>> {
    let m = pairs.len();
    let mut basis = vec![0u8; m * n_qubits];
    let mut coeffs = vec![0f64; m];
    for (i, (w, c)) in pairs.into_iter().enumerate() {
        codes_from_word(&w, &mut basis[i * n_qubits..(i + 1) * n_qubits]);
        coeffs[i] = c;
    }
    let basis_arr = basis
        .into_pyarray(py)
        .reshape([m, n_qubits])
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok((basis_arr, coeffs.into_pyarray(py)))
}

/// Pack `(words, complex coeffs)` into the complex return shape.
pub(crate) fn pack_pauli_map_complex<'py, const W: usize>(
    py: Python<'py>,
    words: &[Word<W>],
    coeffs: &[Complex<f64>],
    n_qubits: usize,
) -> PyResult<PyPauliMapComplex<'py>> {
    let m = words.len();
    let mut out_basis = vec![0u8; m * n_qubits];
    for (i, w) in words.iter().enumerate() {
        codes_from_word(w, &mut out_basis[i * n_qubits..(i + 1) * n_qubits]);
    }
    let out_coeffs: Vec<Complex64> = coeffs.iter().map(|c| Complex64::new(c.re, c.im)).collect();
    let basis_arr = out_basis
        .into_pyarray(py)
        .reshape([m, n_qubits])
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok((basis_arr, out_coeffs.into_pyarray(py)))
}

/// Shared argument decoding of the triple-encoded constructor inputs.
pub(crate) struct SpecInputs {
    pub(crate) h: Vec<(String, f64)>,
    pub(crate) jumps: Vec<JumpInput>,
    pub(crate) ops: Vec<Vec<(String, Complex<f64>)>>,
    pub(crate) k: Vec<Vec<Complex<f64>>>,
}

impl SpecInputs {
    pub(crate) fn new(
        h_terms: Vec<String>,
        h_coeffs: Vec<f64>,
        jump_lincombs: Vec<Vec<(String, f64, f64)>>,
        jump_rates: Vec<f64>,
        kossakowski_ops: Vec<Vec<(String, f64, f64)>>,
        kossakowski_k: Vec<Vec<(f64, f64)>>,
        prefix: &str,
    ) -> PyResult<Self> {
        if h_terms.len() != h_coeffs.len() {
            return Err(PyValueError::new_err(format!(
                "{prefix}h_terms and h_coeffs must have the same length"
            )));
        }
        if jump_lincombs.len() != jump_rates.len() {
            return Err(PyValueError::new_err(format!(
                "{prefix}jump_lincombs and jump_rates must have the same length"
            )));
        }
        let h: Vec<(String, f64)> = h_terms.into_iter().zip(h_coeffs).collect();
        let jumps: Vec<JumpInput> = jump_lincombs
            .into_iter()
            .zip(jump_rates)
            .map(|(lincomb, rate)| JumpInput {
                lincomb: lincomb
                    .into_iter()
                    .map(|(s, re, im)| (s, Complex::new(re, im)))
                    .collect(),
                rate,
            })
            .collect();
        let ops: Vec<Vec<(String, Complex<f64>)>> = kossakowski_ops
            .into_iter()
            .map(|op| {
                op.into_iter()
                    .map(|(s, re, im)| (s, Complex::new(re, im)))
                    .collect()
            })
            .collect();
        let k: Vec<Vec<Complex<f64>>> = kossakowski_k
            .into_iter()
            .map(|row| row.into_iter().map(|(re, im)| Complex::new(re, im)).collect())
            .collect();
        Ok(Self { h, jumps, ops, k })
    }

    /// Build the spec at width `W` (the Kossakowski block added when given).
    pub(crate) fn build<const W: usize>(&self, n_qubits: usize) -> PyResult<CoreSpec<W>> {
        let mut inner = CoreSpec::<W>::new(n_qubits, &self.h, &self.jumps).map_err(map_err)?;
        if !self.ops.is_empty() || !self.k.is_empty() {
            inner.add_kossakowski(&self.ops, &self.k).map_err(map_err)?;
        }
        Ok(inner)
    }
}

/// Decode an optional protected array at width `W`.
fn protected_words<const W: usize>(
    protected: Option<&PyReadonlyArray2<'_, u8>>,
    n_qubits: usize,
) -> PyResult<Vec<Word<W>>> {
    match protected {
        Some(p) => decode_basis::<W>(&p.as_array(), n_qubits),
        None => Ok(Vec::new()),
    }
}

/// A 1-D float64 array of length `n` as a `Vec<f64>` (copying a
/// non-contiguous view); `ValueError` naming `what` on a mismatch.
pub(crate) fn f64_vec(arr: &Bound<'_, PyAny>, n: usize, what: &str) -> PyResult<Vec<f64>> {
    let arr: PyReadonlyArray1<'_, f64> = arr.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{what}: expected a 1-D float64 ndarray of shape ({n},)"
        ))
    })?;
    let got = arr.len()?;
    if got != n {
        return Err(PyValueError::new_err(format!(
            "{what}: expected {n} entries, got {got}"
        )));
    }
    Ok(arr.as_array().iter().copied().collect())
}

/// The compiled maximum `n_qubits` of `LindbladSpec` / `BlockLindbladSpec`
/// (the block spec spends `ceil(log4 n_species)` of these on species digits).
#[pyfunction]
pub fn lindblad_max_qubits() -> usize {
    max_qubits()
}

/// The compiled Pauli-word widths as `[(w_chunks, max_qubits), ...]`, ascending.
#[pyfunction]
pub fn lindblad_word_widths() -> Vec<(usize, usize)> {
    WORD_WIDTHS.iter().map(|&w| (w, max_qubits_for(w))).collect()
}

/// Bits per Pauli-word chunk (64 on 64-bit targets).
#[pyfunction]
pub fn lindblad_chunk_bits() -> usize {
    CHUNK_BITS
}

/// The width `LindbladSpec(n_qubits)` dispatches to, as `(w_chunks,
/// max_qubits)`; `ValueError` above the compiled maximum (the constructor's
/// message).
#[pyfunction]
pub fn lindblad_word_width_for(n_qubits: usize) -> PyResult<(usize, usize)> {
    let w = pick_width(n_qubits, None, "LindbladSpec")?;
    Ok((w, max_qubits_for(w)))
}

type PyMaskFlat<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, i64>,
);
type PyPrivateFlat<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, Complex64>,
    PyReadonlyArray1<'py, i64>,
);

/// One block's slice of a concatenated block-local CSR (`indptr` slices by
/// `offsets`, `indices` / `data` by the running nnz).
fn csr_block<'a, T: Copy>(
    what: &str,
    b: usize,
    s: usize,
    indptr: &[i64],
    indices: &[i64],
    data: &'a [T],
    offsets: &[i64],
    nnz_off: &mut usize,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<T>)> {
    let (lo, hi) = (offsets[b] as usize, offsets[b + 1] as usize);
    if hi > indptr.len() || hi < lo || hi - lo != s + 1 {
        return Err(PyValueError::new_err(format!(
            "kossakowski factor: block {b}: {what} indptr slice [{lo}, {hi}) does not hold size + 1 = {} entries",
            s + 1
        )));
    }
    let ptr: Vec<usize> = indptr[lo..hi].iter().map(|&v| v as usize).collect();
    let nnz = ptr[s];
    if *nnz_off + nnz > indices.len() || indices.len() != data.len() {
        return Err(PyValueError::new_err(format!(
            "kossakowski factor: block {b}: {what} indices / data too short for {nnz} entries at offset {}",
            *nnz_off
        )));
    }
    let idx: Vec<usize> = indices[*nnz_off..*nnz_off + nnz].iter().map(|&v| v as usize).collect();
    let dat: Vec<T> = data[*nnz_off..*nnz_off + nnz].to_vec();
    *nnz_off += nnz;
    Ok((ptr, idx, dat))
}

/// Decode the flat `nmr-csim-kfactor/1` layout (`sizes`, `ranks`, `G_values`
/// block after block row-major, `D_values` block after block, optional masks /
/// private parts as concatenated block-local CSR with `*_offsets` into
/// `indptr`, optional `G_neg` as `G`) into per-block inputs.
#[allow(clippy::too_many_arguments)]
fn factor_blocks_from_flat(
    sizes: &[usize],
    ranks: &[usize],
    g_values: &[Complex64],
    d_values: &[f64],
    mask: Option<&PyMaskFlat<'_>>,
    neg_ranks: Option<&[usize]>,
    g_neg_values: Option<&[Complex64]>,
    private: Option<&PyPrivateFlat<'_>>,
) -> PyResult<Vec<FactorBlockInput>> {
    let nb = sizes.len();
    if ranks.len() != nb {
        return Err(PyValueError::new_err(format!("kossakowski factor: {nb} block sizes but {} ranks", ranks.len())));
    }
    let g_total: usize = sizes.iter().zip(ranks).map(|(s, r)| s * r).sum();
    let d_total: usize = sizes.iter().sum();
    if g_values.len() != g_total {
        return Err(PyValueError::new_err(format!(
            "kossakowski factor: G_values has {} entries, the blocks need sum(size x rank) = {g_total}",
            g_values.len()
        )));
    }
    if d_values.len() != d_total {
        return Err(PyValueError::new_err(format!(
            "kossakowski factor: D_values has {} entries, the blocks need sum(size) = {d_total}",
            d_values.len()
        )));
    }
    let neg_ranks: Vec<usize> = match neg_ranks {
        None => vec![0; nb],
        Some(r) if r.len() == nb => r.to_vec(),
        Some(r) => {
            return Err(PyValueError::new_err(format!("kossakowski factor: {nb} blocks but {} G_neg ranks", r.len())));
        }
    };
    let gneg_total: usize = sizes.iter().zip(&neg_ranks).map(|(s, r)| s * r).sum();
    let g_neg_values = g_neg_values.unwrap_or(&[]);
    if g_neg_values.len() != gneg_total {
        return Err(PyValueError::new_err(format!(
            "kossakowski factor: G_neg_values has {} entries, the blocks need sum(size x rank_neg) = {gneg_total}",
            g_neg_values.len()
        )));
    }
    let mask_sl = match mask {
        None => None,
        Some((a, b, c, d)) => Some((a.as_slice()?, b.as_slice()?, c.as_slice()?, d.as_slice()?)),
    };
    let priv_sl = match private {
        None => None,
        Some((a, b, c, d)) => Some((a.as_slice()?, b.as_slice()?, c.as_slice()?, d.as_slice()?)),
    };
    for (name, off) in [("mask", mask_sl.map(|t| t.3)), ("private", priv_sl.map(|t| t.3))] {
        if let Some(off) = off {
            if off.len() != nb + 1 {
                return Err(PyValueError::new_err(format!(
                    "kossakowski factor: {name}_offsets has {} entries, expected n_blocks + 1 = {}",
                    off.len(),
                    nb + 1
                )));
            }
        }
    }
    let mut out = Vec::with_capacity(nb);
    let (mut g_off, mut gn_off, mut d_off, mut m_nnz, mut p_nnz) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (b, (&s, &r)) in sizes.iter().zip(ranks).enumerate() {
        let rn = neg_ranks[b];
        let g: Vec<Complex<f64>> = g_values[g_off..g_off + s * r].iter().map(|c| Complex::new(c.re, c.im)).collect();
        let g_neg: Vec<Complex<f64>> =
            g_neg_values[gn_off..gn_off + s * rn].iter().map(|c| Complex::new(c.re, c.im)).collect();
        let d: Vec<f64> = d_values[d_off..d_off + s].to_vec();
        let mask = match mask_sl {
            Some((ip, ix, dt, off)) => {
                let (indptr, indices, data) = csr_block("mask", b, s, ip, ix, dt, off, &mut m_nnz)?;
                Some(Csr { indptr, indices, data })
            }
            None => None,
        };
        let private = match priv_sl {
            Some((ip, ix, dt, off)) => {
                let (indptr, indices, data) = csr_block("private", b, s, ip, ix, dt, off, &mut p_nnz)?;
                Some(CsrC { indptr, indices, data: data.into_iter().map(|c| Complex::new(c.re, c.im)).collect() })
            }
            None => None,
        };
        out.push(FactorBlockInput { size: s, rank: r, g, d, mask, rank_neg: rn, g_neg, private });
        g_off += s * r;
        gn_off += s * rn;
        d_off += s;
    }
    Ok(out)
}

/// PyO3 facade exposing [`ppvm_lindblad::LindbladSpec`] to Python.
#[pyclass]
pub struct LindbladSpec {
    inner: AnySpec,
}

// ────────────────── width-generic method bodies ──────────────────

fn action_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    py: Python<'py>,
    p: PyReadonlyArray1<'py, u8>,
) -> PyResult<PyPauliMap<'py>> {
    let p_slice = p.as_slice()?;
    let p_word = word_from_codes::<W>(p_slice).map_err(map_err)?;
    let pairs = spec.action(&p_word);
    pack_pauli_map(py, pairs, spec.n_qubits())
}

fn leakage_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, u8>,
    coeffs: PyReadonlyArray1<'py, f64>,
    protected: Option<PyReadonlyArray2<'py, u8>>,
) -> PyResult<PyPauliMap<'py>> {
    let n_q = spec.n_qubits();
    let basis_view = basis.as_array();
    let basis_words = decode_basis::<W>(&basis_view, n_q)?;
    let coeffs_slice = coeffs.as_slice()?;
    if coeffs_slice.len() != basis_words.len() {
        return Err(PyValueError::new_err(format!(
            "coeffs has length {} but basis has {} rows",
            coeffs_slice.len(),
            basis_words.len()
        )));
    }
    let prot = protected_words::<W>(protected.as_ref(), n_q)?;
    let pairs = spec
        .leakage(&basis_words, coeffs_slice, &prot)
        .map_err(map_err)?;
    pack_pauli_map(py, pairs, n_q)
}

#[allow(clippy::too_many_arguments)]
fn pc_step_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, u8>,
    coeffs: PyReadonlyArray1<'py, f64>,
    dt: f64,
    cfg: ppvm_lindblad::PcStepConfig,
    protected: Option<PyReadonlyArray2<'py, u8>>,
    timed: bool,
) -> PyResult<(PyPauliMap<'py>, Option<ppvm_lindblad::PcStepTimings>, ppvm_lindblad::PcStepStats)> {
    let n_q = spec.n_qubits();
    let basis_view = basis.as_array();
    let mut basis_words = decode_basis::<W>(&basis_view, n_q)?;
    assert_basis_unique(&basis_words)?;
    let mut coeffs_vec = coeffs.as_slice()?.to_vec();
    if coeffs_vec.len() != basis_words.len() {
        return Err(PyValueError::new_err(format!(
            "coeffs has length {} but basis has {} rows",
            coeffs_vec.len(),
            basis_words.len()
        )));
    }
    let prot = protected_words::<W>(protected.as_ref(), n_q)?;
    let (timings, stats) = if timed {
        let (t, st) = spec
            .pc_step_timed(&mut basis_words, &mut coeffs_vec, dt, &prot, &cfg)
            .map_err(map_err)?;
        (Some(t), st)
    } else {
        let st = spec
            .pc_step(&mut basis_words, &mut coeffs_vec, dt, &prot, &cfg)
            .map_err(map_err)?;
        (None, st)
    };
    // Pack output. Basis may have grown; coeffs has the same new length.
    let pairs: Vec<(Word<W>, f64)> = basis_words.into_iter().zip(coeffs_vec).collect();
    Ok((pack_pauli_map(py, pairs, n_q)?, timings, stats))
}

/// The step account ([`ppvm_lindblad::PcStepStats`]) as a Python dict:
/// `max_weight` (None = off), `depolarize_eps` (0.0 = off), `admitted1` /
/// `admitted2`, `weight_dropped_terms` / `weight_dropped_l1`, `trim_dropped` /
/// `trim_dropped_l2sq`, `basis_max_weight`, `basis_len`.
pub(crate) fn stats_dict<'py>(
    py: Python<'py>,
    st: &ppvm_lindblad::PcStepStats,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("max_weight", st.max_weight)?;
    d.set_item("depolarize_eps", st.depolarize_eps)?;
    d.set_item("admitted1", st.admitted1)?;
    d.set_item("admitted2", st.admitted2)?;
    d.set_item("weight_dropped_terms", st.weight_dropped_terms)?;
    d.set_item("weight_dropped_l1", st.weight_dropped_l1)?;
    d.set_item("trim_dropped", st.trim_dropped)?;
    d.set_item("trim_dropped_l2sq", st.trim_dropped_l2sq)?;
    d.set_item("basis_max_weight", st.basis_max_weight)?;
    d.set_item("basis_len", st.basis_len)?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn pc_step_orbit_rep_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, u8>,
    coeffs: PyReadonlyArray1<'py, Complex64>,
    dt: f64,
    cfg: ppvm_lindblad::PcStepConfig,
    group: &crate::symmetry::TranslationGroup,
    momentum: PyReadonlyArray1<'py, i32>,
    protected: Option<PyReadonlyArray2<'py, u8>>,
    canonicalize_first: bool,
) -> PyResult<PyPauliMapComplex<'py>> {
    use ppvm_lindblad::orbit_rep;

    let n_q = spec.n_qubits();
    let basis_view = basis.as_array();
    let mut basis_words = decode_basis::<W>(&basis_view, n_q)?;
    let coeffs_slice = coeffs.as_slice()?;
    if coeffs_slice.len() != basis_words.len() {
        return Err(PyValueError::new_err(format!(
            "coeffs has length {} but basis has {} rows",
            coeffs_slice.len(),
            basis_words.len()
        )));
    }
    let mut coeffs_vec: Vec<Complex<f64>> = coeffs_slice
        .iter()
        .map(|c| Complex::new(c.re, c.im))
        .collect();
    let prot = protected_words::<W>(protected.as_ref(), n_q)?;
    let k_slice = momentum.as_slice()?;
    if k_slice.len() != group.core().n_generators() {
        return Err(PyValueError::new_err(format!(
            "momentum has {} entries but group has {} generators",
            k_slice.len(),
            group.core().n_generators()
        )));
    }
    if canonicalize_first {
        orbit_rep::canonicalize_basis_to_rep(&mut basis_words, group.core());
    }
    orbit_rep::pc_step_orbit_rep(
        spec,
        &mut basis_words,
        &mut coeffs_vec,
        dt,
        &prot,
        group.core(),
        k_slice,
        &cfg,
    )
    .map_err(map_err)?;
    pack_pauli_map_complex(py, &basis_words, &coeffs_vec, n_q)
}

fn generator_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, u8>,
) -> PyResult<PyCoo<'py>> {
    let n_q = spec.n_qubits();
    let basis_view = basis.as_array();
    let basis_words = decode_basis::<W>(&basis_view, n_q)?;
    assert_basis_unique(&basis_words)?;
    let triplets = spec.generator(&basis_words);
    let total = triplets.len();
    let mut rows = Vec::with_capacity(total);
    let mut cols = Vec::with_capacity(total);
    let mut vals = Vec::with_capacity(total);
    for (r, c, v) in triplets {
        rows.push(r as u64);
        cols.push(c as u64);
        vals.push(v);
    }
    Ok((
        rows.into_pyarray(py),
        cols.into_pyarray(py),
        vals.into_pyarray(py),
    ))
}

fn set_kossakowski_impl<const W: usize>(
    spec: &mut CoreSpec<W>,
    k: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let m = spec.num_kossakowski_ops();
    if m == 0 {
        return Err(PyValueError::new_err(
            "set_kossakowski: this LindbladSpec was built without a Kossakowski block",
        ));
    }
    let k: PyReadonlyArray2<'_, Complex64> = k.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "set_kossakowski: expected a 2-D complex128 ndarray of shape ({m}, {m})"
        ))
    })?;
    let view = k.as_array();
    if view.shape() != [m, m] {
        return Err(PyValueError::new_err(format!(
            "set_kossakowski: expected shape ({m}, {m}) (num_kossakowski_ops), got {:?}",
            view.shape()
        )));
    }
    let rows: Vec<Vec<Complex<f64>>> = (0..m)
        .map(|i| (0..m).map(|j| Complex::new(view[[i, j]].re, view[[i, j]].im)).collect())
        .collect();
    spec.set_kossakowski(&rows).map_err(map_err)
}

fn action_census_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    basis: PyReadonlyArray2<'py, u8>,
) -> PyResult<Vec<(u64, u64, u64, u64, u64, u64)>> {
    let n_q = spec.n_qubits();
    let words = decode_basis::<W>(&basis.as_array(), n_q)?;
    use rayon::prelude::*;
    Ok(words
        .par_iter()
        .map(|w| {
            let c = ppvm_lindblad::locality::action_census(spec, w);
            (c[0], c[1], c[2], c[3], c[4], c[5])
        })
        .collect())
}

fn action_locality_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    py: Python<'py>,
    p: PyReadonlyArray1<'py, u8>,
) -> PyResult<PyPauliMap<'py>> {
    let p_slice = p.as_slice()?;
    let p_word = word_from_codes::<W>(p_slice).map_err(map_err)?;
    let mut sc = ppvm_lindblad::support::ActScratch::<W>::default();
    let pairs = spec.action_terms_locality_pub(&p_word, &mut sc);
    pack_pauli_map(py, pairs, spec.n_qubits())
}

fn action_locality_stats_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    basis: PyReadonlyArray2<'py, u8>,
) -> PyResult<Vec<Vec<u64>>> {
    let n_q = spec.n_qubits();
    let words = decode_basis::<W>(&basis.as_array(), n_q)?;
    let mut sc = ppvm_lindblad::support::ActScratch::<W>::default();
    Ok(words
        .iter()
        .map(|w| {
            let _ = spec.action_terms_locality_pub(w, &mut sc);
            sc.stats.to_vec()
        })
        .collect())
}

fn action_locality_profile_impl<'py, const W: usize>(
    spec: &CoreSpec<W>,
    basis: PyReadonlyArray2<'py, u8>,
    mode: u8,
) -> PyResult<Vec<Vec<u64>>> {
    let n_q = spec.n_qubits();
    let words = decode_basis::<W>(&basis.as_array(), n_q)?;
    let _ = spec.support_stats();
    use rayon::prelude::*;
    Ok(words
        .par_iter()
        .map_init(
            || {
                let mut sc = ppvm_lindblad::support::ActScratch::<W>::default();
                sc.prof_mode = mode;
                sc
            },
            |sc, w| {
                let t0 = std::time::Instant::now();
                let _ = spec.action_terms_locality_pub(w, sc);
                let wall = t0.elapsed().as_micros() as u64;
                vec![
                    sc.phase_us[0],
                    sc.phase_us[1],
                    sc.phase_us[2],
                    sc.phase_us[3],
                    sc.stats[1],
                    sc.stats[2],
                    sc.stats[3],
                    wall,
                ]
            },
        )
        .collect())
}

#[pymethods]
impl LindbladSpec {
    /// Construct a Lindbladian spec from Hamiltonian terms and jump operators.
    ///
    /// `jump_lincombs[k]` is a list of `(pauli_string, real, imag)` triples
    /// encoding `L_k = Σ_a (re + i·im) P_a`. A length-1 jump with `im == 0`
    /// is routed to the Hermitian-Pauli fast path (with rate scaled by `re²`).
    ///
    /// `kossakowski_ops` / `kossakowski_k` optionally add a Kossakowski-form
    /// dissipator `D*(O) = Σ_nm K_nm (A_n† O A_m − ½{A_n†A_m, O})`:
    /// `kossakowski_ops[i]` is the Pauli lincomb of `A_i` in the same triple
    /// encoding, `kossakowski_k[n][m] = (re, im)` the Hermitian pair matrix.
    /// `kossakowski_coo = (rows, cols, vals)` (int64, int64, complex128 1-D
    /// arrays over the operator index) gives `K` as COO triples instead, so
    /// a block-sparse `K` over thousands of operators is never materialised
    /// (same rules as `set_kossakowski_sparse`); `kossakowski_k` must then
    /// be empty.
    ///
    /// `word_width` (chunks of `lindblad_chunk_bits()` bits) forces a
    /// compiled Pauli-word width; `None` picks the smallest one that holds
    /// `n_qubits` (`lindblad_word_width_for`).
    #[new]
    #[pyo3(signature = (n_qubits, h_terms, h_coeffs, jump_lincombs, jump_rates,
                        kossakowski_ops = vec![], kossakowski_k = vec![],
                        kossakowski_coo = None, word_width = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_qubits: usize,
        h_terms: Vec<String>,
        h_coeffs: Vec<f64>,
        jump_lincombs: Vec<Vec<(String, f64, f64)>>,
        jump_rates: Vec<f64>,
        kossakowski_ops: Vec<Vec<(String, f64, f64)>>,
        kossakowski_k: Vec<Vec<(f64, f64)>>,
        kossakowski_coo: Option<&Bound<'_, PyAny>>,
        word_width: Option<usize>,
    ) -> PyResult<Self> {
        let width = pick_width(n_qubits, word_width, "LindbladSpec")?;
        let coo_entries_opt = match kossakowski_coo {
            Some(coo) => {
                if !kossakowski_k.is_empty() {
                    return Err(PyValueError::new_err(
                        "kossakowski_k and kossakowski_coo are alternatives; give one of them",
                    ));
                }
                if kossakowski_ops.is_empty() {
                    return Err(PyValueError::new_err(
                        "kossakowski_coo needs a non-empty kossakowski_ops",
                    ));
                }
                let (rows, cols, vals): (
                    PyReadonlyArray1<'_, i64>,
                    PyReadonlyArray1<'_, i64>,
                    PyReadonlyArray1<'_, Complex64>,
                ) = coo.extract().map_err(|_| {
                    PyValueError::new_err(
                        "kossakowski_coo: expected a tuple (rows: int64[nnz], cols: int64[nnz], vals: complex128[nnz])",
                    )
                })?;
                Some(coo_entries(&rows, &cols, &vals, "kossakowski_coo")?)
            }
            None => None,
        };
        let inputs = SpecInputs::new(
            h_terms,
            h_coeffs,
            jump_lincombs,
            jump_rates,
            kossakowski_ops,
            kossakowski_k,
            "",
        )?;
        let inner = match coo_entries_opt {
            Some(entries) => with_width!(width, W => {
                let mut inner = CoreSpec::<W>::new(n_qubits, &inputs.h, &inputs.jumps).map_err(map_err)?;
                inner.add_kossakowski_sparse(&inputs.ops, &entries).map_err(map_err)?;
                AnySpec::from(inner)
            }),
            None => with_width!(width, W => { AnySpec::from(inputs.build::<W>(n_qubits)?) }),
        };
        Ok(Self { inner })
    }

    /// The Pauli-word width this spec was instantiated with (chunks).
    #[getter]
    fn word_width(&self) -> usize {
        self.inner.word_width()
    }

    /// The largest `n_qubits` this spec's word width holds.
    #[getter]
    fn max_qubits(&self) -> usize {
        max_qubits_for(self.inner.word_width())
    }

    /// Bytes of one packed Pauli word at this width (`16·W + 16` on 64-bit
    /// targets); a basis string costs this plus its 8-byte coefficient.
    #[getter]
    fn word_bytes(&self) -> usize {
        dispatch!(&self.inner, s => word_bytes_of(s))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        dispatch!(&self.inner, s => s.n_qubits())
    }

    #[getter]
    fn num_h_terms(&self) -> usize {
        dispatch!(&self.inner, s => s.num_h_terms())
    }

    #[getter]
    fn num_jump_terms(&self) -> usize {
        dispatch!(&self.inner, s => s.num_jump_terms())
    }

    /// Current Hamiltonian coefficients, shape `(num_h_terms,)` float64 (a copy).
    #[getter]
    fn h_coeffs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dispatch!(&self.inner, s => s.h_coeffs().into_pyarray(py))
    }

    /// Overwrite the Hamiltonian coefficients in place (`O(num_h_terms)`).
    ///
    /// `coeffs` must be a 1-D float64 array of length `num_h_terms`;
    /// otherwise `ValueError`. Pauli words, jump operators and the
    /// Kossakowski block are untouched (no PSD re-check).
    fn set_h_coeffs(&mut self, coeffs: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.num_h_terms();
        let v = f64_vec(coeffs, n, "set_h_coeffs").map_err(|_| {
            PyValueError::new_err(format!(
                "set_h_coeffs: expected a 1-D float64 ndarray of shape ({n},) (num_h_terms)"
            ))
        })?;
        dispatch!(&mut self.inner, s => s.set_h_coeffs(&v).map_err(map_err))
    }

    /// Number of jump operators given at construction (the length of
    /// `jump_rates`); does not count Kossakowski pairs.
    #[getter]
    fn num_jump_ops(&self) -> usize {
        dispatch!(&self.inner, s => s.num_jump_ops())
    }

    /// Current GKSL rates of the jump operators given at construction,
    /// shape `(num_jump_ops,)` float64 (a copy).
    #[getter]
    fn jump_rates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dispatch!(&self.inner, s => s.jump_rates().into_pyarray(py))
    }

    /// Overwrite the jump rates in place (`O(num_jump_ops)`).
    ///
    /// `rates` must be a 1-D float64 array of length `num_jump_ops` with
    /// non-negative entries; otherwise `ValueError`. Jump operators,
    /// Hamiltonian and Kossakowski block are untouched.
    fn set_jump_rates(&mut self, rates: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.num_jump_ops();
        let v = f64_vec(rates, n, "set_jump_rates").map_err(|_| {
            PyValueError::new_err(format!(
                "set_jump_rates: expected a 1-D float64 ndarray of shape ({n},) (num_jump_ops)"
            ))
        })?;
        dispatch!(&mut self.inner, s => s.set_jump_rates(&v).map_err(map_err))
    }

    /// Number `M` of Kossakowski operators (0 when built without a block).
    #[getter]
    fn num_kossakowski_ops(&self) -> usize {
        dispatch!(&self.inner, s => s.num_kossakowski_ops())
    }

    /// Current Kossakowski matrix, shape `(M, M)` complex128 (a copy;
    /// `(0, 0)` when built without a block).
    #[getter]
    fn kossakowski<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let k = dispatch!(&self.inner, s => s.kossakowski());
        let m = k.len();
        let flat: Vec<Complex64> = k
            .into_iter()
            .flatten()
            .map(|c| Complex64::new(c.re, c.im))
            .collect();
        flat.into_pyarray(py)
            .reshape([m, m])
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))
    }

    /// Replace the Kossakowski matrix in place, keeping the operator basis.
    ///
    /// `k` must be a 2-D complex128 array of shape `(M, M)` with
    /// `M = num_kossakowski_ops > 0`, Hermitian to `1e-10·max(‖K‖, 1)`;
    /// otherwise `ValueError`. Positive semi-definiteness is checked in the
    /// Python layer. The precompiled pair tables are rebuilt from the
    /// stored operators, so later results are bit-identical to a fresh
    /// construction with `k`.
    fn set_kossakowski(&mut self, k: &Bound<'_, PyAny>) -> PyResult<()> {
        dispatch!(&mut self.inner, s => set_kossakowski_impl(s, k))
    }

    /// Replace the Kossakowski matrix in place from COO triples over the
    /// operator index: `rows`, `cols` (int64, `nnz` each) and `vals`
    /// (complex128, `nnz`). Any order; duplicates are summed; exact zeros
    /// are ignored; an off-diagonal entry given in one triangle only is
    /// Hermitian-completed, one given in both must match its mirror's
    /// conjugate and a diagonal entry must be real (to
    /// `1e-10·max(‖K‖, 1)`), otherwise `ValueError`. The dense `(M, M)`
    /// matrix is never formed; the result is bit-identical to
    /// `set_kossakowski` with it. PSD is checked in the Python layer.
    fn set_kossakowski_sparse(
        &mut self,
        rows: PyReadonlyArray1<'_, i64>,
        cols: PyReadonlyArray1<'_, i64>,
        vals: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<()> {
        if self.num_kossakowski_ops() == 0 {
            return Err(PyValueError::new_err(
                "set_kossakowski_sparse: this LindbladSpec was built without a Kossakowski block",
            ));
        }
        let entries = coo_entries(&rows, &cols, &vals, "set_kossakowski_sparse")?;
        dispatch!(&mut self.inner, s => s.set_kossakowski_sparse(&entries).map_err(map_err))
    }

    /// Replace the Kossakowski matrix in place from blocks: `blocks[b] =
    /// (indices, K_block)` with `indices` an int64 array of distinct
    /// operator indices and `K_block` a Hermitian complex128
    /// `(len, len)` array placed at those indices (`K[indices[i],
    /// indices[j]] += K_block[i, j]`; blocks sharing operators add up).
    /// `ValueError` on an index out of range or repeated within a block,
    /// a shape mismatch or a non-Hermitian block, leaving the spec
    /// unchanged. The dense matrix is never formed; the result is
    /// bit-identical to `set_kossakowski` with it. PSD is checked in the
    /// Python layer.
    fn set_kossakowski_blocks<'py>(
        &mut self,
        blocks: Vec<(PyReadonlyArray1<'py, i64>, PyReadonlyArray2<'py, Complex64>)>,
    ) -> PyResult<()> {
        if self.num_kossakowski_ops() == 0 {
            return Err(PyValueError::new_err(
                "set_kossakowski_blocks: this LindbladSpec was built without a Kossakowski block",
            ));
        }
        let mut core: Vec<(Vec<usize>, Vec<Vec<Complex<f64>>>)> = Vec::with_capacity(blocks.len());
        for (b, (idx, kb)) in blocks.iter().enumerate() {
            let idx_view = idx.as_array();
            let mut indices: Vec<usize> = Vec::with_capacity(idx_view.len());
            for &i in idx_view.iter() {
                if i < 0 {
                    return Err(PyValueError::new_err(format!(
                        "set_kossakowski_blocks: block {b} has a negative operator index {i}"
                    )));
                }
                indices.push(i as usize);
            }
            let kv = kb.as_array();
            let shape = kv.shape();
            if shape[0] != indices.len() || shape[1] != indices.len() {
                return Err(PyValueError::new_err(format!(
                    "set_kossakowski_blocks: block {b} has {} indices but K_block has shape {:?}",
                    indices.len(),
                    shape
                )));
            }
            let mat: Vec<Vec<Complex<f64>>> = (0..shape[0])
                .map(|i| (0..shape[1]).map(|j| Complex::new(kv[[i, j]].re, kv[[i, j]].im)).collect())
                .collect();
            core.push((indices, mat));
        }
        dispatch!(&mut self.inner, s => s.set_kossakowski_blocks(&core).map_err(map_err))
    }

    /// Current Kossakowski matrix as COO arrays `(rows, cols, vals)`
    /// (uint64, uint64, complex128): every nonzero entry, both triangles,
    /// sorted by `(row, col)`. The sparse twin of `kossakowski`.
    #[getter]
    fn kossakowski_coo<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<Complex64>>,
    ) {
        let entries = dispatch!(&self.inner, s => s.kossakowski_entries());
        let rows: Vec<u64> = entries.iter().map(|e| e.0 as u64).collect();
        let cols: Vec<u64> = entries.iter().map(|e| e.1 as u64).collect();
        let vals: Vec<Complex64> = entries.iter().map(|e| Complex64::new(e.2.re, e.2.im)).collect();
        (rows.into_pyarray(py), cols.into_pyarray(py), vals.into_pyarray(py))
    }

    /// Number of stored nonzero entries of `K` (both triangles).
    #[getter]
    fn num_kossakowski_nonzeros(&self) -> usize {
        dispatch!(&self.inner, s => s.num_kossakowski_nonzeros())
    }

    /// Number of distinct pair templates: operator-pattern relationships
    /// (the Pauli structure of `(A_n, A_m)` up to a qubit relabelling)
    /// shared by the compiled pairs.
    #[getter]
    fn num_pair_templates(&self) -> usize {
        dispatch!(&self.inner, s => s.num_pair_templates())
    }

    /// Number of compiled Kossakowski pairs `n ≤ m` (`|K_nm| > 1e-14·max|K|`).
    #[getter]
    fn num_compiled_pairs(&self) -> usize {
        dispatch!(&self.inner, s => s.num_compiled_pairs())
    }

    /// Heap bytes of the Kossakowski compilation: `(templates, pairs, entries)`.
    #[getter]
    fn kossakowski_memory_bytes(&self) -> (usize, usize, usize) {
        dispatch!(&self.inner, s => s.kossakowski_memory_bytes())
    }

    /// The locality-aware step's structures: `(index_built, index_bytes,
    /// index_entries, index_keys, max_delta, cache_len)`.
    #[getter]
    fn locality_stats(&self) -> (bool, usize, usize, usize, u32, usize) {
        dispatch!(&self.inner, s => s.locality_stats())
    }

    /// Build the transfer index now (it is otherwise built by the first
    /// locality step); returns its heap bytes.
    fn build_locality_index(&self) -> usize {
        dispatch!(&self.inner, s => s.transfer_index().bytes)
    }

    /// Per-word census of the full action over the rows of `basis`:
    /// `(h candidates, dissipator candidates, one-sided pairs, both-sided
    /// pairs, table entries visited, distinct output words)`.
    fn action_census<'py>(
        &self,
        basis: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Vec<(u64, u64, u64, u64, u64, u64)>> {
        dispatch!(&self.inner, s => action_census_impl(s, basis))
    }

    /// Apply `L*` to a single Pauli string `p` through the support-grouped
    /// action of the locality step (value-identical to `action`).
    fn action_locality<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<PyPauliMap<'py>> {
        dispatch!(&self.inner, s => action_locality_impl(s, py, p))
    }

    /// Per-row counters of the grouped action (`ActScratch::stats`): dense groups, pairs in groups, dense
    /// emissions, deferred emissions from groups, generic candidates, generic emissions, Hamiltonian emissions,
    /// distinct deferred words.
    fn action_locality_stats<'py>(
        &self,
        basis: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Vec<Vec<u64>>> {
        dispatch!(&self.inner, s => action_locality_stats_impl(s, basis))
    }

    /// Profiling of the grouped action over the rows of `basis`, in parallel (the thread pool's size): per row
    /// `[merge_us, dense_us, generic_us, final_us, pairs, dense emissions, deferred emissions, wall_us]` in
    /// `mode` 0 (normal), 1 (values computed, not accumulated) or 2 (entry loops skipped). Modes 1–2 give wrong
    /// actions and are for the attribution only.
    fn action_locality_profile<'py>(
        &self,
        basis: PyReadonlyArray2<'py, u8>,
        mode: u8,
    ) -> PyResult<Vec<Vec<u64>>> {
        dispatch!(&self.inner, s => action_locality_profile_impl(s, basis, mode))
    }

    /// `(tuples, sets, exclusive sets, bytes)` of the support tables (builds the index).
    fn support_stats(&self) -> (usize, usize, usize, usize) {
        dispatch!(&self.inner, s => s.support_stats())
    }

    // ── factor-form Kossakowski dissipator (ppvm_lindblad::factor) ──

    /// Add a factor-form Kossakowski dissipator `K_b = (G_b G_b†) ∘ W_b + diag(D_b)`
    /// per block (the `nmr-csim-kfactor/1` flat layout): `ops` the operator
    /// lincombs in block order, `sizes` / `ranks` per block, `g_values` the
    /// row-major `G_b` concatenated (complex128), `d_values` the `D_b`
    /// concatenated (float64), `mask` = `(indptr, indices, data, offsets)` the
    /// block-local CSR masks concatenated (int64, int64, float64, int64) or
    /// `None`. PSD is the caller's responsibility, as for `kossakowski_k`.
    #[pyo3(signature = (ops, sizes, ranks, g_values, d_values, mask = None, neg_ranks = None, g_neg_values = None, private = None))]
    #[allow(clippy::too_many_arguments)]
    fn add_kossakowski_factor(
        &mut self,
        ops: Vec<Vec<(String, f64, f64)>>,
        sizes: Vec<usize>,
        ranks: Vec<usize>,
        g_values: PyReadonlyArray1<'_, Complex64>,
        d_values: PyReadonlyArray1<'_, f64>,
        mask: Option<PyMaskFlat<'_>>,
        neg_ranks: Option<Vec<usize>>,
        g_neg_values: Option<PyReadonlyArray1<'_, Complex64>>,
        private: Option<PyPrivateFlat<'_>>,
    ) -> PyResult<()> {
        let ops = lincombs_from_triples(ops);
        let gneg = match &g_neg_values {
            Some(a) => Some(a.as_slice()?),
            None => None,
        };
        let blocks = factor_blocks_from_flat(
            &sizes, &ranks, g_values.as_slice()?, d_values.as_slice()?, mask.as_ref(), neg_ranks.as_deref(), gneg, private.as_ref(),
        )?;
        dispatch!(&mut self.inner, s => s.add_kossakowski_factor(&ops, &blocks).map_err(map_err))
    }

    /// Replace `G` / `D` / masks of the factor dissipator in place (same
    /// operator table and block sizes); arguments as `add_kossakowski_factor`
    /// without `ops`.
    #[pyo3(signature = (sizes, ranks, g_values, d_values, mask = None, neg_ranks = None, g_neg_values = None, private = None))]
    #[allow(clippy::too_many_arguments)]
    fn set_kossakowski_factor(
        &mut self,
        sizes: Vec<usize>,
        ranks: Vec<usize>,
        g_values: PyReadonlyArray1<'_, Complex64>,
        d_values: PyReadonlyArray1<'_, f64>,
        mask: Option<PyMaskFlat<'_>>,
        neg_ranks: Option<Vec<usize>>,
        g_neg_values: Option<PyReadonlyArray1<'_, Complex64>>,
        private: Option<PyPrivateFlat<'_>>,
    ) -> PyResult<()> {
        let gneg = match &g_neg_values {
            Some(a) => Some(a.as_slice()?),
            None => None,
        };
        let blocks = factor_blocks_from_flat(
            &sizes, &ranks, g_values.as_slice()?, d_values.as_slice()?, mask.as_ref(), neg_ranks.as_deref(), gneg, private.as_ref(),
        )?;
        dispatch!(&mut self.inner, s => s.set_kossakowski_factor(&blocks).map_err(map_err))
    }

    /// Number of operators of the factor dissipator (0 without one).
    #[getter]
    fn num_kossakowski_factor_ops(&self) -> usize {
        dispatch!(&self.inner, s => s.num_kossakowski_factor_ops())
    }

    /// Number of factor blocks (0 without one).
    #[getter]
    fn num_kossakowski_factor_blocks(&self) -> usize {
        dispatch!(&self.inner, s => s.num_kossakowski_factor_blocks())
    }

    /// `(factor_bytes, ops_bytes)` held by the factor dissipator.
    #[getter]
    fn kossakowski_factor_memory_bytes(&self) -> (usize, usize) {
        dispatch!(&self.inner, s => s.kossakowski_factor_memory_bytes())
    }

    /// The absolute compile floor of the factor dissipator (`1e-14 · max_n K_nn`).
    #[getter]
    fn kossakowski_factor_floor(&self) -> f64 {
        dispatch!(&self.inner, s => s.kossakowski_factor_floor())
    }

    /// Block `b` of the factor dissipator as a dense `(s_b, s_b)` complex128
    /// array `K_b = (G G†) ∘ W + diag(D)` (inspection).
    fn kossakowski_factor_dense_block<'py>(
        &self,
        py: Python<'py>,
        b: usize,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let k = dispatch!(&self.inner, s => s.kossakowski_factor_dense_block(b)).map_err(map_err)?;
        let s = k.len();
        let flat: Vec<Complex64> = k.into_iter().flatten().map(|c| Complex64::new(c.re, c.im)).collect();
        flat.into_pyarray(py)
            .reshape([s, s])
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))
    }

    /// Apply `L*` to a single Pauli string `p`.
    fn action<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<PyPauliMap<'py>> {
        dispatch!(&self.inner, s => action_impl(s, py, p))
    }

    /// Off-basis component of `L*( Σ_j coeffs[j] · basis[j] )`.
    #[pyo3(signature = (basis, coeffs, protected = None))]
    fn leakage<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        coeffs: PyReadonlyArray1<'py, f64>,
        protected: Option<PyReadonlyArray2<'py, u8>>,
    ) -> PyResult<PyPauliMap<'py>> {
        dispatch!(&self.inner, s => leakage_impl(s, py, basis, coeffs, protected))
    }

    /// One predictor-corrector adaptive step.
    ///
    /// Internally: expand basis with first-hop leakage, predictor step
    /// (`exp(dt·M)`), expand again with second-hop leakage from the
    /// predicted state, then redo the step from the pre-step coefficients
    /// on the doubly-enlarged basis. The matrix exponential is computed in
    /// Rust via `quspin-expm`; no scipy required.
    ///
    /// Returns `(new_basis, new_coeffs)`.
    ///
    /// `max_basis` is a hard rank cap on the retained basis; `admit_basis`
    /// (when `> max_basis`) bounds in-step enrichment instead, so the final
    /// cap selects the top-`max_basis` strings over the whole union
    /// (displacement truncation). `drop_tol` prunes by magnitude after the
    /// step; `tau_add` filters leakage admission by inflow rate. Protected
    /// words are never dropped. `max_weight` (None = off) refuses leakage
    /// candidates of Pauli weight above it at admission; `depolarize_eps`
    /// (0.0 = off) adds the per-site depolarizing diagonal `-eps·w(P)` to the
    /// generator of both exponentials. See `pc_step_stats` for the account.
    #[pyo3(signature = (
        basis, coeffs, dt, max_basis,
        drop_tol = 0.0,
        protected = None,
        num_threads = None,
        admit_basis = None,
        tau_add = None,
        locality = false,
        max_weight = None,
        depolarize_eps = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pc_step<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        coeffs: PyReadonlyArray1<'py, f64>,
        dt: f64,
        max_basis: usize,
        drop_tol: f64,
        protected: Option<PyReadonlyArray2<'py, u8>>,
        num_threads: Option<usize>,
        admit_basis: Option<usize>,
        tau_add: Option<f64>,
        locality: bool,
        max_weight: Option<u32>,
        depolarize_eps: f64,
    ) -> PyResult<PyPauliMap<'py>> {
        let cfg = ppvm_lindblad::PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            num_threads,
            locality,
            max_weight,
            depolarize_eps,
        };
        let (map, _, _) =
            dispatch!(&self.inner, s => pc_step_impl(s, py, basis, coeffs, dt, cfg, protected, false))?;
        Ok(map)
    }

    /// Same as [`Self::pc_step`] but also returns the step account as a dict
    /// (the keys of `stats_dict`): what the weight cap refused, what the two
    /// admission passes added, what the end-of-step trim removed, the largest
    /// weight in the retained basis, and the two settings. The step itself is
    /// the one `pc_step` runs (no timers).
    #[pyo3(signature = (
        basis, coeffs, dt, max_basis,
        drop_tol = 0.0,
        protected = None,
        num_threads = None,
        admit_basis = None,
        tau_add = None,
        locality = false,
        max_weight = None,
        depolarize_eps = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pc_step_stats<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        coeffs: PyReadonlyArray1<'py, f64>,
        dt: f64,
        max_basis: usize,
        drop_tol: f64,
        protected: Option<PyReadonlyArray2<'py, u8>>,
        num_threads: Option<usize>,
        admit_basis: Option<usize>,
        tau_add: Option<f64>,
        locality: bool,
        max_weight: Option<u32>,
        depolarize_eps: f64,
    ) -> PyResult<(PyPauliMap<'py>, Bound<'py, pyo3::types::PyDict>)> {
        let cfg = ppvm_lindblad::PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            num_threads,
            locality,
            max_weight,
            depolarize_eps,
        };
        let (map, _, stats) =
            dispatch!(&self.inner, s => pc_step_impl(s, py, basis, coeffs, dt, cfg, protected, false))?;
        Ok((map, stats_dict(py, &stats)?))
    }

    /// Same as [`Self::pc_step`] but also returns a dict mapping phase
    /// name → microseconds spent in that phase, for profiling, plus the
    /// step-account keys of `pc_step_stats`.
    #[pyo3(signature = (
        basis, coeffs, dt, max_basis,
        drop_tol = 0.0,
        protected = None,
        num_threads = None,
        admit_basis = None,
        tau_add = None,
        locality = false,
        max_weight = None,
        depolarize_eps = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pc_step_timed<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        coeffs: PyReadonlyArray1<'py, f64>,
        dt: f64,
        max_basis: usize,
        drop_tol: f64,
        protected: Option<PyReadonlyArray2<'py, u8>>,
        num_threads: Option<usize>,
        admit_basis: Option<usize>,
        tau_add: Option<f64>,
        locality: bool,
        max_weight: Option<u32>,
        depolarize_eps: f64,
    ) -> PyResult<(PyPauliMap<'py>, Bound<'py, pyo3::types::PyDict>)> {
        let cfg = ppvm_lindblad::PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            num_threads,
            locality,
            max_weight,
            depolarize_eps,
        };
        let (map, timings, stats) =
            dispatch!(&self.inner, s => pc_step_impl(s, py, basis, coeffs, dt, cfg, protected, true))?;
        let timings = timings.expect("timed step returns timings");
        let d = stats_dict(py, &stats)?;
        d.set_item("leakage1_us", timings.leakage1_us)?;
        d.set_item("expand1_us", timings.expand1_us)?;
        d.set_item("expm1_us", timings.expm1_us)?;
        d.set_item("leakage2_us", timings.leakage2_us)?;
        d.set_item("expand2_us", timings.expand2_us)?;
        d.set_item("expm2_us", timings.expm2_us)?;
        d.set_item("build1_us", timings.build1_us)?;
        d.set_item("build2_us", timings.build2_us)?;
        d.set_item("prune_cap_us", timings.prune_cap_us)?;
        let l = &timings.locality;
        d.set_item("leak_action_us", l.leak_action_us)?;
        d.set_item("leak_cache_us", l.leak_cache_us)?;
        d.set_item("leak_merge_us", l.leak_merge_us)?;
        d.set_item("leak_cap_us", l.leak_cap_us)?;
        d.set_item("leak_sort_us", l.leak_sort_us)?;
        d.set_item("build_index_us", l.build_index_us)?;
        d.set_item("build_hit_us", l.build_hit_us)?;
        d.set_item("build_miss_us", l.build_miss_us)?;
        d.set_item("leak_words_n", l.leak_words)?;
        d.set_item("leak_merged_n", l.leak_merged)?;
        d.set_item("leak_strings_n", l.leak_strings)?;
        d.set_item("build_hits_n", l.build_hits)?;
        d.set_item("build_misses_n", l.build_misses)?;
        Ok((map, d))
    }

    /// Per-step orbit-rep predictor-corrector evolution under
    /// translation symmetry. State lives entirely in **orbit-rep form**:
    /// basis contains only canonical orbit representatives, coefficients
    /// are complex. The action is phase-aware: output Paulis canonicalize
    /// to their orbit rep with momentum-character weight.
    ///
    /// Per-step memory benefit: basis is ~|group|× smaller than the
    /// full-basis representation, and the reduction persists through
    /// every step.
    ///
    /// **Pre-condition**: every row of `basis` must be the canonical
    /// orbit representative of its translation orbit under `group`.
    /// Pass `canonicalize_first=True` to enforce this on entry (rewrites
    /// each basis row to its canonical rep; coefficients unchanged).
    /// Default `False` — the caller is trusted.
    ///
    /// `max_basis` is a hard rank cap on the live orbit-rep basis:
    /// enrichment adds at most `max_basis − basis.len()` of the largest
    /// leakage reps and the post-step basis is trimmed to the top-`max_basis`
    /// by `|c|` (protected reps always kept). Pass a large value for the
    /// near-exact case. `drop_tol` additionally prunes by magnitude.
    #[pyo3(signature = (
        basis, coeffs, dt, max_basis,
        group, momentum,
        drop_tol = 0.0,
        protected = None,
        canonicalize_first = false,
        admit_basis = None,
        tau_add = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pc_step_orbit_rep<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        coeffs: PyReadonlyArray1<'py, Complex64>,
        dt: f64,
        max_basis: usize,
        group: &crate::symmetry::TranslationGroup,
        momentum: PyReadonlyArray1<'py, i32>,
        drop_tol: f64,
        protected: Option<PyReadonlyArray2<'py, u8>>,
        canonicalize_first: bool,
        admit_basis: Option<usize>,
        tau_add: Option<f64>,
    ) -> PyResult<PyPauliMapComplex<'py>> {
        // The orbit-rep step has neither the weight cap nor the depolarizing
        // channel (W1-FORK: the plain and block steps only).
        let cfg = ppvm_lindblad::PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            num_threads: None,
            locality: false,
            max_weight: None,
            depolarize_eps: 0.0,
        };
        dispatch!(&self.inner, s => pc_step_orbit_rep_impl(
            s, py, basis, coeffs, dt, cfg, group, momentum, protected, canonicalize_first))
    }

    /// Sparse generator matrix in COO form: `(rows, cols, vals)`.
    fn generator<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<PyCoo<'py>> {
        dispatch!(&self.inner, s => generator_impl(s, py, basis))
    }
}
