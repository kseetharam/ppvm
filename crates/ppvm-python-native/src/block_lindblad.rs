// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! PyO3 wrapper around [`ppvm_lindblad::BlockLindbladSpec`]: `M` species of
//! the same `N` qubits coupled by first-order chemical exchange.
//!
//! Rows cross the boundary as a `(B, n_qubits)` uint8 code array plus a
//! length-`B` uint8 species array; the Rust side packs them into row words
//! (species digits appended) and splits them again on output. All physics
//! lives in `ppvm_lindblad::block`. The Pauli-word width is dispatched on
//! the row length `n_qubits + ceil(log4 n_species)` exactly as `lindblad.rs`
//! dispatches `LindbladSpec` on `n_qubits` (`word_width=` forces one).

use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ppvm_lindblad::{
    BlockLindbladSpec as CoreBlock, LindbladSpec as CoreSpec, Word, max_qubits_for,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::lindblad::{SpecInputs, f64_vec, map_err, pick_width, with_width};

/// `(basis codes (B, n_qubits), species (B,), coeffs (B,))`.
type PyBlockMap<'py> = (
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<f64>>,
);
type PyCoo<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<f64>>,
);

/// A `BlockLindbladSpec` at one of the compiled widths.
pub(crate) enum AnyBlock {
    W2(CoreBlock<2>),
    W4(CoreBlock<4>),
    W8(CoreBlock<8>),
    W16(CoreBlock<16>),
    W17(CoreBlock<17>),
}

macro_rules! impl_from_block {
    ($($v:ident = $w:literal),*) => {
        $(impl From<CoreBlock<$w>> for AnyBlock {
            fn from(s: CoreBlock<$w>) -> Self {
                AnyBlock::$v(s)
            }
        })*
    };
}
impl_from_block!(W2 = 2, W4 = 4, W8 = 8, W16 = 16, W17 = 17);

macro_rules! dispatch {
    ($e:expr, $s:ident => $body:expr) => {
        match $e {
            AnyBlock::W2($s) => $body,
            AnyBlock::W4($s) => $body,
            AnyBlock::W8($s) => $body,
            AnyBlock::W16($s) => $body,
            AnyBlock::W17($s) => $body,
        }
    };
}

fn word_width_of<const W: usize>(_: &CoreBlock<W>) -> usize {
    W
}

fn word_bytes_of<const W: usize>(_: &CoreBlock<W>) -> usize {
    std::mem::size_of::<Word<W>>()
}

/// Species digits a block of `n_species` appends to every row (`⌈log₄ M⌉`),
/// the same rule as `BlockLindbladSpec::with_resets`.
fn species_digits(n_species: usize) -> usize {
    let mut n_digits = 0usize;
    while (1usize << (2 * n_digits)) < n_species.max(1) {
        n_digits += 1;
    }
    n_digits
}

/// Build the block at width `W` from the per-species inputs.
fn build_block<const W: usize>(
    n_qubits: usize,
    inputs: &[SpecInputs],
    exchange: &[Vec<f64>],
    matchings: &[(usize, usize, Vec<usize>)],
    resets: &[(usize, usize, Vec<usize>)],
) -> PyResult<CoreBlock<W>> {
    let mut species: Vec<CoreSpec<W>> = Vec::with_capacity(inputs.len());
    for inp in inputs {
        species.push(inp.build::<W>(n_qubits)?);
    }
    CoreBlock::<W>::with_resets(n_qubits, species, exchange, matchings, resets).map_err(map_err)
}

/// Decode `(B, n_qubits)` codes plus `B` species indices into row words.
fn decode_rows<const W: usize>(
    spec: &CoreBlock<W>,
    basis: &numpy::ndarray::ArrayView2<u8>,
    species: &[u8],
) -> PyResult<Vec<Word<W>>> {
    let n_q = spec.n_qubits();
    let n_rows = basis.shape()[0];
    if basis.shape()[1] != n_q {
        return Err(PyValueError::new_err(format!(
            "basis has {} columns but n_qubits = {n_q}",
            basis.shape()[1]
        )));
    }
    if species.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "species has length {} but basis has {n_rows} rows",
            species.len()
        )));
    }
    let mut out = Vec::with_capacity(n_rows);
    let mut row_buf = vec![0u8; n_q];
    for i in 0..n_rows {
        let row = basis.row(i);
        for (q, slot) in row_buf.iter_mut().enumerate() {
            *slot = row[q];
        }
        out.push(spec.encode_row(&row_buf, species[i] as usize).map_err(map_err)?);
    }
    Ok(out)
}

fn assert_rows_unique<const W: usize>(rows: &[Word<W>]) -> PyResult<()> {
    let mut seen: HashMap<&Word<W>, usize> = HashMap::with_capacity(rows.len());
    for (i, w) in rows.iter().enumerate() {
        if let Some(prev) = seen.insert(w, i) {
            return Err(PyValueError::new_err(format!(
                "basis contains the same (species, Pauli word) row at row {prev} and row {i}"
            )));
        }
    }
    Ok(())
}

fn pack_rows<'py, const W: usize>(
    py: Python<'py>,
    spec: &CoreBlock<W>,
    pairs: Vec<(Word<W>, f64)>,
) -> PyResult<PyBlockMap<'py>> {
    let n_q = spec.n_qubits();
    let m = pairs.len();
    let mut basis = vec![0u8; m * n_q];
    let mut species = vec![0u8; m];
    let mut coeffs = vec![0f64; m];
    for (i, (w, c)) in pairs.into_iter().enumerate() {
        species[i] = spec.decode_row(&w, &mut basis[i * n_q..(i + 1) * n_q]) as u8;
        coeffs[i] = c;
    }
    let basis_arr = basis
        .into_pyarray(py)
        .reshape([m, n_q])
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok((basis_arr, species.into_pyarray(py), coeffs.into_pyarray(py)))
}

fn protected_rows<const W: usize>(
    spec: &CoreBlock<W>,
    basis: Option<&PyReadonlyArray2<'_, u8>>,
    species: Option<&PyReadonlyArray1<'_, u8>>,
) -> PyResult<Vec<Word<W>>> {
    match (basis, species) {
        (None, None) => Ok(Vec::new()),
        (Some(b), Some(s)) => decode_rows(spec, &b.as_array(), s.as_slice()?),
        _ => Err(PyValueError::new_err(
            "protected_basis and protected_species must be given together",
        )),
    }
}

// ────────────────── width-generic method bodies ──────────────────

fn action_impl<'py, const W: usize>(
    spec: &CoreBlock<W>,
    py: Python<'py>,
    species: usize,
    p: PyReadonlyArray1<'py, u8>,
) -> PyResult<PyBlockMap<'py>> {
    let row = spec.encode_row(p.as_slice()?, species).map_err(map_err)?;
    pack_rows(py, spec, spec.action(&row))
}

fn leakage_impl<'py, const W: usize>(
    spec: &CoreBlock<W>,
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, u8>,
    species: PyReadonlyArray1<'py, u8>,
    coeffs: PyReadonlyArray1<'py, f64>,
    protected_basis: Option<PyReadonlyArray2<'py, u8>>,
    protected_species: Option<PyReadonlyArray1<'py, u8>>,
) -> PyResult<PyBlockMap<'py>> {
    let rows = decode_rows(spec, &basis.as_array(), species.as_slice()?)?;
    let coeffs_slice = coeffs.as_slice()?;
    if coeffs_slice.len() != rows.len() {
        return Err(PyValueError::new_err(format!(
            "coeffs has length {} but basis has {} rows",
            coeffs_slice.len(),
            rows.len()
        )));
    }
    let protected = protected_rows(spec, protected_basis.as_ref(), protected_species.as_ref())?;
    let pairs = spec
        .leakage(&rows, coeffs_slice, &protected)
        .map_err(map_err)?;
    pack_rows(py, spec, pairs)
}

#[allow(clippy::too_many_arguments)]
fn pc_step_impl<'py, const W: usize>(
    spec: &CoreBlock<W>,
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, u8>,
    species: PyReadonlyArray1<'py, u8>,
    coeffs: PyReadonlyArray1<'py, f64>,
    dt: f64,
    cfg: ppvm_lindblad::PcStepConfig,
    protected_basis: Option<PyReadonlyArray2<'py, u8>>,
    protected_species: Option<PyReadonlyArray1<'py, u8>>,
) -> PyResult<(PyBlockMap<'py>, ppvm_lindblad::PcStepStats)> {
    let mut rows = decode_rows(spec, &basis.as_array(), species.as_slice()?)?;
    assert_rows_unique(&rows)?;
    let mut coeffs_vec = coeffs.as_slice()?.to_vec();
    if coeffs_vec.len() != rows.len() {
        return Err(PyValueError::new_err(format!(
            "coeffs has length {} but basis has {} rows",
            coeffs_vec.len(),
            rows.len()
        )));
    }
    let protected = protected_rows(spec, protected_basis.as_ref(), protected_species.as_ref())?;
    let stats = spec
        .pc_step(&mut rows, &mut coeffs_vec, dt, &protected, &cfg)
        .map_err(map_err)?;
    let pairs: Vec<(Word<W>, f64)> = rows.into_iter().zip(coeffs_vec).collect();
    Ok((pack_rows(py, spec, pairs)?, stats))
}

fn generator_impl<'py, const W: usize>(
    spec: &CoreBlock<W>,
    py: Python<'py>,
    basis: PyReadonlyArray2<'py, u8>,
    species: PyReadonlyArray1<'py, u8>,
) -> PyResult<PyCoo<'py>> {
    let rows = decode_rows(spec, &basis.as_array(), species.as_slice()?)?;
    assert_rows_unique(&rows)?;
    let triplets = spec.generator(&rows);
    let total = triplets.len();
    let mut r = Vec::with_capacity(total);
    let mut c = Vec::with_capacity(total);
    let mut v = Vec::with_capacity(total);
    for (row, col, val) in triplets {
        r.push(row as u64);
        c.push(col as u64);
        v.push(val);
    }
    Ok((r.into_pyarray(py), c.into_pyarray(py), v.into_pyarray(py)))
}

/// PyO3 facade exposing [`ppvm_lindblad::BlockLindbladSpec`] to Python.
#[pyclass]
pub struct BlockLindbladSpec {
    inner: AnyBlock,
}

#[pymethods]
impl BlockLindbladSpec {
    /// `M = n_species` species on `n_qubits`: per-species Hamiltonian terms,
    /// jump operators and Kossakowski blocks in the same encodings as
    /// `LindbladSpec` (one outer list level per species), the exchange rate
    /// matrix `exchange[c][d] = k_cd` (`c → d`, diagonal ignored) and the
    /// matchings `(c, d, perm)` with `perm[p]` = position in `d` of the qubit
    /// at position `p` in `c` (a missing reverse is the inverse permutation;
    /// a missing pair is the identity). `resets` (optional) lists
    /// `(c, d, positions)`: the qubit positions of species `c` whose state is
    /// replaced by the unit state on the transition `c → d` (strings with a
    /// non-identity letter there are not carried; `block.rs` module docs).
    /// `word_width` forces a compiled Pauli-word width (the row length
    /// `n_qubits + ceil(log4 n_species)` must fit); `None` dispatches.
    #[new]
    #[pyo3(signature = (n_qubits, n_species, h_terms, h_coeffs, jump_lincombs, jump_rates,
                        kossakowski_ops, kossakowski_k, exchange, matchings, resets=None,
                        word_width=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_qubits: usize,
        n_species: usize,
        h_terms: Vec<Vec<String>>,
        h_coeffs: Vec<Vec<f64>>,
        jump_lincombs: Vec<Vec<Vec<(String, f64, f64)>>>,
        jump_rates: Vec<Vec<f64>>,
        kossakowski_ops: Vec<Vec<Vec<(String, f64, f64)>>>,
        kossakowski_k: Vec<Vec<Vec<(f64, f64)>>>,
        exchange: Vec<Vec<f64>>,
        matchings: Vec<(usize, usize, Vec<usize>)>,
        resets: Option<Vec<(usize, usize, Vec<usize>)>>,
        word_width: Option<usize>,
    ) -> PyResult<Self> {
        for (name, len) in [
            ("h_terms", h_terms.len()),
            ("h_coeffs", h_coeffs.len()),
            ("jump_lincombs", jump_lincombs.len()),
            ("jump_rates", jump_rates.len()),
            ("kossakowski_ops", kossakowski_ops.len()),
            ("kossakowski_k", kossakowski_k.len()),
        ] {
            if len != n_species {
                return Err(PyValueError::new_err(format!(
                    "{name} must have one entry per species ({n_species}), got {len}"
                )));
            }
        }
        let row_len = n_qubits + species_digits(n_species);
        let width = pick_width(row_len, word_width, "BlockLindbladSpec (n_qubits + species digits)")?;
        let mut inputs = Vec::with_capacity(n_species);
        for (s, (((((h, c), jl), jr), ko), kk)) in h_terms
            .into_iter()
            .zip(h_coeffs)
            .zip(jump_lincombs)
            .zip(jump_rates)
            .zip(kossakowski_ops)
            .zip(kossakowski_k)
            .enumerate()
        {
            inputs.push(SpecInputs::new(h, c, jl, jr, ko, kk, &format!("species {s}: "))?);
        }
        let resets = resets.unwrap_or_default();
        let inner = with_width!(width, W => {
            AnyBlock::from(build_block::<W>(n_qubits, &inputs, &exchange, &matchings, &resets)?)
        });
        Ok(Self { inner })
    }

    /// The Pauli-word width this block was instantiated with (chunks).
    #[getter]
    fn word_width(&self) -> usize {
        dispatch!(&self.inner, s => word_width_of(s))
    }

    /// The largest row length (`n_qubits + n_digits`) this width holds.
    #[getter]
    fn max_qubits(&self) -> usize {
        max_qubits_for(self.word_width())
    }

    /// Bytes of one packed row word at this width.
    #[getter]
    fn word_bytes(&self) -> usize {
        dispatch!(&self.inner, s => word_bytes_of(s))
    }

    /// The reset masks as given: `[(c, d, positions of c reset on c → d)]`.
    #[getter]
    fn resets(&self) -> Vec<(usize, usize, Vec<usize>)> {
        dispatch!(&self.inner, s => s.resets().to_vec())
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        dispatch!(&self.inner, s => s.n_qubits())
    }

    #[getter]
    fn n_species(&self) -> usize {
        dispatch!(&self.inner, s => s.n_species())
    }

    /// Species digits appended to each row word internally.
    #[getter]
    fn n_digits(&self) -> usize {
        dispatch!(&self.inner, s => s.n_digits())
    }

    /// `exchange[c][d] = k_cd` as a nested list (diagonal zero).
    #[getter]
    fn exchange(&self) -> Vec<Vec<f64>> {
        dispatch!(&self.inner, s => s.exchange().to_vec())
    }

    /// `Σ_d k_cd` per species.
    #[getter]
    fn rate_out(&self) -> Vec<f64> {
        dispatch!(&self.inner, s => s.rate_out().to_vec())
    }

    /// Number of Hamiltonian terms of one species.
    fn num_h_terms(&self, species: usize) -> PyResult<usize> {
        dispatch!(&self.inner, s => Ok(s.species(species).map_err(map_err)?.num_h_terms()))
    }

    /// Number of jump entries (jumps + Kossakowski pairs) of one species.
    fn num_jump_terms(&self, species: usize) -> PyResult<usize> {
        dispatch!(&self.inner, s => Ok(s.species(species).map_err(map_err)?.num_jump_terms()))
    }

    /// Current Hamiltonian coefficients of one species (a copy).
    fn h_coeffs<'py>(&self, py: Python<'py>, species: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        dispatch!(&self.inner, s => Ok(s
            .species(species)
            .map_err(map_err)?
            .h_coeffs()
            .into_pyarray(py)))
    }

    /// Overwrite the Hamiltonian coefficients of one species in place
    /// (`O(num_h_terms)`); `coeffs` is a 1-D float64 array of that length.
    fn set_h_coeffs(&mut self, species: usize, coeffs: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.num_h_terms(species)?;
        let v = f64_vec(coeffs, n, "set_h_coeffs").map_err(|_| {
            PyValueError::new_err(format!(
                "set_h_coeffs: species {species} has {n} Hamiltonian terms; expected a 1-D \
                 float64 ndarray of shape ({n},)"
            ))
        })?;
        dispatch!(&mut self.inner, s => s.set_h_coeffs(species, &v).map_err(map_err))
    }

    /// `L*_block` on the single row `(species, p)`.
    fn action<'py>(
        &self,
        py: Python<'py>,
        species: usize,
        p: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<PyBlockMap<'py>> {
        dispatch!(&self.inner, s => action_impl(s, py, species, p))
    }

    /// Off-basis component of `L*_block( Σ_j coeffs[j] · (species[j], basis[j]) )`.
    #[pyo3(signature = (basis, species, coeffs, protected_basis = None, protected_species = None))]
    fn leakage<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        species: PyReadonlyArray1<'py, u8>,
        coeffs: PyReadonlyArray1<'py, f64>,
        protected_basis: Option<PyReadonlyArray2<'py, u8>>,
        protected_species: Option<PyReadonlyArray1<'py, u8>>,
    ) -> PyResult<PyBlockMap<'py>> {
        dispatch!(&self.inner, s => leakage_impl(
            s, py, basis, species, coeffs, protected_basis, protected_species))
    }

    /// One predictor-corrector step on block rows; truncation knobs as
    /// `LindbladSpec.pc_step`, `max_basis` counting rows over all species.
    /// Returns `(basis, species, coeffs)`.
    #[pyo3(signature = (
        basis, species, coeffs, dt, max_basis,
        drop_tol = 0.0,
        protected_basis = None,
        protected_species = None,
        num_threads = None,
        admit_basis = None,
        tau_add = None,
        max_weight = None,
        depolarize_eps = 0.0,
        tail_mode = None,
        tail_threshold = 0.0,
        tail_rule = "abs",
        tail_seed = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pc_step<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        species: PyReadonlyArray1<'py, u8>,
        coeffs: PyReadonlyArray1<'py, f64>,
        dt: f64,
        max_basis: usize,
        drop_tol: f64,
        protected_basis: Option<PyReadonlyArray2<'py, u8>>,
        protected_species: Option<PyReadonlyArray1<'py, u8>>,
        num_threads: Option<usize>,
        admit_basis: Option<usize>,
        tau_add: Option<f64>,
        max_weight: Option<u32>,
        depolarize_eps: f64,
        tail_mode: Option<&str>,
        tail_threshold: f64,
        tail_rule: &str,
        tail_seed: u64,
    ) -> PyResult<PyBlockMap<'py>> {
        let cfg = ppvm_lindblad::PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            num_threads,
            locality: false,
            max_weight,
            depolarize_eps,
            tail: crate::lindblad::tail_config(tail_mode, tail_threshold, tail_rule, tail_seed)?,
        };
        let (map, _) = dispatch!(&self.inner, s => pc_step_impl(
            s, py, basis, species, coeffs, dt, cfg, protected_basis, protected_species))?;
        Ok(map)
    }

    /// Same as [`Self::pc_step`] but also returns the step account dict
    /// (`LindbladSpec.pc_step_stats`); the weight of a row is taken over the
    /// physical sites, not the species digits.
    #[pyo3(signature = (
        basis, species, coeffs, dt, max_basis,
        drop_tol = 0.0,
        protected_basis = None,
        protected_species = None,
        num_threads = None,
        admit_basis = None,
        tau_add = None,
        max_weight = None,
        depolarize_eps = 0.0,
        tail_mode = None,
        tail_threshold = 0.0,
        tail_rule = "abs",
        tail_seed = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn pc_step_stats<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        species: PyReadonlyArray1<'py, u8>,
        coeffs: PyReadonlyArray1<'py, f64>,
        dt: f64,
        max_basis: usize,
        drop_tol: f64,
        protected_basis: Option<PyReadonlyArray2<'py, u8>>,
        protected_species: Option<PyReadonlyArray1<'py, u8>>,
        num_threads: Option<usize>,
        admit_basis: Option<usize>,
        tau_add: Option<f64>,
        max_weight: Option<u32>,
        depolarize_eps: f64,
        tail_mode: Option<&str>,
        tail_threshold: f64,
        tail_rule: &str,
        tail_seed: u64,
    ) -> PyResult<(PyBlockMap<'py>, Bound<'py, pyo3::types::PyDict>)> {
        let cfg = ppvm_lindblad::PcStepConfig {
            max_basis,
            admit_basis,
            drop_tol,
            tau_add,
            num_threads,
            locality: false,
            max_weight,
            depolarize_eps,
            tail: crate::lindblad::tail_config(tail_mode, tail_threshold, tail_rule, tail_seed)?,
        };
        let (map, stats) = dispatch!(&self.inner, s => pc_step_impl(
            s, py, basis, species, coeffs, dt, cfg, protected_basis, protected_species))?;
        Ok((map, crate::lindblad::stats_dict(py, &stats)?))
    }

    /// Sparse generator on the given rows in COO form `(rows, cols, vals)`.
    fn generator<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        species: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<PyCoo<'py>> {
        dispatch!(&self.inner, s => generator_impl(s, py, basis, species))
    }
}
