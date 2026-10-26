// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! PyO3 wrapper around [`ppvm_lindblad::BlockLindbladSpec`]: `M` species of
//! the same `N` qubits coupled by first-order chemical exchange.
//!
//! Rows cross the boundary as a `(B, n_qubits)` uint8 code array plus a
//! length-`B` uint8 species array; the Rust side packs them into row words
//! (species digits appended) and splits them again on output. All physics
//! lives in `ppvm_lindblad::block`.

use std::collections::HashMap;

use num::Complex;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ppvm_lindblad::{BlockLindbladSpec as CoreBlock, JumpInput, LindbladSpec as CoreSpec, Word};
use pyo3::{exceptions::PyValueError, prelude::*};

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

fn map_err(e: ppvm_lindblad::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// One species' `LindbladSpec` from the same triple-encoded inputs as
/// `LindbladSpec.__init__` (`lindblad.rs`).
#[allow(clippy::too_many_arguments)]
fn build_species(
    n_qubits: usize,
    h_terms: Vec<String>,
    h_coeffs: Vec<f64>,
    jump_lincombs: Vec<Vec<(String, f64, f64)>>,
    jump_rates: Vec<f64>,
    kossakowski_ops: Vec<Vec<(String, f64, f64)>>,
    kossakowski_k: Vec<Vec<(f64, f64)>>,
    species: usize,
) -> PyResult<CoreSpec> {
    if h_terms.len() != h_coeffs.len() {
        return Err(PyValueError::new_err(format!(
            "species {species}: h_terms and h_coeffs must have the same length"
        )));
    }
    if jump_lincombs.len() != jump_rates.len() {
        return Err(PyValueError::new_err(format!(
            "species {species}: jump_lincombs and jump_rates must have the same length"
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
    let mut spec = CoreSpec::new(n_qubits, &h, &jumps).map_err(map_err)?;
    if !kossakowski_ops.is_empty() || !kossakowski_k.is_empty() {
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
        spec.add_kossakowski(&ops, &k).map_err(map_err)?;
    }
    Ok(spec)
}

/// Decode `(B, n_qubits)` codes plus `B` species indices into row words.
fn decode_rows(
    spec: &CoreBlock,
    basis: &numpy::ndarray::ArrayView2<u8>,
    species: &[u8],
) -> PyResult<Vec<Word>> {
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

fn assert_rows_unique(rows: &[Word]) -> PyResult<()> {
    let mut seen: HashMap<&Word, usize> = HashMap::with_capacity(rows.len());
    for (i, w) in rows.iter().enumerate() {
        if let Some(prev) = seen.insert(w, i) {
            return Err(PyValueError::new_err(format!(
                "basis contains the same (species, Pauli word) row at row {prev} and row {i}"
            )));
        }
    }
    Ok(())
}

fn pack_rows<'py>(
    py: Python<'py>,
    spec: &CoreBlock,
    pairs: Vec<(Word, f64)>,
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

fn protected_rows(
    spec: &CoreBlock,
    basis: Option<&PyReadonlyArray2<'_, u8>>,
    species: Option<&PyReadonlyArray1<'_, u8>>,
) -> PyResult<Vec<Word>> {
    match (basis, species) {
        (None, None) => Ok(Vec::new()),
        (Some(b), Some(s)) => decode_rows(spec, &b.as_array(), s.as_slice()?),
        _ => Err(PyValueError::new_err(
            "protected_basis and protected_species must be given together",
        )),
    }
}

/// PyO3 facade exposing [`ppvm_lindblad::BlockLindbladSpec`] to Python.
#[pyclass]
pub struct BlockLindbladSpec {
    inner: CoreBlock,
}

#[pymethods]
impl BlockLindbladSpec {
    /// `M = n_species` species on `n_qubits`: per-species Hamiltonian terms,
    /// jump operators and Kossakowski blocks in the same encodings as
    /// `LindbladSpec` (one outer list level per species), the exchange rate
    /// matrix `exchange[c][d] = k_cd` (`c → d`, diagonal ignored) and the
    /// matchings `(c, d, perm)` with `perm[p]` = position in `d` of the qubit
    /// at position `p` in `c` (a missing reverse is the inverse permutation;
    /// a missing pair is the identity).
    #[new]
    #[pyo3(signature = (n_qubits, n_species, h_terms, h_coeffs, jump_lincombs, jump_rates,
                        kossakowski_ops, kossakowski_k, exchange, matchings))]
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
        let mut species = Vec::with_capacity(n_species);
        for (s, (((((h, c), jl), jr), ko), kk)) in h_terms
            .into_iter()
            .zip(h_coeffs)
            .zip(jump_lincombs)
            .zip(jump_rates)
            .zip(kossakowski_ops)
            .zip(kossakowski_k)
            .enumerate()
        {
            species.push(build_species(n_qubits, h, c, jl, jr, ko, kk, s)?);
        }
        let inner = CoreBlock::new(n_qubits, species, &exchange, &matchings).map_err(map_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn n_species(&self) -> usize {
        self.inner.n_species()
    }

    /// Species digits appended to each row word internally.
    #[getter]
    fn n_digits(&self) -> usize {
        self.inner.n_digits()
    }

    /// `exchange[c][d] = k_cd` as a nested list (diagonal zero).
    #[getter]
    fn exchange(&self) -> Vec<Vec<f64>> {
        self.inner.exchange().to_vec()
    }

    /// `Σ_d k_cd` per species.
    #[getter]
    fn rate_out(&self) -> Vec<f64> {
        self.inner.rate_out().to_vec()
    }

    /// Number of Hamiltonian terms of one species.
    fn num_h_terms(&self, species: usize) -> PyResult<usize> {
        Ok(self.inner.species(species).map_err(map_err)?.num_h_terms())
    }

    /// Number of jump entries (jumps + Kossakowski pairs) of one species.
    fn num_jump_terms(&self, species: usize) -> PyResult<usize> {
        Ok(self.inner.species(species).map_err(map_err)?.num_jump_terms())
    }

    /// Current Hamiltonian coefficients of one species (a copy).
    fn h_coeffs<'py>(&self, py: Python<'py>, species: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .species(species)
            .map_err(map_err)?
            .h_coeffs()
            .into_pyarray(py))
    }

    /// Overwrite the Hamiltonian coefficients of one species in place
    /// (`O(num_h_terms)`); `coeffs` is a 1-D float64 array of that length.
    fn set_h_coeffs(&mut self, species: usize, coeffs: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.species(species).map_err(map_err)?.num_h_terms();
        let coeffs: PyReadonlyArray1<'_, f64> = coeffs.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "set_h_coeffs: expected a 1-D float64 ndarray of shape ({n},)"
            ))
        })?;
        let got = coeffs.len()?;
        if got != n {
            return Err(PyValueError::new_err(format!(
                "set_h_coeffs: species {species} has {n} Hamiltonian terms, got {got} coefficients"
            )));
        }
        match coeffs.as_slice() {
            Ok(sl) => self.inner.set_h_coeffs(species, sl).map_err(map_err),
            Err(_) => {
                let v: Vec<f64> = coeffs.as_array().iter().copied().collect();
                self.inner.set_h_coeffs(species, &v).map_err(map_err)
            }
        }
    }

    /// `L*_block` on the single row `(species, p)`.
    fn action<'py>(
        &self,
        py: Python<'py>,
        species: usize,
        p: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<PyBlockMap<'py>> {
        let row = self
            .inner
            .encode_row(p.as_slice()?, species)
            .map_err(map_err)?;
        pack_rows(py, &self.inner, self.inner.action(&row))
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
        let rows = decode_rows(&self.inner, &basis.as_array(), species.as_slice()?)?;
        let coeffs_slice = coeffs.as_slice()?;
        if coeffs_slice.len() != rows.len() {
            return Err(PyValueError::new_err(format!(
                "coeffs has length {} but basis has {} rows",
                coeffs_slice.len(),
                rows.len()
            )));
        }
        let protected = protected_rows(&self.inner, protected_basis.as_ref(), protected_species.as_ref())?;
        let pairs = self
            .inner
            .leakage(&rows, coeffs_slice, &protected)
            .map_err(map_err)?;
        pack_rows(py, &self.inner, pairs)
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
    ) -> PyResult<PyBlockMap<'py>> {
        let mut rows = decode_rows(&self.inner, &basis.as_array(), species.as_slice()?)?;
        assert_rows_unique(&rows)?;
        let mut coeffs_vec = coeffs.as_slice()?.to_vec();
        if coeffs_vec.len() != rows.len() {
            return Err(PyValueError::new_err(format!(
                "coeffs has length {} but basis has {} rows",
                coeffs_vec.len(),
                rows.len()
            )));
        }
        let protected = protected_rows(&self.inner, protected_basis.as_ref(), protected_species.as_ref())?;
        self.inner
            .pc_step(
                &mut rows,
                &mut coeffs_vec,
                dt,
                &protected,
                &ppvm_lindblad::PcStepConfig {
                    max_basis,
                    admit_basis,
                    drop_tol,
                    tau_add,
                    num_threads,
                    locality: false,
                },
            )
            .map_err(map_err)?;
        let pairs: Vec<(Word, f64)> = rows.into_iter().zip(coeffs_vec).collect();
        pack_rows(py, &self.inner, pairs)
    }

    /// Sparse generator on the given rows in COO form `(rows, cols, vals)`.
    fn generator<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
        species: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<PyCoo<'py>> {
        let rows = decode_rows(&self.inner, &basis.as_array(), species.as_slice()?)?;
        assert_rows_unique(&rows)?;
        let triplets = self.inner.generator(&rows);
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
}
