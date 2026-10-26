// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! PyO3 wrapper around [`ppvm_lindblad::LindbladSpec`].
//!
//! All algorithmic work — Pauli arithmetic, active-site iteration, and the
//! dissipator branches (Hermitian Pauli fast path vs general complex Pauli
//! sum) — lives in the [`ppvm_lindblad`] crate. This module is responsible
//! only for the Python boundary: decoding the `(N, n_qubits)` numpy uint8
//! arrays into [`ppvm_lindblad::Word`] vectors, and re-encoding outputs
//! back into numpy.

use std::collections::HashMap;

use num::Complex;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use ppvm_lindblad::{JumpInput, LindbladSpec as CoreSpec, Word, codes_from_word, word_from_codes};
use pyo3::{exceptions::PyValueError, prelude::*};

type PyPauliMap<'py> = (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<f64>>);
type PyPauliMapComplex<'py> = (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<Complex64>>);
type PyCoo<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<f64>>,
);

fn map_err(e: ppvm_lindblad::Error) -> PyErr {
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
fn assert_basis_unique(basis: &[Word]) -> PyResult<()> {
    let mut seen: HashMap<&Word, usize> = HashMap::with_capacity(basis.len());
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
pub(crate) fn decode_basis(
    view: &numpy::ndarray::ArrayView2<u8>,
    n_qubits: usize,
) -> PyResult<Vec<Word>> {
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
        out.push(word_from_codes(&row_buf).map_err(map_err)?);
    }
    Ok(out)
}

/// Pack `Vec<(Word, f64)>` into the standard PyO3 return shape.
fn pack_pauli_map<'py>(
    py: Python<'py>,
    pairs: Vec<(Word, f64)>,
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

/// PyO3 facade exposing [`ppvm_lindblad::LindbladSpec`] to Python.
#[pyclass]
pub struct LindbladSpec {
    inner: CoreSpec,
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
    #[new]
    #[pyo3(signature = (n_qubits, h_terms, h_coeffs, jump_lincombs, jump_rates,
                        kossakowski_ops = vec![], kossakowski_k = vec![], kossakowski_coo = None))]
    fn new(
        n_qubits: usize,
        h_terms: Vec<String>,
        h_coeffs: Vec<f64>,
        jump_lincombs: Vec<Vec<(String, f64, f64)>>,
        jump_rates: Vec<f64>,
        kossakowski_ops: Vec<Vec<(String, f64, f64)>>,
        kossakowski_k: Vec<Vec<(f64, f64)>>,
        kossakowski_coo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if h_terms.len() != h_coeffs.len() {
            return Err(PyValueError::new_err(
                "h_terms and h_coeffs must have the same length",
            ));
        }
        if jump_lincombs.len() != jump_rates.len() {
            return Err(PyValueError::new_err(
                "jump_lincombs and jump_rates must have the same length",
            ));
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
        let mut inner = CoreSpec::new(n_qubits, &h, &jumps).map_err(map_err)?;
        if let Some(coo) = kossakowski_coo {
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
            let entries = coo_entries(&rows, &cols, &vals, "kossakowski_coo")?;
            let ops = lincombs_from_triples(kossakowski_ops);
            inner.add_kossakowski_sparse(&ops, &entries).map_err(map_err)?;
        } else if !kossakowski_ops.is_empty() || !kossakowski_k.is_empty() {
            let ops = lincombs_from_triples(kossakowski_ops);
            let k: Vec<Vec<Complex<f64>>> = kossakowski_k
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|(re, im)| Complex::new(re, im))
                        .collect()
                })
                .collect();
            inner.add_kossakowski(&ops, &k).map_err(map_err)?;
        }
        Ok(Self { inner })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn num_h_terms(&self) -> usize {
        self.inner.num_h_terms()
    }

    #[getter]
    fn num_jump_terms(&self) -> usize {
        self.inner.num_jump_terms()
    }

    /// Current Hamiltonian coefficients, shape `(num_h_terms,)` float64 (a copy).
    #[getter]
    fn h_coeffs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.h_coeffs().into_pyarray(py)
    }

    /// Overwrite the Hamiltonian coefficients in place (`O(num_h_terms)`).
    ///
    /// `coeffs` must be a 1-D float64 array of length `num_h_terms`;
    /// otherwise `ValueError`. Pauli words, jump operators and the
    /// Kossakowski block are untouched (no PSD re-check).
    fn set_h_coeffs(&mut self, coeffs: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.num_h_terms();
        let coeffs: PyReadonlyArray1<'_, f64> = coeffs.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "set_h_coeffs: expected a 1-D float64 ndarray of shape ({n},)"
            ))
        })?;
        let got = coeffs.len()?;
        if got != n {
            return Err(PyValueError::new_err(format!(
                "set_h_coeffs: expected {n} coefficients (num_h_terms), got {got}"
            )));
        }
        // `as_slice` fails on non-contiguous views; fall back to a copy.
        match coeffs.as_slice() {
            Ok(sl) => self.inner.set_h_coeffs(sl).map_err(map_err),
            Err(_) => {
                let v: Vec<f64> = coeffs.as_array().iter().copied().collect();
                self.inner.set_h_coeffs(&v).map_err(map_err)
            }
        }
    }

    /// Number of jump operators given at construction (the length of
    /// `jump_rates`); does not count Kossakowski pairs.
    #[getter]
    fn num_jump_ops(&self) -> usize {
        self.inner.num_jump_ops()
    }

    /// Current GKSL rates of the jump operators given at construction,
    /// shape `(num_jump_ops,)` float64 (a copy).
    #[getter]
    fn jump_rates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.jump_rates().into_pyarray(py)
    }

    /// Overwrite the jump rates in place (`O(num_jump_ops)`).
    ///
    /// `rates` must be a 1-D float64 array of length `num_jump_ops` with
    /// non-negative entries; otherwise `ValueError`. Jump operators,
    /// Hamiltonian and Kossakowski block are untouched.
    fn set_jump_rates(&mut self, rates: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.num_jump_ops();
        let rates: PyReadonlyArray1<'_, f64> = rates.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "set_jump_rates: expected a 1-D float64 ndarray of shape ({n},)"
            ))
        })?;
        let got = rates.len()?;
        if got != n {
            return Err(PyValueError::new_err(format!(
                "set_jump_rates: expected {n} rates (num_jump_ops), got {got}"
            )));
        }
        match rates.as_slice() {
            Ok(sl) => self.inner.set_jump_rates(sl).map_err(map_err),
            Err(_) => {
                let v: Vec<f64> = rates.as_array().iter().copied().collect();
                self.inner.set_jump_rates(&v).map_err(map_err)
            }
        }
    }

    /// Number `M` of Kossakowski operators (0 when built without a block).
    #[getter]
    fn num_kossakowski_ops(&self) -> usize {
        self.inner.num_kossakowski_ops()
    }

    /// Current Kossakowski matrix, shape `(M, M)` complex128 (a copy;
    /// `(0, 0)` when built without a block).
    #[getter]
    fn kossakowski<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let k = self.inner.kossakowski();
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
        let m = self.inner.num_kossakowski_ops();
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
        self.inner.set_kossakowski(&rows).map_err(map_err)
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
        if self.inner.num_kossakowski_ops() == 0 {
            return Err(PyValueError::new_err(
                "set_kossakowski_sparse: this LindbladSpec was built without a Kossakowski block",
            ));
        }
        let entries = coo_entries(&rows, &cols, &vals, "set_kossakowski_sparse")?;
        self.inner.set_kossakowski_sparse(&entries).map_err(map_err)
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
        if self.inner.num_kossakowski_ops() == 0 {
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
        self.inner.set_kossakowski_blocks(&core).map_err(map_err)
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
        let entries = self.inner.kossakowski_entries();
        let rows: Vec<u64> = entries.iter().map(|e| e.0 as u64).collect();
        let cols: Vec<u64> = entries.iter().map(|e| e.1 as u64).collect();
        let vals: Vec<Complex64> = entries.iter().map(|e| Complex64::new(e.2.re, e.2.im)).collect();
        (rows.into_pyarray(py), cols.into_pyarray(py), vals.into_pyarray(py))
    }

    /// Number of stored nonzero entries of `K` (both triangles).
    #[getter]
    fn num_kossakowski_nonzeros(&self) -> usize {
        self.inner.num_kossakowski_nonzeros()
    }

    /// Number of distinct pair templates: operator-pattern relationships
    /// (the Pauli structure of `(A_n, A_m)` up to a qubit relabelling)
    /// shared by the compiled pairs.
    #[getter]
    fn num_pair_templates(&self) -> usize {
        self.inner.num_pair_templates()
    }

    /// Number of compiled Kossakowski pairs `n ≤ m` (`|K_nm| > 1e-14·max|K|`).
    #[getter]
    fn num_compiled_pairs(&self) -> usize {
        self.inner.num_compiled_pairs()
    }

    /// Heap bytes of the Kossakowski compilation: `(templates, pairs, entries)`.
    #[getter]
    fn kossakowski_memory_bytes(&self) -> (usize, usize, usize) {
        self.inner.kossakowski_memory_bytes()
    }

    /// Apply `L*` to a single Pauli string `p`.
    fn action<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<PyPauliMap<'py>> {
        let p_slice = p.as_slice()?;
        let p_word = word_from_codes(p_slice).map_err(map_err)?;
        let pairs = self.inner.action(&p_word);
        pack_pauli_map(py, pairs, self.inner.n_qubits())
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
        let n_q = self.inner.n_qubits();
        let basis_view = basis.as_array();
        let basis_words = decode_basis(&basis_view, n_q)?;
        let coeffs_slice = coeffs.as_slice()?;
        if coeffs_slice.len() != basis_words.len() {
            return Err(PyValueError::new_err(format!(
                "coeffs has length {} but basis has {} rows",
                coeffs_slice.len(),
                basis_words.len()
            )));
        }
        let protected_words: Vec<Word> = if let Some(ref prot) = protected {
            let pv = prot.as_array();
            decode_basis(&pv, n_q)?
        } else {
            Vec::new()
        };
        let pairs = self
            .inner
            .leakage(&basis_words, coeffs_slice, &protected_words)
            .map_err(map_err)?;
        pack_pauli_map(py, pairs, n_q)
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
    /// words are never dropped.
    #[pyo3(signature = (
        basis, coeffs, dt, max_basis,
        drop_tol = 0.0,
        protected = None,
        num_threads = None,
        admit_basis = None,
        tau_add = None,
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
    ) -> PyResult<PyPauliMap<'py>> {
        let n_q = self.inner.n_qubits();
        let basis_view = basis.as_array();
        let mut basis_words = decode_basis(&basis_view, n_q)?;
        assert_basis_unique(&basis_words)?;
        let mut coeffs_vec = coeffs.as_slice()?.to_vec();
        if coeffs_vec.len() != basis_words.len() {
            return Err(PyValueError::new_err(format!(
                "coeffs has length {} but basis has {} rows",
                coeffs_vec.len(),
                basis_words.len()
            )));
        }
        let protected_words: Vec<Word> = if let Some(ref p) = protected {
            decode_basis(&p.as_array(), n_q)?
        } else {
            Vec::new()
        };
        self.inner
            .pc_step(
                &mut basis_words,
                &mut coeffs_vec,
                dt,
                &protected_words,
                &ppvm_lindblad::PcStepConfig {
                    max_basis,
                    admit_basis,
                    drop_tol,
                    tau_add,
                    num_threads,
                },
            )
            .map_err(map_err)?;

        // Pack output. Basis may have grown; coeffs has the same new length.
        let pairs: Vec<(Word, f64)> = basis_words.into_iter().zip(coeffs_vec).collect();
        pack_pauli_map(py, pairs, n_q)
    }

    /// Same as [`Self::pc_step`] but also returns a dict mapping phase
    /// name → microseconds spent in that phase, for profiling.
    #[pyo3(signature = (
        basis, coeffs, dt, max_basis,
        drop_tol = 0.0,
        protected = None,
        num_threads = None,
        admit_basis = None,
        tau_add = None,
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
    ) -> PyResult<(PyPauliMap<'py>, Bound<'py, pyo3::types::PyDict>)> {
        let n_q = self.inner.n_qubits();
        let basis_view = basis.as_array();
        let mut basis_words = decode_basis(&basis_view, n_q)?;
        assert_basis_unique(&basis_words)?;
        let mut coeffs_vec = coeffs.as_slice()?.to_vec();
        if coeffs_vec.len() != basis_words.len() {
            return Err(PyValueError::new_err(format!(
                "coeffs has length {} but basis has {} rows",
                coeffs_vec.len(),
                basis_words.len()
            )));
        }
        let protected_words: Vec<Word> = if let Some(ref p) = protected {
            decode_basis(&p.as_array(), n_q)?
        } else {
            Vec::new()
        };
        let timings = self
            .inner
            .pc_step_timed(
                &mut basis_words,
                &mut coeffs_vec,
                dt,
                &protected_words,
                &ppvm_lindblad::PcStepConfig {
                    max_basis,
                    admit_basis,
                    drop_tol,
                    tau_add,
                    num_threads,
                },
            )
            .map_err(map_err)?;

        let pairs: Vec<(Word, f64)> = basis_words.into_iter().zip(coeffs_vec).collect();
        let map = pack_pauli_map(py, pairs, n_q)?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("leakage1_us", timings.leakage1_us)?;
        d.set_item("expand1_us", timings.expand1_us)?;
        d.set_item("expm1_us", timings.expm1_us)?;
        d.set_item("leakage2_us", timings.leakage2_us)?;
        d.set_item("expand2_us", timings.expand2_us)?;
        d.set_item("expm2_us", timings.expm2_us)?;
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
        use num::Complex;
        use ppvm_lindblad::orbit_rep;

        let n_q = self.inner.n_qubits();
        let basis_view = basis.as_array();
        let mut basis_words = decode_basis(&basis_view, n_q)?;
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
        let protected_words: Vec<Word> = if let Some(ref p) = protected {
            decode_basis(&p.as_array(), n_q)?
        } else {
            Vec::new()
        };
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
            &self.inner,
            &mut basis_words,
            &mut coeffs_vec,
            dt,
            &protected_words,
            group.core(),
            k_slice,
            &ppvm_lindblad::PcStepConfig {
                max_basis,
                admit_basis,
                drop_tol,
                tau_add,
                num_threads: None,
            },
        )
        .map_err(map_err)?;

        let m = basis_words.len();
        let mut out_basis = vec![0u8; m * n_q];
        for (i, w) in basis_words.iter().enumerate() {
            codes_from_word(w, &mut out_basis[i * n_q..(i + 1) * n_q]);
        }
        let out_coeffs: Vec<Complex64> = coeffs_vec
            .iter()
            .map(|c| Complex64::new(c.re, c.im))
            .collect();
        let basis_arr = out_basis
            .into_pyarray(py)
            .reshape([m, n_q])
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
        Ok((basis_arr, out_coeffs.into_pyarray(py)))
    }

    /// Sparse generator matrix in COO form: `(rows, cols, vals)`.
    fn generator<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<PyCoo<'py>> {
        let n_q = self.inner.n_qubits();
        let basis_view = basis.as_array();
        let basis_words = decode_basis(&basis_view, n_q)?;
        assert_basis_unique(&basis_words)?;
        let triplets = self.inner.generator(&basis_words);
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
}
