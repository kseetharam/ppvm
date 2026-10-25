// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! Error type for [`crate::LindbladSpec`] construction and stepping.

use crate::MAX_QUBITS;
use std::fmt;

/// Errors raised when constructing a [`crate::LindbladSpec`].
#[derive(Debug, Clone)]
pub enum Error {
    TooManyQubits {
        got: usize,
    },
    LengthMismatch {
        what: &'static str,
        a: usize,
        b: usize,
    },
    InvalidPauliCode {
        code: u8,
    },
    InvalidPauliChar {
        c: char,
    },
    WrongLength {
        expected: usize,
        got: usize,
    },
    NegativeRate {
        index: usize,
        rate: f64,
    },
    EmptyLincomb {
        index: usize,
    },
    KMatrixNotHermitian {
        n: usize,
        m: usize,
    },
    /// `set_kossakowski` on a spec built without a Kossakowski block.
    NoKossakowski,
    /// Sparse / block Kossakowski input: an operator index at or beyond
    /// the operator table.
    KIndexOutOfRange {
        what: &'static str,
        index: usize,
        m_ops: usize,
    },
    /// Block Kossakowski input: the same operator listed twice in one block.
    KBlockDuplicateIndex {
        block: usize,
        index: usize,
    },
    /// `set_jump_rates` on a spec built without jump operators.
    NoJumpOps,
    /// Block Lindbladian: a species index at or beyond `n_species`.
    SpeciesOutOfRange {
        got: usize,
        n_species: usize,
    },
    /// Block Lindbladian: a matching that is not a permutation of the qubits.
    InvalidMatching {
        c: usize,
        d: usize,
    },
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TooManyQubits { got } => {
                write!(
                    f,
                    "LindbladSpec supports n_qubits ≤ {MAX_QUBITS}; got {got}"
                )
            }
            Error::LengthMismatch { what, a, b } => {
                write!(f, "{what}: expected matching lengths, got {a} and {b}")
            }
            Error::InvalidPauliCode { code } => write!(
                f,
                "Pauli code must be 0 (I), 1 (X), 2 (Z), or 3 (Y); got {code}"
            ),
            Error::InvalidPauliChar { c } => {
                write!(f, "invalid Pauli character '{c}'; expected I, X, Y, or Z")
            }
            Error::WrongLength { expected, got } => {
                write!(f, "Pauli string has length {got} but n_qubits = {expected}")
            }
            Error::NegativeRate { index, rate } => {
                write!(f, "jump rate must be non-negative; got γ_{index} = {rate}")
            }
            Error::EmptyLincomb { index } => {
                write!(
                    f,
                    "jump {index}: lincomb must contain at least one Pauli term"
                )
            }
            Error::KMatrixNotHermitian { n, m } => write!(
                f,
                "Kossakowski matrix is not Hermitian: K[{n}][{m}] != conj(K[{m}][{n}])"
            ),
            Error::NoKossakowski => write!(
                f,
                "set_kossakowski: this LindbladSpec was built without a Kossakowski block"
            ),
            Error::NoJumpOps => write!(
                f,
                "set_jump_rates: this LindbladSpec was built without jump operators"
            ),
            Error::KIndexOutOfRange { what, index, m_ops } => write!(
                f,
                "{what}: Kossakowski operator index {index} out of range for {m_ops} operators"
            ),
            Error::KBlockDuplicateIndex { block, index } => write!(
                f,
                "set_kossakowski_blocks: block {block} lists operator {index} more than once"
            ),
            Error::SpeciesOutOfRange { got, n_species } => {
                write!(f, "species index {got} out of range for n_species = {n_species}")
            }
            Error::InvalidMatching { c, d } => write!(
                f,
                "matching ({c}, {d}) is not a permutation of the qubit positions"
            ),
            Error::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
