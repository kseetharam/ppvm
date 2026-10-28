// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! The Pauli-word width `W` (chunk count) is a const generic of every type of
//! the crate. These tests pin (i) the capacities and sizes per width, (ii) the
//! refusal above a width's capacity, (iii) that a wider instantiation gives
//! the SAME numbers as the default `W = 2` on a system that fits both (the
//! bit-ops loop over `W` chunks, the extra chunks are zero), and (iv) that a
//! word with more than 128 qubits really evolves.

use num::Complex;
use ppvm_lindblad::{
    CHUNK_BITS, Error, JumpInput, LindbladSpec, MAX_QUBITS, PcStepConfig, W_CHUNKS, Word,
    codes_from_word, max_qubits_for, parse_pauli_string, word_from_codes,
};
use std::collections::HashMap;

#[test]
fn capacities_and_sizes_per_width() {
    assert_eq!(MAX_QUBITS, max_qubits_for(W_CHUNKS));
    assert_eq!(CHUNK_BITS, 8 * std::mem::size_of::<ppvm_lindblad::Chunk>());
    for (w, n) in [(2usize, 128usize), (4, 256), (8, 512), (16, 1024), (17, 1088)] {
        assert_eq!(max_qubits_for(w), n * CHUNK_BITS / 64);
    }
    // 16·W + 16 bytes per word on a 64-bit target: x bits, z bits, n_qubits, hash.
    assert_eq!(std::mem::size_of::<Word<2>>(), 2 * 2 * CHUNK_BITS / 8 + 16);
    assert_eq!(std::mem::size_of::<Word<4>>(), 2 * 4 * CHUNK_BITS / 8 + 16);
    assert_eq!(std::mem::size_of::<Word<17>>(), 2 * 17 * CHUNK_BITS / 8 + 16);
}

#[test]
fn refusal_names_the_width() {
    let n = max_qubits_for(2) + 1;
    let codes = vec![0u8; n];
    match word_from_codes::<2>(&codes) {
        Err(Error::TooManyQubits { got, max }) => {
            assert_eq!((got, max), (n, max_qubits_for(2)));
        }
        other => panic!("expected TooManyQubits, got {other:?}"),
    }
    let msg = match LindbladSpec::<2>::new(n, &[], &[]) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected a refusal above the width"),
    };
    assert!(msg.starts_with(&format!("LindbladSpec supports n_qubits ≤ {}", max_qubits_for(2))), "{msg}");
    assert!(msg.contains("W = 2 chunks"), "{msg}");
    // the same n fits the next width
    assert!(LindbladSpec::<4>::new(n, &[], &[]).is_ok());
    assert!(word_from_codes::<4>(&codes).is_ok());
    assert!(LindbladSpec::<17>::new(max_qubits_for(17), &[], &[]).is_ok());
    assert!(LindbladSpec::<17>::new(max_qubits_for(17) + 1, &[], &[]).is_err());
}

/// A small dissipative chain: XX+YY+Z field, σ⁻ damping on site 0, a
/// Kossakowski block of σ⁻ on sites 1 and 2 with cross-correlations.
fn chain_inputs(n: usize) -> (Vec<(String, f64)>, Vec<JumpInput>, Vec<Vec<(String, Complex<f64>)>>, Vec<Vec<Complex<f64>>>) {
    let mut h = Vec::new();
    for j in 0..n - 1 {
        for op in ['X', 'Y'] {
            let mut s = vec!['I'; n];
            s[j] = op;
            s[j + 1] = op;
            h.push((s.into_iter().collect::<String>(), 0.7 + 0.1 * j as f64));
        }
    }
    for j in 0..n {
        let mut s = vec!['I'; n];
        s[j] = 'Z';
        h.push((s.into_iter().collect::<String>(), 0.3 * (j as f64 + 1.0)));
    }
    let sm = |site: usize| {
        let mut x = vec!['I'; n];
        x[site] = 'X';
        let mut y = vec!['I'; n];
        y[site] = 'Y';
        vec![
            (x.into_iter().collect::<String>(), Complex::new(0.5, 0.0)),
            (y.into_iter().collect::<String>(), Complex::new(0.0, -0.5)),
        ]
    };
    let jumps = vec![JumpInput { lincomb: sm(0), rate: 0.4 }];
    let ops = vec![sm(1), sm(2)];
    let k = vec![
        vec![Complex::new(1.0, 0.0), Complex::new(0.3, 0.2)],
        vec![Complex::new(0.3, -0.2), Complex::new(0.8, 0.0)],
    ];
    (h, jumps, ops, k)
}

fn build<const W: usize>(n: usize) -> LindbladSpec<W> {
    let (h, jumps, ops, k) = chain_inputs(n);
    let mut spec = LindbladSpec::<W>::new(n, &h, &jumps).unwrap();
    spec.add_kossakowski(&ops, &k).unwrap();
    spec
}

fn codes_of(s: &str) -> Vec<u8> {
    s.chars()
        .map(|c| match c {
            'I' => 0,
            'X' => 1,
            'Z' => 2,
            _ => 3,
        })
        .collect()
}

/// Run `n_steps` of the capped predictor-corrector at width `W`; return the
/// final `(codes, coeff)` map keyed by the code bytes (width-independent).
fn evolve<const W: usize>(n: usize, n_steps: usize, max_basis: usize) -> HashMap<Vec<u8>, f64> {
    let spec = build::<W>(n);
    let mut basis: Vec<Word<W>> = (0..n)
        .map(|j| {
            let mut s = vec!['I'; n];
            s[j] = 'Z';
            let st: String = s.into_iter().collect();
            parse_pauli_string::<W>(&st, n).unwrap().0
        })
        .collect();
    let mut coeffs: Vec<f64> = (0..n).map(|j| 1.0 / (j as f64 + 1.0)).collect();
    let cfg = PcStepConfig {
        max_basis,
        admit_basis: Some(3 * max_basis),
        drop_tol: 0.0,
        tau_add: None,
        num_threads: Some(1),
    };
    for _ in 0..n_steps {
        spec.pc_step(&mut basis, &mut coeffs, 0.05, &[], &cfg).unwrap();
    }
    let mut out = HashMap::new();
    let mut buf = vec![0u8; n];
    for (w, c) in basis.iter().zip(coeffs.iter()) {
        codes_from_word(w, &mut buf);
        out.insert(buf.clone(), *c);
    }
    out
}

fn assert_same_map_bitwise(a: &HashMap<Vec<u8>, f64>, b: &HashMap<Vec<u8>, f64>) {
    assert_eq!(a.len(), b.len(), "basis sizes differ: {} vs {}", a.len(), b.len());
    for (k, va) in a {
        let vb = b.get(k).expect("word missing at the other width");
        assert_eq!(va.to_bits(), vb.to_bits(), "{va} != {vb} (bitwise)");
    }
}

/// Same basis SET and the same coefficients to `tol`; returns the largest
/// absolute difference.
fn assert_same_map_close(a: &HashMap<Vec<u8>, f64>, b: &HashMap<Vec<u8>, f64>, tol: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "basis sizes differ: {} vs {}", a.len(), b.len());
    let mut worst = 0.0_f64;
    for (k, va) in a {
        let vb = b.get(k).expect("word missing at the other width");
        worst = worst.max((va - vb).abs());
        assert!((va - vb).abs() <= tol, "{va} vs {vb}");
    }
    worst
}

fn action_map<const W: usize>(spec: &LindbladSpec<W>, p: &str, n: usize) -> HashMap<Vec<u8>, f64> {
    spec.action(&parse_pauli_string::<W>(p, n).unwrap().0)
        .into_iter()
        .map(|(w, c)| {
            let mut b = vec![0u8; n];
            codes_from_word(&w, &mut b);
            (b, c)
        })
        .collect()
}

/// On a system that fits every width: the action of every word is identical
/// bit for bit across W = 2 / 4 / 8 / 16 / 17 (each output coefficient is
/// accumulated in the same candidate order; only the ORDER of the returned
/// list follows the width-dependent hash), and a truncated predictor-
/// corrector trajectory keeps the same basis set with coefficients equal to
/// round-off (the leakage merge and the CSC matvec associate in hash order,
/// and a word hashes over its `16·W` storage bytes). The W = 2 path itself is
/// the historical code path bit for bit (the Python-side fixtures pin that).
#[test]
fn wider_words_reproduce_the_default_width() {
    let n = 6;
    let s2 = build::<2>(n);
    let s4 = build::<4>(n);
    let s8 = build::<8>(n);
    let s16 = build::<16>(n);
    let s17 = build::<17>(n);
    for p in ["ZIIIII", "IXIIII", "XYZIII", "IIZZII", "YIIIIX", "XXYYZZ"] {
        let a2 = action_map(&s2, p, n);
        assert_same_map_bitwise(&a2, &action_map(&s4, p, n));
        assert_same_map_bitwise(&a2, &action_map(&s8, p, n));
        assert_same_map_bitwise(&a2, &action_map(&s16, p, n));
        assert_same_map_bitwise(&a2, &action_map(&s17, p, n));
    }
    // a truncated trajectory (rank cap binding): same basis set, round-off coefficients
    let e2 = evolve::<2>(n, 6, 40);
    let mut worst = 0.0_f64;
    worst = worst.max(assert_same_map_close(&e2, &evolve::<4>(n, 6, 40), 1e-12));
    worst = worst.max(assert_same_map_close(&e2, &evolve::<8>(n, 6, 40), 1e-12));
    worst = worst.max(assert_same_map_close(&e2, &evolve::<16>(n, 6, 40), 1e-12));
    worst = worst.max(assert_same_map_close(&e2, &evolve::<17>(n, 6, 40), 1e-12));
    eprintln!("largest cross-width coefficient difference after 6 capped steps: {worst:e}");
    // and W = 2 is deterministic run to run
    assert_same_map_bitwise(&e2, &evolve::<2>(n, 6, 40));
}

/// Words above 128 qubits: codec round trip at every width that holds them,
/// and a 200-site XX chain whose observable on the last site evolves under
/// the action exactly as the same 200-site operator does on site 0 (translation
/// of the chain = the bits moving across chunk boundaries).
#[test]
fn words_beyond_128_qubits_round_trip_and_evolve() {
    for n in [129usize, 256, 257, 512, 1000, 1024, 1088] {
        let codes: Vec<u8> = (0..n).map(|q| ((q * 7 + 3) % 4) as u8).collect();
        macro_rules! roundtrip {
            ($w:literal) => {
                if n <= max_qubits_for($w) {
                    let w: Word<$w> = word_from_codes(&codes).unwrap();
                    let mut back = vec![0u8; n];
                    codes_from_word(&w, &mut back);
                    assert_eq!(back, codes, "W = {} n = {n}", $w);
                } else {
                    assert!(matches!(word_from_codes::<$w>(&codes), Err(Error::TooManyQubits { .. })));
                }
            };
        }
        roundtrip!(2);
        roundtrip!(4);
        roundtrip!(8);
        roundtrip!(16);
        roundtrip!(17);
    }
    // XX + YY chain of 200 sites with a Z field: i[H, Z_j] at the first and
    // the last site give mirror-image outputs (the chain is reflection symmetric).
    let n = 200;
    let mut h = Vec::new();
    for j in 0..n - 1 {
        for op in ['X', 'Y'] {
            let mut s = vec!['I'; n];
            s[j] = op;
            s[j + 1] = op;
            h.push((s.into_iter().collect::<String>(), 1.0));
        }
    }
    let spec = LindbladSpec::<4>::new(n, &h, &[]).unwrap();
    let z_at = |j: usize| {
        let mut s = vec!['I'; n];
        s[j] = 'Z';
        parse_pauli_string::<4>(&s.into_iter().collect::<String>(), n).unwrap().0
    };
    let first = spec.action(&z_at(0));
    let last = spec.action(&z_at(n - 1));
    assert_eq!(first.len(), 2);
    assert_eq!(last.len(), 2);
    let mirror = |w: &Word<4>| {
        let mut b = vec![0u8; n];
        codes_from_word(w, &mut b);
        b.reverse();
        b
    };
    let mut lm: HashMap<Vec<u8>, f64> = last.iter().map(|(w, c)| (mirror(w), *c)).collect();
    for (w, c) in &first {
        let mut b = vec![0u8; n];
        codes_from_word(w, &mut b);
        let v = lm.remove(&b).expect("mirror word missing");
        assert_eq!(v.to_bits(), c.to_bits());
    }
    assert!(lm.is_empty());
    // the bits really sit past the second chunk
    let w = z_at(n - 1);
    assert_eq!(w.zbits.data[(n - 1) / CHUNK_BITS] >> ((n - 1) % CHUNK_BITS) & 1, 1);
    let _ = codes_of("I");
}
