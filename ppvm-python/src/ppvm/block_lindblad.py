# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0

"""Chemical exchange between ``M`` species of the same qubits as a block-
structured Pauli Lindbladian.

Species ``c`` (a conformer, a ligand pool, a maturation stage) has its own
Hamiltonian, jump operators and Kossakowski block on the ``N`` qubits; the
species interconvert by first-order kinetics with rates ``k_cd`` (``c → d``)
and spin matchings ``Π_cd`` (which qubit of ``c`` is which qubit of ``d``):

    d ρ_c / dt = L_c ρ_c + Σ_{d≠c} ( k_dc Π_dc ρ_d Π_dc† − k_cd ρ_c ).

The observable is a direct sum ``O = ⊕_c O_c`` with ``⟨O⟩ = Σ_c Tr[O_c ρ_c]``
and evolves in the Heisenberg picture as

    d O_c / dt = L_c* O_c + Σ_{d≠c} k_cd ( Π_cd† O_d Π_cd − O_c ),

so on a Pauli string of species ``d`` the exchange produces the matched
string in every species ``c`` with ``k_cd > 0`` (weight ``+k_cd``, letters
permuted by the matching, no phase) and ``−Σ_c k_dc`` on the string itself.

A basis row is a Pauli string **and** a species index; the ``*_arr`` methods
take a ``(B, n_qubits)`` uint8 code array (``0=I, 1=X, 2=Z, 3=Y``) together
with a length-``B`` uint8 species array and return the same triple
``(basis, species, coeffs)``. Internally a row is one packed word with the
species index appended as base-4 digits, so the cost per step is that of a
single species' Lindbladian per row: ``M ×`` the strings of one species, not
``4^(N + a)`` strings of an ancilla encoding. Truncation (``max_basis``,
``admit_basis``, ``drop_tol``, ``tau_add``, protected rows) has exactly the
semantics of `Lindbladian.pc_step_arr`, with ``max_basis`` counting rows
over all species.

**Reset masks** (species of unequal size padded to a common qubit set): a
transition ``c → d`` may name positions of ``c`` whose state is not carried
but replaced by the unit state (spins that exist only in ``c`` are traced
out, spins that exist only in ``d`` arrive unpolarised),

    ρ_d += k_cd Π_cd [ Tr_R(ρ_c) ⊗ 1_R / 2^|R| ].

On Pauli strings this is the projector onto strings that are the identity on
``R`` followed by the permutation (self-adjoint), so in the Heisenberg
picture the inflow ``+k_cd (P ∘ π_cd)`` is dropped whenever ``P ∘ π_cd`` has
a non-identity letter on a masked position; the drain is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ._core import BlockLindbladSpec as _BlockLindbladSpec
from .lindblad import MAX_QUBITS, PauliLincomb, _check_n_qubits, _codes_to_basis, _normalize_jump, _string_to_codes, _tail_args


def _per_species(value, n_species, name):
    """``value`` as a list with one entry per species (``None`` -> all ``None``)."""
    if value is None:
        return [None] * n_species
    out = list(value)
    if len(out) != n_species:
        raise ValueError(f"{name} must have one entry per species ({n_species}), got {len(out)}")
    return out


class BlockLindbladian:
    """Pre-compiled adjoint Lindbladian of ``n_species`` exchanging species.

    Parameters
    ----------
    n_qubits, n_species:
        Qubits per species and number of species ``M``.
    h_terms:
        One iterable of ``(pauli_string, coefficient)`` pairs per species
        (the species' Hermitian Hamiltonian).
    jump_terms:
        ``None`` or one iterable of ``(jump_op, rate)`` pairs per species
        (``None`` / empty for a species without jumps); same format as
        `Lindbladian`.
    kossakowski:
        ``None`` or one ``(ops, K)`` pair (or ``None``) per species, as the
        ``kossakowski`` argument of `Lindbladian`; ``K`` must be Hermitian
        positive semidefinite.
    exchange:
        ``(M, M)`` array of first-order rates, ``exchange[c, d] = k_cd`` the
        rate ``c → d`` in the units of the Hamiltonian (diagonal ignored,
        entries non-negative); ``None`` for no exchange.
    matchings:
        ``{(c, d): perm}`` with ``perm[p]`` the position in species ``d`` of
        the qubit at position ``p`` of species ``c``; a missing reverse pair
        uses the inverse permutation, a missing pair the identity.
    word_width:
        Optional Pauli-word width in chunks (one of `ppvm.lindblad.WORD_WIDTHS`)
        to force; ``None`` dispatches on the row length ``n_qubits +
        ceil(log4 n_species)`` (the species digits share the word), which must
        not exceed `ppvm.lindblad.MAX_QUBITS`.
    resets:
        ``None`` or ``{(c, d): positions}``: the qubit positions **of species
        c** whose state is replaced by the unit state on the transition
        ``c → d`` (module docs). Only pairs with ``exchange[c, d] > 0`` may
        appear; no reverse pair is inferred. ``None`` / all-empty is the plain
        permutation exchange, bit for bit.

    Examples
    --------
    Two conformers of a two-qubit system, symmetric exchange:

    >>> BlockLindbladian(2, 2, [[("ZZ", 1.0)], [("XX", 0.5)]],
    ...                  exchange=[[0.0, 100.0], [100.0, 0.0]])
    """

    #: relative tolerance for the Hermiticity and PSD checks on ``K``.
    _K_TOL = 1e-10

    def __init__(
        self,
        n_qubits: int,
        n_species: int,
        h_terms: Sequence[Iterable[tuple[str, float]]],
        jump_terms: Sequence[Iterable[tuple[str | PauliLincomb, float]] | None] | None = None,
        kossakowski: Sequence[tuple[Sequence[str | PauliLincomb], object] | None] | None = None,
        exchange=None,
        matchings: dict[tuple[int, int], Sequence[int]] | None = None,
        resets: dict[tuple[int, int], Iterable[int]] | None = None,
        word_width: int | None = None,
    ):
        self.n_qubits = int(n_qubits)
        self.n_species = int(n_species)
        if self.n_species < 1:
            raise ValueError("n_species must be at least 1")
        n_digits = 0
        while (1 << (2 * n_digits)) < self.n_species:
            n_digits += 1
        _check_n_qubits(self.n_qubits + n_digits, "BlockLindbladian (n_qubits + species digits)")
        h_terms = _per_species(h_terms, self.n_species, "h_terms")
        jump_terms = _per_species(jump_terms, self.n_species, "jump_terms")
        kossakowski = _per_species(kossakowski, self.n_species, "kossakowski")
        h_strs, h_coeffs, j_lincombs, j_rates, k_ops, k_mats = [], [], [], [], [], []
        for s in range(self.n_species):
            strs, coeffs = [], []
            for p, c in h_terms[s] or ():
                strs.append(p)
                coeffs.append(float(c))
            h_strs.append(strs)
            h_coeffs.append(coeffs)
            lincombs, rates = [], []
            for jump_op, rate in jump_terms[s] or ():
                lincombs.append(_normalize_jump(jump_op))
                rates.append(float(rate))
            j_lincombs.append(lincombs)
            j_rates.append(rates)
            ops_s, k_s = [], []
            if kossakowski[s] is not None:
                ops, K = kossakowski[s]
                ops_s = [_normalize_jump(op) for op in ops]
                K = np.asarray(K, dtype=np.complex128)
                if K.shape != (len(ops_s), len(ops_s)):
                    raise ValueError(
                        f"species {s}: kossakowski K has shape {K.shape} but ops has length {len(ops_s)}"
                    )
                if len(ops_s):
                    scale = max(float(np.abs(K).max()), 1.0)
                    if float(np.abs(K - K.conj().T).max()) > self._K_TOL * scale:
                        raise ValueError(f"species {s}: kossakowski K is not Hermitian")
                    evals = np.linalg.eigvalsh(K)
                    if float(evals.min()) < -self._K_TOL * scale:
                        raise ValueError(
                            f"species {s}: kossakowski K is not positive semidefinite "
                            f"(min eigenvalue {evals.min():.3e})"
                        )
                k_s = [[(float(v.real), float(v.imag)) for v in row] for row in K]
            k_ops.append(ops_s)
            k_mats.append(k_s)
        if exchange is None:
            ex = np.zeros((self.n_species, self.n_species))
        else:
            ex = np.array(exchange, dtype=np.float64)
            if ex.shape != (self.n_species, self.n_species):
                raise ValueError(f"exchange must have shape ({self.n_species}, {self.n_species}), got {ex.shape}")
        ex = ex.copy()
        np.fill_diagonal(ex, 0.0)
        match_list = []
        for (c, d), perm in (matchings or {}).items():
            match_list.append((int(c), int(d), [int(q) for q in perm]))
        reset_list = []
        for (c, d), positions in (resets or {}).items():
            positions = sorted({int(q) for q in positions})
            if positions:                                   # an empty mask is no mask
                reset_list.append((int(c), int(d), positions))
        self._spec = _BlockLindbladSpec(
            self.n_qubits, self.n_species, h_strs, h_coeffs, j_lincombs, j_rates, k_ops, k_mats,
            ex.tolist(), match_list, reset_list or None,
            None if word_width is None else int(word_width),
        )

    @property
    def word_width(self) -> int:
        """The Pauli-word width (chunks) the rows are packed in."""
        return self._spec.word_width

    @property
    def max_qubits(self) -> int:
        """The largest row length (``n_qubits + n_digits``) this width holds."""
        return self._spec.max_qubits

    @property
    def word_bytes(self) -> int:
        """Bytes of one packed row word at this width."""
        return self._spec.word_bytes

    # ── bookkeeping ──

    @property
    def n_digits(self) -> int:
        """Species digits appended to each row word internally (``⌈log₄ M⌉``)."""
        return self._spec.n_digits

    @property
    def exchange(self) -> np.ndarray:
        """``(M, M)`` rate matrix ``k_cd`` (a copy, diagonal zero)."""
        return np.array(self._spec.exchange, dtype=np.float64)

    @property
    def rate_out(self) -> np.ndarray:
        """``Σ_d k_cd`` per species."""
        return np.array(self._spec.rate_out, dtype=np.float64)

    @property
    def resets(self) -> dict[tuple[int, int], list[int]]:
        """``{(c, d): positions of c reset on c → d}`` (empty without masks)."""
        return {(int(c), int(d)): [int(q) for q in qs] for c, d, qs in self._spec.resets}

    def num_h_terms(self, species: int) -> int:
        return self._spec.num_h_terms(int(species))

    def num_jump_terms(self, species: int) -> int:
        return self._spec.num_jump_terms(int(species))

    def h_coeffs(self, species: int) -> np.ndarray:
        """Current Hamiltonian coefficients of one species (a copy)."""
        return self._spec.h_coeffs(int(species))

    def set_h_coeffs(self, species: int, coeffs) -> None:
        """Overwrite one species' Hamiltonian coefficients in place
        (construction order of its ``h_terms``); the other species, the
        jumps, the Kossakowski blocks and the exchange are untouched.
        Afterwards every method is bit-identical to a fresh construction."""
        arr = np.ascontiguousarray(coeffs, dtype=np.float64)
        n = self._spec.num_h_terms(int(species))
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"set_h_coeffs: species {species} expects shape ({n},), got {arr.shape}")
        self._spec.set_h_coeffs(int(species), arr)

    # ── ndarray hot path ──

    @staticmethod
    def _species_arr(species, n_rows):
        arr = np.ascontiguousarray(species, dtype=np.uint8)
        if arr.ndim != 1 or arr.shape[0] != n_rows:
            raise ValueError(f"species must be a 1-D array of length {n_rows}, got shape {arr.shape}")
        return arr

    def _protected(self, protected_arr, protected_species):
        if protected_arr is None and protected_species is None:
            return None, None
        if protected_arr is None or protected_species is None:
            raise ValueError("protected_arr and protected_species must be given together")
        p = np.ascontiguousarray(protected_arr, dtype=np.uint8)
        return p, self._species_arr(protected_species, p.shape[0])

    def action_arr(self, species: int, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``L*_block`` on the row ``(species, p)``; ``p`` as uint8 codes.
        Returns ``(out_basis, out_species, out_coeffs)``."""
        return self._spec.action(int(species), np.ascontiguousarray(p, dtype=np.uint8))

    def leakage_arr(
        self,
        basis_arr: np.ndarray,
        species: np.ndarray,
        coeffs: np.ndarray,
        protected_arr: np.ndarray | None = None,
        protected_species: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Off-basis component of ``L*_block( Σ_j coeffs[j] (species[j], basis[j]) )``."""
        basis_arr = np.ascontiguousarray(basis_arr, dtype=np.uint8)
        p_arr, p_sp = self._protected(protected_arr, protected_species)
        return self._spec.leakage(
            basis_arr,
            self._species_arr(species, basis_arr.shape[0]),
            np.ascontiguousarray(coeffs, dtype=np.float64),
            p_arr,
            p_sp,
        )

    def pc_step_arr(
        self,
        basis_arr: np.ndarray,
        species: np.ndarray,
        coeffs: np.ndarray,
        dt: float,
        max_basis: int,
        drop_tol: float = 1e-12,
        protected_arr: np.ndarray | None = None,
        protected_species: np.ndarray | None = None,
        num_threads: int | None = None,
        admit_basis: int | None = None,
        tau_add: float | None = None,
        max_weight: int | None = None,
        depolarize_eps: float = 0.0,
        return_stats: bool = False,
        tail_mode: str | None = None,
        tail_threshold: float = 0.0,
        tail_rule: str = "abs",
        tail_seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """One predictor-corrector step ``O ← exp(dt·L*_block) O`` on block rows.

        Same algorithm and truncation semantics as `Lindbladian.pc_step_arr`
        (``max_basis`` rank cap over all rows, ``admit_basis`` working-set
        bound, ``drop_tol`` magnitude prune, ``tau_add`` admission filter,
        protected rows never dropped); the protected rows are given as a
        ``(protected_arr, protected_species)`` pair. ``max_weight`` /
        ``depolarize_eps`` / ``return_stats`` and the tail keywords
        ``tail_mode`` / ``tail_threshold`` / ``tail_rule`` / ``tail_seed`` as
        `Lindbladian.pc_step_arr` (the stochastic rounding acts on rows over
        all species; protected rows are never randomised); the Pauli weight of
        a row is taken over the physical sites (the species index is not a
        letter). Returns
        ``(new_basis, new_species, new_coeffs)`` (plus the stats dict with
        ``return_stats=True``).
        """
        basis_arr = np.ascontiguousarray(basis_arr, dtype=np.uint8)
        p_arr, p_sp = self._protected(protected_arr, protected_species)
        args = (
            basis_arr,
            self._species_arr(species, basis_arr.shape[0]),
            np.ascontiguousarray(coeffs, dtype=np.float64),
            float(dt),
            int(max_basis),
            float(drop_tol),
            p_arr,
            p_sp,
            None if num_threads is None else int(num_threads),
            None if admit_basis is None else int(admit_basis),
            None if tau_add is None else float(tau_add),
            None if max_weight is None else int(max_weight),
            float(depolarize_eps),
            *_tail_args(tail_mode, tail_threshold, tail_rule, tail_seed),
        )
        if return_stats:
            (b, sp, c), stats = self._spec.pc_step_stats(*args)
            return b, sp, c, stats
        return self._spec.pc_step(*args)

    def generator_arr(self, basis_arr: np.ndarray, species: np.ndarray):
        """Generator on the given rows as COO triples ``(rows, cols, vals)``."""
        basis_arr = np.ascontiguousarray(basis_arr, dtype=np.uint8)
        return self._spec.generator(basis_arr, self._species_arr(species, basis_arr.shape[0]))

    # ── string-keyed convenience ──

    def action(self, species: int, p: str) -> dict[tuple[int, str], float]:
        """``L*_block`` on ``(species, p)`` as ``{(species, string): coeff}``."""
        out_basis, out_species, out_coeffs = self.action_arr(species, _string_to_codes(p, self.n_qubits))
        keys = _codes_to_basis(out_basis)
        return {(int(s), k): float(v) for s, k, v in zip(out_species, keys, out_coeffs) if v != 0.0}
