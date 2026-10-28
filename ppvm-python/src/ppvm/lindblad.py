# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct Pauli-Lindbladian time evolution on an adaptive Pauli-string basis.

Given a Hermitian Pauli Hamiltonian H = Σ c_i P_i and jump operators
L_k = Σ_a λ_{k,a} P_{k,a} (each a complex linear combination of Pauli
strings) with rates γ_k ≥ 0, this module exposes the primitives for
adaptive Heisenberg-picture evolution:

- ``pc_step(...)`` / ``pc_step_arr(...)``: one adaptive predictor-corrector
  step ``O ← exp(dt·L*) O``; ``pc_step_orbit_rep(...)`` is the
  translation-symmetric (momentum-sector) variant
- ``action(p)`` / ``action_arr(p)``: L*(p) for one Pauli string p
- ``leakage(basis, coeffs)`` / ``leakage_arr(...)``: off-basis component of
  L*(Σ c_j p_j), driving basis expansion
- ``generator(basis)``: COO triples ``(rows, cols, vals)`` for the generator
  matrix M such that L* restricted to ``basis`` is ``M @ coeffs``. Users
  wanting a sparse matrix can wrap them — e.g.
  ``scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()``

The ``*_arr`` variants pass Pauli strings as ``(N, n_qubits)`` ``uint8``
arrays of Pauli codes (``0=I, 1=X, 2=Z, 3=Y``) and skip string
construction entirely — at ~10^5 basis rows per evolution step, per-row
``str.join`` dominates wall time.

Each jump term can be either:

- a single Hermitian Pauli (`("ZZII", γ)`), routed to a fast diagonal path,
  or
- a complex Pauli sum (`([("XIII", 0.5+0j), ("YIII", 0+0.5j)], γ)`) to
  describe e.g. amplitude-damping (`σ⁻`) and excitation (`σ⁺`) operators.

For the general case the dissipator
``γ ( L† p L − ½ {L†L, p} )`` is evaluated directly; the L†L Pauli
expansion is precomputed once at construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np

from ._core import LindbladSpec as _LindbladSpec

_PAULI_CODE = {"I": 0, "X": 1, "Z": 2, "Y": 3}
# Lookup table mapping code -> ASCII byte for vectorised string output.
_CODE_TO_ASCII = np.array([ord("I"), ord("X"), ord("Z"), ord("Y")], dtype=np.uint8)

# A jump operator is either a Hermitian Pauli (single string) or a complex
# linear combination of Pauli strings.
PauliLincomb = Iterable[tuple[str, complex]]
JumpSpec = Union[tuple[str, float], tuple[PauliLincomb, float]]


def _string_to_codes(s: str, n_qubits: int) -> np.ndarray:
    """Encode a Pauli string ``"IXYZ..."`` as a length-``n_qubits`` uint8 array.

    Underscores in the input are ignored, matching the Rust parser
    (``parse_pauli_string`` in `ppvm-lindblad`) so users can write
    ``"X_Y_Z"`` for readability.
    """
    s_clean = s.replace("_", "")
    if len(s_clean) != n_qubits:
        raise ValueError(
            f"Pauli string {s!r} has length {len(s_clean)} (after stripping '_') "
            f"!= n_qubits {n_qubits}"
        )
    try:
        return np.array([_PAULI_CODE[c] for c in s_clean], dtype=np.uint8)
    except KeyError as exc:
        bad = exc.args[0]
        raise ValueError(
            f"Pauli string {s!r} contains invalid character {bad!r}; "
            f"expected one of 'I', 'X', 'Y', 'Z' (and '_' is allowed as a separator)"
        ) from None


def _codes_to_string(codes: np.ndarray) -> str:
    """Decode one length-``n_qubits`` row of Pauli codes back to a string."""
    return _CODE_TO_ASCII[codes].tobytes().decode("ascii")


def _basis_to_codes(basis: Sequence[str], n_qubits: int) -> np.ndarray:
    """Stack a sequence of Pauli strings into an ``(N, n_qubits)`` uint8 array."""
    arr = np.zeros((len(basis), n_qubits), dtype=np.uint8)
    for i, s in enumerate(basis):
        arr[i] = _string_to_codes(s, n_qubits)
    return arr


def _codes_to_basis(arr: np.ndarray) -> list[str]:
    """Inverse of :func:`_basis_to_codes`. One call into C per row."""
    bytes_per_row = _CODE_TO_ASCII[arr].tobytes()
    n = arr.shape[1]
    return [bytes_per_row[i * n : (i + 1) * n].decode("ascii") for i in range(arr.shape[0])]


def sigma_plus(site: int, n_qubits: int) -> list[tuple[str, complex]]:
    """``σ⁺_q = (X_q + i Y_q) / 2``. Use as a Lindblad jump for excitation."""
    if not 0 <= site < n_qubits:
        raise ValueError(f"site {site} out of range for n_qubits={n_qubits}")
    x_str = "I" * site + "X" + "I" * (n_qubits - site - 1)
    y_str = "I" * site + "Y" + "I" * (n_qubits - site - 1)
    return [(x_str, 0.5 + 0.0j), (y_str, 0.0 + 0.5j)]


def sigma_minus(site: int, n_qubits: int) -> list[tuple[str, complex]]:
    """``σ⁻_q = (X_q − i Y_q) / 2``. Use as a Lindblad jump for amplitude damping."""
    if not 0 <= site < n_qubits:
        raise ValueError(f"site {site} out of range for n_qubits={n_qubits}")
    x_str = "I" * site + "X" + "I" * (n_qubits - site - 1)
    y_str = "I" * site + "Y" + "I" * (n_qubits - site - 1)
    return [(x_str, 0.5 + 0.0j), (y_str, 0.0 - 0.5j)]


def _normalize_jump(jump_op: str | PauliLincomb) -> list[tuple[str, float, float]]:
    """Convert a user-supplied jump operator to ``[(pauli_str, re, im), ...]``.

    Accepts either a single Pauli string (treated as a Hermitian-Pauli jump
    with coefficient 1) or an iterable of ``(pauli_str, complex_coeff)``
    pairs.
    """
    if isinstance(jump_op, str):
        return [(jump_op, 1.0, 0.0)]
    out: list[tuple[str, float, float]] = []
    for term in jump_op:
        s, c = term
        cc = complex(c)
        out.append((str(s), float(cc.real), float(cc.imag)))
    if not out:
        raise ValueError("jump operator lincomb must contain at least one Pauli term")
    return out


def _csr_parts(m, what: str):
    """``(indptr, indices, data)`` of a CSR-like object (attributes or a 3-tuple), copied to int64 / int64 / ndarray."""
    if isinstance(m, tuple) and len(m) == 3:
        indptr, indices, data = m
    else:
        try:
            indptr, indices, data = m.indptr, m.indices, m.data
        except AttributeError:
            raise TypeError(f"{what}: expected a CSR-like object with indptr / indices / data, got {type(m).__name__}") from None
    return (
        np.ascontiguousarray(indptr, dtype=np.int64),
        np.ascontiguousarray(indices, dtype=np.int64),
        np.asarray(data),
    )


def _concat_csr(parts, sizes, what: str, dtype):
    """Concatenate per-block CSR parts (``None`` = empty row set) into the flat ``(indptr, indices, data, offsets)``."""
    indptr, indices, data, offsets = [], [], [], [0]
    for b, (m, s) in enumerate(zip(parts, sizes)):
        if m is None:
            ip = np.zeros(s + 1, dtype=np.int64)
            ix = np.zeros(0, dtype=np.int64)
            dt = np.zeros(0, dtype=dtype)
        else:
            ip, ix, dt = _csr_parts(m, f"{what} block {b}")
            if ip.shape != (s + 1,):
                raise ValueError(f"{what} block {b}: indptr has shape {ip.shape}, expected ({s + 1},)")
            dt = np.ascontiguousarray(dt, dtype=dtype)
        indptr.append(ip)
        indices.append(ix)
        data.append(dt)
        offsets.append(offsets[-1] + s + 1)
    return (
        np.concatenate(indptr) if indptr else np.zeros(0, dtype=np.int64),
        np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
        np.concatenate(data) if data else np.zeros(0, dtype=dtype),
        np.asarray(offsets, dtype=np.int64),
    )


def _normalize_factor(factor, m: int, *, what: str):
    """The flat arguments of ``_core.LindbladSpec.add_kossakowski_factor`` from a factor-form object (docstring of
    `Lindbladian`): ``(sizes, ranks, G_values, D_values, mask, neg_ranks, G_neg_values, private)``; validated shapes,
    ``D ≥ 0``, finite entries, ``Σ sizes == m``."""
    get = (lambda k: factor.get(k)) if isinstance(factor, dict) else (lambda k: getattr(factor, k, None))
    if get("G_values") is not None:
        # the flat kfactor/1 layout
        g_values = np.ascontiguousarray(get("G_values"), dtype=np.complex128).reshape(-1)
        d_values = np.ascontiguousarray(get("D_values"), dtype=np.float64).reshape(-1)
        d_off = np.asarray(get("D_offsets"), dtype=np.int64)
        ranks = [int(r) for r in np.asarray(get("G_ranks"), dtype=np.int64)]
        sizes = [int(x) for x in np.diff(d_off)]
        g_off = np.asarray(get("G_offsets"), dtype=np.int64)
        if len(ranks) != len(sizes) or g_off.shape != (len(sizes) + 1,):
            raise ValueError(f"{what}: G_ranks / G_offsets / D_offsets disagree on the block count")
        if any(int(g_off[b + 1] - g_off[b]) != sizes[b] * ranks[b] for b in range(len(sizes))):
            raise ValueError(f"{what}: G_offsets do not match size x rank per block")
        mask = None
        if get("mask_indptr") is not None:
            mask = tuple(
                np.ascontiguousarray(get(k), dtype=t)
                for k, t in (("mask_indptr", np.int64), ("mask_indices", np.int64), ("mask_data", np.float64), ("mask_offsets", np.int64))
            )
        neg_ranks = g_neg = None
        if get("G_neg_values") is not None:
            neg_ranks = [int(r) for r in np.asarray(get("G_neg_ranks"), dtype=np.int64)]
            g_neg = np.ascontiguousarray(get("G_neg_values"), dtype=np.complex128).reshape(-1)
        private = None
        if get("private_offdiag_indptr") is not None:
            private = tuple(
                np.ascontiguousarray(get(k), dtype=t)
                for k, t in (("private_offdiag_indptr", np.int64), ("private_offdiag_indices", np.int64),
                             ("private_offdiag_data", np.complex128), ("private_offdiag_offsets", np.int64))
            )
    else:
        G, D = get("G"), get("D")
        if G is None or D is None:
            raise TypeError(f"{what}: expected G and D per block (or the flat G_values / D_values layout)")
        G = [np.ascontiguousarray(g, dtype=np.complex128) for g in G]
        D = [np.ascontiguousarray(d, dtype=np.float64).reshape(-1) for d in D]
        if len(G) != len(D):
            raise ValueError(f"{what}: {len(G)} G blocks but {len(D)} D blocks")
        sizes, ranks = [], []
        for b, (g, d) in enumerate(zip(G, D)):
            if g.ndim != 2 or g.shape[0] != d.shape[0]:
                raise ValueError(f"{what} block {b}: G has shape {g.shape}, D has {d.shape}")
            sizes.append(int(g.shape[0]))
            ranks.append(int(g.shape[1]))
        g_values = np.concatenate([g.reshape(-1) for g in G]) if G else np.zeros(0, dtype=np.complex128)
        d_values = np.concatenate(D) if D else np.zeros(0, dtype=np.float64)
        masks = get("mask")
        mask = None if masks is None or all(x is None for x in masks) else _concat_csr(list(masks), sizes, f"{what} mask", np.float64)
        neg = get("G_neg")
        neg_ranks = g_neg = None
        if neg is not None and any(x is not None and np.asarray(x).shape[1] for x in neg):
            neg = [np.zeros((s, 0), dtype=np.complex128) if x is None else np.ascontiguousarray(x, dtype=np.complex128) for x, s in zip(neg, sizes)]
            for b, (x, s) in enumerate(zip(neg, sizes)):
                if x.ndim != 2 or x.shape[0] != s:
                    raise ValueError(f"{what} block {b}: G_neg has shape {x.shape}, expected ({s}, r_neg)")
            neg_ranks = [int(x.shape[1]) for x in neg]
            g_neg = np.concatenate([x.reshape(-1) for x in neg])
        po = get("private_offdiag")
        private = None if po is None or all(x is None for x in po) else _concat_csr(list(po), sizes, f"{what} private_offdiag", np.complex128)
    if sum(sizes) != m:
        raise ValueError(f"{what}: the blocks hold {sum(sizes)} operators but ops has length {m}")
    if d_values.size and (not np.all(np.isfinite(d_values)) or d_values.min() < 0.0):
        raise ValueError(f"{what}: D must be finite and non-negative (min {d_values.min() if d_values.size else 0})")
    if not np.all(np.isfinite(g_values.view(np.float64))):
        raise ValueError(f"{what}: G has a non-finite entry")
    return sizes, ranks, g_values, d_values, mask, neg_ranks, g_neg, private


class Lindbladian:
    """Pre-compiled adjoint Pauli-Lindbladian acting on Pauli strings.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    h_terms:
        Iterable of ``(pauli_string, coefficient)`` pairs for the
        Hermitian Hamiltonian ``H = Σ c_i P_i``. Each ``pauli_string`` is
        a length-``n_qubits`` ``str`` over ``"IXYZ"``.
    jump_terms:
        Iterable of ``(jump_op, rate)`` pairs. ``jump_op`` is either a
        Pauli string ``"XYZI..."`` (treated as a Hermitian-Pauli jump
        with coefficient 1, hitting the fast path) or an iterable of
        ``(pauli_string, complex_coeff)`` pairs for a general complex
        Pauli linear combination such as `sigma_plus` or
        `sigma_minus`. ``rate`` is the non-negative GKSL rate
        ``γ_k``.
    kossakowski:
        Optional ``(ops, K)`` pair adding a Kossakowski-form dissipator
        ``D*(O) = Σ_nm K_nm (A_n† O A_m − ½{A_n†A_m, O})``. ``ops`` is a
        length-``M`` sequence of Pauli lincombs (same format as a
        ``jump_op``, e.g. ``[sigma_minus(j, n) for j in range(n)]``);
        ``K`` is an ``(M, M)`` Hermitian positive-semidefinite array
        (e.g. the collective-decay pair matrix ``Γ_nm``). Mathematically
        identical to passing the eigenmode jumps
        ``L_ν = √γ_ν Σ_j V*_jν A_j`` of ``K = V diag(γ) V†``, but the
        per-string action cost scales with the number of nonzero
        ``K_nm`` entries instead of carrying an extra factor of ``M``.
        May coexist with ``jump_terms`` (both contribute). Raises
        ``ValueError`` if ``K`` is not Hermitian or has an eigenvalue
        below ``−tol·‖K‖`` (a non-PSD ``K`` is not a valid GKSL
        generator; no silent clipping).
    kossakowski_factor:
        Optional ``(ops, factor)`` pair adding a Kossakowski-form dissipator
        given per block in **factor form**, applied without a per-pair
        table (memory ``Σ_b s_b r_b`` instead of ``Σ_b s_b²``):
        ``K_b = (G_b G_b† − G_neg,b G_neg,b†) ∘ W_b + P_b + diag(D_b)``, blocks
        partitioning ``ops`` in order. ``factor`` is either an object / dict
        with ``G`` (per block, ``(s_b, r_b)`` complex), ``D`` (per block,
        ``(s_b,)`` float ≥ 0) and optional ``mask`` (per block, a CSR-like
        object with ``indptr`` / ``indices`` / ``data`` or ``None``;
        ``W_b`` real, diagonal present), ``G_neg`` (per block, ``(s_b,
        r_neg)``) and ``private_offdiag`` (per block, complex CSR or
        ``None``), or the flat ``nmr-csim-kfactor/1`` arrays ``G_values /
        G_offsets / G_ranks / D_values / D_offsets`` with the optional
        ``mask_*``, ``G_neg_*`` and ``private_offdiag_*`` members. PSD is
        the caller's responsibility (``check_psd`` has no eigensolve here:
        a Gram form plus a non-negative diagonal is PSD; masked / unclipped
        forms carry their own records). May coexist with ``kossakowski``
        (a dense block for a few extra operators) and ``jump_terms``.

    Examples
    --------
    Dephasing (Hermitian Pauli):

    >>> Lindbladian(2, [("XX", 1.0)], [("ZI", 0.3), ("IZ", 0.3)])

    Amplitude damping on site 0 (non-Hermitian):

    >>> jumps = [(sigma_minus(0, 2), 0.5)]
    >>> Lindbladian(2, [("XX", 1.0)], jumps)

    Collective decay from a pair matrix:

    >>> ops = [sigma_minus(j, 2) for j in range(2)]
    >>> gamma = [[1.0, 0.6], [0.6, 1.0]]
    >>> Lindbladian(2, [("XX", 1.0)], kossakowski=(ops, gamma))
    """

    #: relative tolerance for the Hermiticity and PSD checks on ``K``.
    _K_TOL = 1e-10

    def __init__(
        self,
        n_qubits: int,
        h_terms: Iterable[tuple[str, float]],
        jump_terms: Iterable[tuple[str | PauliLincomb, float]] = (),
        kossakowski: tuple[Sequence[str | PauliLincomb], object] | None = None,
        kossakowski_factor: tuple[Sequence[str | PauliLincomb], object] | None = None,
    ):
        self.n_qubits = int(n_qubits)
        h_strs: list[str] = []
        h_coeffs: list[float] = []
        for s, c in h_terms:
            h_strs.append(s)
            h_coeffs.append(float(c))
        j_lincombs: list[list[tuple[str, float, float]]] = []
        j_rates: list[float] = []
        for jump_op, rate in jump_terms:
            j_lincombs.append(_normalize_jump(jump_op))
            j_rates.append(float(rate))
        k_ops: list[list[tuple[str, float, float]]] = []
        k_mat: list[list[tuple[float, float]]] = []
        if kossakowski is not None:
            ops, K = kossakowski
            k_ops = [_normalize_jump(op) for op in ops]
            K = self._validate_k(K, len(k_ops), check_psd=True, what="kossakowski K")
            k_mat = [[(float(v.real), float(v.imag)) for v in row] for row in K]
        self._spec = _LindbladSpec(
            self.n_qubits, h_strs, h_coeffs, j_lincombs, j_rates, k_ops, k_mat
        )
        if kossakowski_factor is not None:
            f_ops, factor = kossakowski_factor
            f_ops = [_normalize_jump(op) for op in f_ops]
            flat = _normalize_factor(factor, len(f_ops), what="kossakowski_factor")
            self._spec.add_kossakowski_factor(f_ops, *flat)

    @classmethod
    def _validate_k(cls, K, m: int, *, check_psd: bool, what: str) -> np.ndarray:
        """Return ``K`` as a C-contiguous ``(m, m)`` complex128 array.

        Raises ``ValueError`` on a shape mismatch, when ``K`` is not
        Hermitian to ``_K_TOL·max(‖K‖, 1)`` and, with ``check_psd``, when
        its smallest eigenvalue is below ``−_K_TOL·max(‖K‖, 1)``.
        """
        K = np.ascontiguousarray(K, dtype=np.complex128)
        if K.shape != (m, m):
            raise ValueError(f"{what} has shape {K.shape} but ops has length {m}")
        if m == 0:
            return K
        scale = max(float(np.abs(K).max()), 1.0)
        if float(np.abs(K - K.conj().T).max()) > cls._K_TOL * scale:
            raise ValueError(f"{what} is not Hermitian")
        if check_psd:
            evals = np.linalg.eigvalsh(K)
            if float(evals.min()) < -cls._K_TOL * scale:
                raise ValueError(
                    f"{what} is not positive semidefinite "
                    f"(min eigenvalue {evals.min():.3e}); a non-PSD pair matrix "
                    f"is not a valid GKSL generator"
                )
        return K

    @property
    def num_h_terms(self) -> int:
        return self._spec.num_h_terms

    @property
    def num_jump_terms(self) -> int:
        return self._spec.num_jump_terms

    @property
    def h_coeffs(self) -> np.ndarray:
        """Current Hamiltonian coefficients, shape ``(num_h_terms,)`` float64.

        Returns a copy; mutating it does not change the Lindbladian.
        """
        return self._spec.h_coeffs

    def set_h_coeffs(self, coeffs) -> None:
        """Overwrite the Hamiltonian coefficients in place.

        ``coeffs`` is converted to a 1-D float64 array that must have length
        ``num_h_terms`` (same order as the ``h_terms`` given at construction);
        otherwise ``ValueError``. The Pauli words, jump operators and the
        Kossakowski block are left untouched (no PSD re-check). Cost is
        ``O(num_h_terms)``; afterwards every method gives results
        bit-identical to a freshly built ``Lindbladian`` with ``coeffs``.
        """
        arr = np.ascontiguousarray(coeffs, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self._spec.num_h_terms:
            raise ValueError(
                f"set_h_coeffs: expected shape ({self._spec.num_h_terms},), got {arr.shape}"
            )
        self._spec.set_h_coeffs(arr)

    @property
    def num_jump_ops(self) -> int:
        """Number of ``jump_terms`` given at construction (Kossakowski pairs excluded).

        ``num_jump_terms`` counts the compiled dissipator entries, which
        include one entry per retained Kossakowski pair.
        """
        return self._spec.num_jump_ops

    @property
    def jump_rates(self) -> np.ndarray:
        """Current GKSL rates ``γ_k`` of the ``jump_terms``, shape ``(num_jump_ops,)``.

        Construction order; a copy, so mutating it does not change the
        Lindbladian. Empty when built without ``jump_terms``.
        """
        return self._spec.jump_rates

    def set_jump_rates(self, rates) -> None:
        """Overwrite the jump rates in place.

        ``rates`` is converted to a 1-D float64 array that must have length
        ``num_jump_ops`` (same order as the ``jump_terms`` given at
        construction) with non-negative entries; otherwise ``ValueError``,
        also when the object was built without ``jump_terms``. The jump
        operators, the Hamiltonian and the Kossakowski block are left
        untouched. Cost is ``O(num_jump_ops)``; afterwards every method
        gives results bit-identical to a freshly built ``Lindbladian`` with
        these rates (a zero rate keeps its term, as the constructor does).
        """
        m = self._spec.num_jump_ops
        if m == 0:
            raise ValueError("set_jump_rates: this Lindbladian was built without jump_terms")
        arr = np.ascontiguousarray(rates, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != m:
            raise ValueError(f"set_jump_rates: expected shape ({m},), got {arr.shape}")
        if arr.size and float(arr.min()) < 0.0:
            k = int(np.argmin(arr))
            raise ValueError(f"set_jump_rates: rates must be non-negative; got rates[{k}] = {arr[k]}")
        self._spec.set_jump_rates(arr)

    @property
    def num_kossakowski_ops(self) -> int:
        """Number ``M`` of operators in the ``kossakowski`` block (0 without one)."""
        return self._spec.num_kossakowski_ops

    @property
    def kossakowski(self) -> np.ndarray | None:
        """Current Kossakowski matrix ``K``, shape ``(M, M)`` complex128.

        Same operator basis and order as the ``ops`` given at construction.
        A copy; ``None`` when built without a ``kossakowski`` block.
        """
        if self._spec.num_kossakowski_ops == 0:
            return None
        return self._spec.kossakowski

    def set_kossakowski(self, K, check_psd: bool = True) -> None:
        """Replace the Kossakowski matrix in place, keeping the operator basis.

        ``K`` is converted to an ``(M, M)`` complex128 array over the same
        ``ops`` (same order) as given at construction, ``M =
        num_kossakowski_ops``. ``ValueError`` on a shape mismatch, when the
        object was built without a ``kossakowski`` block, when ``K`` is not
        Hermitian to ``_K_TOL·max(‖K‖, 1)`` and, with ``check_psd=True``
        (default), when it has an eigenvalue below ``−_K_TOL·max(‖K‖, 1)``;
        ``check_psd=False`` skips the eigensolve (the caller vouches for
        PSD). The Hamiltonian and the ``jump_terms`` are left untouched.

        Rust folds ``K`` into the precompiled per-pair tables and drops
        pairs with ``|K_nm| ≤ 1e-14·max|K|``, so the setter rebuilds those
        tables from the stored operators — the Kossakowski part of
        construction without any string parsing or Python-side work —
        and afterwards every method gives results bit-identical to a
        freshly built ``Lindbladian`` with ``K``.
        """
        m = self._spec.num_kossakowski_ops
        if m == 0:
            raise ValueError(
                "set_kossakowski: this Lindbladian was built without a kossakowski block"
            )
        K = self._validate_k(K, m, check_psd=check_psd, what="set_kossakowski: K")
        self._spec.set_kossakowski(K)

    # ── factor-form Kossakowski dissipator ──

    @property
    def num_kossakowski_factor_ops(self) -> int:
        """Number of operators of the ``kossakowski_factor`` dissipator (0 without one)."""
        return self._spec.num_kossakowski_factor_ops

    @property
    def num_kossakowski_factor_blocks(self) -> int:
        """Number of blocks of the ``kossakowski_factor`` dissipator (0 without one)."""
        return self._spec.num_kossakowski_factor_blocks

    @property
    def kossakowski_factor_memory_bytes(self) -> tuple[int, int]:
        """``(factor_bytes, ops_bytes)``: the bytes of ``G`` / ``G_neg`` / ``D`` / masks / private parts, and of the parsed operators."""
        return tuple(self._spec.kossakowski_factor_memory_bytes)

    @property
    def kossakowski_factor_floor(self) -> float:
        """The absolute compile floor of the factor dissipator (pairs with ``|K_nm|`` at or below it are skipped)."""
        return float(self._spec.kossakowski_factor_floor)

    def kossakowski_factor_dense_block(self, b: int) -> np.ndarray:
        """Block ``b`` of the factor dissipator re-formed as a dense ``(s_b, s_b)`` complex128 array (inspection, tests)."""
        return np.asarray(self._spec.kossakowski_factor_dense_block(int(b)))

    def set_kossakowski_factor(self, factor) -> None:
        """Replace ``G`` / ``G_neg`` / ``D`` / masks / private parts of the factor dissipator in place, keeping its operator
        table (same number of blocks and block sizes; ``factor`` as in the constructor). ``ValueError`` when the object was
        built without a ``kossakowski_factor`` or the shapes disagree. Later results are bit-identical to a fresh build."""
        m = self._spec.num_kossakowski_factor_ops
        if m == 0:
            raise ValueError("set_kossakowski_factor: this Lindbladian was built without a kossakowski_factor block")
        flat = _normalize_factor(factor, m, what="set_kossakowski_factor")
        self._spec.set_kossakowski_factor(*flat)

    # ── Pure-ndarray hot path ──

    def action_arr(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply ``L*`` to a single Pauli string given as uint8 codes.

        Returns ``(out_basis, out_coeffs)``: a ``(M, n_qubits)`` uint8
        array and a length-``M`` float64 array.
        """
        return self._spec.action(np.ascontiguousarray(p, dtype=np.uint8))

    def leakage_arr(
        self,
        basis_arr: np.ndarray,
        coeffs: np.ndarray,
        protected_arr: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Off-basis component of ``L*( Σ_j coeffs[j] basis[j] )``.

        ``basis_arr``: ``(N, n_qubits)`` uint8. ``coeffs``: length-N float64.
        ``protected_arr``: optional ``(K, n_qubits)`` uint8 of Pauli strings
        that must NEVER appear in the leakage output.

        Returns ``(out_basis, out_coeffs)`` packed the same way as
        :meth:`action_arr`.
        """
        n = self.n_qubits
        if protected_arr is None:
            protected_arr = np.zeros((0, n), dtype=np.uint8)
        return self._spec.leakage(
            np.ascontiguousarray(basis_arr, dtype=np.uint8),
            np.ascontiguousarray(coeffs, dtype=np.float64),
            np.ascontiguousarray(protected_arr, dtype=np.uint8),
        )

    def pc_step_arr(
        self,
        basis_arr: np.ndarray,
        coeffs: np.ndarray,
        dt: float,
        max_basis: int,
        drop_tol: float = 1e-12,
        protected_arr: np.ndarray | None = None,
        num_threads: int | None = None,
        admit_basis: int | None = None,
        tau_add: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One predictor-corrector adaptive step.

        All work — leakage expansion, matrix-exponential step, second-hop
        re-expansion, corrector — runs in Rust; SciPy is not required.
        The matrix-exponential action is computed matrix-free via the
        external ``quspin-expm`` crate.

        Truncation. ``max_basis`` is a hard rank cap on the live basis:
        enrichment adds at most ``max_basis - len(basis)`` of the largest
        leakage strings, and the post-step basis is trimmed to the
        top-``max_basis`` entries by ``|coeff|`` (``protected`` words always
        kept). Pass a large value (e.g. ``10_000_000``) for the near-exact,
        uncapped case. ``drop_tol`` additionally prunes basis entries whose
        absolute coefficient is below the threshold after the corrector
        (unless the word is ``protected``).

        ``num_threads``, when set, pins this call to a freshly-built rayon
        pool of that size — useful for benchmarking parallel scaling.

        ``tau_add``, when set, filters leakage admission by an absolute
        coefficient-rate threshold (a dt- and drop_tol-independent
        parameterization). Off by default — with cap-based truncation it
        is at most a modest wall optimization.

        ``admit_basis``, when set (must be >= ``max_basis``), bounds the
        enriched working set during the step instead of ``max_basis``: the
        step may hold up to ``admit_basis`` strings transiently, and the
        final truncation keeps the top-``max_basis`` by evolved ``|coeff|``
        over the whole union (retained + newly admitted) — rank
        displacement, so no ``drop_tol`` is needed to sustain membership
        turnover. With the default ``None``, admission is bounded by
        ``max_basis`` and turnover requires ``drop_tol > 0``.

        Returns ``(new_basis_arr, new_coeffs)``; the basis may have grown
        (or shrunk, if ``max_basis`` / ``drop_tol`` pruned entries).
        """
        n = self.n_qubits
        if protected_arr is None:
            protected_arr = np.zeros((0, n), dtype=np.uint8)
        return self._spec.pc_step(
            np.ascontiguousarray(basis_arr, dtype=np.uint8),
            np.ascontiguousarray(coeffs, dtype=np.float64),
            float(dt),
            int(max_basis),
            float(drop_tol),
            np.ascontiguousarray(protected_arr, dtype=np.uint8),
            None if num_threads is None else int(num_threads),
            None if admit_basis is None else int(admit_basis),
            None if tau_add is None else float(tau_add),
        )

    def pc_step_orbit_rep(
        self,
        basis_arr: np.ndarray,
        coeffs: np.ndarray,
        dt: float,
        max_basis: int,
        group,
        momentum: np.ndarray,
        drop_tol: float = 1e-12,
        protected_arr: np.ndarray | None = None,
        canonicalize_first: bool = False,
        admit_basis: int | None = None,
        tau_add: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-step orbit-representative pc evolution.

        State lives entirely in orbit-rep form throughout: ``basis_arr``
        contains only canonical translation-orbit representatives,
        ``coeffs`` are complex, and the action is phase-aware. The basis
        is ~``|group|×`` smaller than the equivalent full-basis complex
        evolution, and the reduction persists across every step.

        Truncation. ``max_basis`` is a hard rank cap on the live orbit-rep
        basis: enrichment adds at most ``max_basis - len(basis)`` of the
        largest leakage reps, and the post-step basis is trimmed to the
        top-``max_basis`` reps by ``|c|`` (``protected`` reps always kept).
        Pass a large value (e.g. ``10_000_000``) for the near-exact,
        uncapped case. ``drop_tol`` additionally prunes reps whose absolute
        coefficient is below the threshold after the corrector.

        ``admit_basis``, when set (>= ``max_basis``), bounds the enriched
        working set instead of ``max_basis``: the step may hold up to
        ``admit_basis`` reps transiently and the final truncation keeps the
        top-``max_basis`` by evolved ``|c|`` over the whole union — the
        displacement scheme, matching the real-space ``pc_step_arr``.

        ``basis_arr`` is assumed to contain canonical reps only. Pass
        ``canonicalize_first=True`` to rewrite each row to its canonical
        rep on entry (coefficients unchanged).
        """
        n = self.n_qubits
        if protected_arr is None:
            protected_arr = np.zeros((0, n), dtype=np.uint8)
        return self._spec.pc_step_orbit_rep(
            np.ascontiguousarray(basis_arr, dtype=np.uint8),
            np.ascontiguousarray(coeffs, dtype=np.complex128),
            float(dt),
            int(max_basis),
            group,
            np.ascontiguousarray(momentum, dtype=np.int32),
            float(drop_tol),
            np.ascontiguousarray(protected_arr, dtype=np.uint8),
            bool(canonicalize_first),
            None if admit_basis is None else int(admit_basis),
            None if tau_add is None else float(tau_add),
        )

    def pc_step(
        self,
        basis: Sequence[str],
        coeffs: np.ndarray,
        dt: float,
        max_basis: int,
        drop_tol: float = 1e-12,
        protected: Sequence[str] | None = None,
        num_threads: int | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """String-keyed variant of :meth:`pc_step_arr`."""
        n = self.n_qubits
        basis_arr = _basis_to_codes(basis, n)
        protected_arr = (
            _basis_to_codes(list(protected), n) if protected else np.zeros((0, n), dtype=np.uint8)
        )
        new_basis_arr, new_coeffs = self.pc_step_arr(
            basis_arr,
            coeffs,
            dt,
            max_basis,
            drop_tol,
            protected_arr,
            num_threads,
        )
        return _codes_to_basis(new_basis_arr), new_coeffs

    def generator_arr(self, basis_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generator matrix as COO triples ``(rows, cols, vals)``.

        Basis given as uint8 codes. To get a SciPy sparse matrix:

        >>> import scipy.sparse as sp
        >>> rows, cols, vals = L_op.generator_arr(basis_arr)
        >>> M = sp.coo_matrix(
        ...     (vals, (rows, cols)), shape=(len(basis_arr), len(basis_arr))
        ... ).tocsc()
        """
        return self._spec.generator(np.ascontiguousarray(basis_arr, dtype=np.uint8))

    # ── String-keyed convenience API (slower; for tests / display) ──

    def action(self, p: str) -> dict[str, float]:
        """Apply ``L*`` to a single Pauli string ``p`` (string-keyed dict)."""
        codes = _string_to_codes(p, self.n_qubits)
        out_basis, out_coeffs = self._spec.action(codes)
        keys = _codes_to_basis(out_basis)
        return {k: float(v) for k, v in zip(keys, out_coeffs) if v != 0.0}

    def leakage(
        self,
        basis: Sequence[str],
        coeffs: np.ndarray,
        protected: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Off-basis leakage as a ``dict[str, float]`` (slower API)."""
        n = self.n_qubits
        basis_arr = _basis_to_codes(basis, n)
        protected_arr = (
            _basis_to_codes(list(protected), n) if protected else np.zeros((0, n), dtype=np.uint8)
        )
        out_basis, out_coeffs = self._spec.leakage(
            basis_arr,
            np.ascontiguousarray(coeffs, dtype=np.float64),
            protected_arr,
        )
        keys = _codes_to_basis(out_basis)
        return {k: float(v) for k, v in zip(keys, out_coeffs) if v != 0.0}

    def generator(self, basis: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generator matrix as COO triples ``(rows, cols, vals)``,
        basis given as strings. See :meth:`generator_arr` for the conversion
        to a SciPy sparse matrix."""
        n = self.n_qubits
        basis_arr = _basis_to_codes(basis, n)
        return self.generator_arr(basis_arr)
