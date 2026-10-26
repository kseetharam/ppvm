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


def _k_form(K) -> str:
    """Classify the ``K`` of ``kossakowski=(ops, K)``: ``"dense"``, ``"blocks"`` or ``"coo"``.

    Blocks: an object with a ``blocks`` attribute (a sequence of
    ``(op_indices, K_block)`` pairs) or a ``dict`` with a ``"blocks"`` key.
    COO: a ``scipy.sparse`` matrix (anything with ``tocoo``), an object
    with ``to_coo()`` returning ``(rows, cols, vals)``, or a ``dict`` with
    ``"rows"``, ``"cols"`` and ``"vals"`` keys. Everything else (an array
    or nested sequence) is the dense ``(M, M)`` matrix. A bare tuple of
    three arrays is deliberately *not* read as COO: a ``3 × 3`` dense
    matrix given as rows looks the same.
    """
    if isinstance(K, dict):
        if "blocks" in K:
            return "blocks"
        if {"rows", "cols", "vals"} <= set(K):
            return "coo"
        raise ValueError(
            "kossakowski K given as a dict needs the key 'blocks' or the keys 'rows', 'cols', 'vals'"
        )
    if hasattr(K, "blocks") and not isinstance(K, np.ndarray):
        return "blocks"
    if (hasattr(K, "tocoo") or hasattr(K, "to_coo")) and not isinstance(K, np.ndarray):
        return "coo"
    return "dense"


def _k_blocks(K) -> list:
    """The ``[(op_indices, K_block), ...]`` list behind a blocks-form ``K``."""
    if isinstance(K, dict):
        return list(K["blocks"])
    if hasattr(K, "blocks") and not isinstance(K, (list, tuple, np.ndarray)):
        return list(K.blocks)
    return list(K)


def _k_coo(K) -> tuple:
    """The ``(rows, cols, vals)`` behind a COO-form ``K``."""
    if isinstance(K, dict):
        return K["rows"], K["cols"], K["vals"]
    if hasattr(K, "tocoo"):
        c = K.tocoo()
        return c.row, c.col, c.data
    return tuple(K.to_coo())


def _blocks_to_coo(blocks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated blocks → COO arrays (int64, int64, complex128), nonzero entries only."""
    rows, cols, vals = [], [], []
    for idx, K_b in blocks:
        r, c = np.meshgrid(idx, idx, indexing="ij")
        keep = K_b != 0
        rows.append(r[keep])
        cols.append(c[keep])
        vals.append(K_b[keep])
    if not rows:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros(0, dtype=np.complex128)
    return (
        np.ascontiguousarray(np.concatenate(rows), dtype=np.int64),
        np.ascontiguousarray(np.concatenate(cols), dtype=np.int64),
        np.ascontiguousarray(np.concatenate(vals), dtype=np.complex128),
    )


def _coo_components(rows: np.ndarray, cols: np.ndarray, m: int) -> np.ndarray:
    """Component label per operator of the undirected graph with edges ``(rows[i], cols[i])``.

    ``scipy.sparse.csgraph.connected_components`` when scipy is importable,
    else a union-find pass over the edges (``O(nnz)`` Python steps).
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        g = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(m, m))
        return connected_components(g, directed=False)[1]
    except ImportError:
        parent = np.arange(m)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for r, c in zip(rows.tolist(), cols.tolist()):
            a, b = find(r), find(c)
            if a != b:
                parent[max(a, b)] = min(a, b)
        return np.array([find(i) for i in range(m)])


def _check_coo_psd(rows, cols, vals, m: int, tol: float, what: str) -> None:
    """Raise ``ValueError`` when a connected block of the COO matrix is not PSD to ``tol``.

    Each component is assembled densely (duplicates summed, one-sided
    entries Hermitian-completed as the Rust side does) and eigensolved;
    the ``(m, m)`` matrix is never formed.
    """
    labels = _coo_components(rows, cols, m)
    scale = max(float(np.abs(vals).max()), 1.0)
    for lab in np.unique(labels[rows]):
        members = np.flatnonzero(labels == lab)
        pos = np.full(m, -1, dtype=np.int64)
        pos[members] = np.arange(members.size)
        sel = labels[rows] == lab
        K_b = np.zeros((members.size, members.size), dtype=np.complex128)
        np.add.at(K_b, (pos[rows[sel]], pos[cols[sel]]), vals[sel])
        given = np.zeros_like(K_b, dtype=bool)
        given[pos[rows[sel]], pos[cols[sel]]] = True
        one_sided = given & ~given.T
        K_b = np.where(one_sided.T, K_b.conj().T, K_b)
        evals = np.linalg.eigvalsh(0.5 * (K_b + K_b.conj().T))
        if float(evals.min()) < -tol * scale:
            raise ValueError(
                f"{what} is not positive semidefinite (block on operators "
                f"{members[:8].tolist()}{'...' if members.size > 8 else ''}: min eigenvalue "
                f"{evals.min():.3e}); a non-PSD pair matrix is not a valid GKSL generator"
            )


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
        k_coo = None
        if kossakowski is not None:
            ops, K = kossakowski
            k_ops = [_normalize_jump(op) for op in ops]
            form = _k_form(K)
            if form == "dense":
                K = self._validate_k(K, len(k_ops), check_psd=True, what="kossakowski K")
                k_mat = [[(float(v.real), float(v.imag)) for v in row] for row in K]
            elif form == "blocks":
                blocks = self._validate_blocks(
                    _k_blocks(K), len(k_ops), check_psd=True, what="kossakowski K"
                )
                k_coo = _blocks_to_coo(blocks)
            else:
                rows, cols, vals = _k_coo(K)
                k_coo = self._validate_coo(
                    rows, cols, vals, len(k_ops), check_psd=True, what="kossakowski K"
                )
        self._spec = _LindbladSpec(
            self.n_qubits,
            h_strs,
            h_coeffs,
            j_lincombs,
            j_rates,
            k_ops,
            k_mat,
            kossakowski_coo=k_coo,
        )

    @classmethod
    def _validate_blocks(cls, blocks, m: int, *, check_psd: bool, what: str):
        """Return ``[(indices int64[b], K_b complex128[b, b]), ...]`` after checking each block.

        ``ValueError`` on an index outside ``range(m)`` or repeated within a
        block, a shape mismatch, a block that is not Hermitian to
        ``_K_TOL·max(‖K_b‖, 1)`` and, with ``check_psd``, one with an
        eigenvalue below ``−_K_TOL·max(‖K_b‖, 1)``. Each block is checked
        on its own: blocks sharing operators add up, and a sum of PSD
        blocks is PSD.
        """
        out = []
        for b, (indices, K_b) in enumerate(blocks):
            idx = np.ascontiguousarray(indices, dtype=np.int64).reshape(-1)
            if idx.size and (idx.min() < 0 or idx.max() >= m):
                raise ValueError(
                    f"{what}: block {b} has an operator index outside range({m}): {idx.tolist()}"
                )
            if np.unique(idx).size != idx.size:
                raise ValueError(f"{what}: block {b} lists an operator more than once")
            K_b = cls._validate_k(K_b, idx.size, check_psd=check_psd, what=f"{what}: block {b}")
            out.append((idx, K_b))
        return out

    @classmethod
    def _validate_coo(cls, rows, cols, vals, m: int, *, check_psd: bool, what: str):
        """Return ``(rows, cols, vals)`` as int64, int64, complex128 1-D arrays after the checks.

        Equal lengths and indices in ``range(m)`` are checked here; the Rust
        side sums duplicates, Hermitian-completes one-sided entries and
        checks Hermiticity (``ValueError``). With ``check_psd`` the
        connected components of the sparsity pattern (the blocks of ``K``
        under any permutation) are each checked for an eigenvalue below
        ``−_K_TOL·max(‖K‖, 1)`` — exact, since ``K`` is PSD iff every
        block is, and never forms the ``(m, m)`` matrix.
        """
        rows = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1)
        cols = np.ascontiguousarray(cols, dtype=np.int64).reshape(-1)
        vals = np.ascontiguousarray(vals, dtype=np.complex128).reshape(-1)
        if not (rows.size == cols.size == vals.size):
            raise ValueError(
                f"{what}: rows, cols and vals must have the same length; "
                f"got {rows.size}, {cols.size} and {vals.size}"
            )
        if rows.size and (
            rows.min() < 0 or cols.min() < 0 or rows.max() >= m or cols.max() >= m
        ):
            raise ValueError(f"{what}: operator indices must lie in range({m})")
        if check_psd and rows.size:
            _check_coo_psd(rows, cols, vals, m, cls._K_TOL, what)
        return rows, cols, vals

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

    def set_kossakowski_blocks(self, blocks, check_psd: bool = True) -> None:
        """Replace the Kossakowski matrix in place from blocks, never forming ``K``.

        ``blocks`` is a sequence of ``(op_indices, K_block)`` pairs (or an
        object with a ``blocks`` attribute holding one): ``op_indices`` an
        integer array of distinct operator indices into the ``ops`` given
        at construction, ``K_block`` a Hermitian PSD ``(len, len)`` array
        placed there, ``K[np.ix_(op_indices, op_indices)] += K_block``.
        Blocks normally partition the operators (a block-diagonal ``K``);
        blocks sharing operators add up. Operators in no block get zero
        rows; an empty sequence is the zero dissipator. ``ValueError`` on
        an index out of range or repeated within a block, a shape
        mismatch, a non-Hermitian block (``_K_TOL`` relative) and, with
        ``check_psd=True`` (default), a block with an eigenvalue below
        ``−_K_TOL·max(‖K_block‖, 1)`` — each block is eigensolved on its
        own, so the check costs ``Σ_b len_b³``, not ``M³``. Nothing changes
        on an error.

        Rust assembles the entries, keeps the pair templates whose
        ``(n, m)`` stay nonzero and templates the new pairs, so a drive
        rescaling a fixed block pattern costs one complex multiply per
        stored coefficient; the result is bit-identical to
        ``set_kossakowski`` with the assembled dense matrix (which is never
        formed: memory ``O(Σ_b len_b²)`` plus the templates).
        """
        m = self._spec.num_kossakowski_ops
        if m == 0:
            raise ValueError(
                "set_kossakowski_blocks: this Lindbladian was built without a kossakowski block"
            )
        checked = self._validate_blocks(
            _k_blocks(blocks), m, check_psd=check_psd, what="set_kossakowski_blocks"
        )
        self._spec.set_kossakowski_blocks(checked)

    def set_kossakowski_sparse(
        self, rows, cols, vals, n_ops: int | None = None, check_psd: bool = True
    ) -> None:
        """Replace the Kossakowski matrix in place from COO triples, never forming ``K``.

        ``rows``, ``cols`` (integer, ``nnz`` each) and ``vals`` (complex,
        ``nnz``) describe ``K[rows[i], cols[i]] = vals[i]`` over the
        operator index of the ``ops`` given at construction, in any order;
        duplicate ``(row, col)`` pairs are summed (the COO convention) and
        exact zeros ignored. ``n_ops``, when given, must equal
        ``num_kossakowski_ops`` (a consistency check for callers that
        carry it). An off-diagonal entry given in one triangle only is
        Hermitian-completed (its mirror is the conjugate), so a single
        triangle is a valid input; one given in both triangles must match
        its mirror's conjugate and a diagonal entry must be real, to
        ``_K_TOL·max(‖K‖, 1)`` — otherwise ``ValueError``, as for an index
        outside ``range(num_kossakowski_ops)`` or mismatched lengths. With
        ``check_psd=True`` (default) the connected components of the
        pattern are eigensolved one by one (exact: ``K`` is PSD iff every
        block is); ``check_psd=False`` skips it. Nothing changes on an
        error. Bit-identical to ``set_kossakowski`` with the dense matrix
        these triples describe; see `set_kossakowski_blocks` for the cost.
        """
        m = self._spec.num_kossakowski_ops
        if m == 0:
            raise ValueError(
                "set_kossakowski_sparse: this Lindbladian was built without a kossakowski block"
            )
        if n_ops is not None and int(n_ops) != m:
            raise ValueError(
                f"set_kossakowski_sparse: n_ops = {n_ops} but this Lindbladian has "
                f"num_kossakowski_ops = {m}"
            )
        rows, cols, vals = self._validate_coo(
            rows, cols, vals, m, check_psd=check_psd, what="set_kossakowski_sparse"
        )
        self._spec.set_kossakowski_sparse(rows, cols, vals)

    @property
    def kossakowski_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Current ``K`` as COO arrays ``(rows, cols, vals)``: every nonzero entry, both triangles.

        uint64, uint64, complex128, sorted by ``(row, col)``; copies.
        ``None`` when built without a ``kossakowski`` block. The sparse twin
        of `kossakowski`, which materialises the ``(M, M)`` matrix.
        """
        if self._spec.num_kossakowski_ops == 0:
            return None
        return self._spec.kossakowski_coo

    @property
    def num_kossakowski_nonzeros(self) -> int:
        """Number of stored nonzero entries of ``K`` (both triangles)."""
        return self._spec.num_kossakowski_nonzeros

    @property
    def num_pair_templates(self) -> int:
        """Number of distinct pair templates among the compiled Kossakowski pairs.

        A template is the ``K``-independent compilation of one operator-pattern
        relationship: the Pauli structure of ``(A_n, A_m)`` up to a qubit
        relabelling (both operators' terms, in order, with their exact
        coefficients, over the pair's qubits numbered by first appearance).
        Pairs with the same structure on other qubits — the same tensor
        components of a dipolar / CSA dissipator on other spins — share one
        template, so this count is bounded by the number of distinct patterns,
        not by `num_compiled_pairs`, and the compiled memory scales with it.
        """
        return self._spec.num_pair_templates

    @property
    def num_compiled_pairs(self) -> int:
        """Number of compiled Kossakowski pairs ``n ≤ m`` (``|K_nm| > 1e-14·max|K|``), one record each."""
        return self._spec.num_compiled_pairs

    @property
    def kossakowski_memory_bytes(self) -> dict[str, int]:
        """Heap bytes of the Kossakowski compilation: ``{'templates', 'pairs', 'entries'}``.

        ``templates``: the shared pair templates (Pauli tables, keys and index);
        ``pairs``: the per-pair records (template index, qubit map, ``K_nm``,
        support slots); ``entries``: the stored nonzero ``K`` entries (24 B
        each). Vector capacities, i.e. what was asked of the allocator; the
        operator table is not counted.
        """
        t, p, e = self._spec.kossakowski_memory_bytes
        return {"templates": int(t), "pairs": int(p), "entries": int(e)}

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
