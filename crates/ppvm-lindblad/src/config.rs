// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! Configuration objects for the predictor-corrector stepper.

/// Truncation and execution policy for a single predictor-corrector step
/// ([`crate::LindbladSpec::pc_step`], [`crate::LindbladSpec::pc_step_timed`],
/// [`crate::orbit_rep::pc_step_orbit_rep`]).
///
/// These are the per-run *tuning knobs*, kept separate from the per-call data
/// (`basis`, `coeffs`, `dt`, `protected`, and — on the orbit path — the
/// translation group and momentum).
///
/// `max_basis` is the primary accuracy/cost dial; `admit_basis` selects the
/// displacement scheme; `drop_tol` is the churn valve of the admission-bound
/// scheme; `tau_add` is a wall optimization at most.
#[derive(Debug, Clone, Copy)]
pub struct PcStepConfig {
    /// Hard rank cap on the retained basis: after the corrector, only the
    /// top-`max_basis` strings by `|coeff|` are kept (protected words always
    /// survive). The primary convergence dial — verify by re-running at 2×.
    pub max_basis: usize,
    /// Working-set (admission) bound. When `Some(a)` with `a > max_basis`,
    /// enrichment may grow the live basis to `a` and the final cap performs a
    /// genuine top-`max_basis`-of-union rank displacement (the analog of
    /// two-site TDVP truncation at `χ_max`); `drop_tol` is then not needed
    /// for membership turnover. `None` bounds admission by `max_basis`
    /// itself — the valve scheme, which requires `drop_tol > 0` to keep the
    /// basis adapting once it fills.
    pub admit_basis: Option<usize>,
    /// Magnitude prune applied after the corrector: basis entries whose
    /// `|coeff|` is below `drop_tol` are discarded (protected words are
    /// always kept). `<= 0.0` disables pruning — valid only with
    /// `admit_basis` set, otherwise the basis freezes once it fills the cap.
    pub drop_tol: f64,
    /// Optional absolute rate threshold on leakage admission: a candidate is
    /// admitted only if its inflow rate exceeds `tau_add`. This is the
    /// natural (dt- and drop_tol-independent) parameterization — the
    /// admission accuracy cliff sits at a fixed `tau_add`. `None` = no
    /// filter, the recommended default with cap-based truncation.
    pub tau_add: Option<f64>,
    /// When `Some(n)`, run the entire step inside a freshly built rayon
    /// thread pool of `n` threads (useful for benchmarking parallel
    /// scaling). When `None`, the global rayon pool is used.
    pub num_threads: Option<usize>,
    /// Run the locality-aware step ([`crate::locality`]): the generator is
    /// built from the dissipator's transfer index and a per-word raw / diag
    /// cache instead of a full action pass, the leakage merge is sharded
    /// across threads, the discarded second leakage pass is skipped and the
    /// corrector reuses the predictor's generator when nothing was admitted.
    /// Every action coefficient, generator entry and leakage value is the
    /// legacy one bit for bit; the admitted strings enter the basis sorted by
    /// word rather than in hash-map order, which changes the association
    /// order of the matvec reduction (round-off level). `false` (default) is
    /// the legacy step.
    pub locality: bool,
    /// Pauli-weight cap on admission (`Some(l)`): a leakage candidate whose
    /// weight (number of non-identity sites; on the block spec the physical
    /// sites only, not the species digits) exceeds `l` is never admitted. The
    /// filter runs before the room cap, so the admission slots go to strings
    /// of weight `<= l`, and the dropped contributions are counted in
    /// [`crate::PcStepStats`] (`weight_dropped_terms` / `weight_dropped_l1`).
    /// Strings already in the basis (the observable, protected words) are
    /// not touched. `None` (default): no cap, the step unchanged bit for bit.
    pub max_weight: Option<u32>,
    /// Per-site depolarizing channel of rate `eps` (when `> 0.0`): the exact
    /// Lindblad dissipator with jumps `sqrt(eps/4)·{X_i, Y_i, Z_i}` on every
    /// site, which is diagonal in the Pauli basis, `D*(P) = −eps·w(P)·P`. It
    /// enters the in-basis generator of both exponentials of the step (so a
    /// string of weight `w` decays as `exp(−eps·w·t)` exactly for `H = 0`,
    /// and the channel is the exact generator term, not a split factor, for
    /// `H ≠ 0`); being diagonal it never leaks. `0.0` (default): off, the
    /// generator unchanged bit for bit.
    pub depolarize_eps: f64,
    /// The end-of-step truncation ("tail") mode. `None` (default): the
    /// deterministic trim — the `drop_tol` prune followed by the top-`max_basis`
    /// rank cap — bit for bit the step of every earlier build. `Some(tail)`:
    /// stochastic rounding ([`TailConfig`]): the strings below the step's
    /// threshold τ are kept with probability |c|/τ at coefficient sign(c)·τ or
    /// removed, so the expectation of every coefficient is preserved and the
    /// propagated operator is unbiased step by step; strings with |c| ≥ τ and
    /// protected words are never randomised; the `drop_tol` prune and the rank
    /// cap are NOT applied (the rounding is the whole trim; the working set stays
    /// bounded by `admit_basis`). The admission room cap of the leakage passes
    /// stays deterministic in both modes and is counted
    /// ([`crate::PcStepStats::room_refused_terms`]).
    pub tail: Option<TailConfig>,
}

/// How the stochastic tail's threshold τ is set per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailRule {
    /// τ = `threshold` (an absolute coefficient).
    Abs,
    /// τ = `threshold` × max_i |c_i| over the whole basis after the corrector
    /// (protected words included).
    Rel,
    /// τ solves Σ_i min(1, |c_i|/τ) = `max_basis` − n_protected over the
    /// non-protected strings — the expected number of retained strings equals
    /// the budget; τ = 0 (nothing rounded) when the basis already fits.
    Budget,
}

/// The stochastic tail: threshold rule, threshold and the step's seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailConfig {
    pub rule: TailRule,
    /// The absolute threshold (`Abs`), the fraction of the largest
    /// coefficient (`Rel`); ignored by `Budget`.
    pub threshold: f64,
    /// Seed of the step's generator (SplitMix64; one uniform per rounded
    /// string, drawn in basis order). The caller derives it per
    /// (replica, step) so a trajectory is reproducible and replicas are
    /// independent.
    pub seed: u64,
}

impl Default for PcStepConfig {
    /// Uncapped, unfiltered, no pruning: the near-exact reference
    /// configuration. Production runs should set `max_basis` (and usually
    /// `admit_basis ≈ 2-3×` it).
    fn default() -> Self {
        Self {
            max_basis: usize::MAX,
            admit_basis: None,
            drop_tol: 0.0,
            tau_add: None,
            num_threads: None,
            locality: false,
            max_weight: None,
            depolarize_eps: 0.0,
            tail: None,
        }
    }
}
