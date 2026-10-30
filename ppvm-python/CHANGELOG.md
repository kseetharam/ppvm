# Changelog of this fork's engine builds (the `+…` version suffix)

The wheel version is `0.1.0+<suffix>`; each suffix component names one fork line merged into the build, in order:
`hcoeffs` (coefficient setters for time-dependent Hamiltonians), `koss` (Kossakowski dissipators), `chem` (block Lindbladian for
chemical exchange), `ww` (word widths beyond 128 sites), `blocks` / `templates` (block-sparse Kossakowski input compiled through shared
pair templates), `locality` (the locality-aware predictor-corrector step), `factor` (factor-form Kossakowski dissipator), `weight`
and `weight.hist` (below). A wheel installed from a file reports no commit through pip; its record carries the fork commit (the pin)
and the wheel's sha256.

## `.weight` / `.weight.hist` — branch `nc-weight` (commits 8ba69022 → 97e4c189 → 4dad2359)

Two knobs in the predictor-corrector step, both default off, in the Rust core (`PcStepConfig`) and the Python bindings
(`Lindbladian.pc_step_arr` / `BlockLindbladian.pc_step_arr`), with a per-step account:

- `max_weight` (`Option<u32>` / `max_weight=None`): a leakage candidate whose Pauli weight (non-identity sites; physical sites only
  on the block spec) exceeds it is refused at admission — before the room cap, so the admission slots go to lower-weight strings;
  strings already in the basis (the observable, protected words) are untouched. The refused contributions are counted.
- `depolarize_eps` (`f64` / `depolarize_eps=0.0`): the exact Lindblad depolarizing channel of rate ε per site, jumps
  √(ε/4)·{X_i, Y_i, Z_i} on every site. Diagonal in the Pauli basis, D*(P) = −ε·w(P)·P, it enters the in-basis generator of both
  exponentials of the step (a string of weight w decays as exp(−ε·w·t) for H = 0; the exact generator term, not a split factor,
  for H ≠ 0); it never leaks and leaves the identity alone. Verified byte-identical to the same channel written as explicit jumps.
- `PcStepStats` (returned by `pc_step` / `pc_step_timed` / the block `pc_step`; Python `return_stats=True` → a `stats` dict, or
  `_core.LindbladSpec.pc_step_stats`): `max_weight`, `depolarize_eps`, `admitted1` / `admitted2`, `weight_dropped_terms` /
  `weight_dropped_l1` (refused contributions and Σ of their magnitudes — an upper bound on the L1 norm of the refused leakage; × dt the
  first-order norm the cap removed in the step), `trim_dropped` / `trim_dropped_l2sq` (what the end-of-step prune + rank cap removed),
  `basis_max_weight`, `basis_len`.
- `.hist` (4dad2359) adds the per-weight-class arrays, numpy float64 of length `n_bins = n_sites + 1` (index w = Pauli weight w,
  index 0 = the identity): `cap_refused_l1_by_w` (sums to `weight_dropped_l1`), `trim_l2sq_by_w` (sums to `trim_dropped_l2sq`),
  `pre_l2sq_by_w` / `post_l2sq_by_w` (Σ c² by weight of the corrector's input — the state at the start of the step on the enlarged
  basis — and of its output before the trim), `retained_l2sq_by_w` (after the trim; `post − trim = retained` per class) and
  `channel_damp_l2sq_by_w` (the channel's norm removal per class over the step, ε·w·dt·(pre_w + post_w): the trapezoid rule on the two
  snapshots, exact to third order in ε·w·dt; for H = 0 the exact value is pre_w·(1 − e^{−2εw·dt}); zeros when the channel is off).
- `ppvm.STEP_FEATURES` for guarded imports: `('locality', 'max_weight', 'depolarize_eps', 'step_stats', 'step_hist')`; the
  authority on a wheel is the `pc_step_arr` signature. The orbit-rep step (`pc_step_orbit_rep`) has neither knob.
- With both knobs off the step is the `.factor` step bit for bit (the same leakage, generator and trim arithmetic; the exponential is the
  same `mf_cache_full` + `apply_cache` sequence as before) — checked on thirteen stored propagation records against the previous wheel.
- Tests: `cargo test -p ppvm-lindblad` (`depolarize_*`, `max_weight_cap_refuses_and_counts`, `trim_account_counts_and_sums`,
  `pauli_weight_upto_masks_trailing_positions`, `per_class_arrays_h0_and_cap`) and `test/lindblad/test_weight_depolarize.py`.
- Installing the wheel: with its dependencies (`kirin`, `bloqade-circuit`; `import ppvm` fails without them); running the test suite
  needs `pytest-benchmark` (the package's `addopts`).
