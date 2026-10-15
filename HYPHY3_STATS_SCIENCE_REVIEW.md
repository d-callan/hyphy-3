# HyPhy 3 Statistical & Scientific Code Review

**Scope.** This review audits the HyPhy 3 port (`include/hyphy/**`) for three classes of issues:

1. **Parity against HyPhy 2.5** — verified by reading the corresponding 2.5 batch files (`res/TemplateBatchFiles/SelectionAnalyses/*.bf`, `modules/*.ibf`).
2. **Methodological weaknesses or gaps** — independent of whether 2.5 does the same thing.
3. **Porting errors** — places where the intent appears correct but the implementation deviates.

Coverage: core likelihood engine (`core/likelihood.hpp`), rate matrices (`core/rate_matrix.hpp`), genetic code (`core/genetic_code.hpp`), alignment (`core/alignment.hpp`), optimizers (`opt/optimizer.hpp`, `opt/nelder_mead.hpp`, `opt/squarem.hpp`), analyses (`analyses/fel.hpp`, `meme.hpp`, `busted.hpp`, `absrel.hpp`), the test suite (`tests/*.cpp`), and README parity claims.

**Review-date note:** file/line citations refer to the checked-out working tree as of Oct 2026.

---

## 1. Executive Summary

The port reproduces the *model structure* of all four analyses faithfully: every LRT p-value formula I checked is exactly correct against the 2.5 source (MEME boundary mixture, aBSREL `0.5·(0.4χ²₁+0.6χ²₂)`, BUSTED `0.5·(1−CChi2(LRT,2))`, FEL χ²₁, MEME-vs-FEL χ²₂). The inside-outside machinery, EM responsibilities, SQUAREM acceleration, Holm correction, and the MG94→nucleotide branch-length conversion are all implemented correctly.

The substantive problems are, in rough severity order:

- **aBSREL silently cannot fit more than 2 rate classes per branch** — `optimize_branch_mixture` returns −1e20 for M≥3, so the default `max_rate_classes=3` is unreachable and the AICc step-up always stops at 2. HyPhy 2.5 fits up to 5. This is the single largest methodological gap: real datasets routinely produce 3–5-class branches, and the port will systematically under-model them (and emit LRT/p-values computed under a reduced model).
- **No likelihood scaling anywhere** — all pruning and inside-outside recursion runs in raw linear space with a `-1e20` zero-likelihood penalty. Products of transition-matrix entries down long trees will underflow to denormal/zero, producing false `-1e20` penalties that corrupt both point estimates and gradients. This will not show on the small benchmark alignments tested so far but is a genuine robustness gap versus 2.5 (which operates with scaled conditional likelihoods).
- **BUSTED branch-length "refinement" runs a single L-BFGS iteration** (`refine_branch_lengths(..., max_iters=1)` at `busted.hpp:1070`). One L-BFGS step per model is almost certainly the dominant source of the small lnL gaps the README reports versus 2.5.
- **BUSTED has no foreground/background partition** — `is_test_branch` is hardwired all-true, so only the all-branches-as-test variant of BUSTED exists. The plumbing (test-branch masking in `compute_expected_weights`) is present but unreachable.
- **BUSTED "optimized null" evidence ratios are fake** — the JSON emits the constrained-null ER vector under *both* the `constrained` and `optimized null` keys; the third model fit required to produce the optimized-null ER is never performed.
- **Partial ambiguity is over-marginalized** — `char_to_nuc` maps every IUPAC ambiguity code to -1, so a codon like `ATN` (4 compatible codons) is treated as *fully* unknown (all 61 states) rather than the correct 4-state leaf vector. Sites/sequences with partial ambiguity get inflated likelihoods relative to 2.5.
- **AICc sample size is inconsistent** — aBSREL correctly uses `N = taxa × codons` (matching 2.5's `sample_size = sites × sequences`), but FEL/MEME/BUSTED/GTR use `N = sites` (or nucleotide sites), understating N by a factor of `num_taxa` and inflating the small-sample penalty.
- **Numerous small output/reporting bugs** — MEME's `total_branch_length` column is always 0; aBSREL's GTR parameter count is hardcoded 39; aBSREL `do_srv`/`syn_rate_classes` settings are dead; FEL null α is capped at 50 while the alternative can reach 1000.

None of the verified statistical formulas (LRTs, mixture distributions, EBF/ER ratios, Holm correction) were found to be wrong. The problems are concentrated in **feature completeness** (rate-class ceilings, branch sets, third-model fits), **numerical robustness** (no scaling), and **optimization depth** (1-iteration refinement, fixed-cycle GTR).

---

## 2. Verified-Parity Items (checked against 2.5 source)

These were confirmed correct against the corresponding `.bf`/`.ibf` files — included so the problems below are interpreted in context.

| Item | HyPhy 3 | HyPhy 2.5 | Status |
|---|---|---|---|
| FEL LRT p-value | `erfc(sqrt(lrt/2))` = χ²₁ tail | `1 - CChi2(lrt,1)` | ✅ identical |
| FEL "Total branch length" | `alpha_null × baseline_tree_length` | α·T | ✅ |
| MEME mixture p-value | `(2/3)·(0.45·χ²₁ + 0.55·χ²₂)` | `2/3 − 2/3·(0.45·CChi2(x,1) + 0.55·CChi2(x,2))` | ✅ identical |
| MEME null model | β⁺ := α constraint | `beta2 := alpha` | ✅ |
| MEME branch EBF | `L_+/L_1` conditional-likelihood ratio | posterior-odds/prior-odds via `compute_branch_EBF` | ✅ algebraically identical for 2-class |
| MEME-vs-FEL p | `exp(-Δll)` = χ²₂ tail | `1 - CChi2(2·Δ,2)` | ✅ |
| BUSTED LRT p-value | `0.5·exp(-lrt/2)` | `0.5·(1-CChi2(lrt,2))` in `busted.ComputeLRT` | ✅ identical |
| BUSTED site Evidence Ratio | `exp(alt_ll − null_ll)` | `ha["Exp(ha-h0)"]` in `selection.EvidenceRatios` | ✅ formula identical |
| aBSREL branch-test p-value | `0.5·(0.4·χ²₁ + 0.6·χ²₂)` | `(1-0.4·CChi2(lrt,1)-0.6·CChi2(lrt,2))·0.5` | ✅ identical |
| aBSREL step-up AICc sample size | `taxa × codons` | `sites × sequences` (`shared-load-file.bf:33`) | ✅ |
| aBSREL Holm-Bonferroni | step-down, running max | Holm | ✅ |
| aBSREL site EBF | posterior odds / prior odds per branch-mixture class | `absrel.branch.GetMixtureParameters` + EBF | ✅ |
| MG94 nucleotide→codon conversion | `bl × 3/scale_factor` | same convention | ✅ |
| SQUAREM | Varadhan–Roland S1 scheme, stabilized | same | ✅ |
| Inside-outside responsibilities | `V·(P_k·D)` per (site,branch) class | equivalent EM | ✅ |
| MG94 multi-hit Q | `rate_modifier × δ/ψ × Πr × Ππ_target` | BUSTED-PH model | ✅ consistent |
| GTR branch-length init | 0.05 + scale stage | distance-based init | ⚠️ different but convergent |

---

## 3. Defects & Errors

### 3.1 aBSREL rate classes hard-capped at 2 (HIGH)

`include/hyphy/analyses/absrel.hpp:669–672`:

```cpp
// Fallback for M >= 3
LocalBranchFit res;
res.log_l = -1e20;
return res;
```

`optimize_branch_mixture` only implements M=1 and M=2. The step-up loop (`absrel.hpp:750–763`) iterates `M = 2 … max_rate_classes` and compares candidate AICc — with `log_l = -1e20` the M=3 candidate always loses, so the loop `break`s after M=2 fails or succeeds once. Consequences:

- `settings.max_rate_classes` (default 3) is **unreachable**; `test_absrel.cpp` sets `max_rate_classes = 3` and the test passes only because M=3 silently fails rather than being exercised.
- HyPhy 2.5 `absrel.max_rate_classes = 5` and real datasets (including β-globin, the test's own benchmark) do yield 3–5-class branches. The port will systematically under-model complex branches → attenuated LRTs and conservative p-values on exactly the branches that matter most.
- The null fits in Phase 5 inherit the same ceiling (`optimize_branch_mixture(bid, rate_classes, …, constrain_null=true)`), so the null/alt pairing is internally consistent, but both are wrong-size relative to 2.5.

### 3.2 aBSREL hardcoded parameter count (LOW, cosmetic)

`absrel.hpp:684` — `result.gtr_fit = {gtr_log_l, 39, gtr_aicc}`. The `39` is literal. For the β-globin test (17 taxa → 33 branches) the true count is 5 rates + 33 branch lengths = 38; for any other tree it's something else. JSON `"estimated parameters"` for the GTR fit is wrong for every dataset.

### 3.3 aBSREL baseline parameter-count magic number (LOW)

`absrel.hpp:448` — `baseline_params = 5 + 9 + 2 * num_branches`. The `+9` is presumably the 9 free F3x4 positional frequencies (3 positions × (4−1)), which is defensible, but it is undocumented, and the frequencies are never refit during the adaptive phase while the +9 persists in `current_params` throughout step-up — internally consistent, but worth a comment and cross-check against 2.5's accounting.

### 3.4 BUSTED branch refinement = 1 L-BFGS iteration (HIGH)

`busted.hpp:1070` — `bl = refine_branch_lengths(bl, omegas, weights, syn_rates, syn_weights, delta, psi, 1)`. The function's default is `max_iters=10` and it is structurally a proper L-BFGS (two-loop recursion, Armijo line search, correct adjoint gradients at `busted.hpp:512–614`), but the call site caps it at **one iteration** per model fit. Every unconstrained and constrained fit therefore leaves most of the available branch-length likelihood on the table. This is the most plausible single explanation for the residual ~1–3-unit lnL gaps versus 2.5 noted in the README parity table for larger datasets.

### 3.5 BUSTED has no foreground/background partition (HIGH — capability gap)

`busted.hpp:109` — `is_test_branch.assign(num_nodes, true)`; `num_test_branches = num_nodes - 1`. There is no API, CLI flag, or constructor argument to mark a subset of branches as background. 2.5 BUSTED tests a user-selected branch set against a separate background ω distribution; the port can only run the all-branches-as-test variant (BUSTED-E semantics). The masking logic in `compute_expected_weights` (`busted.hpp:348`) is already written for the general case — only the wiring is missing — so this is a gap rather than a defect, but it changes what scientific questions the tool can answer.

### 3.6 BUSTED "optimized null" Evidence Ratios duplicate the constrained ones (MEDIUM)

`busted.hpp:1385–1388`:

```cpp
j["Evidence Ratios"] = {
    {"constrained", {res.evidence_ratios}},
    {"optimized null", {res.evidence_ratios}}
};
```

2.5 computes ER against **two different nulls**: `site_logl.constrained` (ω₃=1 fixed) and `site_logl.optimized_null` (a third fit where the top class is allowed to estimate freely under ω≤1, `BUSTED.bf:1384–1385`). The port never fits the optimized null — it emits the constrained ER vector twice under different keys. Datamonkey/hyphy-vision consumers that distinguish the two series will see silently identical data.

Related: `run()` only computes `res.evidence_ratios` once (`busted.hpp:1231–1249`), and clamps ER to ≥1.0 for sites where the null fits better (`res.evidence_ratios[s_idx] = 1.0`), whereas 2.5 reports the raw `exp(ha−h0)` which can be <1. Minor, but it loses information.

Also note `busted.hpp:1243–1244` — `post_odds = exp(delta) * prior_odds; bf = post_odds / prior_odds` — is dead algebra (the prior cancels); harmless but confusing.

### 3.7 MEME `total_branch_length` always reported 0 (LOW)

`meme.hpp:175` and `meme.hpp:349` — `res.total_branch_length = 0.0` unconditionally; the field is emitted into the per-site table (`meme.hpp:485`). FEL computes the same column correctly (`fel.hpp:268` → `alpha_null × baseline_tree_length`). In 2.5 MEME this column carries the site's synonymous tree length; downstream consumers reading it will get a column of zeros.

### 3.8 Ambiguity codes over-marginalized (MEDIUM)

`types.hpp:34–42` — `char_to_nuc` returns -1 for *any* non-ACGT character (R, Y, N, W, K, B, D, H, V, gaps). `triplet_to_sense` then returns -1 for the whole codon (`alignment.hpp` → `triplet_to_sense = -1`), and the leaf vector becomes all-ones (`likelihood.hpp`, `busted.hpp` leaf paths). Consequences:

- `ATN` should be a 4-state leaf {ATI, ATM, ATC, ATT}; the port makes it a 61-state leaf — inflated site likelihood and diluted signal at every partially-ambiguous column.
- 2.5 resolves all unambiguous IUPAC expansions before evaluating the pattern.
- For datasets with even modest ambiguity content (common in viral alignments), both global fits and per-site inferences shift.

Stop codons and `---` correctly produce fully-missing leaves — that part is right; the issue is only *partial* ambiguity.

### 3.9 AICc sample-size inconsistency (MEDIUM)

2.5 convention (`shared-load-file.bf:33`): `sample_size = sites × sequences`. The port is inconsistent:

- aBSREL: `compute_aicc` uses `num_taxa × num_codons` (`absrel.hpp:333`) — **correct parity**.
- FEL/MEME GTR & MG94 AICc: `N = nuc_aln.num_nucleotides` / `aln.num_codons` — undercounted by ×num_taxa.
- BUSTED `fit.aicc` (`busted.hpp:1139–1145`): `N_codons = aln.num_codons` — same undercount.

Effect: the `2K(K+1)/(N−K−1)` correction is overstated; reported AICc values won't match 2.5, and BUSTED's `auto_select_k` (which compares ΔAICc) operates on differently-scaled corrections than 2.5's would. For typical small viral alignments the absolute AICc difference is modest, but it is a systematic deviation.

### 3.10 FEL null-model bound asymmetry (LOW)

`fel.hpp` null path — Brent search `ax=1e-4, bx=clamp(best_null_a,0.01,45), cx=50` caps the null rate at 50 while the alternative's Nelder-Mead box runs to α=1000, β=2000. If a site's true null optimum exceeds 50 (highly saturated/duplicated columns), the null is artificially capped → inflated LRT → false-positive selection call. Rare in practice but asymmetric by construction.

### 3.11 FEL/MEME per-pattern evaluator rebuilds the eigendecomposition (perf, not correctness)

`fel.hpp` `eval_pattern_lnl` / `compute_pattern_likelihood` calls `MG94Matrix::update` (a 61×61 eigendecomposition) **per likelihood evaluation** inside the Nelder-Mead/Brent loops. 2.5 evaluates these via a shared eigensystem with rate-multiplier rescaling (α,β enter linearly in Q so a single eigendecomp is not possible — but 2.5 reuses the base system differently per site). Not a bug; a ~10–100× per-site cost driver. Worth noting only because site-wise profiling is the dominant runtime term.

### 3.12 `-1e20` zero-likelihood penalty is unsafe (MEDIUM — robustness)

Every pruning/inside-outside path works in raw probability space:

- `likelihood.hpp`, `busted.hpp:257`, `absrel.hpp:302`, `absrel.hpp:417`, etc.: `pat_L > 0 ? log(pat_L) : -1e20`.
- `compute_inside_outside` products `P × D` accumulate multiplicatively down the tree with **no per-node scaling** (2.5's `LF_SMOOTHING_SCALER` / per-site scaling machinery is absent).

On trees with many deep nodes, conditional-likelihood vectors legitimately underflow below denormal range → `pat_L == 0` → the site contributes `weight × -1e20` to the total. That is not a graceful `-inf`; it is a *finite, huge* penalty that (a) corrupts the reported lnL, and (b) inside optimizers acts as a cliff that distorts simplex/BFGS steps. On the small benchmarks (17–30 taxa, ~150–300 codons) it likely never triggers; on 100+ taxa or deep divergence it will. This is the port's most important *numerical* gap.

### 3.13 BUSTED SRV degenerates for M ≠ 3 (LOW)

`busted.hpp:1026–1042` — continuous synonymous-rate optimization runs only `if (srv && M == 3)`. For other `num_syn_rate_classes` values, `syn_rates` stays at its all-1.0 initialization (`busted.hpp:836`) and only the weights move via EM — a degenerate SRV that silently does nothing. Also note the E[α]=1 constraint (`a3 = (1 − q1·a1 − q2·a2)/q3`) is correct but only enforced inside the M==3 branch.

### 3.14 BUSTED negative-LRT handling (LOW)

`busted.hpp:1220–1223` — when `unconstrained < constrained`, the port copies the constrained lnL into the unconstrained and reports `lrt=0, p=0.5`. 2.5 detects the same condition and **refits the alternative from the null's parameter estimates** (`BUSTED.bf:1387–1389`, "Negative test LRT (convergence problem)"). The port's shortcut loses the chance to recover a better alternative optimum; the reported `site_log_likelihoods` and branch lengths also become inconsistent with the overwritten total.

### 3.15 aBSREL `do_srv`/`syn_rate_classes` are dead settings (LOW)

`ABSRELSettings` declares `do_srv`/`syn_rate_classes` (`absrel.hpp:65–66`) which are never referenced in `run()`. Either implement BUSTED-SR-style synonymous variation for aBSREL or remove the fields — as-is they promise a feature that silently doesn't exist.

### 3.16 Test-suite issues (MEDIUM)

- **Hardcoded absolute paths**: `tests/test_busted.cpp:18–19,56,89,118` use `/Users/sergei/Development/hyphy*/...` with no fallback — the tests simply cannot run outside the author's machine. `test_absrel.cpp:18–23` has a fallback list but still contains the absolute paths. Same pattern presumably in `test_meme.cpp`, `test_fel*.cpp`, `test_busted_mh.cpp` (verify all).
- **`test_absrel.cpp` sets `max_rate_classes = 3`** — passing only because of §3.1 (the M≥3 path is unreachable); the test would also pass with `max_rate_classes = 2`, so it gives false confidence that 3-class support works.
- **Weak assertions**: several tests assert only structural facts (p ∈ [0,1], LRT ≥ 0, sizes) rather than numeric parity targets. `test_absrel.cpp` comments expect Node15/Node18/SHEEP positives but asserts only `positive_branches.size() > 0`. The `test_busted.cpp` K=2 assertion (`optimal_k == 2`, `p ≤ 0.05`) is the strongest parity check present.
- No test exercises the partial-ambiguity path (§3.8), the long-branch underflow path (§3.12), or `is_test_branch = false` semantics (§3.5, unsurprising since unreachable).

---

## 4. Methodological Gaps (regardless of 2.5 parity)

- **GTR fitter is fixed-iteration coordinate descent** (`optimizer.hpp`, `GTRFitter::fit`): 3 cycles of per-rate Brent + per-branch Brent. No convergence test, no distance-based initialization (branches start at 0.05). It apparently reaches ~0.01 lnL of 2.5 on the benchmark set, but on harder landscapes (deep trees, many taxa) a fixed 3-cycle schedule will underfit; there is no mechanism to detect that.
- **MG94Fitter L-BFGS uses finite-difference gradients** on an expensive closed-form objective; exact adjoint gradients already exist (`compute_branch_length_gradients`, `compute_busted_branch_gradients`) and could feed the same solver. Both a speed and an accuracy issue at the same time.
- **MEME `p_value = 2/3` shortcut** (`meme.hpp`) when the alternative doesn't find β⁺>α: mathematically consistent with the mixture at LRT=0, but the port *skips the null fit and the branch-EBF computation entirely* for those sites — 2.5 still reports EBFs. Sites that are "MEME-negative" therefore have no branch-level posterior information at all.
- **`p_plus > 1e-6` gating** for the MEME LRT means near-boundary fits report `p=2/3` without testing; combined with the above, the effective power characterization of the port's MEME differs from 2.5's at the edges.
- **MEME `resample` (parametric bootstrap) option absent** — 2.5 supports `--resample N` per-site null simulation; useful for small alignments. Flagged as a documented gap, not a defect.
- **BUSTED `auto_select_k` uses an undocumented ΔAICc ≥ 2.0 accept threshold** (`busted.hpp:1167`) on *null-model* fits — a reasonable heuristic but arbitrary; standard practice would be plain argmin-AICc, and the choice is not described in the README.
- **`error-sink`, `mss` settings hardcoded off** in the BUSTED JSON — unimplemented 2.5 options.
- **No partition support** anywhere — all analyzers assume a single partition/tree; 2.5 analyses accept multi-partition alignments (multiple `filter_names`).
- **No CI/profile-likelihood output** for FEL (`"ci": 0` hardcoded), and FEL JSON claims `"srv": 1`-style settings without implementing synonymous rate variation per site (FEL-S).
- **Branch-set selection in aBSREL limited to All/Internal/Leaves** — 2.5 also honors `{label}` Newick tags for named branch sets; the tree parser strips `{}` tags (treated as comments), so label-based testing is impossible without parser changes.

---

## 5. Numerical & Engine Concerns

- **Symmetrization trick is valid but fragile**: `rate_matrix.hpp:220` symmetrizes `S = diag(√π)Qdiag(1/√π)` — correct *iff* Q is reversible w.r.t. π. For F3x4/CF3x4 product-form π and the single/double/triple-hit Q construction, reversibility holds exactly (verified: `π_i·Q_ij = π_j·Q_ji` because the θ's are symmetric and π factors over positions). If a future non-product frequency estimator or non-symmetric exchangeability is added, the `0.5(S+Sᵀ)` hack will silently produce wrong eigensystems. `eigenvalues.cwiseMin(0)` is a safe guard.
- **`sqrt_pi` floor of `1e-12`** (`rate_matrix.hpp:165`): for codons with estimated frequency 0 (possible under CF3x4 when a nucleotide is absent at a position), `inv_sqrt_pi = 1e6` and Q entries involving that state can be nonzero → symmetrized matrix gets spurious ±(Q·1e-6) cross terms. Rare but a latent instability; 2.5 handles zero-frequency codons explicitly.
- **`find_taxon_index` per-evaluation leaf remap**: every `evaluate_log_l`/`compute_inside_outside` call rebuilds `leaf_to_taxon` by name lookup — O(leaves × taxa) per eval inside hot loops. Perf, not correctness.
- **Pattern `pll = -1e20` vs `-inf`** — see §3.12; also note the same sentinel is used inside EM responsibilities (`pat_L > 0` gates), which is fine, but combined with underflow it produces *wrong* responsibilities rather than NaNs — silent corruption is worse than loud failure.
- **OpenMP reductions are correct** (per-thread `node_L`, `#pragma omp critical` accumulators, `reduction(+:total_log_l)`), but `compute_inside_outside` results allocate full inside-outside tables per pattern per syn class — memory pressure on large SRV analyses.

---

## 6. Parity Gaps vs HyPhy 2.5 (features not ported)

- FEL: parametric `resample` bootstrap; `--srv` synonymous-rate-variation site model (FEL-S); `--multihit`; per-site CIs; multi-partition input; the JSON hardcodes `srv/ci/resample/multihit` settings rather than reflecting capability.
- MEME: `--resample`; `rate-classes` >2 (2.5 supports general K rate classes in the mixture; port is hardwired 2); EBF output on non-significant sites.
- BUSTED: branch-set (FG/BG) testing; the optimized-null third fit; `error-sink`; `mss`; `srv` for M≠3 classes; negative-LRT refit.
- aBSREL: M≥3 rate classes (§3.1 — implemented for M≤2 only vs 2.5's 5); `{label}`-based branch subsets; SRV options (dead settings); per-branch `Baseline MG94xREV omega ratio` is present; "Sites @ EBF≥100" present.
- General: SLAC, FUBAR, RELAX, BGM, FADE, GARD, aBSREL-MH, Contrast-FEL, FitMultiModel — none ported (FUBAR planned separately).
- Output schema drift: aBSREL `analysis.version` "3.0.0" vs BUSTED "4.7-next"; aBSREL `"tested"` map omits background entries that 2.5 emits; BUSTED `"Evidence Ratios"` duplicated keys (§3.6).

---

## 7. Test & README Assessment

- The README parity table (lnL within ~0.01–2 units on ADH/CD2/β-globin-class datasets) is plausible given the code: the *structure* is right; residual gaps are consistent with (a) 1-iteration BUSTED refinement (§3.4), (b) fixed-cycle GTR (§4.1), (c) finite-difference MG94 gradients (§4.2), and (d) AICc N-convention (§3.9) for reported AICc values.
- Claims of "exact" formula parity for LRT/p-values are substantiated (§2).
- Claims about aBSREL rate-class adaptivity should be tempered until §3.1 is fixed — the advertised `max_rate_classes` cannot exceed 2 in practice.
- Tests assert structure more than numbers; two of the strongest parity assertions (BUSTED `optimal_k==2`, p≤0.05) are genuine, but there is no regression coverage on ambiguity handling, deep-tree scaling, MEME null-report fields, or the unreachable aBSREL M=3 path.

---

## 8. Recommended Actions (priority order)

1. **Implement aBSREL M≥3** (`optimize_branch_mixture` general-M branch): generalize the 2-class parameterization to ordered ωs + softmax weights; this is the only HIGH-severity *correctness-of-capability* item.
2. **Add likelihood scaling** to `compute_inside_outside` and the pruning paths (per-node scaling factors or log-space accumulation): protects every analysis on realistic trees and removes the `-1e20` cliff.
3. **Raise BUSTED `refine_branch_lengths` to ~10 iterations** (its own default) and re-run the parity benchmarks — likely closes most of the reported lnL gap.
4. **Wire foreground/background selection** through `BUSTEDAnalyzer`/`is_test_branch` (plumbing already exists).
5. **Resolve IUPAC partial ambiguity** at `triplet_to_sense`/leaf-vector construction (enumerate compatible codons); add a test.
6. **Fit the BUSTED optimized-null model** (ω₃ free under ω≤1) or stop emitting the second ER key.
7. **Unify AICc sample size** to `sites × taxa` everywhere (aBSREL is already right).
8. **Fix small things**: MEME `total_branch_length`, aBSREL `gtr_fit` params=39, aBSREL dead SRV settings, FEL null α bound, MEME skipped-EBF path, test path fallbacks, `test_absrel` max_rate_classes, negative-LRT refit in BUSTED, SRV M≠3.
9. **Strengthen tests**: numeric parity assertions against stored 2.5 outputs (not just ranges), a deep-tree scaling test, an ambiguity test, an aBSREL 3-class test (once implemented).

---

## Appendix: Files reviewed

- `include/hyphy/core/types.hpp`, `tree.hpp`, `alignment.hpp`, `genetic_code.hpp`, `rate_matrix.hpp`, `likelihood.hpp`
- `include/hyphy/opt/nelder_mead.hpp`, `squarem.hpp`, `optimizer.hpp`
- `include/hyphy/analyses/fel.hpp`, `meme.hpp`, `busted.hpp`, `absrel.hpp`
- `src/apps/hyphy3.cpp`
- `tests/test_absrel.cpp`, `test_busted.cpp` (spot-checked others by grep)
- 2.5 references: `SelectionAnalyses/FEL.bf`, `MEME.bf`, `BUSTED.bf`, `aBSREL.bf`, `modules/selection_lib.ibf`, `modules/shared-load-file.bf`, `libv3/convenience/math.bf`
