# FUBAR Implementation Plan for HyPhy 3

**Goal.** Port FUBAR (Fast Unbiased Bayesian AppRoximation; Murrell et al. 2013,
*Mol Biol Evol* 30:1196–1205) from HyPhy 2.5 to the HyPhy 3 C++20 core, following
the existing `hyphy::analyses` pattern used by `FELAnalyzer`, `MEMEAnalyzer`,
`BUSTEDAnalyzer`, and `ABSRELAnalyzer`. Deliverables: a header-only analyzer, a
CLI subcommand, Python bindings, Datamonkey-compatible JSON output, and a parity
test suite against HyPhy 2.5.

**Reference implementation.** The authoritative HyPhy 2.5 sources live in the
local clone at `/home/danielle/Documents/veg/hyphy`:

- `res/TemplateBatchFiles/SelectionAnalyses/FUBAR.bf` — orchestration,
  grid definition, reporting, JSON emission.
- `res/TemplateBatchFiles/SelectionAnalyses/modules/grid_compute.ibf` —
  grid evaluation, `ConvertToConditionals`, the three posterior samplers
  (`ExecuteMCMC` / `RunCollapsedGibbs` / `RunVariationalBayes`), and
  `ComputeRandNeff`.

Read both before writing any code. Every formula below was extracted verbatim
from these files; do not re-derive from the paper alone (the code contains
implementation-specific conventions, including one quirk noted in §3.3 that must
be replicated or consciously diverged from).

---

## 1. What FUBAR does (2.5 semantics)

FUBAR places a $D\times D$ grid of $(\alpha,\beta)$ rate pairs over the MG94 codon
model — $\alpha$ scales synonymous substitution rates, $\beta$ scales
non-synonymous rates — computes per-site conditional likelihoods at every grid
point, then infers posterior weights $w$ on the grid under a symmetric
$\mathrm{Dirichlet}(\delta)$ prior. Per-site quantities reported: posterior mean
$\alpha$, $\beta$, $\beta-\alpha$, $P[\alpha>\beta]$, $P[\alpha<\beta]$,
empirical Bayes factor for $\beta>\alpha$, and (MH only) PSRF / effective sample
size.

### 1.1 Default settings (`fubar.run_settings`)

| Setting | Default | CLI flag to add |
|---|---|---|
| `grid size` (D) | 20 (5–50) | `--grid-points` |
| `method` | `Variational-Bayes` (also `Collapsed-Gibbs`, `Metropolis-Hastings`) | `--method vb\|cg\|mh` |
| `chains` | 5 (MH only; forced 1 otherwise) | `--chains` |
| `chain-length` | 2e6 | `--chain-length` |
| `burn-in` | 1e6 | `--burn-in` |
| `samples` | 100 | `--samples` |
| `concentration` ($\delta$) | 0.5 | `--concentration` |
| `posterior` (reporting threshold) | 0.9 | `--posterior` |
| `non-zero` | false | `--non-zero` |

### 1.2 Phase 1 — nucleotide GTR fit

`doGTR("fubar")` fits a nucleotide GTR model (5 relative rates + empirical nucleotide
frequencies + all branch lengths) on the codon alignment treated as nucleotides.

HyPhy-3 equivalent: `hyphy::opt::GTRFitter` (in `include/hyphy/opt/optimizer.hpp`),
exactly as `FELAnalyzer::create_and_fit` does (`include/hyphy/analyses/fel.hpp:67`).
Retain `gtr_res.tree` (fitted branch lengths), `gtr_res.params` (θ rates),
`gtr_res.log_likelihood`, `gtr_res.aicc` for JSON output.

### 1.3 Phase 2 — the (α, β) grid and conditional likelihoods

#### 1.3.1 Branch-length semantics (critical)

In 2.5, each branch carries local synonymous/nonsynonymous rates
$\alpha_b, \beta_b$. `fubar.scalers.SetBranchLength` solves, per branch, for
$\alpha_b=\beta_b=s_b$ such that expected substitutions per codon site equals
$3\cdot t_b$ where $t_b$ is the GTR nucleotide branch length.
`parameters.SetProprtionalConstraint(id, scaler)` sets
`id := scaler · (current value of id)`, i.e. the current local value becomes the
coefficient on a global scaler. So at grid point $(\alpha,\beta)$ the effective
branch rates are $(\alpha\cdot s_b,\ \beta\cdot s_b)$.

**HyPhy-3 mapping (verified against `rate_matrix.hpp`).** `MG94Parameters::alpha`
and `beta` are absolute rate multipliers on Q entries
(`rate_matrix.hpp:183`), and `MG94Matrix::transition_matrix` uses the raw
eigenvalues without dividing by `scale_factor` — so α/β scale absolute rates, not
normalized ones. The equivalent construction is:

1. Build a neutral reference `MG94Matrix` with `alpha = beta = 1` and GTR θ rates.
2. `conversion_factor = 3.0 / mat.scale_factor` — identical to `s_b`, uniform
   across branches because the multipliers are per-rate not per-branch.
3. Multiply every branch length by `conversion_factor`
   (same as `FELAnalyzer::prepare_branch_lengths`, `fel.hpp:102-115`).
4. At grid point $(\alpha,\beta)$: `params.alpha = α_eff`, `params.beta = β_eff`,
   rebuild `MG94Matrix`, compute $P_b(t)$ per branch, run pruning.

Do **not** use `MG94Fitter::fit_omega_and_scale` — FUBAR never fits a global
codon model; the GTR tree scaled by the neutral conversion factor *is* the
baseline.

#### 1.3.2 Grid definition (`fubar.DefineAlphaBetaGrid`, FUBAR.bf:799)

For `D = max(grid_size, 5)` points per dimension:

- `neg_pts = floor(D·0.7 + 0.5)`; `pos_pts = floor((D-1)·0.3)`; if
  `neg_pts + pos_pts ≠ D`, set `pos_pts = D - neg_pts`. For D=20: 14 + 6.
- 1-D grid `g[0..D-1]`:
  - `g[k] = k / neg_pts` for `k = 0 .. neg_pts-1`, then overwrite `g[neg_pts-1] = 1`.
    (D=20 → 0, 1/14, …, 12/14, 1.)
  - `pos_step = 49^(1/3) / pos_pts`; `g[neg_pts + k - 1] = 1 + (pos_step·k)^3`
    for `k = 1 .. pos_pts`. (Cubic spacing on (1, 50]; top point exactly 50.)
- 2-D grid, row-major: `grid[r·D + c] = (g[r], g[c])` — `grid[i][0]` is α,
  `grid[i][1]` is β. Total `G = D²` points (400 for D=20).
- If `non_zero = true`: clamp α entries to `max(g[r], 1e-3)`.

#### 1.3.3 Pass 1 — tree rescaling (`fubar.pass1`, FUBAR.bf:280-297)

Evaluate **total alignment log-likelihood** at each of the G grid points
(α_eff = α, β_eff = β for pass 1). Find the argmax point $(\alpha^*,\beta^*)$.
Then **re-constrain**: in 2.5 this freezes the per-branch rates at
$\alpha^*\cdot s_b$, $\beta^*\cdot s_b$ and re-attaches the global scalers, so
pass-2 grid point $(\alpha,\beta)$ produces effective rates
$(\alpha\cdot\alpha^*,\ \beta\cdot\beta^*)$.

HyPhy-3 equivalent: pass 2 simply evaluates `params.alpha = α·α*`,
`params.beta = β·β*`. No tree mutation is needed — fold the rescale into the
parameter mapping. (Verify this equivalence with the parity test in §6: on
datasets where the argmax is (1,1) the passes are identical anyway, so test at
least one dataset where the argmax differs.)

#### 1.3.4 Pass 2 — per-site conditional likelihoods

At each grid point, compute the **log-likelihood of every site column**
(`ConstructCategoryMatrix(..., SITE_LOG_LIKELIHOODS)` in 2.5). Store a
`G × num_sites` matrix of log-likelihoods, clamped below at `-1e200`
(`pass2.evaluator`, grid_compute.ibf:70) to avoid `-inf`.

HyPhy-3 equivalent: hyphy-3 compresses identical columns into `SitePattern`s.
Compute likelihoods **per pattern** (`G × num_patterns`), then expand to per-site
columns afterwards (site s ↔ pattern via `aln` pattern index mapping). The 2.5
code does not deduplicate, but the conditional math is identical per column —
dedup is a pure optimization.

Efficient evaluation: for each grid point, one post-order traversal per pattern
with `LikelihoodEngine` internals (`include/hyphy/core/likelihood.hpp`,
`compute_inside_outside` or the Felsenstein path used by
`FELAnalyzer::compute_pattern_likelihood`, `fel.hpp:118`). Parallelize over grid
points with OpenMP (`#pragma omp parallel for`, dynamic schedule) — this is the
dominant compute cost of the whole analysis: `G × P` pruning passes.

#### 1.3.5 `ConvertToConditionals` (grid_compute.ibf:414)

Convert the `G × S` log-likelihood matrix `L` (S = sites, expanded) into
normalized conditionals `C` and per-site scalers `m`:

```
for each site s:
    M_s        = max_g L[g,s]
    e[g]       = exp(L[g,s] − M_s)
    n_s        = Σ_g e[g]
    C[g,s]     = e[g] / n_s          // column-stochastic over grid points
    scalers[s] = M_s + log(n_s)      // needed for the MH joint log-L
```

Cache `C` (`G × S`), `scalers` (`S`), and the `G × 2` grid matrix. These three
objects are the entire input to Phase 3.

### 1.4 Phase 3 — posterior estimation of grid weights

All three methods estimate the posterior mean of $w$ over the $G$ simplex under
prior $w \sim \mathrm{Dirichlet}(\delta\cdot\mathbf{1})$. Implement all three;
`vb` (default), `cg`, `mh`.

#### 1.4.1 VB0 — 0th-order Variational Bayes (`RunVariationalBayes`, ibf:355)

Deterministic fixed-point iteration — implement first (it is the default and the
easiest parity target):

```
w_g ← Σ_s C[g,s]                                   // init
repeat:
    W[g,s]   = w_g · C[g,s]
    norm_s   = Σ_g W[g,s]
    w_g     ← Σ_s W[g,s]/norm_s + δ
    w       ← w / Σ_g w_g
until |w − w_prev|_1 < 1e-8 or 100000 iterations
return w
```

#### 1.4.2 Collapsed Gibbs (`RunCollapsedGibbs`, ibf:277)

Single chain (`chains` forced to 1 in 2.5):

```
w ~ Dirichlet(δ·1_G)                               // initial draw
for step = 1 .. chain_length:
    W[g,s]   = w_g · C[g,s]
    norm_s   = Σ_g W[g,s]
    post_g   = Σ_s W[g,s]/norm_s
    w ~ Dirichlet(post + δ·1)                      // fresh draw each step
    if step > burn_in and (step − burn_in + 1) % thin == 0: record w
thin = floor((chain_length − burn_in) / samples)
```

Requires a Dirichlet sampler: draw `z_g ~ Gamma(shape = post_g + δ, scale = 1)`
(Marsaglia–Tsang), set `w_g = z_g/Σz`.

#### 1.4.3 Metropolis-Hastings (`ExecuteMCMC`, ibf:95)

Per chain (default 5, run in parallel):

```
load_g     = Σ_s C[g,s]                            // = 1 per point since C is column-normalized; compute anyway
w_g ~ U(0.8·load_g, 1.2·load_g);  w ← w/Σw
baseline_step = max(min(0.001, 1/S), median(w))
L[g,s] = w_g·C[g,s];  site_log_sum = Σ_{g,s} log L[g,s]
logL_data = Σ_s [log(Σ_g w_g C[g,s]) + scalers[s]]
logL_prior = Dirichlet log-density(w; δ)           // −1e10 if any w_g ≤ 1e-10
for steps = 0 .. chain_length−1:
    idx, idx2 = two distinct uniform draws in [0, G)
    change ~ U(0, baseline_step)
    if w[idx] > change:
        diff_s        = (C[idx2,s] − C[idx,s])·change        // vector over s
        logLDiff      = Σ_{g,s} log(L[g,s]+diff[g,s]) − site_log_sum   // only rows idx,idx2 change
        diffPrior     = (δ−1)·[log((w_i−δc)/w_i) + log((w_j+δc)/w_j)]
        if U(0,1) ≤ exp(logLDiff + diffPrior): accept; update L, w, sums
    if steps > burn_in and (steps − burn_in + 1) % thin == 0:
        record w and logL_data
```

**Quirk to replicate** (`ExecuteMCMC`, ibf:150, 177): the step-size adaptation
checks `(steps+1) % (chain_length·50) == 0`, which can never trigger — the
baseline step is effectively **fixed** in 2.5. Do not "fix" this silently:
replicate the fixed-step behavior for parity, and document the divergence option
(a `--adaptive-steps` flag may be added later, but must default to 2.5 behavior).

Dirichlet log-density (`LogDrichletDensity`, ibf:81):
`Σ_g (δ−1)·log w_g + lnΓ(δG) − G·lnΓ(δ)`; return `-1e10` if `min(w) ≤ 1e-10`;
return `0` if `δ == 1`.

### 1.5 Phase 4 — per-site report

Let `K` = total recorded weight samples (`samples` per chain × `chains`; for VB0,
`K = 1` with `w` = the converged mean). For sample `k` and site `s`:

```
P_ks[k,s]   = Σ_g w_k[g]·C[g,s]
pos[k,s]    = Σ_g w_k[g]·C[g,s]·1[α_g < β_g] / P_ks[k,s]      // P[β>α]
neg[k,s]    = Σ_g w_k[g]·C[g,s]·1[α_g > β_g] / P_ks[k,s]      // P[α>β]
mα[k,s]     = Σ_g w_k[g]·α_g·C[g,s] / P_ks[k,s]
mβ[k,s]     = Σ_g w_k[g]·β_g·C[g,s] / P_ks[k,s]
```

Efficient form (what 2.5 does): `pos_ks = W · (stencil₊ ∘ C) / (W·C)` with
`W` = `K×G` weight samples, `stencil₊[g,s] = 1[α_g<β_g]` broadcast.
Same for `neg_ks`, and `mα = W·diag(α)·C`, `mβ = W·diag(β)·C`.

Per-site table values (per partition; hyphy-3 has a single partition):

| Col | Content | Source |
|---|---|---|
| 0 | `alpha` | pooled mean of `mα` over samples/chains |
| 1 | `beta` | pooled mean of `mβ` |
| 2 | `beta-alpha` | col1 − col0 |
| 3 | `Prob[alpha>beta]` | pooled mean of `neg` |
| 4 | `Prob[alpha<beta]` | pooled mean of `pos` |
| 5 | `BayesFactor[alpha<beta]` | see below |
| 6 | `PSRF` | MH only |
| 7 | `N.eff` | MH only |

For MCMC methods, "pooled mean" is the mean over per-chain sample means via
`ComputeRandNeff` (ibf:246) — standard Gelman-Rubin: per-chain mean `x̄_c`,
within-chain variance `W_c`; `B = n·Var(x̄_c)`, `W = mean(W_c)`,
`VarEst = (n−1)/n·W + B/n`; returns `(overall_mean, sqrt(VarEst/W)=PSRF,
VarEst·C·n/B = Neff)`. For a single chain returns `(mean, 0, 0)` — so CG
reports PSRF = Neff = 0 columns in the 8-col matrix but the headers omit them.
For VB0, values are used directly (no pooling).

`posterior_mean_over_grid` = mean of all recorded weight samples (K×G column
means). For VB0 it is `w` itself.

`weight_non_positive = Σ_g posterior_mean_over_grid[g]·1[α_g ≥ β_g]`

`BayesFactor[s] = pos_odds / prior_odds` where
`pos_odds = p4/(1−p4)` and `prior_odds = (1−w_np)/w_np` — i.e.
`BF = p4/(1−p4) · w_np/(1−w_np)`; if `w_np` is exactly 0 or 1 → `BF = 1`
(FUBAR.bf:591-595). Guard `p4 = 1` division by zero.

Also compute per-site posterior loading vectors for JSON
`posterior` section: `pp[g] = w̄[g]·C[g,s]`, normalized by `Σpp` — a `G`-vector
per site (FUBAR.bf:556-557).

Sites flagged: `p4 ≥ posterior` threshold (0.9). Expected false positives
`E = Σ_flagged (1 − p4_s)` and 95% CI via Poisson-binomial convolution
(`fubar.ComputeENFP_CI`, FUBAR.bf:652): iterate the PDF
`PDF_new[x] = PDF[x]·(1−p_i) + PDF[x−1]·p_i`; take the CDF to find indices at
`sig/2` and `1−sig/2`.

### 1.6 RNG

2.5 uses HyPhy's RNG; exact draw-for-draw parity is impossible and not expected —
parity is **statistical** for MCMC (posterior means agree within MC error) and
**deterministic** for VB0. Use `std::mt19937_64` seeded by `--seed` (default:
fixed constant, e.g. 0x5EED, for reproducible runs). Implement:

- `U(a,b)` uniform — `std::uniform_real_distribution`
- `Gamma(k,1)` — Marsaglia–Tsang (handles `k<1` via the `k+1` trick)
- `Dirichlet(α vector)` — normalized independent gammas
- Random permutation — `std::shuffle` (used to permute MH draw order, ibf:483)

Keep all RNG behind a small `FubarRng` struct so a single seed reproduces an
entire run.

---

## 2. Files to create / modify

New files:

| Path | Content |
|---|---|
| `include/hyphy/analyses/fubar.hpp` | `FUBARSettings`, `FUBARSiteResult`, `FUBARAnalyzer` — full implementation, header-only, matching existing analyzer style |
| `src/apps/hyphy_fubar.cpp` | CLI driver + `run_fubar(argc, argv)` |
| `tests/test_fubar.cpp` | unit + parity tests |

Modified files:

| Path | Change |
|---|---|
| `src/apps/hyphy3.cpp` | add `fubar` dispatch (lines ~65-76); add `run_fubar` extern decl; update usage text |
| `src/python/bindings.cpp` | bind `FUBARSettings`, `FUBARSiteResult`, `FUBARAnalyzer` mirroring the FEL block (lines 114-127) |
| `python/hyphy3/__init__.py` | add names to the `from _hyphy3 import (...)` list |
| `CMakeLists.txt` | `hyphy3-fubar` executable + `hyphy-fubar` alias (mirror lines 94-109); add `hyphy_fubar.cpp` to the `hyphy3` multi-call binary sources (line ~83); `hyphy_fubar_test` + `add_test` (mirror lines 49-59) |
| `README.md` | add FUBAR to Implemented Analyses + CLI example |
| `AGENT.md` | remove FUBAR from the "implement a new model" example list if desired |

No changes needed in `hyphy::core` — `MG94Parameters`, `MG94Matrix`,
`LikelihoodEngine`, `GTRFitter` already provide everything. Do **not** modify
`rate_matrix.hpp`.

---

## 3. `FUBARAnalyzer` design

```cpp
#pragma once
#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/opt/optimizer.hpp"
#include <nlohmann/json.hpp>

namespace hyphy::analyses {

enum class FUBARMethod { VariationalBayes, CollapsedGibbs, MetropolisHastings };

struct FUBARSettings {
    int         grid_points    = 20;
    FUBARMethod method         = FUBARMethod::VariationalBayes;
    int         chains         = 5;            // MH only; forced to 1 for CG
    long        chain_length   = 2'000'000;
    long        burn_in        = 1'000'000;
    int         samples        = 100;
    Scalar      concentration  = 0.5;          // Dirichlet delta
    Scalar      posterior      = 0.9;          // reporting threshold
    bool        non_zero       = false;        // clamp alpha >= 1e-3
    uint64_t    seed           = 0x5EED;
};

struct FUBARSiteResult {
    int    site = 0;              // 1-based codon index
    Scalar alpha = 0, beta = 0;   // posterior means
    Scalar beta_minus_alpha = 0;
    Scalar prob_neg = 0;          // P[alpha > beta]
    Scalar prob_pos = 0;          // P[alpha < beta]
    Scalar bayes_factor = 1.0;
    Scalar psrf = 0.0;            // MH only
    Scalar n_eff = 0.0;           // MH only
};

struct FUBARResult {
    std::vector<FUBARSiteResult> sites;
    std::vector<std::array<Scalar,3>> grid_with_weights; // (alpha, beta, mean posterior weight)
    std::vector<std::vector<Scalar>> site_posteriors;    // G-vector per site
    std::vector<int>   flagged_sites;                    // prob_pos >= threshold
    Scalar expected_false_positives = 0.0;
    std::pair<int,int> enfp_ci_95 = {0,0};
};

class FUBARAnalyzer {
public:
    Tree tree;                                   // GTR tree, branch lengths in codon units
    Alignment aln;
    MG94Parameters base_params;                  // GTR thetas; alpha=beta=1 for conversion
    Scalar conversion_factor = 1.0;
    bool   has_gtr_fit = false;
    Scalar gtr_log_l = 0.0, gtr_aicc = 0.0;
    GTRParameters gtr_rates;
    std::unordered_map<std::string, Scalar> gtr_branch_lengths;

    // Phase-2 products
    Matrix grid;                    // G x 2
    Matrix conditionals;            // G x S (expanded per-site)
    Vector site_scalers;            // S
    Scalar rescale_alpha = 1.0, rescale_beta = 1.0;  // pass-1 argmax

    static FUBARAnalyzer create_and_fit(Tree tree, Alignment aln,
        std::function<void(const std::string&, double)> progress_cb = nullptr);

    FUBARResult run(FUBARSettings settings = {}) const;
    nlohmann::json to_json(const FUBARResult& res) const;
};
} // namespace hyphy::analyses
```

### 3.1 `create_and_fit`

1. `GTRFitter(input_tree, aln).fit()` → tree, θ, logL, AICc (mirror `fel.hpp:67-98`).
2. `base_params`: θ from GTR; `alpha = beta = 1`.
3. `conversion_factor = 3 / MG94Matrix(base_params).scale_factor`; scale all
   branch lengths. Store GTR branch lengths separately for JSON.

### 3.2 `run(settings)` — phases

1. **Grid** — `build_grid(D, non_zero)` per §1.3.2.
2. **Pass 1** — total log-likelihood per grid point (parallel over G):
   `params.alpha = grid[g][0]`, `params.beta = grid[g][1]`; reuse
   `LikelihoodEngine::compute_mg94_log_likelihood(tree, aln, mat)`
   (`likelihood.hpp:107`). Argmax → `rescale_alpha/beta`.
3. **Pass 2** — per-**pattern** log-likelihoods `G × P`, clamp `max(l, -1e200)`;
   expand to `G × S`; `ConvertToConditionals` per §1.3.5.
   Reuse `FELAnalyzer::compute_pattern_likelihood`-style pruning — consider
   lifting it to `LikelihoodEngine` as a shared static if it is FEL-private.
4. **Posterior** — dispatch on `settings.method` (§1.4). For MH run chains under
   `#pragma omp parallel for` with per-chain `FubarRng(seed + chain_id)` so runs
   are reproducible regardless of thread scheduling.
5. **Report** — §1.5. Implement `compute_rand_neff` and `enfp_ci` as file-local
   statics.

### 3.3 Replicate-or-diverge decisions

- MH step-size adaptation never fires in 2.5 (§1.4.3 quirk). **Replicate** fixed
  `baseline_step`; add a comment citing `grid_compute.ibf:150`.
- `normalize_by_site`/`sum_by_site` in `ExecuteMCMC` are algebraically redundant
  (C is already column-normalized). Compute them anyway — trivial cost, exact
  parity of initialization.
- Grid indexing is row-major `(α_outer, β_inner)` — getting this wrong silently
  transposes the grid. Unit-test it (§5).
- `sample` thinning: `thin = (chain_length − burn_in) / samples` (integer).
  2.5 records when `(step − burn_in + 1) % thin == 0` (note the `+1`); replicate
  exactly.
- 2.5 writes a `.FUBAR.cache` file for checkpointing. **Skip** caching in the
  port (hyphy-3 has no cache infrastructure); note it in README as a deliberate
  omission.

---

## 4. CLI (`src/apps/hyphy_fubar.cpp`)

Mirror `hyphy_fel.cpp` structure: banner ("Fast Unbiased Bayesian
AppRoximation", cite Murrell 2013), `print_usage`, `run_fubar(argc, argv)`.

```
Required:
  --alignment <file>     codon alignment (FASTA/NEXUS)
Optional:
  --tree <file>          Newick tree (or embedded in NEXUS)
  --code <name>          genetic code (default Universal)
  --grid-points <int>    D, points per dimension (default 20)
  --method <vb|cg|mh>    posterior estimator (default vb)
  --chains <int>         MH chains (default 5)
  --chain-length <int>   steps per chain (default 2000000)
  --burn-in <int>        burn-in steps (default 1000000)
  --samples <int>        posterior samples to record (default 100)
  --concentration <f>    Dirichlet concentration (default 0.5)
  --posterior <f>        reporting threshold (default 0.9)
  --non-zero             clamp alpha grid >= 1e-3
  --seed <int>           RNG seed (default fixed)
  --threads <N>
  --output <file>        default <alignment>.FUBAR.json
```

Register `fubar` in `hyphy3.cpp` dispatch and usage text.

## 5. JSON output schema

Follow `FELAnalyzer::to_json` (`fel.hpp:305-330`) and the 2.5 emission
(FUBAR.bf:50-59, 624-638):

```json
{
  "analysis": {"info": "...", "version": "3.0.0", "reference": "FUBAR ... MBE 30:1196", "authors": "...", "contact": "...", "requirements": "in-frame codon alignment and a phylogenetic tree"},
  "input":    {"file": ..., "sequences": ..., "sites": ..., "tree": ..., "trees": ...},
  "fits": {
    "Nucleotide GTR": {
      "log-likelihood": ..., "AIC-c": ..., "parameters": ...,
      "Rate Distributions": ..., "Equilibrium frequencies": ...,
      "Branch lengths": {...}, "display order": 0
    }
  },
  "grid":   [[alpha, beta, posterior_mean_weight], ...],   // G rows
  "MLE": {
    "headers": [["alpha","..."],["beta","..."],["beta-alpha","..."],
                ["Prob[alpha>beta]","..."],["Prob[alpha<beta]","..."],
                ["BayesFactor[alpha<beta]","..."],
                ["PSRF","..."],["Neff","..."]],              // last two MH only
    "content": {"0": [[col0..col5(,col6,col7)] per site]}
  },
  "posterior": {"0": [[w_1..w_G] per site]},               // per-site grid loadings
  "settings": { ...all FUBARSettings... },
  "timers":  {...}, "runtime": "3.0.0"
}
```

Datamonkey consumers key on `"MLE".headers`/`content`, `"fits"["Nucleotide GTR"]`,
and `"grid"`. Match key names exactly.

---

## 6. Verification plan

### 6.1 Unit tests (`tests/test_fubar.cpp`)

1. `build_grid`: assert for D=20 the exact 1-D vector `{0, 1/14,…,12/14, 1, 1+(49^(1/3)/6)^3,…,50}`; G=400; row-major ordering `grid[r*20+c]=(g[r],g[c])`; `non_zero` clamp.
2. `convert_to_conditionals` on a small synthetic L matrix: columns sum to 1; scalers equal log-sum-exp.
3. `variational_bayes`: converges on a synthetic C (e.g. 4×5 random column-stochastic), fixed point satisfies `w ∝ (post + δ)` within 1e-10; result independent of init.
4. `dirichlet`/`gamma` sampler: moment checks (mean = α/Σα, tolerance 1%) over 1e5 draws with fixed seed; reproducibility under same seed.
5. `compute_rand_neff`: feed 2 chains with known means/variances; verify mean/PSRF/Neff vs hand-computed Gelman-Rubin.
6. `enfp_ci`: small hand-verified Poisson-binomial (e.g. p = {0.5,0.5}).
7. End-to-end smoke: CD2 or small test alignment (reuse whatever dataset
   `test_fel.cpp`/`test_fel_cd2_parity.cpp` load — **note:** existing tests have
   hardcoded `/Users/sergei/...` paths; use repo-relative test data instead, and
   do not propagate the hardcoded-path pattern).
8. RNG determinism: two `run()` calls with same seed → identical JSON for `cg`/`mh`; `vb` identical regardless of seed.

### 6.2 Parity vs HyPhy 2.5

1. Generate reference output once (outside the test): run HyPhy 2.5 `FUBAR.bf` on
   2–3 datasets (CD2 `data/`, `pol`-like viral set, one larger e.g. β-globin) with
   `--method Variational-Bayes` and `--method Metropolis-Hastings` default
   settings; store `.FUBAR.json` under `tests/data/` (gitignored or committed
   small).
2. **VB0 parity (primary gate)**: `posterior_mean_over_grid` elementwise
   `|Δ| < 1e-6`; per-site `prob_pos |Δ| < 1e-4`; identical flagged-site sets at
   the 0.9 threshold; `alpha`/`beta` means within 1e-3.
   Rationale: VB0 is deterministic; residual differences come only from
   floating-point traversal order and the pass-1 argmax (assert identical argmax).
   Per §1.3.3, include at least one dataset where pass-1 argmax ≠ (1,1) to
   exercise the rescale equivalence.
3. **GTR fit parity**: `|Δ ln L| < 1e-2`, branch lengths `< 1e-4` — same gate as
   FEL (`AGENT.md` verification protocol).
4. **MH statistical parity**: per-site `prob_pos` within ~0.02 of 2.5 on the same
   settings (MC error bound), flagged-site sets equal, PSRF/Neff in same range.
   Run both with fixed seeds but do **not** expect draw-for-draw equality.
5. **CG parity**: same statistical gate as MH.

### 6.3 Performance target

`G × P` pruning passes dominates. On CD2 (10 taxa, 187 codons, G=400): target
< 5 s for VB0 end-to-end single-threaded; document actual numbers in README
benchmark table style. MH chains should scale linearly with `--threads`.

---

## 7. Implementation order (suggested)

1. `build_grid` + `convert_to_conditionals` + unit tests (no likelihood needed; testable standalone).
2. `create_and_fit` (GTR + conversion) — verify against 2.5 GTR JSON.
3. Pass 1 + pass 2 grid evaluation — dump `L` and compare a few columns against a 2.5 `fubar.cache` conditional matrix if obtainable, else verify total log-likelihood at (1,1) matches 2.5 MG94 logL.
4. VB0 + Phase-4 reporting + JSON — **first end-to-end milestone**, run full VB0 parity gate.
5. Dirichlet/Gamma RNG + Collapsed Gibbs.
6. Metropolis-Hastings + `compute_rand_neff`.
7. CLI, bindings, CMake, README.
8. Parity runs for CG/MH; performance measurement; docs.

## 8. Known risks / gotchas

- **Grid ordering**: row-major `(α_outer, β_inner)` — a transposed grid silently
  swaps α/β posteriors. Unit test pins this.
- **Pass-1 rescale**: verify equivalence of "fold α*β* into params" vs 2.5
  proportional constraints (§1.3.3) — the parity datasets must include a case
  with argmax ≠ (1,1).
- **Pattern vs site expansion**: `C` is `G × S` expanded, not `G × P`; the
  Dirichlet/MH math counts site multiplicity. Expand before Phase 3 or multiply
  pattern terms by pattern weight — expanding is simpler and matches 2.5 exactly.
- **`-inf` handling**: clamp at −1e200 like 2.5; `exp(l − max)` then yields 0,
  consistent.
- **`scalers` column** is required by MH `logL_data` (ibf:139) — easy to drop
  accidentally.
- **`weight_non_positive` uses `α ≥ β`** (includes the diagonal) — off-by-one on
  strictness changes the Bayes factor.
- **`ComputeRandNeff` single-chain** returns zeros for PSRF/Neff — CG output must
  still produce the 8-column matrix with zeros, headers omit cols 6-7.
- **JSON "content"** is keyed by partition index (`"0"`) containing the per-site
  rows — match the structure other analyzers emit for Datamonkey compatibility.
