# HyPhy 3 Code Health Review

**Scope.** A code-health review of the HyPhy 3 port — factoring, brittleness, edge cases, error handling, memory safety, API design, threading, build, and tests. This is a companion to `HYPHY3_STATS_SCIENCE_REVIEW.md` (statistical/methodological findings) and deliberately does not repeat those; where a finding has both a correctness and a code-health dimension, it is cross-referenced.

Coverage: `include/hyphy/core/{types,alignment,tree,genetic_code,rate_matrix,likelihood,param_map}.hpp`, `include/hyphy/opt/{nelder_mead,squarem,optimizer}.hpp`, `include/hyphy/autograd/{var,tree_likelihood,optimizer}.hpp`, `include/hyphy/analyses/{fel,meme,busted,absrel}.hpp`, `src/apps/*.cpp`, `src/python/bindings.cpp`, `CMakeLists.txt`, `tests/*.cpp`.

**Revision note (2026-10-10).** Updated for upstream commits `9521355`–`6e22db3`: RELAX landed as a fifth analyzer (`analyses/relax.hpp` — not yet audited beyond its `to_json`); threshold-based likelihood scaling landed in `likelihood.hpp` (`e1352b6`); all five CLIs were rewritten onto `console.hpp` (`e724bd3`); and a versioned v3.0 JSON schema landed for FEL only (`6e22db3`). §9 is rewritten against the landed code; other sections are annotated where the delta affects them. Cited line numbers may drift further.

**Overall assessment.** The statistical core is carefully written — the inside-outside adjoint, EM responsibilities, and optimizer wiring are structurally sound and mostly correct. The systemic weaknesses are not algorithmic; they are of a consistent type:

- **Silent degradation everywhere.** Missing tree taxa become all-ones leaves, malformed branch lengths become 0.0, partial ambiguity becomes full missingness, non-converged optimizers report `converged=true`, and M≥3 aBSREL returns a sentinel instead of an error. The codebase almost never fails loudly.
- **No NaN hygiene.** Not a single `isnan`/`isfinite` check exists in `include/hyphy/`. Every optimizer treats NaN objective values as ordinary numbers through comparisons that silently misbehave.
- **Input validation stops at file-open.** After `ifstream::is_open`, nearly everything downstream is trusted: malformed Newick, mismatched tree↔alignment taxa, degenerate CLI values, and sub-3-codon alignments all produce silently wrong results or crashes deep in hot loops.
- **Duplication over abstraction.** The same leaf-vector construction, taxon lookup, AICc formula, try/catch wrappers, banner/usage/arg-parse blocks are copy-pasted across files; fixing a bug requires finding every copy.

None of this blocks the current benchmarks, but the failure modes are the worst kind for a scientific tool: *plausible-looking wrong output* rather than crashes.

---

## 1. Crash & UB Risks

### 1.1 Uncaught `std::stoi`/`std::stod` in all five CLIs — crash on bad numeric args (HIGH, trivially triggered)

`src/apps/hyphy_fel.cpp:81,85`, `hyphy_meme.cpp:77,81`, `hyphy_busted.cpp:90`, `hyphy_absrel.cpp:72,74,76`, `hyphy_relax.cpp:72,74` — `std::stoi(argv[++i])` / `std::stod(argv[++i])` called with no try/catch. The `e724bd3` console-UI rewrite preserved the unguarded pattern verbatim and `ec9e5d1` added a fifth copy (`hyphy_relax.cpp`); `parse_json_format` (`provenance.hpp:301`) adds a silent-default variant — `--json-format garbage` is accepted as `modern` rather than rejected. `hyphy3 fel --threads abc` or `--pvalue foo` → `std::invalid_argument` → `terminate`. No validation of parsed ranges either: `--rates 0` → `num_rate_classes=0` → downstream `omegas[K-1]`/`omegas[2]` indexing is UB; `--threads -5` → `omp_set_num_threads(-5)` (behavior undefined, typically clamps or errors silently); `--pvalue 99` silently accepted.

### 1.2 Tree↔alignment taxon mismatch is silently absorbed (HIGH)

`likelihood.hpp:20-33` — `find_taxon_index` returns `size_t(-1)` for unmatched leaves; every leaf initializer (`likelihood.hpp:74,146,204`; equivalent paths in `busted.hpp`, `absrel.hpp`) treats that as a fully-ambiguous `setOnes()` leaf. A tree leaf name that doesn't match any alignment taxon → **that taxon's entire sequence is replaced by an uninformative leaf** → biased likelihood, no warning. A NEXUS quoting difference, a FASTA-name truncation (`alignment.hpp:84-87` cuts names at first whitespace while tree files keep the full name), or a stray `.`/`_` difference silently degrades an entire analysis. There is also no check that `tree.num_leaves() == aln.num_taxa`, and no check that every *alignment* taxon is *used* (extra sequences are simply ignored). This should be a hard error listing unmatched names on both sides.

### 1.3 Newick parser accepts malformed input silently (HIGH)

`tree.hpp`:

- **`std::stod` exceptions swallowed** (`tree.hpp:84-88`): `catch(...) { length = 0.0; }` — `:abc`, `:1e+`, `:` parse failures become a real 0.0 branch length indistinguishable from data.
- **No balance validation**: unmatched `(` leaves `parent_stack` non-empty at EOF — the tree is accepted anyway. Extra `)` hits `if (!parent_stack.empty())` and is silently ignored → detached subtrees.
- **Multiple roots**: `for (auto& node : nodes) if (parent_id == INVALID) root_id = node.id;` (`tree.hpp:146-149`) — picks the **last** rootless node; a forest or dangling subtree silently re-roots the tree.
- **Whitespace is stripped globally before tokenization** (`tree.hpp:59`): quoted Newick names `'My Species'` lose their spaces → `'MySpecies'` → leaf-name mismatch → §1.2.
- **`;` only stripped at end** — a mid-string semicolon (multi-tree file) gets parsed into a token name.
- **`process_token` splits at first `:`** (`tree.hpp:79`) — names containing colons break silently.
- **Duplicate leaf names** silently overwrite in `leaf_name_to_id` (`tree.hpp:159`).
- **Recursive post-order** (`tree.hpp:182-187`) → stack overflow on deeply ladderized trees (~10k+ taxa); should be iterative.
- **Auto-named nodes `Node1..N`** (`tree.hpp:151`) can collide with genuine leaf names → `branches[name]` maps (aBSREL) silently merge results.

### 1.4 Alignment parsing silent corruptions (MEDIUM–HIGH)

`alignment.hpp`:

- **`load()` format sniffing** (`alignment.hpp:239-261`): a FASTA file starting with a `#` comment line → dispatched to `from_nexus` → confusing "MATRIX not found" error. Any first line containing "begin" → NEXUS. File is opened twice (once to sniff, once to parse) — wasteful and racy.
- **FASTA names truncated at first space** (`alignment.hpp:84-87`) — `>seq1 some description` becomes `seq1`; two sequences `>x 1` / `>x 2` collide → `taxon_to_index` overwrite (see below), sequence count inflated, patterns sized by `sequences.size()` while lookups resolve differently.
- **Duplicate taxon names** (`alignment.hpp:263-269`): `taxon_to_index[name]` overwritten but both `Sequence` objects kept → `num_taxa` counts both, name lookup returns only the last → the first sequence is unreachable shadow data.
- **NEXUS `.` mapped to `-`** (`alignment.hpp:183`): in NEXUS, `.` means *match the first taxon's character* — not a gap. Dot-notation matrices are silently corrupted (a real correctness bug, not just an edge case).
- **NEXUS `MATRIX` keyword match** (`alignment.hpp:136`): case-insensitive substring search can hit `MATRIX` inside unrelated block text; `mat_end` is the first `;` after it — a `;` inside quoted content ends the matrix early.
- **NEXUS seq reader stops at first whitespace** (`alignment.hpp:185-193`): works for interleaved format via the taxon-dedup-append path (`alignment.hpp:199-206`, which is good), but digits/symbols inside sequences are silently *skipped* (`idx++` at line 195) — a sequence containing e.g. a stray `1` is silently shortened.
- **No `num_codons > 0` / `num_taxa ≥ 2` guards in `finalize`** (`alignment.hpp:272-309`): an all-gap or single-sequence alignment loads "successfully"; downstream `N - K - 1` denominators in every AICc produce NaN/Inf (§1.5); likelihood of an empty pattern set is exactly `0.0` logL — a *plausible-looking* result.

### 1.5 AICc denominator `N − K − 1` unguarded → NaN/Inf (MEDIUM)

`optimizer.hpp:137`, `hyphy_fel.cpp:191`, `absrel.hpp:333`, `busted.hpp:1139-1145` — every AICc computes `2·K·N/(N−K−1)` with no check that `N > K+1`. For a small alignment (e.g., 10 codons, 40 params) → negative denominator → negative AICc → garbage ordering and garbage JSON, silently.

### 1.6 Unchecked indexing — latent UB (MEDIUM)

- **`param_map.hpp:35,57,123`** — `x[opt_index]`, `grad_x_out[c.opt_index]` via `std::span`: **no bounds checking**; a misregistered `opt_index` → out-of-bounds read/write UB. (The whole class is unused — §4.1 — but it's a loaded gun for whoever wires it in.)
- **`bindings.cpp:63-66`** — `set_branch_lengths` silently ignores a short `bl` vector (`i < bl.size()` truncates): partial update, no error.
- **`tree_likelihood.hpp:28-31`** — `branch_lengths.size() < num_nodes` → silently assigns 0.0-length branches (P(0)=I) rather than erroring.
- **`likelihood.hpp:333`** — `branch_lengths[node.id]` silently falls back to `node.branch_length` for out-of-range indices — a partially-specified vector mixes tree state and overrides silently.
- **`busted.hpp` `evaluate_log_l(bl[...])`** — indexes `bl[node.id]` assuming dense node ids and correct size; no assert.
- **`genetic_code.hpp` overrides** — `tbl[idx] = aa` with `std::array` unchecked write; a typo'd override index >63 → OOB write at static-init time. `hyphy_aa[aa_idx]` unchecked read.
- **`P_branches[root_id]` never initialized** (universal pattern) — `std::vector<Matrix> P_branches(num_nodes)` leaves root's entry a default `Matrix()` (0×0); every loop hand-guards `node.id != root_id`. It works today, but it is a latent crash: the first caller who forgets the guard gets an Eigen dimension mismatch (assert in debug, UB in release).

### 1.7 Zero-π division → inf/NaN asymmetry (LOW–MEDIUM)

`rate_matrix.hpp:38-40` — `GTRMatrix::update`: `inv_sqrt_pi = 1/sqrt(pi)` with **no floor**. An alignment missing a nucleotide entirely (e.g., all-A/C/G) → `inv_sqrt_pi = inf` → `S` gets `0·inf = NaN` → eigendecomp throws. Loud, but avoidable — `MG94Matrix` floors `pi` at `1e-12` (`rate_matrix.hpp:~165`); the two classes handle the same edge inconsistently.

---

## 2. Silent Failures & False Reporting

### 2.1 Optimizers lie about convergence (HIGH for a stats tool)

- **`optimizer.hpp:249,284`** — `res.converged = true` is set **unconditionally**, including the path where `solver.minimize` threw (`optimizer.hpp:240-242` catches, prints to `cerr`, continues with whatever `x` happened to be — possibly the initial guess — and reports it as a converged optimum). `res.iterations` is likewise fabricated: `= 3` (hardcoded, `optimizer.hpp:133`) and `= 50` (`optimizer.hpp:283`) instead of actual counts.
- **`nelder_mead.hpp`** — `NelderMead2D`/`NelderMeadND` return best point + evals but no `converged` flag; hitting `max_iter` is indistinguishable from convergence. Callers cannot detect non-convergence at all.
- **`squarem.hpp`** — does report `res.converged`, but on non-monotone steps (`x2_lnL < cur_lnL`) the fallback update is taken unconditionally and `res.log_likelihood` tracks the *last* point, not the **best** point — a regressing trajectory returns a worse result than an intermediate iterate.
- **`total_evals` (`optimizer.hpp:101,123`)** — incremented by the magic constant `20` (Brent's max_iter) rather than actual evaluations; dead code that also misreports.

### 2.2 Silent fallback constants (MEDIUM)

- `likelihood.hpp:98,170`, `busted.hpp:257`, `absrel.hpp:302,417` — `pat_L ≤ 0 → -1e20`. The same sentinel doubles as "this pattern has likelihood 0" (real, deserves penalty) and "this path can't be evaluated" — and in `absrel.hpp:672` as "M≥3 unimplemented." Three semantically different conditions share one magic number, so no caller can distinguish failure modes.
- `absrel.hpp:669-672` — `optimize_branch_mixture` returns `log_l = -1e20` for M≥3 (statistics review §3.1) rather than throwing — the AICc loop reads it as "model is terrible" and `break`s. A capability gap disguised as a model comparison.
- `optimizer.hpp:172` — `if (fit_rates && x.size() >= 6)` silently ignores rates when x is too short.
- `busted.hpp:1220-1223` — `unconstrained < constrained` → overwrites `unconstrained` with constrained lnL; 2.5 *refits*; here the site-level arrays and totals go inconsistent silently.
- `meme.hpp:175,349` — `total_branch_length = 0.0` always (stats review §3.7).
- `settings.test_branches` (`absrel`) / `settings.multiple_hits` (`busted`) — any unrecognized string silently defaults to `"All"`/`"None"`: a typo (`"internal"`, `"double"`) silently runs a different analysis than requested.

### 2.3 Autograd traps (MEDIUM — experimental code, but bound to a test)

- **`var.hpp:64`** — `node->grad = grad_output` sets the root grad but **parent grads are accumulated into without zeroing**: calling `backward()` twice on the same graph doubles parent gradients. There is no graph-wide `zero_grad()`; `Adam::zero_grad` only clears *parameter* leaves. The test calls work because each iteration builds a fresh graph — the footgun fires only in real use.
- **`var.hpp:131-135,153-155,186-187,201-202`** — backward closures capture `a.val()`/`b.val()` **at graph-construction time**. If a leaf's `set_val` is called between forward and backward (which is exactly what a hand-rolled optimizer does), the captured value is stale → silently wrong gradients.
- **`var.hpp:148`** — `operator/` clamps `b` to ±1e-15 *then* differentiates through the clamped value: forward computes `a/1e-15` (a huge finite number, not a domain error) and the gradient is `−g·a/b²` at the clamp — the gradient claims the output still depends on `b` where the forward says it doesn't. Same silent-clamp in `log()` (`var.hpp:180`): `log(x<0)` returns `log(1e-15)` with gradient `1e15`. These hide domain errors instead of surfacing NaN.
- **`var.hpp:69`** — `std::abs(curr->grad) > 0.0` skips `backward_fn` on zero grad — correct shortcut, but combined with the stale-capture issue means the only safe usage pattern (fresh graph per backward) is implicit and undocumented.
- **`tree_likelihood.hpp:28,41,48`** — three separate `i < num_nodes && i < branch_lengths.size()` truncations — a short vector silently zero-pads (see §1.6).
- **`Adam` (`autograd/optimizer.hpp`)** — stores `std::vector<Var*>` raw pointers (dangling if the Var dies first), no parameter projection (nothing stops branch lengths going negative → NaN likelihood → no recovery), `beta1_corr` can divide by ~0 if `beta1` is mis-set.

### 2.4 Inconsistent likelihood paths (LOW–MEDIUM)

`compute_mg94_log_likelihood` penalizes `L≤0` with `-1e20` per site; `compute_branch_length_gradients` (`likelihood.hpp:361`) **skips** `L≤0` patterns entirely — contributes 0 to both ll and gradient. Same model, same data → different log-likelihoods depending on which entry point you call. A consumer comparing them will see a discrepancy that isn't a bug in either function alone.

---

## 3. Factoring & Duplication

### 3.1 The CLI apps are ~85% identical boilerplate

`hyphy_fel.cpp`, `hyphy_meme.cpp`, `hyphy_busted.cpp`, `hyphy_absrel.cpp`, `hyphy_relax.cpp` each re-implement the same skeleton (the `e724bd3` console-UI rewrite kept the per-app argument-parser structure): banner/usage text, the same `for`-loop argument parser (same uncaught-stoi bug ×5), the same `--threads` block, the same genetic-code try/catch, the same `Alignment::load` try/catch, the same tree-or-embedded-tree resolution (4 of 5 versions wrap the embedded-tree `from_newick` in try/catch — `hyphy_meme.cpp:152` still does **not**, so a malformed embedded tree escapes as an uncaught exception there and only there), the same default-output-name rule. This is one shared `cli_common.hpp` of ~150 lines; the current structure guarantees the apps drift (they already have — see the meme try/catch gap, and JSON-write-failure returns `1` in fel/meme but warns-and-returns-`0` in busted/absrel).

### 3.2 Duplicated likelihood scaffolding

- `compute_gtr_log_likelihood` vs `compute_mg94_log_likelihood` (`likelihood.hpp:36-176`) — two ~70-line near-identical functions differing only in matrix/vector types and state count. A single template on `P_branches`/`pi`/`S` would halve the surface area for bugs like §1.2/§2.4.
- The leaf one-hot/`setOnes()` block is written **three times** in `likelihood.hpp` plus again in `busted.hpp` and `absrel.hpp` leaf paths — five copies of the ambiguity-handling logic to keep in sync.
- `leaf_to_taxon` map construction (`find_taxon_index` over leaves) is rebuilt **per likelihood evaluation** in `compute_mg94_log_likelihood`, `compute_branch_length_gradients`, `busted.hpp`'s `evaluate_log_l`, `absrel.hpp`'s loops — an O(leaves×taxa) name-map built inside the hottest loops. Build once per analysis.
- `eval_lnl`/`eval_at` lambdas rebuilding `MG94Matrix` (a 61×61 eigendecomposition) per objective call appear in `optimizer.hpp:180-184`, `fel.hpp`, `meme.hpp`, `busted.hpp` — the same "make matrix then likelihood" idiom copy-pasted ~6 places.

### 3.3 Magic numbers with no constants (HIGH maintenance cost)

A non-exhaustive list of unexplained literals carrying semantics: `-1e20` (×4 files), `1e-20`, `0.05` (branch init), `1e-4` (≈10 sites), `50`/`100`/`1000`/`2000`/`5000` (rate bounds, each a different one in different places), `39` (aBSREL GTR params — stats review §3.2), `+9` (aBSREL baseline params), `0.45/0.55`/`0.4/0.6` (mixture p-values), `2/3` (MEME boundary p), `100` (EBF threshold), `50` iterations (CF3x4), `3` cycles (GTR), `6` passes (aBSREL joint), `0.05`/`1e-3` tolerances. None are named constants or commented with rationale; several should be configuration or computed. Two analyses would benefit from a `constants.hpp` or at minimum doc comments tying each to the 2.5 source line it mirrors.

### 3.4 Magic sentinel doubles as error channel

`-1e20` is simultaneously: (a) a zero-likelihood penalty, (b) the "not implemented for M≥3" return, (c) a constraint-violation penalty in objectives (which use `+1e20`, yet *another* convention). One enum/`std::optional`/`NaN` convention would make failure modes distinguishable.

### 3.5 Hardcoded version strings vs `project(VERSION)`

`CMakeLists.txt:2` declares `3.0.0`; JSON output embeds `"3.0.0"` (aBSREL) and `"4.7-next"` (BUSTED) as literals; app banners hardcode `3.0.0`. No `HYPHY_VERSION` define flows from CMake to code — guaranteed to drift.

---

## 4. Dead Code & Unused Infrastructure

- **`param_map.hpp` (143 lines)** — a complete parameter-constraint engine (FREE/FIXED/TIED/LINEAR/POWER, forward evaluation, gradient backprop) with **zero callers**. Only `#include`d by `optimizer.hpp`, never instantiated. Either wire it in or delete; in its current state it's a documented-looking API whose UB (§1.6) invites misuse.
- **`autograd/` subtree (~330 lines)** — `var.hpp`, `tree_likelihood.hpp`, `optimizer.hpp` (Adam) are used **only by `tests/test_cpp_autograd.cpp`**. Not bound to Python (despite the CMake comment grouping), not used by any analyzer. It's an experiment living in the production include tree — fine to keep, but it should live under `experimental/` or `tests/` and be clearly marked, since §2.3's bugs are real traps.
- **`types.hpp` universal-only helpers** — `triplet_to_sense_codon`, `is_synonymous(codon,codon)`, `count_nucleotide_differences` (hardcoded 61-codon universal table) have **zero callers** and silently give *wrong* answers for non-universal codes — they shadow `GeneticCode` member functions that do it correctly. Delete or deprecate.
- **`num_test_branches` field**, `res.total_branch_length` (MEME), `settings.do_srv`/`syn_rate_classes` (aBSREL), `total_evals` (GTRFitter) — dead state.
- **`GTRFitResult.iterations`, `FitResult.converged` fields** — exist only to be populated with fake values (§2.1).

---

## 5. API Design Brittleness

- **Copy-heavy signatures**: `GTRFitter(Tree t, const Alignment& a)`, `MG94Fitter(Tree t, Alignment a)`, `create_and_fit(tree, std::move(aln))` — `Alignment` (potentially hundreds of MB with per-sequence codon vectors) is copied/moved into every analyzer; `Tree` copied into results and per-evaluation `Tree t_eval = tree` copies (`optimizer.hpp:265`). Pass `const&` + one owned copy where mutation is genuinely needed.
- **Mutable public state**: `FELAnalyzer::p_value_threshold`, `global_log_l`, `has_gtr_fit`, `gtr_rates`, `gtr_branch_lengths` are public fields the apps poke directly (`hyphy_fel.cpp:187-201`) — the analyzer's invariants depend on callers setting them consistently.
- **`std::span` parameters without size contracts** (`param_map.hpp`) — see §1.6.
- **`size_t(-1)` sentinel vs `INVALID_INDEX` (int32, -1)** — two different "not found" conventions coexist (`find_taxon_index` vs `parent_id`); comparisons `t_idx != static_cast<size_t>(-1)` are repeated ~8 times instead of a named `is_valid()`/`NO_TAXON` constant.
- **Callbacks**: `progress_cb` signatures differ subtly between analyzers (`(string,frac)` vs `(string,pct)` — same shape, ok) but aBSREL exposes it and the others bake `std::cout` calls into library code (`absrel.hpp` has 2, `optimizer.hpp` 1) — a library that prints to stdout is unusable embedded/in bindings.
- **`Tree` mutation via public `nodes[]`** — all invariants (post_order, leaf maps) are computed once at parse; `bindings.cpp` `set_branch_lengths` mutates `nodes[i].branch_length` by index with no consistency check; nothing prevents a caller from setting `branch_length < 0`.
- **`num_nodes` dense-id assumption** — `bl[node.id]`, `P_branches[node.id]`, `leaf_to_taxon[node.id]` everywhere assume ids are dense 0..N−1; true for the parser, unenforced for manually constructed trees.

---

## 6. Threading & Numerics

- **OpenMP correctness is mostly right**: per-thread `node_L`/`local_grad`/`local_ll`, `reduction(+:total_log_l)`, `#pragma omp critical` merges — the reductions and privatization are done properly. `schedule(dynamic)` is a reasonable default for variable-cost patterns.
- **The real issue is allocation churn inside parallel loops**: `compute_inside_outside` allocates `U` (num_nodes × S doubles), `child_branch_L` per internal node, plus `V`/`D` — per pattern, per call. `evaluate_log_l` rebuilds `leaf_to_taxon` and `std::vector<MG94Matrix> mg(K)` (K eigendecompositions!) per objective evaluation. On big analyses, allocator + eigendecomp dominates the actual math.
- **`is_test_branch` masking in `compute_expected_weights`** is written for the general case but unreachable (all-true) — dead generality that's confusing to read.
- **No NaN guards** (restated for emphasis — it pervades): a NaN objective into `NelderMead` → `f_r < s[0].f` is false → contraction path → spins to `max_iter` → returns garbage *without any signal*. A NaN `pat_L` → `pat_L > 0` false → `-1e20` → optimizer chases a cliff. First `isnan` check in the codebase would have to be invented.
- **`compute_pattern_mixture_responsibilities` rebuilds `P_mix` per pattern** (`likelihood.hpp:274-282`) — `P_mix` depends only on branch lengths/weights, identical across patterns; should be hoisted per evaluation, not per site.
- **`sqrt_pi` floor inconsistency** between GTR (§1.7) and MG94.

---

## 7. Build & Packaging

- **`CMAKE_CXX_FLAGS` appends `-O3` globally** (`CMakeLists.txt:9`) — ignores `CMAKE_BUILD_TYPE` (Debug builds get -O3 → debugging is painful; no `-O0 -g` path), is non-portable (MSVC), and stomps user flags.
- **No `install()` rules** — binaries and headers can't be installed/packaged.
- **Aliases `hyphy-fel` etc.** — `ALIAS` targets can't be installed; they're build-tree conveniences only.
- **`FetchContent` Eigen** — network required at configure; no vendored/system fallback (`find_package(Eigen3)` first would be more robust).
- **Python bindings silently skipped** if nanobind absent — no `STATUS`/`WARNING` telling the user bindings weren't built; module output goes into the *source* `python/` dir (`CMakeLists.txt:130`).
- **Tests not working-directory-anchored** — `add_test` has no `WORKING_DIRECTORY`; combined with §8's paths, `ctest` results depend on where it's invoked.
- **`#ifdef _OPENMP`** guards are correct throughout (verified `absrel.hpp:22`, `busted.hpp:23`, `likelihood.hpp:12`) — pragmas without include are harmlessly ignored; good.
- **`KMP_DUPLICATE_LIB_OK=TRUE`** (`CMakeLists.txt:138`) — papers over a real libomp/libiomp conflict instead of fixing it.

---

## 8. Test Suite Health

- **Tests fail rather than skip on missing data.** `test_busted.cpp:18-19`, `test_core.cpp:13,96,127`, `test_busted_mh.cpp:18,71-72,111-112`, `test_fel.cpp:14`, `test_fel_cd2_parity.cpp:68-70,122-124`, `test_fel_multi_dataset.cpp:67-130,156-164`, `test_meme.cpp:67-72,106` — hardcoded `/Users/sergei/...` paths passed straight to `Alignment::load`/`from_newick_file` with **no existence check and no fallback**. On any other machine these are not skipped tests, they're exceptions → doctest failures → red suite. Only `test_absrel.cpp:18-23` and `test_cpp_autograd.cpp:214-219` have candidate-path fallback lists (the former still with absolute paths in it).
- **`test_cpp_autograd.cpp` uses raw `assert()`** (`:27,40,41,96,100,140,203`) — compiled out under `-DNDEBUG` → in a Release test build, the entire file is a no-op that always "passes."
- **Weak assertions** (stats review §3.16): `test_absrel` sets `max_rate_classes=3` while the M≥3 path returns -1e20 — it asserts properties that would pass identically at 2; several tests assert ranges (`p ∈ [0,1]`, `LRT ≥ 0`) rather than numeric parity values.
- **No negative/edge tests**: nothing exercises a malformed Newick, a missing leaf name, a `--threads abc` CLI, a duplicated FASTA header, a NEXUS `.` matrix, a zero-codon alignment, a degenerate single-class BUSTED, an M=3 aBSREL — i.e., the entire §1 surface is untested.
- **No CI configuration** in the repo — nothing prevents regressions of exactly the silent-failure class described here.

---

## 9. JSON Output Schema — 2.5 Legacy Everywhere + v3.0 for FEL Only

**Status update (2026-10-10, commits `9521355`–`6e22db3`).** A v3.0 schema landed for **FEL only** (`6e22db3`): `fel.hpp` now has `to_modern_json()`/`to_legacy_json()` with `to_json()` defaulting to v3; `--json-format`/`--legacy-json`/`--modern-json` CLI flags; Python bindings expose both; two committed JSON Schema files (`schemas/v3/`); and a `jsonschema`-validating pytest (`tests/test_json_schema.py`). Meanwhile MEME/BUSTED/aBSREL still emit legacy-only — and **`ec9e5d1` added RELAX, a fifth `to_json` emitter that also emits the legacy 2.5-style format** (fits keyed by display name, `{"0": …}` partition indirection, `*Test*`/`*Reference*` labels inside values). The legacy emitters replicate the HyPhy 2.5 output schema — verified against `benchmarks/ground_truth/*.json` (2.5 outputs used as parity fixtures). So today there are **two disjoint top-level formats and five handwritten emitters**; the newest method, written after this concern was documented, still copied the legacy shape.

### 9.1 No explicit method identifier (the named pain point)

**Landed for FEL-v3**: `analysis.id = "fel"` (enum-pinned in `fel.v3.schema.json`) plus `schema_version`, `software.name`/`version`, and `statistical_tests.site_level_summary.method` — the last is a second, display-string method identifier that can now diverge from `analysis.id` (§9.3). **Still absent everywhere else**: all legacy emitters (FEL-legacy, MEME, BUSTED, aBSREL, and the brand-new RELAX — whose `to_json` has no `id`/`method` field). The key-inference heuristic remains live for 4 of 5 methods, and FEL emits it only in modern format.

### 9.2 Positional `headers`/`content` parallel arrays (the named pain point)

**Landed for FEL-v3**: `site_results.data` is a named columnar map — `data["p_value"][i]` — so positional `row[N]` access is gone and column insertion is non-breaking; `site_results.columns` carries `{type, unit, description}` per column name. Residual brittleness: (a) the schema does **not** require the parallel arrays be equal-length (`test_json_schema.py:52-57` asserts this by hand — the invariant belongs in the schema or emitter); (b) `columns` is an object, so declared column order is a convention, not data; (c) `classification` is recomputed in the writer from `p_value`/`beta`/`alpha` — a derived column that can silently disagree with `statistical_tests` counts; (d) SoA still forces consumers to hand-zip rows, so the record-vs-columnar choice must be settled once for all methods (§9.4). The legacy `MLE.headers`/`content` positional format persists unchanged everywhere else (`fel.hpp` legacy path, `meme.hpp:456-502`).

### 9.3 Additional schema brittleness found in review

- **Space-separated, display-oriented keys throughout**: `"file name"`, `"number of sequences"`, `"Log Likelihood"`, `"Equilibrium frequencies"`, `"Rate Distributions"`, `"display order"`, `"AIC-c"`. Forces `obj["Log Likelihood"]` access, breaks `camelCase`/`snake_case` consumer conventions, and mixes presentation ("display order") with data.
- **Fits keyed by display name**: `"Global MG94xREV"`, `"Nucleotide GTR"`, `"Baseline MG94xREV"`, `"Full adaptive model"` — the label *is* the key. Any wording change is a breaking schema change; iterating fits requires treating arbitrary human-readable strings as identifiers.
- **`{"0": ...}` partition indirection** everywhere (`MLE.content`, `branch attributes`, `tested`, `input.trees`) — every consumer unwraps a constant. There is no non-"0" path; the multi-partition capability is vestigial but consumers must still implement the indirection.
- **Internal inconsistency across what are now five emitters** — this is a partial parity copy, not a shared schema: `"Log-likelihood"` (`fel.hpp:373`) vs `"Log Likelihood"` (`absrel.hpp:104`); `"citation"` (fel) vs `"reference"` (absrel); `version: "3.0-next"` vs `"3.0.0"` vs BUSTED's `"4.7-next"`; FEL emits `attributes` metadata under `branch attributes` while aBSREL does not; FEL omits `tested`/`test results`/`timers`/`runtime`/`data partitions` that 2.5 emits — so files are **not interchangeable with real 2.5 output** either.
- **HTML entities baked into data** (`meme.hpp:457-471`): `&alpha;`, `&beta;<sup>+</sup>`, `p<sup>1</sup>` — presentation markup inside field names; any non-browser consumer must strip it.
- **Untested branches are absent, not false**: aBSREL's `tested` map and per-branch LRT/P-value keys exist only for tested branches — a consumer cannot distinguish "untested" from "missing key" (same absence-ambiguity class as §2).
- **Sentinels serialize as real values**: `-1e20` log-likelihoods and fabricated `converged` results (§2.1) serialize as ordinary numbers; NaN p-values serialize as `null` (nlohmann default) — three different encodings for "not computed", none documented.
- **`runtime` is a bare float in BUSTED but a nested `{timer, order}` object in aBSREL** — same concept, two shapes.
- **No schema version on legacy output**: FEL-v3 emits `$schema`/`schema_version`; nothing in any legacy document identifies its own format.
- **New issues in the v3 emitter itself** (`fel.hpp:490-712`, `provenance.hpp`):
  - `software.git_commit` is hardcoded `"e1352b6"` — stale at birth (the schema commit itself is `6e22db3`); the same literal-drift class as `3.0-next`/`4.7-next` (§8.4), now baked into the 'fixed' schema.
  - The auto-provenance fallback fabricates `cli_command = "hyphy3 fel"`, `wall_time_seconds = 0`, `cpu_threads = 1` — the Python binding calls `to_modern_json()` with no `Provenance`, so library consumers get JSON asserting a fake CLI invocation. Emit `null` or honest values.
  - `parse_json_format("anything")` → `ModernV3` silently (§1.1 sibling).
  - `SHA256::hash_file` returns `""` for an unreadable input → a hash that violates its own schema pattern `^[a-f0-9]{64}$`.
  - `model_fits.*.parameters_count` restates the magic counts (`5 + nodes−1`, `6 + nodes−1`) inside the new emitter (§3.3).
  - `fel.v3.schema.json` copy-pastes ~150 lines of `hyphy-analysis.v3.schema.json` instead of `$ref`-ing it — the shared schema is already drifting from itself.
  - `$schema`/`$id` URLs point at `raw.githubusercontent.com/veg/hyphy-3/...` — not resolvable until published; the pytest loads local files so remote resolution is never exercised.
  - `Tree::to_newick` (added for `phylogeny.newick`) recurses over children — deep ladderized trees that overflow `post_order` (§1.3) will now also overflow the JSON writer.

### 9.4 Landed v3.0 vs the consistency contract

`6e22db3` is a real first pass at the right architecture — versioned schema, `analysis.id`, named columnar data, snake_case, stable fit ids, provenance, committed JSON Schemas, a validating test — implemented for one method only. A consumer writing a v3 parser still learns nothing transferable to MEME (legacy format), and RELAX's brand-new emitter demonstrates that *the default shape for a new method is still the legacy 2.5 clone*. The contract below remains the target; §9.4.2 maps where landed v3 diverges from it.

#### 9.4.1 Consistency contract (target state — invariants every method satisfies)

- **Identical top-level envelope for every method**: `schema_version`, `method`, `tool`, `input`, `fits`, `test_results`, `timers`, `site_results`, `branch_results`. If a block doesn't apply (e.g., `site_results` for aBSREL), it is present as `[]`/`null` — **never omitted**. A consumer can `output["site_results"]` unconditionally for any method.
- **Identical sub-structure within each block**: every fit in `fits` has the same keys (`label`, `log_likelihood`, `aic_c`, `estimated_parameters`, `rate_distributions`); every site record has the same base keys (`site`, `lrt`, `p_value`) plus method-specific extras; every branch record has (`name`, `role`, `lrt`, `uncorrected_p_value`, `corrected_p_value`, `rate_classes`, `omegas`, `weights`) — extras only, never renames.
- **Method-specific content lives in one namespaced block** (`method_results`) — the only place shape may differ; everything above it is a fixed contract.
- **Identical conventions**: snake_case keys, `null`/omitted-value semantics, p-value names (`p_value`, never `p-value`/`P-value`/`pvalue`), likelihood names (`log_likelihood`, never `Log Likelihood`/`Log-likelihood`/`lnL`), test-role vocabulary (`"test"|"background"|"untested"` — a single enum across methods).
- **One `headers`-style table format at most** — if columnar output is retained for size, it uses the *same* `{columns: [...], rows: [...]}` shape in every method, not per-method ad hoc.

#### 9.4.2 Where landed v3 diverges from the contract

- **Coverage**: the v3 envelope exists for 1 of 5 methods; `site_results` isn't even `required` in the base schema. The other four have no `analysis.id`, `schema_version`, or `provenance`.
- **Missing contract blocks**: no `branch_results`, `timers`, `test_results`, or `method_results` namespace — BUSTED's constrained/unconstrained fits, aBSREL's per-branch rate distributions, and RELAX's Test/Reference branch sets have no home under v3 as committed. Extending v3 method-by-method before settling these reproduces per-method ad hoc inside the new schema.
- **No shared emitter code**: `fel.hpp` builds the whole envelope itself; `hyphy-analysis.v3.schema.json` defines the shared contract but no shared builder implements it. The next method to go v3 will copy-paste FEL's ~220-line `to_modern_json` — the same drift mechanism §9.3 documents for legacy (RELAX already copy-pasted legacy).
- **Legacy is `--legacy-json` for FEL and the *only* option elsewhere** — `--json-format` should exist on all five CLIs before defaults flip; FEL's transition shape (dual emitters + flag + schema + validating test) is the template to replicate.
- **Format name**: the flag takes `modern|legacy`, the schema says `3.0.0`, the doc/proposal says `v2`/`v3` — pick one externally-visible vocabulary before consumers write parsers.

#### 9.4.3 Canonical skeleton (contract illustration)

```json
{
  "schema_version": 2,
  "method": "fel",
  "tool": { "name": "hyphy3", "version": "3.0.0", "build": "…" },
  "input": {
    "file_name": "…", "genetic_code": "Universal",
    "num_sequences": 17, "num_sites": 187,
    "partition_count": 1, "tree": "(…);"
  },
  "settings": { "pvalue_threshold": 0.1, "srv": true, "multiple_hits": "none" },
  "fits": {
    "nucleotide_gtr":   { "label": "Nucleotide GTR", "log_likelihood": -1234.5, "aic_c": 2500.1,
                          "estimated_parameters": { "theta_ac": 0.4, "…": "…" },
                          "rate_distributions": { "…": "…" } },
    "global_mg94xrev":  { "label": "Global MG94xREV", "log_likelihood": -2400.0, "aic_c": 5000.0 }
  },
  "test_results": { "p_value_threshold": 0.1, "tested": 187, "positive": 6, "lrt": null },
  "site_results": [
    { "site": 1, "alpha": 0.01, "beta": 1.1, "alpha_null": 0.2,
      "lrt": 2.1, "p_value": 0.15, "total_branch_length": 5.2 }
  ],
  "branch_results": [
    { "name": "HUMAN", "role": "test", "branch_length": 0.1,
      "baseline_omega": null, "rate_classes": 1, "lrt": null, "p_value": null }
  ],
  "timers": [ { "label": "model_fitting", "seconds": 1.2 }, { "label": "site_analysis", "seconds": 30.0 } ],
  "method_results": { }
}
```

Per-method `method_results` carries only what that method adds — FEL: nothing (empty); MEME: `{ "fel_comparison": true }` + extra site columns already in `site_results`; BUSTED: `{ "unconstrained_fit": "…", "constrained_fit": "…", "site_evidence_ratios": [...] }`; aBSREL: `{ "step_up": [...] }` + per-branch rate distributions already in `branch_results`.

Note that landed v3 chose different envelope spellings (`analysis.id`, `model_fits`, `statistical_tests`, `site_results`) than this illustration — the contract's intent is structural uniformity, not these exact names; whichever vocabulary wins should be adopted wholesale across all five methods rather than inventing a third.

#### 9.4.4 Remaining rules (contract deltas vs landed v3)

- **Self-describing envelope**: `method` + `schema_version` eliminate the inference heuristic entirely.
- **Record-oriented data**: `site_results`/`branch_results` are arrays of objects — values carry names; adding a field appends a key, breaking nothing positionally.
- **Stable machine ids as keys** (`nucleotide_gtr`, `global_mg94xrev`), display labels as *values*; `display_order` under an optional `ui` block or dropped; **no HTML entities in field names**.
- **No `{"0": …}` partition indirection** — emit the partition directly; if multi-partition ever ships, `partitions: []`.
- **Explicit status, never absence**: `role` on every branch; uncomputed quantities are `null`, never `-1e20` — emitter asserts `isfinite` before writing any number.
- **Backwards compatibility**: FEL's `to_json(format)` + `--legacy-json`/`--modern-json` landed — replicate on all methods; the committed `schemas/v3/*.json` should be extended per-method and **every method's output validated in test** — conformance as a test, not a convention. The same fixture should then exercise the cross-method contract (§9.4.1) as one parameterized check: each method, every envelope block present, every `site_results`/`branch_results` record satisfying the shared sub-schema. `tests/test_json_schema.py` is the right shape but still points at `../hyphy/hyphy-next/` (Phase 0 item 1) and module-skips without `jsonschema`.

---

## 10. Hardening Plan — Tests First, Then Refactor

> **Explicit ordering requirement**: before *any* of the refactoring below lands, the test suite must be fleshed out enough to catch regressions *introduced by the refactor itself*. Refactoring this codebase without a regression net is exactly how silent-failure bugs (the dominant failure class here) get *introduced*, not just missed. Do not refactor first and test after.

### Phase 0 — Test infrastructure prerequisites (must precede Phase 1+)

1. **Commit the benchmark data into the repo** (or a `data/` submodule): `cd2.fna`, `cd2.nwk`, `cd2_fitted.nwk`, `adh.nex`, `bglobin.nex`, `COXI.nex` + the `ground_truth/*.json` files the tests already reference. Until the data lives in-repo, every test is machine-specific and "works on my machine" is the only CI signal.
2. **Add a `test_data_path()` helper** (`tests/helpers.hpp`) that resolves `tests/data/…` relative to a compile-time `TEST_DATA_DIR` define passed from CMake (not cwd, not fallbacks-to-absolute-paths). Migrate every test to it; delete all `/Users/sergei` strings.
3. **Make tests skip, not fail, on missing data**: a doctest fixture/wrapper returning `WARN` + early return when a file is absent — a red suite caused by missing data hides real regressions.
4. **Replace `assert()` in `test_cpp_autograd.cpp`** with doctest `CHECK`s.
5. **Pin `WORKING_DIRECTORY`** on all `add_test`s and wire CTest so `ctest` is green from a clean build on a machine that isn't the author's.
6. **Store the 2.5 reference outputs** (`benchmarks/ground_truth/*.json` referenced by tests) as committed fixtures with a README noting provenance (HyPhy version, command line, dataset).

### Phase 1 — Regression net over current behavior (golden tests, no code changes)

Goal: capture *current* outputs as the baseline, so the refactors below can be checked for *behavior* regressions rather than re-derived correctness (the stats review already lists which current behaviors are wrong vs 2.5 — those get fixed later deliberately, not silently during refactor).

7. **End-to-end golden tests** for all four analyses on the committed data: run `run_fel`/`run_meme`/`run_busted`/`run_absrel` (or the analyzers directly), parse the emitted JSON, and `CHECK` a fixed set of scalars: global/baseline/adaptive lnL, LRTs, p-values, rate-class counts, per-branch omegas, site p-values for a handful of known sites. Tolerances: `1e-3`–`1e-2` lnL where optimizer noise exists, tighter where deterministic.
8. **Component-level goldens**: `Tree::from_newick` on a fixed set (balanced, unrooted, named-internal, comments, `{tags}`, scientific-notation lengths); `Alignment::from_fasta`/`from_nexus` on fixtures exercising every parser branch (interleaved NEXUS, quoted names, `.` chars, duplicate taxa, mixed case, gaps); `CF3x4` frequencies on 2–3 alignments checked against stored values.
9. **Likelihood unit tests**: `compute_gtr_log_likelihood`/`compute_mg94_log_likelihood`/`compute_inside_outside`/`compute_branch_length_gradients` on tiny hand-computable trees (2–3 taxa, 1–2 patterns) with closed-form expected values; a **finite-difference gradient check** against `compute_branch_length_gradients` (this is the cheapest way to catch broken adjoint math and should be a permanent test).
10. **Optimizer unit tests**: `Brent1D`, `NelderMead2D`, `NelderMeadND`, `SQUAREM` on analytic functions (quadratic, Rosenbrock, constrained simplex) — assert recovered optima and, critically, *convergence flags* once they're made truthful.
11. **CLI smoke tests**: run the built `hyphy3` binary on the committed data via CTest; assert exit codes and that JSON parses and contains required keys.
12. **Determinism check**: run FEL twice with `OMP_NUM_THREADS=4`, assert identical lnL/p-values — OpenMP reductions are float-order-sensitive; the test documents the expected tolerance (should be bitwise-stable if reductions are ordered the same way each run — verify and pin down).

### Phase 2 — Edge/negative tests for the §1/§2 failure surface

13. Malformed Newick → must throw (after fix): unbalanced parens, `:bad_length`, multi-root, duplicate leaves, quoted names.
14. Tree leaf absent from alignment + alignment taxon absent from tree → must throw/list mismatches (after fix).
15. `--threads abc`, `--rates 0`, `--pvalue -1`, `--branches bogus`, `--multiple-hits bogus` → clean exit-code-1 errors, not `abort()`.
16. Empty alignment, 1-taxon alignment, non-multiple-of-3 alignment, all-gap column → clean errors.
17. Zero-frequency nucleotide → GTR path behavior pinned (currently throws via eigendecomp; decide & pin).
18. Partial-ambiguity codons (`ATN`, `RAY`) → leaf vector must equal the union of compatible codons (after fix).
19. Deep/ladderized tree → `post_order` must not stack-overflow (after iterative rewrite).
20. NaN-injection test: an objective returning NaN to Nelder-Mead/L-BFGS → documented behavior (currently silent garbage; after fix, should propagate/converge-fail cleanly).

### Phase 3 — Refactors (each gated by Phase 1+2 staying green)

21. `cli_common.hpp` for the four apps (§3.1) — kills 4 copies of the same stoi/try-catch/tree-resolution bugs.
22. Unify the two `compute_*_log_likelihood` bodies + leaf-vector builder (§3.2); hoist `leaf_to_taxon`, `P_mix`, and `MG94Matrix` construction out of per-evaluation paths (§6) — measure, this is also a performance win.
23. Truthful optimizer results: `converged`/`iterations` populated from actual solver state; exceptions → `converged=false`, callers branch on it (§2.1). SQUAREM tracks best-so-far.
24. Named constants for the §3.3 literals, tied to 2.5 source citations.
25. Replace `-1e20` sentinel with a typed result/`std::optional`/NaN convention that distinguishes "likelihood zero" from "path not implemented" (§3.4).
26. Iterative post-order; Newick validation (balance, root uniqueness, length-parse errors → throw).
27. Taxa-set validation at analyzer construction: error on leaf∉alignment or taxon∉tree (§1.2).
28. Decide and delete dead code: `param_map.hpp`, `autograd/` (move to `experimental/` if retained), `types.hpp` universal helpers (§4).
29. Input hygiene: IUPAC-aware ambiguity leaf vectors; NEXUS `.` → first-taxon match (after deciding spec conformance); CLI arg validation via a tiny `parse_int_arg`/`parse_enum_arg` helper.
30. NaN hygiene: `isnan` checks at optimizer entry, in objective evaluation, and in `transition_matrix` outputs.
31. Build: `CMAKE_BUILD_TYPE`-respecting flags, `find_package(Eigen3)` fallback, `install()` rules, nanobind-missing warning.
32. Lift the v3 envelope out of `fel.hpp` into a shared emitter (`analysis`/`software`/`provenance`/`dataset`/`phylogeny`/`model_fits` assembly + `isfinite` guard), roll `to_modern_json` out to MEME/BUSTED/aBSREL/RELAX, settle the missing contract blocks (`branch_results`, `timers`, `test_results`, `method_results`) *first* (§9.4.2), and `$ref` the base schema from per-method schemas — before a second method copy-pastes the ~220-line emitter the way RELAX already copied legacy (§9.4.2).

### Phase 4 — Behavioral fixes (from the stats review; gated on the same net)

These change outputs deliberately; each needs its own updated goldens *with the 2.5 reference as the target*, done one at a time. The v3 rollout for the remaining four methods (§9.4.2) belongs here *after* item 32 — FEL demonstrates the transition shape; flip other methods' defaults only once each can emit both formats, retire legacy last. Remaining: aBSREL M≥3, BUSTED refinement iterations, FG/BG branch sets, optimized-null ER, AICc N convention unification, MEME `total_branch_length`, FEL null bound, ambiguity resolution. (`e1352b6` landed threshold-based scaling in both the post-order and inside-outside passes — it leaves this list and becomes a Phase 2 deep-tree fixture once confirmed correct.)

---

## Appendix: quick reference of highest-value fixes

| # | Fix | Why first |
|---|---|---|
| 1 | `find_taxon_index` miss → error (§1.2) | silent corruption of any real dataset |
| 2 | stoi/stod guards in CLIs (§1.1) | trivial crashes on every user's typo |
| 3 | `converged`/`iterations` truthfulness (§2.1) | everything downstream trusts these |
| 4 | Newick validation (§1.3) | silently wrong trees are uncatchable downstream |
| 5 | Golden tests (Phase 1) | gate everything else |
| 6 | `-1e20` sentinel → typed result (§3.4) | conflates 3 different failure modes |
| 7 | AICc `N>K+1` guard (§1.5) | NaN in JSON output |
| 8 | NEXUS `.` semantics (§1.4) | silent sequence corruption |
| 9 | `P_branches[root]` init/guard contract (§1.6) | latent UB |
| 10 | Hoist per-eval eigendecomps/leaf maps (§6) | performance without behavior change |
| 11 | v3 rollout to the 4 remaining methods + shared emitter (§9.4.2) | `analysis.id`/named data landed for FEL only; RELAX just added a 5th divergent legacy emitter |
