#pragma once

#include "hyphy/core/types.hpp"
#include "hyphy/core/genetic_code.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/opt/optimizer.hpp"
#include "hyphy/analyses/fel.hpp"
#include "hyphy/core/progress_bar.hpp"
#include "hyphy/opt/nelder_mead.hpp"
#include "hyphy/opt/squarem.hpp"
#include "nlohmann/json.hpp"
#include "hyphy/core/provenance.hpp"

#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace hyphy::analyses {

using namespace hyphy::core;
using namespace hyphy::opt;

struct BUSTEDErrorEvent {
    size_t site = 0;         // 1-based codon site
    std::string branch_name; // Node/branch name
    Scalar posterior = 0.0;  // Posterior probability P(sink | b, s)
    Scalar bayes_factor = 0.0; // Empirical Bayes factor for error sink at (b, s)
};

struct BUSTEDRateDistribution {
    std::vector<Scalar> omegas;
    std::vector<Scalar> weights;
    std::vector<Scalar> syn_rates;
    std::vector<Scalar> syn_weights;
    Scalar error_sink_weight = 0.0; // Contamination weight epsilon in [0, error_sink_max]
    std::vector<std::string> annotations; // e.g. "Collapsed rate class"
};

struct BUSTEDFit {
    Scalar log_likelihood = 0.0;
    Scalar aicc = 0.0;
    Scalar tree_scale = 1.0;
    size_t num_rate_classes = 3;
    BUSTEDRateDistribution test_distribution;
    std::vector<Scalar> branch_lengths; // Branch lengths per node id
    std::vector<Scalar> site_log_likelihoods;
    Scalar delta = 0.0;      // Double-hit rate parameter
    Scalar psi = 0.0;        // Triple-hit rate parameter
    Scalar frac_delta = 0.0;  // Fraction of 2-hit substitutions
    Scalar frac_psi = 0.0;    // Fraction of 3-hit substitutions
    Scalar error_sink_weight = 0.0; // Fitted contamination proportion epsilon
};

struct BUSTEDAutoDiagnostics {
    bool enabled = false;
    // Gate 1: Error Sink (Saturation Replacement Kernel)
    Scalar max_sink_surprisal_bits = 0.0; // max_{b,s} log2(L_sink(b,s) / L_0(s))
    Scalar max_sink_bf = 0.0;             // max_{b,s} Empirical Bayes Factor at baseline
    size_t sink_candidate_sites = 0;      // Branch-sites with > 8 bits saturation surprisal
    size_t sink_adjacent_clusters = 0;    // Same-branch adjacent pairs (|s1 - s2| <= 2) with high surprisal
    Scalar sink_delta_ll = 0.0;           // Log-likelihood gain of saturation error sink at baseline
    bool activated_error_sink = false;

    // Gate 2: Multi-Hits (Instantaneous 2-nt / 3-nt Substitutions)
    Scalar double_hit_score = 0.0;        // Quarantined first-order tangent score for delta
    Scalar triple_hit_score = 0.0;        // Quarantined first-order tangent score for psi
    Scalar double_hit_delta_ll = 0.0;     // Log-likelihood gain for Double hits (conditioned on Gate 1)
    Scalar triple_hit_delta_ll = 0.0;     // Log-likelihood gain for Triple hits (conditioned on Double)
    std::string selected_multiple_hits = "None";

    // Gate 3: Synonymous Rate Variation (SRV)
    Scalar srv_overdispersion_z = 0.0;    // Neyman-Scott / Lindsay synonymous dispersion Z-score
    Scalar srv_delta_ll = 0.0;            // 1-step SQUAREM SRV log-likelihood gain
    bool activated_srv = false;
    size_t selected_syn_rates = 1;

    // Gate 4: Omega Rate Classes (K)
    size_t selected_k = 3;
    std::string k_rationale;
};

struct BUSTEDSettings {
    bool full_auto = false;            // Enable 4-gate surprisal & score automatic model selector (BUSTED-Auto)
    size_t num_rate_classes = 3;       // Default K = 3
    bool srv = false;                  // Enable synonymous rate variation (BUSTED-S)
    size_t num_syn_rate_classes = 3;   // Default M = 3 for alpha
    bool auto_select_k = false;        // Automatically select K in {1, ..., max_k} via AICc
    size_t max_k = 3;                  // Max K to test in auto_select_k
    bool refine_branch_lengths = true; // Re-optimize individual branch lengths
    Scalar p_value_threshold = 0.05;
    std::string multiple_hits = "None"; // "None", "Double", "Double+Triple"
    bool error_sink = false;           // Enable Option A saturation/replacement error sink
    Scalar error_sink_max_weight = 0.01; // Upper bound on contamination proportion epsilon
    Scalar error_sink_bf_threshold = 100.0; // Bayes factor cutoff to flag branch-site errors
};

struct BUSTEDResult {
    BUSTEDSettings settings;
    BUSTEDAutoDiagnostics auto_diagnostics;
    size_t optimal_k = 3;
    std::vector<BUSTEDFit> k_null_fits; // Null model fits for K=1, 2, ...
    Scalar lrt = 0.0;
    Scalar p_value = 1.0;
    BUSTEDFit unconstrained;
    BUSTEDFit constrained;
    std::vector<Scalar> evidence_ratios; // Bayes factors per site for omega_K > 1
    std::vector<Scalar> site_error_bfs;  // Max error sink Bayes factor across branches per site
    std::vector<BUSTEDErrorEvent> flagged_error_events; // Branch-site pairs with BF_sink >= threshold
    double runtime_seconds = 0.0;
};

class BUSTEDAnalyzer {
public:
    Tree tree;
    Alignment aln;
    std::shared_ptr<const GeneticCode> code;
    MG94Parameters base_params;

    // Relative branch lengths tau_b (normalized from MG94 baseline)
    std::vector<Scalar> branch_tau;
    std::vector<bool> is_test_branch;
    std::vector<size_t> leaf_to_taxon;
    size_t num_test_branches = 0;

    // Precomputed off-diagonal saturation replacement kernel P_sink(i, j) = pi_j / (1 - pi_i) for i != j, 0 for i == j
    Matrix P_sink;
    Vector inv_one_minus_pi;

    // GTR baseline results
    bool has_gtr_fit = false;
    Scalar gtr_log_l = 0.0;
    Scalar gtr_aicc = 0.0;
    GTRParameters gtr_rates;
    std::unordered_map<std::string, Scalar> gtr_branch_lengths;

    // Global MG94 baseline results
    Scalar mg94_log_l = 0.0;
    Scalar mg94_omega = 1.0;
    Scalar mg94_tree_length = 0.0;

    BUSTEDAnalyzer(
        Tree t,
        Alignment a,
        MG94Parameters base_p = MG94Parameters{}
    ) : tree(std::move(t)), aln(std::move(a)), base_params(base_p) {
        code = aln.code ? aln.code : GeneticCode::universal();
        size_t num_nodes = tree.num_nodes();
        branch_tau.assign(num_nodes, 0.0);
        is_test_branch.assign(num_nodes, true); // Default: all branches are Test
        num_test_branches = num_nodes - 1; // Non-root branches
        leaf_to_taxon = LikelihoodEngine::build_leaf_to_taxon_map(tree, aln);
        build_sink_kernel();
        prepare_baseline_lengths();
    }

    void build_sink_kernel() {
        int S = code->num_sense_codons;
        P_sink = Matrix::Zero(S, S);
        inv_one_minus_pi = Vector::Zero(S);
        for (int i = 0; i < S; ++i) {
            Scalar denom = std::max(1e-12, 1.0 - aln.codon_frequencies_f3x4(i));
            inv_one_minus_pi(i) = 1.0 / denom;
            for (int j = 0; j < S; ++j) {
                if (i != j) {
                    P_sink(i, j) = aln.codon_frequencies_f3x4(j) * inv_one_minus_pi(i);
                }
            }
        }
    }

    // O(S) evaluation of V^T * P_sink * D using the rank-1 off-diagonal identity
    inline Scalar eval_sink_bilinear(const Vector& V, const Vector& D) const {
        Scalar pi_dot_D = aln.codon_frequencies_f3x4.dot(D);
        int S = static_cast<int>(V.size());
        Scalar acc = 0.0;
        for (int i = 0; i < S; ++i) {
            Scalar p_sink_D_i = (pi_dot_D - aln.codon_frequencies_f3x4(i) * D(i)) * inv_one_minus_pi(i);
            acc += V(i) * std::max(0.0, p_sink_D_i);
        }
        return acc;
    }

    static BUSTEDAnalyzer create_and_fit(
        Tree raw_tree,
        Alignment aln
    ) {
        GTRFitter gtr_fitter(raw_tree, aln);
        auto gtr_res = gtr_fitter.fit();

        MG94Parameters base_p;
        base_p.theta_AC = gtr_res.params.theta_AC;
        base_p.theta_AT = gtr_res.params.theta_AT;
        base_p.theta_CG = gtr_res.params.theta_CG;
        base_p.theta_CT = gtr_res.params.theta_CT;
        base_p.theta_GT = gtr_res.params.theta_GT;

        MG94Fitter mg_fitter(gtr_res.tree, aln);
        auto mg_res = mg_fitter.fit_omega_and_scale(1.0, base_p);
        base_p.alpha = 1.0;
        base_p.beta = mg_res.x_opt(0);

        BUSTEDAnalyzer analyzer(mg_res.tree, std::move(aln), base_p);
        analyzer.has_gtr_fit = true;
        analyzer.gtr_log_l = gtr_res.log_likelihood;
        analyzer.gtr_aicc = gtr_res.aicc;
        analyzer.gtr_rates = gtr_res.params;
        for (const auto& node : gtr_res.tree.nodes) {
            if (node.id != gtr_res.tree.root_id) {
                analyzer.gtr_branch_lengths[node.name] = node.branch_length;
            }
        }

        analyzer.mg94_log_l = mg_res.log_likelihood;
        analyzer.mg94_omega = mg_res.x_opt(0);
        return analyzer;
    }

    void prepare_baseline_lengths() {
        MG94Matrix mat;
        mat.update(base_params, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
        Scalar Q_scale = mat.scale_factor;
        Scalar conversion_factor = (Q_scale > 1e-12) ? (3.0 / Q_scale) : 1.0;

        mg94_tree_length = 0.0;
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                Scalar scaled_len = node.branch_length * conversion_factor;
                branch_tau[node.id] = scaled_len;
                mg94_tree_length += scaled_len;
            }
        }
    }

    // Evaluates tree log-likelihood for arbitrary K rate categories, optional SRV (M classes), optional saturation error sink (eps), and custom branch lengths
    Scalar evaluate_log_l(
        Scalar s,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        std::vector<Scalar>* out_pattern_ll = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0
    ) const {
        size_t num_nodes = tree.num_nodes();
        size_t num_patterns = aln.patterns.size();
        int S = code->num_sense_codons;
        size_t K = omegas.size();
        size_t M = syn_rates.size();
        Scalar bio_scale = 1.0 - eps_sink;

        // Build K rate matrices
        std::vector<MG94Matrix> mg(K);
        for (size_t k = 0; k < K; ++k) {
            MG94Parameters p = base_params;
            p.alpha = 1.0;
            p.beta = omegas[k];
            p.delta = delta;
            p.psi = psi;
            mg[k].update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
        }

        // Precompute branch transition matrices P_mix[m][node_id]
        std::vector<std::vector<Matrix>> P_mix(M, std::vector<Matrix>(num_nodes));
        for (size_t m = 0; m < M; ++m) {
            Scalar alpha_m = syn_rates[m];
            for (const auto& node : tree.nodes) {
                if (node.id != tree.root_id) {
                    Scalar bl = custom_branch_lengths ? (*custom_branch_lengths)[node.id] : (s * branch_tau[node.id]);
                    P_mix[m][node.id] = Matrix::Zero(S, S);
                    Scalar use_bio = (eps_sink > 0.0 && is_test_branch[node.id]) ? bio_scale : 1.0;
                    for (size_t k = 0; k < K; ++k) {
                        P_mix[m][node.id] += (use_bio * weights[k]) * mg[k].transition_matrix(alpha_m * bl);
                    }
                    if (eps_sink > 0.0 && is_test_branch[node.id]) {
                        P_mix[m][node.id].noalias() += eps_sink * P_sink;
                    }
                }
            }
        }

        if (out_pattern_ll) {
            out_pattern_ll->assign(num_patterns, 0.0);
        }

        Scalar total_log_l = 0.0;
        #pragma omp parallel
        {
            std::vector<Vector> node_L(num_nodes, Vector::Zero(S));
            std::vector<double> node_log_scale(num_nodes, 0.0);
            #pragma omp for reduction(+:total_log_l) schedule(dynamic)
            for (size_t p = 0; p < num_patterns; ++p) {
                const auto& pattern = aln.patterns[p];
                Scalar pat_L = 0.0;

                for (size_t m = 0; m < M; ++m) {
                    for (int32_t node_id : tree.post_order) {
                        const auto& node = tree.nodes[node_id];
                        if (node.is_leaf) {
                            size_t t_idx = leaf_to_taxon[node_id];
                            if (t_idx != static_cast<size_t>(-1)) {
                                int8_t state = pattern.states[t_idx];
                                if (state >= 0 && state < S) {
                                    node_L[node_id].setZero();
                                    node_L[node_id](state) = 1.0;
                                } else {
                                    node_L[node_id].setOnes();
                                }
                            } else {
                                node_L[node_id].setOnes();
                            }
                            node_log_scale[node_id] = 0.0;
                        } else {
                            double child_scale_sum = 0.0;
                            tree.eval_internal_node(node_id, P_mix[m], node_L, node_log_scale, node_L[node_id], child_scale_sum);
                            node_log_scale[node_id] = child_scale_sum;
                        }
                    }
                    Scalar cond_L = aln.codon_frequencies_f3x4.dot(node_L[tree.root_id]);
                    if (cond_L > 0.0 && node_log_scale[tree.root_id] != 0.0) {
                        cond_L *= std::exp(node_log_scale[tree.root_id]);
                    }
                    pat_L += syn_weights[m] * cond_L;
                }

                Scalar pll = (pat_L > 0.0) ? std::log(pat_L) : -1e20;
                total_log_l += pattern.weight * pll;
                if (out_pattern_ll) {
                    (*out_pattern_ll)[p] = pll;
                }
            }
        }
        return total_log_l;
    }

    // Expectation Step: Computes new mixture weights (and optional saturation sink weight eps_sink) via inside-outside traversal
    std::vector<Scalar> compute_expected_weights(
        Scalar s,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0,
        Scalar* out_new_eps_sink = nullptr,
        Scalar max_eps_sink = 0.01
    ) const {
        size_t K = omegas.size();
        if (K <= 1 && !out_new_eps_sink) return {1.0};

        size_t M = syn_rates.size();
        size_t num_nodes = tree.num_nodes();
        size_t num_patterns = aln.patterns.size();
        int S = code->num_sense_codons;
        Scalar bio_scale = (out_new_eps_sink != nullptr) ? (1.0 - eps_sink) : 1.0;

        std::vector<std::vector<std::vector<Matrix>>> P_comp(M, std::vector<std::vector<Matrix>>(K, std::vector<Matrix>(num_nodes)));
        std::vector<std::vector<Matrix>> P_mix(M, std::vector<Matrix>(num_nodes));
        for (size_t k = 0; k < K; ++k) {
            MG94Parameters p = base_params;
            p.alpha = 1.0;
            p.beta = omegas[k];
            p.delta = delta;
            p.psi = psi;
            MG94Matrix mg;
            mg.update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
            for (size_t m = 0; m < M; ++m) {
                Scalar alpha_m = syn_rates[m];
                for (const auto& node : tree.nodes) {
                    if (node.id != tree.root_id) {
                        Scalar bl = custom_branch_lengths ? (*custom_branch_lengths)[node.id] : (s * branch_tau[node.id]);
                        P_comp[m][k][node.id] = mg.transition_matrix(alpha_m * bl);
                    }
                }
            }
        }

        for (size_t m = 0; m < M; ++m) {
            for (const auto& node : tree.nodes) {
                if (node.id != tree.root_id) {
                    P_mix[m][node.id] = Matrix::Zero(S, S);
                    Scalar use_bio = (out_new_eps_sink != nullptr && is_test_branch[node.id]) ? bio_scale : 1.0;
                    for (size_t k = 0; k < K; ++k) {
                        P_mix[m][node.id] += (use_bio * weights[k]) * P_comp[m][k][node.id];
                    }
                    if (out_new_eps_sink != nullptr && is_test_branch[node.id]) {
                        P_mix[m][node.id].noalias() += eps_sink * P_sink;
                    }
                }
            }
        }

        Vector weight_accum = Vector::Zero(K);
        Scalar sink_accum = 0.0;
        Scalar total_pattern_weight = 0.0;

        #pragma omp parallel
        {
            Vector local_accum = Vector::Zero(K);
            Scalar local_sink_accum = 0.0;
            Scalar local_pat_wt = 0.0;
            #pragma omp for schedule(dynamic)
            for (size_t p = 0; p < num_patterns; ++p) {
                const auto& pattern = aln.patterns[p];
                Scalar pat_L = 0.0;
                std::vector<LikelihoodEngine::InsideOutsideResult> io_m(M);

                for (size_t m = 0; m < M; ++m) {
                    io_m[m] = LikelihoodEngine::compute_inside_outside(
                        tree, pattern, leaf_to_taxon, P_mix[m], aln.codon_frequencies_f3x4, S
                    );
                    pat_L += syn_weights[m] * io_m[m].likelihood;
                }

                if (pat_L > 0.0) {
                    Scalar inv_L = pattern.weight / pat_L;
                    for (const auto& node : tree.nodes) {
                        if (node.id != tree.root_id && is_test_branch[node.id]) {
                            int nid = node.id;
                            for (size_t k = 0; k < K; ++k) {
                                Scalar num = 0.0;
                                for (size_t m = 0; m < M; ++m) {
                                    num += syn_weights[m] * io_m[m].V[nid].dot(P_comp[m][k][nid] * io_m[m].D[nid]);
                                }
                                local_accum(k) += (bio_scale * weights[k]) * inv_L * num;
                            }
                            if (out_new_eps_sink != nullptr) {
                                Scalar sink_num = 0.0;
                                for (size_t m = 0; m < M; ++m) {
                                    sink_num += syn_weights[m] * eval_sink_bilinear(io_m[m].V[nid], io_m[m].D[nid]);
                                }
                                local_sink_accum += eps_sink * inv_L * sink_num;
                            }
                            local_pat_wt += pattern.weight;
                        }
                    }
                }
            }
            #pragma omp critical
            {
                weight_accum += local_accum;
                sink_accum += local_sink_accum;
                total_pattern_weight += local_pat_wt;
            }
        }

        std::vector<Scalar> new_weights(K);
        if (total_pattern_weight > 0.0) {
            Scalar sum_w = 0.0;
            for (size_t k = 0; k < K; ++k) {
                new_weights[k] = std::clamp(weight_accum(k) / total_pattern_weight, 1e-6, 1.0 - 1e-6);
                sum_w += new_weights[k];
            }
            for (size_t k = 0; k < K; ++k) {
                new_weights[k] /= sum_w;
            }
            if (out_new_eps_sink != nullptr) {
                Scalar raw_eps = sink_accum / total_pattern_weight;
                *out_new_eps_sink = std::clamp(raw_eps, 1e-7, max_eps_sink);
            }
        } else {
            new_weights = weights;
            if (out_new_eps_sink != nullptr) {
                *out_new_eps_sink = eps_sink;
            }
        }
        return new_weights;
    }

    // Expectation Step: Computes new synonymous rate weights via site posteriors
    std::vector<Scalar> compute_expected_syn_weights(
        Scalar s,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates,
        const std::vector<Scalar>& syn_weights,
        const std::vector<Scalar>* custom_branch_lengths = nullptr,
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0
    ) const {
        size_t M = syn_rates.size();
        if (M <= 1) return {1.0};

        size_t num_nodes = tree.num_nodes();
        size_t num_patterns = aln.patterns.size();
        int S = code->num_sense_codons;
        size_t K = omegas.size();
        Scalar bio_scale = 1.0 - eps_sink;

        std::vector<MG94Matrix> mg(K);
        for (size_t k = 0; k < K; ++k) {
            MG94Parameters p = base_params;
            p.alpha = 1.0;
            p.beta = omegas[k];
            p.delta = delta;
            p.psi = psi;
            mg[k].update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
        }

        std::vector<std::vector<Matrix>> P_mix(M, std::vector<Matrix>(num_nodes));
        for (size_t m = 0; m < M; ++m) {
            Scalar alpha_m = syn_rates[m];
            for (const auto& node : tree.nodes) {
                if (node.id != tree.root_id) {
                    Scalar bl = custom_branch_lengths ? (*custom_branch_lengths)[node.id] : (s * branch_tau[node.id]);
                    P_mix[m][node.id] = Matrix::Zero(S, S);
                    Scalar use_bio = (eps_sink > 0.0 && is_test_branch[node.id]) ? bio_scale : 1.0;
                    for (size_t k = 0; k < K; ++k) {
                        P_mix[m][node.id] += (use_bio * weights[k]) * mg[k].transition_matrix(alpha_m * bl);
                    }
                    if (eps_sink > 0.0 && is_test_branch[node.id]) {
                        P_mix[m][node.id].noalias() += eps_sink * P_sink;
                    }
                }
            }
        }

        Vector syn_accum = Vector::Zero(M);
        Scalar total_pattern_weight = 0.0;

        #pragma omp parallel
        {
            Vector local_accum = Vector::Zero(M);
            Scalar local_pat_wt = 0.0;
            std::vector<Vector> node_L(num_nodes, Vector::Zero(S));

            #pragma omp for schedule(dynamic)
            for (size_t p = 0; p < num_patterns; ++p) {
                const auto& pattern = aln.patterns[p];
                Scalar pat_L = 0.0;
                std::vector<Scalar> cond_L(M, 0.0);

                for (size_t m = 0; m < M; ++m) {
                    for (int32_t node_id : tree.post_order) {
                        const auto& node = tree.nodes[node_id];
                        if (node.is_leaf) {
                            size_t t_idx = leaf_to_taxon[node_id];
                            if (t_idx != static_cast<size_t>(-1)) {
                                int8_t state = pattern.states[t_idx];
                                if (state >= 0 && state < S) {
                                    node_L[node_id].setZero();
                                    node_L[node_id](state) = 1.0;
                                } else {
                                    node_L[node_id].setOnes();
                                }
                            } else {
                                node_L[node_id].setOnes();
                            }
                        } else {
                            node_L[node_id].setOnes();
                            for (int32_t child_id : node.children) {
                                Vector child_msg = P_mix[m][child_id] * node_L[child_id];
                                node_L[node_id] = node_L[node_id].cwiseProduct(child_msg);
                            }
                        }
                    }
                    cond_L[m] = aln.codon_frequencies_f3x4.dot(node_L[tree.root_id]);
                    pat_L += syn_weights[m] * cond_L[m];
                }

                if (pat_L > 0.0) {
                    for (size_t m = 0; m < M; ++m) {
                        Scalar rho_m = (syn_weights[m] * cond_L[m]) / pat_L;
                        local_accum(m) += pattern.weight * rho_m;
                    }
                    local_pat_wt += pattern.weight;
                }
            }

            #pragma omp critical
            {
                syn_accum += local_accum;
                total_pattern_weight += local_pat_wt;
            }
        }

        std::vector<Scalar> new_syn_weights(M);
        if (total_pattern_weight > 0.0) {
            Scalar sum_w = 0.0;
            for (size_t m = 0; m < M; ++m) {
                new_syn_weights[m] = std::clamp(syn_accum(m) / total_pattern_weight, 1e-6, 1.0 - 1e-6);
                sum_w += new_syn_weights[m];
            }
            for (size_t m = 0; m < M; ++m) {
                new_syn_weights[m] /= sum_w;
            }
        } else {
            new_syn_weights = syn_weights;
        }
        return new_syn_weights;
    }

    // Evaluates log-likelihood and exact analytical branch gradients d ln L / d t_b
    // under the BUSTED mixture model using Inside-Outside adjoints (with optional SRV and error sink)
    std::pair<Scalar, std::vector<Scalar>> compute_busted_branch_gradients(
        const std::vector<Scalar>& branch_lengths,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        Scalar eps_sink = 0.0
    ) const {
        size_t num_nodes = tree.num_nodes();
        size_t num_patterns = aln.patterns.size();
        int S = code->num_sense_codons;
        size_t K = omegas.size();
        size_t M = syn_rates.size();
        Scalar bio_scale = 1.0 - eps_sink;

        // 1. Build K rate matrices
        std::vector<MG94Matrix> mg(K);
        for (size_t k = 0; k < K; ++k) {
            MG94Parameters p = base_params;
            p.alpha = 1.0;
            p.beta = omegas[k];
            p.delta = delta;
            p.psi = psi;
            mg[k].update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
        }

        // 2. Precompute P_mix[m] and dP_mix[m] for each synonymous rate category and branch
        // Note: d/dt_b (eps_sink * P_sink) == 0 because the saturation replacement kernel is branch-length invariant!
        std::vector<std::vector<Matrix>> P_mix(M, std::vector<Matrix>(num_nodes));
        std::vector<std::vector<Matrix>> dP_mix(M, std::vector<Matrix>(num_nodes));
        for (size_t m = 0; m < M; ++m) {
            Scalar alpha_m = syn_rates[m];
            for (const auto& node : tree.nodes) {
                if (node.id != tree.root_id) {
                    int nid = node.id;
                    Scalar bl = branch_lengths[nid];
                    P_mix[m][nid] = Matrix::Zero(S, S);
                    dP_mix[m][nid] = Matrix::Zero(S, S);
                    Scalar use_bio = (eps_sink > 0.0 && is_test_branch[nid]) ? bio_scale : 1.0;
                    for (size_t k = 0; k < K; ++k) {
                        Matrix P_k = mg[k].transition_matrix(alpha_m * bl);
                        P_mix[m][nid] += (use_bio * weights[k]) * P_k;
                        dP_mix[m][nid] += (use_bio * weights[k]) * alpha_m * (mg[k].Q * P_k);
                    }
                    if (eps_sink > 0.0 && is_test_branch[nid]) {
                        P_mix[m][nid].noalias() += eps_sink * P_sink;
                    }
                }
            }
        }

        // 3. Parallel inside-outside traversal
        Scalar total_log_l = 0.0;
        std::vector<Scalar> grad_b(num_nodes, 0.0);

        #pragma omp parallel
        {
            std::vector<Scalar> local_grad(num_nodes, 0.0);
            Scalar local_ll = 0.0;

            #pragma omp for schedule(dynamic)
            for (size_t p = 0; p < num_patterns; ++p) {
                const auto& pattern = aln.patterns[p];
                Scalar pat_L = 0.0;
                std::vector<LikelihoodEngine::InsideOutsideResult> io_m(M);

                for (size_t m = 0; m < M; ++m) {
                    io_m[m] = LikelihoodEngine::compute_inside_outside(
                        tree, pattern, leaf_to_taxon, P_mix[m], aln.codon_frequencies_f3x4, S
                    );
                    pat_L += syn_weights[m] * io_m[m].likelihood;
                }

                if (pat_L > 0.0) {
                    local_ll += pattern.weight * std::log(pat_L);
                    Scalar inv_L = pattern.weight / pat_L;
                    for (const auto& node : tree.nodes) {
                        if (node.id != tree.root_id) {
                            int32_t vid = node.id;
                            Scalar dL_dt = 0.0;
                            for (size_t m = 0; m < M; ++m) {
                                dL_dt += syn_weights[m] * io_m[m].V[vid].dot(dP_mix[m][vid] * io_m[m].D[vid]);
                            }
                            local_grad[vid] += inv_L * dL_dt;
                        }
                    }
                }
            }

            #pragma omp critical
            {
                total_log_l += local_ll;
                for (size_t i = 0; i < num_nodes; ++i) {
                    grad_b[i] += local_grad[i];
                }
            }
        }

        return {total_log_l, grad_b};
    }

    // High-performance joint branch length refinement via exact Inside-Outside adjoint gradients and L-BFGS
    std::vector<Scalar> refine_branch_lengths(
        const std::vector<Scalar>& init_bl,
        const std::vector<Scalar>& omegas,
        const std::vector<Scalar>& weights,
        const std::vector<Scalar>& syn_rates = {1.0},
        const std::vector<Scalar>& syn_weights = {1.0},
        Scalar delta = 0.0,
        Scalar psi = 0.0,
        int max_iters = 10,
        Scalar eps_sink = 0.0
    ) const {
        size_t num_nodes = tree.num_nodes();
        std::vector<Scalar> bl = init_bl;
        for (size_t i = 0; i < num_nodes; ++i) {
            if (i != static_cast<size_t>(tree.root_id)) {
                bl[i] = std::clamp(bl[i], 1e-4, 10.0);
            }
        }

        // L-BFGS state
        const size_t m_history = 5;
        std::vector<std::vector<Scalar>> s_hist;
        std::vector<std::vector<Scalar>> y_hist;
        std::vector<Scalar> rho_hist;

        auto [curr_ll, curr_grad] = compute_busted_branch_gradients(bl, omegas, weights, syn_rates, syn_weights, delta, psi, eps_sink);

        Scalar best_ll = curr_ll;
        std::vector<Scalar> best_bl = bl;

        for (int iter = 0; iter < max_iters; ++iter) {
            // Check convergence
            Scalar max_g = 0.0;
            for (size_t i = 0; i < num_nodes; ++i) {
                if (i != static_cast<size_t>(tree.root_id)) {
                    max_g = std::max(max_g, std::abs(curr_grad[i]));
                }
            }
            if (max_g < 1e-3) break;

            // Two-loop L-BFGS recursion to compute search direction d = H * grad
            std::vector<Scalar> q = curr_grad;
            std::vector<Scalar> alpha(s_hist.size());

            for (int j = static_cast<int>(s_hist.size()) - 1; j >= 0; --j) {
                Scalar s_dot_q = 0.0;
                for (size_t i = 0; i < num_nodes; ++i) {
                    s_dot_q += s_hist[j][i] * q[i];
                }
                alpha[j] = rho_hist[j] * s_dot_q;
                for (size_t i = 0; i < num_nodes; ++i) {
                    q[i] -= alpha[j] * y_hist[j][i];
                }
            }

            // Initial Hessian scaling H0 = gamma * I
            Scalar gamma = 1.0;
            if (!s_hist.empty()) {
                Scalar s_dot_y = 0.0;
                Scalar y_dot_y = 0.0;
                const auto& s_last = s_hist.back();
                const auto& y_last = y_hist.back();
                for (size_t i = 0; i < num_nodes; ++i) {
                    s_dot_y += s_last[i] * y_last[i];
                    y_dot_y += y_last[i] * y_last[i];
                }
                if (y_dot_y > 1e-12) {
                    gamma = std::clamp(s_dot_y / y_dot_y, 1e-4, 10.0);
                }
            }

            std::vector<Scalar> r(num_nodes);
            for (size_t i = 0; i < num_nodes; ++i) {
                r[i] = gamma * q[i];
            }

            for (size_t j = 0; j < s_hist.size(); ++j) {
                Scalar y_dot_r = 0.0;
                for (size_t i = 0; i < num_nodes; ++i) {
                    y_dot_r += y_hist[j][i] * r[i];
                }
                Scalar beta = rho_hist[j] * y_dot_r;
                for (size_t i = 0; i < num_nodes; ++i) {
                    r[i] += s_hist[j][i] * (alpha[j] - beta);
                }
            }

            // Direction d (for maximization, d points along r)
            std::vector<Scalar> d = r;
            d[tree.root_id] = 0.0;

            // Ensure ascent direction (d . grad > 0)
            Scalar dir_dot_grad = 0.0;
            for (size_t i = 0; i < num_nodes; ++i) {
                if (i != static_cast<size_t>(tree.root_id)) {
                    dir_dot_grad += d[i] * curr_grad[i];
                }
            }
            if (dir_dot_grad <= 0.0) {
                d = curr_grad;
                d[tree.root_id] = 0.0;
                dir_dot_grad = 0.0;
                for (size_t i = 0; i < num_nodes; ++i) {
                    if (i != static_cast<size_t>(tree.root_id)) {
                        dir_dot_grad += d[i] * curr_grad[i];
                    }
                }
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
            }

            // Backtracking Armijo line search
            Scalar step_size = 1.0;
            std::vector<Scalar> new_bl(num_nodes);
            Scalar new_ll = curr_ll;
            std::vector<Scalar> new_grad;

            bool line_search_ok = false;
            for (int ls = 0; ls < 8; ++ls) {
                for (size_t i = 0; i < num_nodes; ++i) {
                    if (i != static_cast<size_t>(tree.root_id)) {
                        new_bl[i] = std::clamp(bl[i] + step_size * d[i], 1e-4, 10.0);
                    } else {
                        new_bl[i] = 0.0;
                    }
                }
                auto [test_ll, test_grad] = compute_busted_branch_gradients(new_bl, omegas, weights, syn_rates, syn_weights, delta, psi, eps_sink);
                if (test_ll > curr_ll + 1e-4 * step_size * dir_dot_grad) {
                    new_ll = test_ll;
                    new_grad = test_grad;
                    line_search_ok = true;
                    break;
                }
                step_size *= 0.5;
            }

            if (!line_search_ok) {
                break;
            }

            // Update L-BFGS history
            std::vector<Scalar> s_k(num_nodes);
            std::vector<Scalar> y_k(num_nodes);
            Scalar s_dot_y = 0.0;
            for (size_t i = 0; i < num_nodes; ++i) {
                s_k[i] = new_bl[i] - bl[i];
                y_k[i] = curr_grad[i] - new_grad[i]; // for maximization
                s_dot_y += s_k[i] * y_k[i];
            }

            if (s_dot_y > 1e-8) {
                if (s_hist.size() >= m_history) {
                    s_hist.erase(s_hist.begin());
                    y_hist.erase(y_hist.begin());
                    rho_hist.erase(rho_hist.begin());
                }
                s_hist.push_back(s_k);
                y_hist.push_back(y_k);
                rho_hist.push_back(1.0 / s_dot_y);
            }

            bl = new_bl;
            curr_ll = new_ll;
            curr_grad = new_grad;

            if (curr_ll > best_ll) {
                best_ll = curr_ll;
                best_bl = bl;
            }
        }

        return best_bl;
    }

    // Fits BUSTED model for specified number of rate classes K in {1, 2, 3}, optional SRV, and optional saturation error sink
    BUSTEDFit fit_model(
        bool is_constrained,
        size_t K = 3,
        bool refine_branches = true,
        bool srv = false,
        size_t num_syn_rates = 3,
        const std::string& multiple_hits = "None",
        ProgressBar* pbar = nullptr,
        const std::string& model_desc = "",
        bool use_error_sink = false,
        Scalar max_eps_sink = 0.01
    ) const {
        BUSTEDFit fit;
        fit.num_rate_classes = K;
        Scalar s = 1.0;

        bool allow_double = (multiple_hits == "Double" || multiple_hits == "Double+Triple");
        bool allow_triple = (multiple_hits == "Double+Triple");
        Scalar delta = 0.0;
        Scalar psi = 0.0;
        Scalar eps_sink = use_error_sink ? std::min(0.002, max_eps_sink * 0.5) : 0.0;

        // Initialize rates and weights
        std::vector<Scalar> omegas(K);
        std::vector<Scalar> weights(K);

        if (K == 1) {
            weights[0] = 1.0;
            omegas[0] = is_constrained ? std::min(0.5, mg94_omega) : mg94_omega;
        } else if (K == 2) {
            weights = {0.75, 0.25};
            if (is_constrained) {
                omegas = {0.10, 1.0};
            } else {
                omegas = {0.10, 3.0};
            }
        } else {
            // Default K = 3
            if (is_constrained) {
                omegas = {0.05, 0.50, 1.0};
                weights = {0.45, 0.45, 0.10};
            } else {
                omegas = {0.05, 0.50, 4.0};
                weights = {0.50, 0.40, 0.10};
            }
        }

        // Initialize SRV synonymous rate distribution
        size_t M = srv ? num_syn_rates : 1;
        std::vector<Scalar> syn_rates(M, 1.0);
        std::vector<Scalar> syn_weights(M, 1.0 / M);

        if (srv && M == 2) {
            syn_rates = {0.35, 1.65};
            syn_weights = {0.50, 0.50};
        } else if (srv && M == 3) {
            syn_rates = {0.20, 0.60, 2.20};
            syn_weights = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        }

        if (allow_double) {
            Scalar best_grid_ll = -1e20;
            std::vector<Scalar> d_cands = {0.0, 0.02, 0.08, 0.20};
            std::vector<Scalar> p_cands = allow_triple ? std::vector<Scalar>{0.0, 0.02, 0.05} : std::vector<Scalar>{0.0};
            for (Scalar d_c : d_cands) {
                for (Scalar p_c : p_cands) {
                    Scalar cand_ll = evaluate_log_l(s, omegas, weights, syn_rates, syn_weights, nullptr, nullptr, d_c, p_c, eps_sink);
                    if (cand_ll > best_grid_ll) {
                        best_grid_ll = cand_ll;
                        delta = d_c;
                        psi = p_c;
                    }
                }
            }
        }

        // ECM Optimization Loop
        int max_ecm = (K == 1 && !srv && !allow_double && !use_error_sink) ? 1 : 6;
        for (int ecm_iter = 0; ecm_iter < max_ecm; ++ecm_iter) {
            Scalar* eps_ptr = use_error_sink ? &eps_sink : nullptr;

            // Block 1: SQUAREM acceleration for omega weights (and penalized EM update for eps_sink)
            if (K == 1 && use_error_sink) {
                weights = compute_expected_weights(s, omegas, weights, syn_rates, syn_weights, nullptr, delta, psi, eps_sink, eps_ptr, max_eps_sink);
            } else if (K == 2) {
                for (int sq = 0; sq < 2; ++sq) {
                    Scalar eps0 = eps_sink;
                    auto w0 = weights;
                    Scalar eps1 = eps0;
                    auto w1 = compute_expected_weights(s, omegas, w0, syn_rates, syn_weights, nullptr, delta, psi, eps0, use_error_sink ? &eps1 : nullptr, max_eps_sink);
                    Scalar eps2 = eps1;
                    auto w2 = compute_expected_weights(s, omegas, w1, syn_rates, syn_weights, nullptr, delta, psi, eps1, use_error_sink ? &eps2 : nullptr, max_eps_sink);
                    eps_sink = eps2;
                    Scalar r = w1[0] - w0[0];
                    Scalar v = (w2[0] - w1[0]) - r;
                    if (std::abs(v) > 1e-12) {
                        Scalar step = -std::abs(r) / std::abs(v);
                        Scalar p_acc = std::clamp(w0[0] - 2.0 * step * r + step * step * v, 1e-5, 1.0 - 1e-5);
                        std::vector<Scalar> w_acc = {p_acc, 1.0 - p_acc};
                        Scalar eps_stab = eps_sink;
                        auto w_stab = compute_expected_weights(s, omegas, w_acc, syn_rates, syn_weights, nullptr, delta, psi, eps_sink, use_error_sink ? &eps_stab : nullptr, max_eps_sink);
                        if (evaluate_log_l(s, omegas, w_stab, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_stab) >=
                            evaluate_log_l(s, omegas, w2, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps2)) {
                            weights = w_stab;
                            eps_sink = eps_stab;
                        } else {
                            weights = w2;
                        }
                    } else {
                        weights = w2;
                    }
                }
            } else if (K == 3) {
                for (int sq = 0; sq < 2; ++sq) {
                    Scalar eps0 = eps_sink;
                    auto w0 = weights;
                    Scalar eps1 = eps0;
                    auto w1 = compute_expected_weights(s, omegas, w0, syn_rates, syn_weights, nullptr, delta, psi, eps0, use_error_sink ? &eps1 : nullptr, max_eps_sink);
                    Scalar eps2 = eps1;
                    auto w2 = compute_expected_weights(s, omegas, w1, syn_rates, syn_weights, nullptr, delta, psi, eps1, use_error_sink ? &eps2 : nullptr, max_eps_sink);
                    eps_sink = eps2;
                    Scalar r0 = w1[0] - w0[0], r1 = w1[1] - w0[1];
                    Scalar v0 = (w2[0] - w1[0]) - r0, v1 = (w2[1] - w1[1]) - r1;
                    Scalar r2 = r0 * r0 + r1 * r1;
                    Scalar v2 = v0 * v0 + v1 * v1;
                    if (v2 > 1e-12) {
                        Scalar step = -std::sqrt(r2 / v2);
                        Scalar p1_acc = std::clamp(w0[0] - 2.0 * step * r0 + step * step * v0, 1e-5, 1.0 - 1e-5);
                        Scalar p2_acc = std::clamp(w0[1] - 2.0 * step * r1 + step * step * v1, 1e-5, 1.0 - 1e-5);
                        if (p1_acc + p2_acc > 1.0 - 1e-5) {
                            Scalar sum_p = p1_acc + p2_acc;
                            p1_acc = (p1_acc / sum_p) * 0.99;
                            p2_acc = (p2_acc / sum_p) * 0.99;
                        }
                        Scalar p3_acc = 1.0 - p1_acc - p2_acc;
                        std::vector<Scalar> w_acc = {p1_acc, p2_acc, p3_acc};
                        Scalar eps_stab = eps_sink;
                        auto w_stab = compute_expected_weights(s, omegas, w_acc, syn_rates, syn_weights, nullptr, delta, psi, eps_sink, use_error_sink ? &eps_stab : nullptr, max_eps_sink);
                        if (evaluate_log_l(s, omegas, w_stab, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_stab) >=
                            evaluate_log_l(s, omegas, w2, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps2)) {
                            weights = w_stab;
                            eps_sink = eps_stab;
                        } else {
                            weights = w2;
                        }
                    } else {
                        weights = w2;
                    }
                }
            }

            // Block 1b: SQUAREM acceleration for synonymous weights (if SRV)
            if (srv && M >= 2) {
                for (int sq = 0; sq < 2; ++sq) {
                    auto q0 = syn_weights;
                    auto q1 = compute_expected_syn_weights(s, omegas, weights, syn_rates, q0, nullptr, delta, psi, eps_sink);
                    auto q2 = compute_expected_syn_weights(s, omegas, weights, syn_rates, q1, nullptr, delta, psi, eps_sink);
                    if (M == 2) {
                        Scalar r = q1[0] - q0[0];
                        Scalar v = (q2[0] - q1[0]) - r;
                        if (std::abs(v) > 1e-12) {
                            Scalar step = -std::abs(r) / std::abs(v);
                            Scalar q_acc = std::clamp(q0[0] - 2.0 * step * r + step * step * v, 1e-5, 1.0 - 1e-5);
                            std::vector<Scalar> q_cand = {q_acc, 1.0 - q_acc};
                            auto q_stab = compute_expected_syn_weights(s, omegas, weights, syn_rates, q_cand, nullptr, delta, psi, eps_sink);
                            if (evaluate_log_l(s, omegas, weights, syn_rates, q_stab, nullptr, nullptr, delta, psi, eps_sink) >=
                                evaluate_log_l(s, omegas, weights, syn_rates, q2, nullptr, nullptr, delta, psi, eps_sink)) {
                                syn_weights = q_stab;
                            } else {
                                syn_weights = q2;
                            }
                        } else {
                            syn_weights = q2;
                        }
                    } else if (M == 3) {
                        Scalar r0 = q1[0] - q0[0], r1 = q1[1] - q0[1];
                        Scalar v0 = (q2[0] - q1[0]) - r0, v1 = (q2[1] - q1[1]) - r1;
                        Scalar r2 = r0 * r0 + r1 * r1;
                        Scalar v2 = v0 * v0 + v1 * v1;
                        if (v2 > 1e-12) {
                            Scalar step = -std::sqrt(r2 / v2);
                            Scalar q1_acc = std::clamp(q0[0] - 2.0 * step * r0 + step * step * v0, 1e-5, 1.0 - 1e-5);
                            Scalar q2_acc = std::clamp(q0[1] - 2.0 * step * r1 + step * step * v1, 1e-5, 1.0 - 1e-5);
                            if (q1_acc + q2_acc > 1.0 - 1.e-5) {
                                Scalar sum_q = q1_acc + q2_acc;
                                q1_acc = (q1_acc / sum_q) * 0.99;
                                q2_acc = (q2_acc / sum_q) * 0.99;
                            }
                            Scalar q3_acc = 1.0 - q1_acc - q2_acc;
                            std::vector<Scalar> q_cand = {q1_acc, q2_acc, q3_acc};
                            auto q_stab = compute_expected_syn_weights(s, omegas, weights, syn_rates, q_cand, nullptr, delta, psi, eps_sink);
                            if (evaluate_log_l(s, omegas, weights, syn_rates, q_stab, nullptr, nullptr, delta, psi, eps_sink) >=
                                evaluate_log_l(s, omegas, weights, syn_rates, q2, nullptr, nullptr, delta, psi, eps_sink)) {
                                syn_weights = q_stab;
                            } else {
                                syn_weights = q2;
                            }
                        } else {
                            syn_weights = q2;
                        }
                    } else {
                        syn_weights = q2;
                    }
                }
            }

            // Block 2: Continuous parameter search on rates + scale s
            if (is_constrained) {
                if (K == 1) {
                    auto opt = NelderMead2D::minimize([&](Scalar w1, Scalar s_c) {
                        return -evaluate_log_l(s_c, {w1}, weights, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                    }, omegas[0], s, 1e-4, 1.0, 1e-4, 40);
                    omegas[0] = opt.a;
                    s = opt.b;
                } else if (K == 2) {
                    auto opt = NelderMead2D::minimize([&](Scalar w1, Scalar s_c) {
                        return -evaluate_log_l(s_c, {w1, 1.0}, weights, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                    }, omegas[0], s, 1e-4, 1.0, 1e-4, 40);
                    omegas[0] = opt.a;
                    omegas[1] = 1.0;
                    s = opt.b;
                } else if (K == 3) {
                    auto opt = NelderMeadND<3>::minimize([&](const std::array<Scalar, 3>& x) {
                        if (x[1] > x[2]) return 1e20;
                        return -evaluate_log_l(x[0], {x[1], x[2], 1.0}, weights, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                    }, {s, omegas[0], omegas[1]}, {0.01, 1e-4, 1e-4}, {50.0, 1.0, 1.0}, 1e-4, 40);
                    s = opt.x[0];
                    omegas[0] = opt.x[1];
                    omegas[1] = opt.x[2];
                    omegas[2] = 1.0;
                }
            } else {
                Scalar max_omega = use_error_sink ? 50.0 : 2000.0;
                if (K == 1) {
                    auto opt = NelderMead2D::minimize([&](Scalar w1, Scalar s_c) {
                        return -evaluate_log_l(s_c, {w1}, weights, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                    }, omegas[0], s, 1e-4, max_omega, 1e-4, 40);
                    omegas[0] = opt.a;
                    s = opt.b;
                } else if (K == 2) {
                    auto opt = NelderMeadND<3>::minimize([&](const std::array<Scalar, 3>& x) {
                        return -evaluate_log_l(x[0], {x[1], x[2]}, weights, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                    }, {s, omegas[0], omegas[1]}, {0.01, 1e-4, 1.0}, {50.0, 1.0, max_omega}, 1e-4, 50);
                    s = opt.x[0];
                    omegas[0] = opt.x[1];
                    omegas[1] = opt.x[2];
                } else if (K == 3) {
                    auto opt = NelderMeadND<4>::minimize([&](const std::array<Scalar, 4>& x) {
                        if (x[1] > x[2]) return 1e20;
                        return -evaluate_log_l(x[0], {x[1], x[2], x[3]}, weights, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                    }, {s, omegas[0], omegas[1], omegas[2]}, {0.01, 1e-4, 1e-4, 1.0}, {50.0, 1.0, 1.0, max_omega}, 1e-4, 50);
                    s = opt.x[0];
                    omegas[0] = opt.x[1];
                    omegas[1] = opt.x[2];
                    omegas[2] = opt.x[3];
                }
            }

            // Block 2b: Continuous search on synonymous rates (if SRV && M in {2, 3})
            if (srv && M == 2) {
                auto syn_opt = NelderMead2D::minimize([&](Scalar a1, Scalar /*unused*/) {
                    Scalar q1 = syn_weights[0], q2 = syn_weights[1];
                    Scalar a2 = (1.0 - q1 * a1) / q2;
                    if (a1 < 0.01 || a2 < a1 || a2 > 50.0) return 1e20;
                    return -evaluate_log_l(s, omegas, weights, {a1, a2}, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                }, syn_rates[0], 0.0, 0.01, 2.0, 1e-3, 25);
                Scalar a1 = syn_opt.a;
                Scalar a2 = (1.0 - syn_weights[0] * a1) / syn_weights[1];
                if (a1 >= 0.01 && a2 >= a1) {
                    syn_rates[0] = a1;
                    syn_rates[1] = a2;
                }
            } else if (srv && M == 3) {
                auto syn_opt = NelderMead2D::minimize([&](Scalar a1, Scalar a2) {
                    Scalar q1 = syn_weights[0], q2 = syn_weights[1], q3 = syn_weights[2];
                    Scalar a3 = (1.0 - q1 * a1 - q2 * a2) / q3;
                    if (a1 < 0.01 || a2 < a1 || a3 < a2 || a3 > 50.0) return 1e20;
                    return -evaluate_log_l(s, omegas, weights, {a1, a2, a3}, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                }, syn_rates[0], syn_rates[1], 0.01, 5.0, 1e-3, 30);

                Scalar a1 = syn_opt.a;
                Scalar a2 = syn_opt.b;
                Scalar a3 = (1.0 - syn_weights[0] * a1 - syn_weights[1] * a2) / syn_weights[2];
                if (a1 >= 0.01 && a2 >= a1 && a3 >= a2) {
                    syn_rates[0] = a1;
                    syn_rates[1] = a2;
                    syn_rates[2] = a3;
                }
            }

            // Block 3: Multiple hits optimization
            if (allow_triple) {
                auto mh_opt = NelderMead2D::minimize([&](Scalar d, Scalar p) {
                    if (d < 0.0 || d > 10.0 || p < 0.0 || p > 10.0) return 1e20;
                    return -evaluate_log_l(s, omegas, weights, syn_rates, syn_weights, nullptr, nullptr, d, p, eps_sink);
                }, delta, psi, 0.0, 10.0, 1e-4, 30);
                delta = std::max(0.0, mh_opt.a);
                psi = std::max(0.0, mh_opt.b);
            } else if (allow_double) {
                auto d_opt = NelderMead2D::minimize([&](Scalar d, Scalar /*unused*/) {
                    if (d < 0.0 || d > 10.0) return 1e20;
                    return -evaluate_log_l(s, omegas, weights, syn_rates, syn_weights, nullptr, nullptr, d, 0.0, eps_sink);
                }, delta, 0.0, 0.0, 10.0, 1e-4, 25);
                delta = std::max(0.0, d_opt.a);
            }

            if (pbar) {
                std::ostringstream ss;
                ss << model_desc << " (ECM " << (ecm_iter + 1) << "/" << max_ecm << "): lnL = "
                   << std::fixed << std::setprecision(2) << evaluate_log_l(s, omegas, weights, syn_rates, syn_weights, nullptr, nullptr, delta, psi, eps_sink);
                if (!is_constrained && !omegas.empty()) {
                    ss << ", ω" << K << " = " << std::setprecision(2) << omegas.back();
                }
                if (use_error_sink) {
                    ss << ", ε = " << std::setprecision(4) << eps_sink;
                }
                pbar->set_status(ss.str());
                pbar->tick();
            }
        }

        // Branch length initialization and refinement
        std::vector<Scalar> bl(tree.num_nodes(), 0.0);
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                bl[node.id] = s * branch_tau[node.id];
            }
        }

        if (refine_branches) {
            bl = refine_branch_lengths(bl, omegas, weights, syn_rates, syn_weights, delta, psi, 1, eps_sink);
            if (pbar) {
                std::ostringstream ss;
                ss << model_desc << ": Branch lengths refined";
                pbar->set_status(ss.str());
                pbar->tick();
            }
            if (K >= 2 || use_error_sink) {
                weights = compute_expected_weights(1.0, omegas, weights, syn_rates, syn_weights, &bl, delta, psi, eps_sink, use_error_sink ? &eps_sink : nullptr, max_eps_sink);
            }
            if (srv && M >= 2) {
                syn_weights = compute_expected_syn_weights(1.0, omegas, weights, syn_rates, syn_weights, &bl, delta, psi, eps_sink);
            }
            if (allow_triple) {
                auto mh_polish = NelderMead2D::minimize([&](Scalar d, Scalar p) {
                    if (d < 0.0 || d > 10.0 || p < 0.0 || p > 10.0) return 1e20;
                    return -evaluate_log_l(1.0, omegas, weights, syn_rates, syn_weights, &bl, nullptr, d, p, eps_sink);
                }, delta, psi, 0.0, 10.0, 1e-4, 25);
                delta = std::max(0.0, mh_polish.a);
                psi = std::max(0.0, mh_polish.b);
            } else if (allow_double) {
                auto d_polish = NelderMead2D::minimize([&](Scalar d, Scalar /*unused*/) {
                    if (d < 0.0 || d > 10.0) return 1e20;
                    return -evaluate_log_l(1.0, omegas, weights, syn_rates, syn_weights, &bl, nullptr, d, 0.0, eps_sink);
                }, delta, 0.0, 0.0, 10.0, 1e-4, 25);
                delta = std::max(0.0, d_polish.a);
            }
            if (allow_double && pbar) {
                std::ostringstream ss;
                ss << model_desc << ": Multiple hits refined (δ=" << std::fixed << std::setprecision(3) << delta << ")";
                pbar->set_status(ss.str());
                pbar->tick();
            }
        }

        Scalar best_lnL = evaluate_log_l(1.0, omegas, weights, syn_rates, syn_weights, &bl, &fit.site_log_likelihoods, delta, psi, eps_sink);
        fit.log_likelihood = best_lnL;
        fit.tree_scale = s;
        fit.branch_lengths = bl;
        fit.delta = delta;
        fit.psi = psi;
        fit.error_sink_weight = eps_sink;

        // Compute substitution rates and fractions
        Scalar total_rate = 0.0;
        Scalar double_rate = 0.0;
        Scalar triple_rate = 0.0;
        for (size_t k = 0; k < K; ++k) {
            MG94Parameters p = base_params;
            p.alpha = 1.0;
            p.beta = omegas[k];
            p.delta = delta;
            p.psi = psi;
            MG94Matrix mat;
            mat.update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
            auto sr = mat.compute_substitution_rates(*code);
            total_rate += weights[k] * sr.total_rate;
            double_rate += weights[k] * sr.rate_double;
            triple_rate += weights[k] * sr.rate_triple;
        }
        fit.frac_delta = (total_rate > 1e-12) ? (double_rate / total_rate) : 0.0;
        fit.frac_psi = (total_rate > 1e-12) ? (triple_rate / total_rate) : 0.0;

        fit.test_distribution.omegas = omegas;
        fit.test_distribution.weights = weights;
        fit.test_distribution.error_sink_weight = eps_sink;
        if (srv) {
            fit.test_distribution.syn_rates = syn_rates;
            fit.test_distribution.syn_weights = syn_weights;
        }

        // Detect collapsed rate classes
        for (size_t i = 0; i < K; ++i) {
            if (weights[i] < 0.005) {
                fit.test_distribution.annotations.push_back("Rate class " + std::to_string(i) + " collapsed (weight < 0.5%)");
            }
        }
        for (size_t i = 0; i + 1 < K; ++i) {
            if (std::abs(omegas[i+1] - omegas[i]) < 0.015) {
                fit.test_distribution.annotations.push_back("Rate classes " + std::to_string(i) + " and " + std::to_string(i+1) + " collapsed into identical omega");
            }
        }

        size_t N_codons = aln.num_codons;
        size_t num_branches = tree.num_nodes() - 1;
        size_t num_mixture_params = (2 * K - 1) - (is_constrained ? 1 : 0) + (use_error_sink ? 1 : 0);
        size_t num_syn_params = srv ? 2 * (M - 1) : 0;
        size_t num_mh_params = (allow_double ? 1 : 0) + (allow_triple ? 1 : 0);
        size_t K_params = 5 + (refine_branches ? num_branches : 1) + num_mixture_params + num_syn_params + num_mh_params;
        fit.aicc = -2.0 * fit.log_likelihood + 2.0 * K_params * N_codons / (N_codons - K_params - 1.0);

        return fit;
    }

    // Computes branch-site Error Sink Bayes Factors in a single O(patterns * branches) Inside-Outside pass
    void compute_error_sink_diagnostics(
        const BUSTEDFit& fit,
        Scalar bf_threshold,
        std::vector<Scalar>& out_site_max_bf,
        std::vector<BUSTEDErrorEvent>& out_flagged_events
    ) const {
        size_t num_sites = aln.num_codons;
        out_site_max_bf.assign(num_sites, 0.0);
        out_flagged_events.clear();

        Scalar eps = fit.error_sink_weight;
        if (eps <= 1e-9 || eps >= 1.0 - 1e-9) return;
        Scalar prior_odds = eps / (1.0 - eps);
        Scalar bio_scale = 1.0 - eps;

        size_t num_nodes = tree.num_nodes();
        size_t num_patterns = aln.patterns.size();
        int S = code->num_sense_codons;
        size_t K = fit.test_distribution.omegas.size();
        std::vector<Scalar> syn_rates = fit.test_distribution.syn_rates.empty() ? std::vector<Scalar>{1.0} : fit.test_distribution.syn_rates;
        std::vector<Scalar> syn_weights = fit.test_distribution.syn_weights.empty() ? std::vector<Scalar>{1.0} : fit.test_distribution.syn_weights;
        size_t M = syn_rates.size();

        std::vector<MG94Matrix> mg(K);
        for (size_t k = 0; k < K; ++k) {
            MG94Parameters p = base_params;
            p.alpha = 1.0;
            p.beta = fit.test_distribution.omegas[k];
            p.delta = fit.delta;
            p.psi = fit.psi;
            mg[k].update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
        }

        std::vector<std::vector<Matrix>> P_mix(M, std::vector<Matrix>(num_nodes));
        for (size_t m = 0; m < M; ++m) {
            Scalar alpha_m = syn_rates[m];
            for (const auto& node : tree.nodes) {
                if (node.id != tree.root_id) {
                    Scalar bl = fit.branch_lengths[node.id];
                    P_mix[m][node.id] = Matrix::Zero(S, S);
                    Scalar use_bio = is_test_branch[node.id] ? bio_scale : 1.0;
                    for (size_t k = 0; k < K; ++k) {
                        P_mix[m][node.id] += (use_bio * fit.test_distribution.weights[k]) * mg[k].transition_matrix(alpha_m * bl);
                    }
                    if (is_test_branch[node.id]) {
                        P_mix[m][node.id].noalias() += eps * P_sink;
                    }
                }
            }
        }

        // Per-pattern, per-node posterior probability of the saturation sink
        std::vector<std::vector<Scalar>> pat_node_post(num_patterns, std::vector<Scalar>(num_nodes, 0.0));
        #pragma omp parallel for schedule(dynamic)
        for (size_t p = 0; p < num_patterns; ++p) {
            const auto& pattern = aln.patterns[p];
            Scalar pat_L = 0.0;
            std::vector<LikelihoodEngine::InsideOutsideResult> io_m(M);
            for (size_t m = 0; m < M; ++m) {
                io_m[m] = LikelihoodEngine::compute_inside_outside(
                    tree, pattern, leaf_to_taxon, P_mix[m], aln.codon_frequencies_f3x4, S
                );
                pat_L += syn_weights[m] * io_m[m].likelihood;
            }
            if (pat_L > 0.0) {
                for (const auto& node : tree.nodes) {
                    if (node.id != tree.root_id && is_test_branch[node.id]) {
                        int nid = node.id;
                        Scalar sink_num = 0.0;
                        for (size_t m = 0; m < M; ++m) {
                            sink_num += syn_weights[m] * eval_sink_bilinear(io_m[m].V[nid], io_m[m].D[nid]);
                        }
                        Scalar post = std::clamp((eps * sink_num) / pat_L, 0.0, 1.0 - 1e-9);
                        pat_node_post[p][nid] = post;
                    }
                }
            }
        }

        for (size_t s_idx = 0; s_idx < num_sites; ++s_idx) {
            size_t p_idx = aln.site_to_pattern[s_idx];
            Scalar max_bf = 0.0;
            for (const auto& node : tree.nodes) {
                if (node.id != tree.root_id && is_test_branch[node.id]) {
                    Scalar post = pat_node_post[p_idx][node.id];
                    Scalar post_odds = post / std::max(1e-12, 1.0 - post);
                    Scalar bf = post_odds / prior_odds;
                    if (bf > max_bf) max_bf = bf;
                    if (bf >= bf_threshold) {
                        out_flagged_events.push_back({s_idx + 1, node.name, post, bf});
                    }
                }
            }
            out_site_max_bf[s_idx] = max_bf;
        }
    }

    // Fast 4-Gate Surprisal & Score Automatic Model Selector (BUSTED-Auto)
    // Diagnoses Error Sink (O(1) saturation + spatial adjacency), Multi-Hits (O(t_b) quarantined tangent score),
    // Synonymous Rate Variation (Neyman-Scott overdispersion + SQUAREM probe), and Omega classes K in a single baseline pass.
    BUSTEDAutoDiagnostics diagnose_and_select_model(BUSTEDSettings& settings) const {
        BUSTEDAutoDiagnostics diag;
        diag.enabled = true;

        size_t num_sites = aln.num_codons;
        size_t num_patterns = aln.patterns.size();
        size_t num_nodes = tree.num_nodes();
        int S = code->num_sense_codons;

        // Step 0: Fast K=2 baseline warm-up (M0: M=1, delta=0, psi=0, eps=0)
        std::vector<Scalar> omegas0 = {std::min(0.15, 0.5 * mg94_omega), 1.0};
        std::vector<Scalar> weights0 = {0.75, 0.25};
        for (int it = 0; it < 2; ++it) {
            weights0 = compute_expected_weights(1.0, omegas0, weights0);
        }
        Scalar base_ll = evaluate_log_l(1.0, omegas0, weights0);
        Scalar mean_omega0 = weights0[0] * omegas0[0] + weights0[1] * omegas0[1];

        // Build baseline branch transition matrices P0[node_id]
        std::vector<MG94Matrix> mg0(2);
        for (size_t k = 0; k < 2; ++k) {
            MG94Parameters p = base_params;
            p.alpha = 1.0;
            p.beta = omegas0[k];
            mg0[k].update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *code);
        }
        std::vector<Matrix> P0(num_nodes);
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                Scalar bl = branch_tau[node.id];
                P0[node.id] = weights0[0] * mg0[0].transition_matrix(bl) + weights0[1] * mg0[1].transition_matrix(bl);
            }
        }

        // Build pure 2-hit (Q2_off), 3-hit (Q3_off), and synonymous (Q_syn) tangent operators at mean_omega0
        Scalar r_nuc[4][4] = {
            {0.0, base_params.theta_AC, 1.0, base_params.theta_AT},
            {base_params.theta_AC, 0.0, base_params.theta_CG, base_params.theta_CT},
            {1.0, base_params.theta_CG, 0.0, base_params.theta_GT},
            {base_params.theta_AT, base_params.theta_CT, base_params.theta_GT, 0.0}
        };
        Scalar q_scale = std::max(1e-6, mg0[1].scale_factor);
        Matrix Q2_off = Matrix::Zero(S, S);
        Matrix Q3_off = Matrix::Zero(S, S);
        Matrix Q_syn = Matrix::Zero(S, S);
        for (int i = 0; i < S; ++i) {
            Scalar syn_row_sum = 0.0;
            for (int j = 0; j < S; ++j) {
                if (i == j) continue;
                const auto& diff = code->diff_matrix[i][j];
                Scalar mod = diff.synonymous ? 1.0 : mean_omega0;
                if (diff.num_diffs == 1 && diff.synonymous) {
                    Scalar rate1 = r_nuc[diff.diff_from[0]][diff.diff_to[0]] * aln.pos_nuc_frequencies(diff.diff_pos[0], diff.diff_to[0]) / q_scale;
                    Q_syn(i, j) = rate1;
                    syn_row_sum += rate1;
                } else if (diff.num_diffs == 2) {
                    Scalar rate2 = mod * r_nuc[diff.diff_from[0]][diff.diff_to[0]] * r_nuc[diff.diff_from[1]][diff.diff_to[1]] *
                                   aln.pos_nuc_frequencies(diff.diff_pos[0], diff.diff_to[0]) * aln.pos_nuc_frequencies(diff.diff_pos[1], diff.diff_to[1]) / q_scale;
                    Q2_off(i, j) = rate2;
                } else if (diff.num_diffs == 3) {
                    Scalar rate3 = mod * r_nuc[diff.diff_from[0]][diff.diff_to[0]] * r_nuc[diff.diff_from[1]][diff.diff_to[1]] * r_nuc[diff.diff_from[2]][diff.diff_to[2]] *
                                   aln.pos_nuc_frequencies(diff.diff_pos[0], diff.diff_to[0]) * aln.pos_nuc_frequencies(diff.diff_pos[1], diff.diff_to[1]) * aln.pos_nuc_frequencies(diff.diff_pos[2], diff.diff_to[2]) / q_scale;
                    Q3_off(i, j) = rate3;
                }
            }
            Q_syn(i, i) = -syn_row_sum;
        }

        // Single parallel Inside-Outside pass on M0
        std::vector<std::vector<Scalar>> pat_R_sink(num_patterns, std::vector<Scalar>(num_nodes, 0.0));
        std::vector<std::vector<Scalar>> pat_T2(num_patterns, std::vector<Scalar>(num_nodes, 0.0));
        std::vector<std::vector<Scalar>> pat_T3(num_patterns, std::vector<Scalar>(num_nodes, 0.0));
        std::vector<Scalar> pat_syn_score(num_patterns, 0.0);
        std::vector<Scalar> pat_syn_info(num_patterns, 0.0);

        #pragma omp parallel for schedule(dynamic)
        for (size_t p = 0; p < num_patterns; ++p) {
            const auto& pattern = aln.patterns[p];
            auto io = LikelihoodEngine::compute_inside_outside(
                tree, pattern, leaf_to_taxon, P0, aln.codon_frequencies_f3x4, S
            );
            Scalar L_p = io.likelihood;
            if (L_p > 0.0) {
                Scalar inv_L = 1.0 / L_p;
                Scalar u_syn = 0.0;
                Scalar v_syn = 0.0;
                for (const auto& node : tree.nodes) {
                    if (node.id != tree.root_id) {
                        int nid = node.id;
                        Scalar bl = branch_tau[nid];
                        if (is_test_branch[nid]) {
                            Scalar sink_val = eval_sink_bilinear(io.V[nid], io.D[nid]);
                            pat_R_sink[p][nid] = sink_val * inv_L;
                            pat_T2[p][nid] = bl * io.V[nid].dot(Q2_off * io.D[nid]) * inv_L;
                            pat_T3[p][nid] = bl * io.V[nid].dot(Q3_off * io.D[nid]) * inv_L;
                        }
                        Scalar branch_u = bl * io.V[nid].dot(Q_syn * (P0[nid] * io.D[nid])) * inv_L;
                        u_syn += branch_u;
                        v_syn += branch_u * branch_u;
                    }
                }
                pat_syn_score[p] = u_syn;
                pat_syn_info[p] = std::max(1e-6, v_syn);
            }
        }

        // Gate 1: Evaluate Error Sink Surprisal & Spatial Adjacency
        Scalar max_R_sink = 0.0;
        size_t candidate_count = 0;
        size_t adjacent_clusters = 0;
        Scalar eps_init = 0.002;
        Scalar sink_resp_sum = 0.0;
        Scalar total_branch_sites = static_cast<Scalar>(num_sites * std::max<size_t>(1, num_test_branches));

        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id && is_test_branch[node.id]) {
                int nid = node.id;
                int prev_high_site = -10;
                for (size_t s_idx = 0; s_idx < num_sites; ++s_idx) {
                    size_t p_idx = aln.site_to_pattern[s_idx];
                    Scalar r_val = pat_R_sink[p_idx][nid];
                    if (r_val > max_R_sink) max_R_sink = r_val;
                    Scalar bits = (r_val > 1.0) ? std::log2(r_val) : 0.0;
                    if (bits > 8.0) {
                        candidate_count++;
                    }
                    if (bits > 6.5) {
                        if (static_cast<int>(s_idx) - prev_high_site <= 2) {
                            adjacent_clusters++;
                        }
                        prev_high_site = static_cast<int>(s_idx);
                    }
                    Scalar resp = (eps_init * r_val) / ((1.0 - eps_init) + eps_init * r_val);
                    sink_resp_sum += resp;
                }
            }
        }

        diag.max_sink_bf = max_R_sink;
        diag.max_sink_surprisal_bits = (max_R_sink > 1.0) ? std::log2(max_R_sink) : 0.0;
        diag.sink_candidate_sites = candidate_count;
        diag.sink_adjacent_clusters = adjacent_clusters;

        Scalar eps_probe = std::clamp(sink_resp_sum / total_branch_sites, 1e-6, settings.error_sink_max_weight);
        Scalar sink_probe_ll = (diag.max_sink_surprisal_bits > 6.5)
            ? evaluate_log_l(1.0, omegas0, weights0, {1.0}, {1.0}, nullptr, nullptr, 0.0, 0.0, eps_probe)
            : base_ll;
        diag.sink_delta_ll = std::max(0.0, sink_probe_ll - base_ll);

        // Activate Error Sink if Delta LL > 2.2 AND either extreme short-branch surprisal (>= 13.0 bits, BF >= 8192) or spatial adjacency cluster
        if (diag.sink_delta_ll > 2.2 && (diag.max_sink_surprisal_bits >= 13.0 || (diag.sink_adjacent_clusters >= 1 && diag.max_sink_surprisal_bits >= 8.0))) {
            diag.activated_error_sink = true;
        }

        // Gate 2: Evaluate Multi-Hit Tangent Score & Probe (Quarantining high-R_sink branch-sites if Error Sink is active)
        Scalar eps_active = diag.activated_error_sink ? eps_probe : 0.0;
        Scalar s_delta = 0.0;
        Scalar s_psi = 0.0;
        for (size_t p = 0; p < num_patterns; ++p) {
            Scalar w_p = aln.patterns[p].weight;
            for (const auto& node : tree.nodes) {
                if (node.id != tree.root_id && is_test_branch[node.id]) {
                    int nid = node.id;
                    Scalar r_val = pat_R_sink[p][nid];
                    Scalar r_sink = (eps_active > 0.0) ? ((eps_active * r_val) / ((1.0 - eps_active) + eps_active * r_val)) : 0.0;
                    Scalar clean_wt = w_p * (1.0 - r_sink);
                    s_delta += clean_wt * pat_T2[p][nid];
                    s_psi += clean_wt * pat_T3[p][nid];
                }
            }
        }
        diag.double_hit_score = s_delta;
        diag.triple_hit_score = s_psi;

        Scalar mh_base_ll = diag.activated_error_sink ? sink_probe_ll : base_ll;
        if (s_delta > 0.15) {
            Scalar ll_d1 = evaluate_log_l(1.0, omegas0, weights0, {1.0}, {1.0}, nullptr, nullptr, 0.06, 0.0, eps_active);
            Scalar ll_d2 = evaluate_log_l(1.0, omegas0, weights0, {1.0}, {1.0}, nullptr, nullptr, 0.20, 0.0, eps_active);
            Scalar best_d = (ll_d2 > ll_d1) ? 0.20 : 0.06;
            Scalar best_d_ll = std::max(ll_d1, ll_d2);
            diag.double_hit_delta_ll = std::max(0.0, best_d_ll - mh_base_ll);

            // Check if diffuse 2-hit evolution (with eps=0) strictly outperforms an isolated non-clustered error sink
            if (diag.activated_error_sink && diag.sink_adjacent_clusters == 0 && diag.max_sink_surprisal_bits < 16.0) {
                Scalar pure_d_ll = evaluate_log_l(1.0, omegas0, weights0, {1.0}, {1.0}, nullptr, nullptr, best_d, 0.0, 0.0);
                if (pure_d_ll > sink_probe_ll + 0.5 && (pure_d_ll - base_ll) > 2.2) {
                    diag.activated_error_sink = false;
                    eps_active = 0.0;
                    mh_base_ll = base_ll;
                    best_d_ll = pure_d_ll;
                    diag.double_hit_delta_ll = pure_d_ll - base_ll;
                }
            }

            if (diag.double_hit_delta_ll > 2.2) {
                diag.selected_multiple_hits = "Double";
                if (s_psi > 0.05) {
                    Scalar ll_p = evaluate_log_l(1.0, omegas0, weights0, {1.0}, {1.0}, nullptr, nullptr, best_d, 0.06, eps_active);
                    diag.triple_hit_delta_ll = std::max(0.0, ll_p - best_d_ll);
                    if (diag.triple_hit_delta_ll > 2.2) {
                        diag.selected_multiple_hits = "Double+Triple";
                    }
                }
            }
        }

        // Gate 3: Evaluate Synonymous Rate Variation (SRV) Overdispersion & Probe
        Scalar num_disp = 0.0;
        Scalar den_disp = 0.0;
        for (size_t p = 0; p < num_patterns; ++p) {
            Scalar w_p = aln.patterns[p].weight;
            Scalar u2 = pat_syn_score[p] * pat_syn_score[p];
            Scalar v = pat_syn_info[p];
            num_disp += w_p * (u2 - v);
            den_disp += w_p * v * v;
        }
        diag.srv_overdispersion_z = (den_disp > 1e-12) ? (num_disp / std::sqrt(2.0 * den_disp)) : 0.0;

        Scalar active_d = (diag.selected_multiple_hits != "None") ? 0.10 : 0.0;
        Scalar active_p = (diag.selected_multiple_hits == "Double+Triple") ? 0.04 : 0.0;
        Scalar pre_srv_ll = evaluate_log_l(1.0, omegas0, weights0, {1.0}, {1.0}, nullptr, nullptr, active_d, active_p, eps_active);

        std::vector<Scalar> syn_r3 = {0.20, 0.75, 2.05};
        std::vector<Scalar> syn_w3 = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
        for (int it = 0; it < 2; ++it) {
            syn_w3 = compute_expected_syn_weights(1.0, omegas0, weights0, syn_r3, syn_w3, nullptr, active_d, active_p, eps_active);
        }
        Scalar srv3_ll = evaluate_log_l(1.0, omegas0, weights0, syn_r3, syn_w3, nullptr, nullptr, active_d, active_p, eps_active);
        diag.srv_delta_ll = std::max(0.0, srv3_ll - pre_srv_ll);

        if (diag.srv_delta_ll > 4.5 && syn_w3[0] >= 0.02 && syn_w3[1] >= 0.02 && syn_w3[2] >= 0.02) {
            diag.activated_srv = true;
            diag.selected_syn_rates = 3;
        } else if (diag.srv_delta_ll > 2.2) {
            std::vector<Scalar> syn_r2 = {0.35, 1.65};
            std::vector<Scalar> syn_w2 = {0.50, 0.50};
            syn_w2 = compute_expected_syn_weights(1.0, omegas0, weights0, syn_r2, syn_w2, nullptr, active_d, active_p, eps_active);
            Scalar srv2_ll = evaluate_log_l(1.0, omegas0, weights0, syn_r2, syn_w2, nullptr, nullptr, active_d, active_p, eps_active);
            if (srv2_ll - pre_srv_ll > 2.2 && syn_w2[0] >= 0.02 && syn_w2[1] >= 0.02) {
                diag.activated_srv = true;
                diag.selected_syn_rates = 2;
                diag.srv_delta_ll = srv2_ll - pre_srv_ll;
            }
        }

        // Gate 4: Fast Unrefined Null Screen for Optimal Omega Classes K in {2, 3}
        BUSTEDFit null_k2 = fit_model(true, 2, false, diag.activated_srv, diag.selected_syn_rates, diag.selected_multiple_hits, nullptr, "", diag.activated_error_sink, settings.error_sink_max_weight);
        BUSTEDFit null_k3 = fit_model(true, 3, false, diag.activated_srv, diag.selected_syn_rates, diag.selected_multiple_hits, nullptr, "", diag.activated_error_sink, settings.error_sink_max_weight);

        bool k3_collapsed = (std::abs(null_k3.test_distribution.omegas[1] - null_k3.test_distribution.omegas[0]) < 0.025) ||
                            (null_k3.test_distribution.weights[0] < 0.015) ||
                            (null_k3.test_distribution.weights[1] < 0.015);
        if (!k3_collapsed && (null_k3.log_likelihood - null_k2.log_likelihood > 2.5)) {
            diag.selected_k = 3;
            diag.k_rationale = "K=3 retained (distinct two-rate purifying background, delta lnL = +" +
                               std::to_string(null_k3.log_likelihood - null_k2.log_likelihood) + ")";
        } else {
            diag.selected_k = 2;
            diag.k_rationale = k3_collapsed
                ? "K=2 selected (pruned collapsed/low-weight 3rd background rate class)"
                : "K=2 selected via AICc parsimony (K=3 gain < threshold)";
        }

        // Apply selected specification to settings
        settings.error_sink = diag.activated_error_sink;
        settings.multiple_hits = diag.selected_multiple_hits;
        settings.srv = diag.activated_srv;
        settings.num_syn_rate_classes = diag.selected_syn_rates;
        settings.num_rate_classes = diag.selected_k;
        settings.auto_select_k = false;

        return diag;
    }

    // Step-up model selection across K in {1, ..., max_k} using AICc on the Null model
    std::pair<size_t, std::vector<BUSTEDFit>> select_optimal_k(
        size_t max_k = 3,
        bool refine_branches = true,
        bool srv = false,
        size_t num_syn_rates = 3,
        const std::string& multiple_hits = "None",
        ProgressBar* pbar = nullptr,
        bool use_error_sink = false,
        Scalar max_eps_sink = 0.01
    ) const {
        std::vector<BUSTEDFit> null_fits;
        null_fits.reserve(max_k);
        for (size_t k = 1; k <= max_k; ++k) {
            std::string desc = "Null K=" + std::to_string(k);
            null_fits.push_back(fit_model(true, k, refine_branches, srv, num_syn_rates, multiple_hits, pbar, desc, use_error_sink, max_eps_sink));
        }

        size_t optimal_k = 1;
        for (size_t k = 2; k <= max_k; ++k) {
            // Delta AICc threshold: improvement must be at least 2.0 to justify adding a rate class
            if (null_fits[k - 1].aicc < null_fits[optimal_k - 1].aicc - 2.0) {
                optimal_k = k;
            }
        }
        return {optimal_k, null_fits};
    }

    // Run complete BUSTED analysis with options
    BUSTEDResult run(
        BUSTEDSettings settings = {},
        bool show_progress = false,
        bool force_progress = false
    ) const {
        auto t0 = std::chrono::high_resolution_clock::now();
        BUSTEDResult res;
        if (settings.full_auto) {
            res.auto_diagnostics = diagnose_and_select_model(settings);
        }
        res.settings = settings;

        bool allow_double = (settings.multiple_hits == "Double" || settings.multiple_hits == "Double+Triple");
        auto count_steps = [&](size_t K) -> size_t {
            size_t max_ecm = (K == 1 && !settings.srv && !allow_double && !settings.error_sink) ? 1 : 6;
            return max_ecm + (settings.refine_branch_lengths ? 1 : 0) + ((settings.refine_branch_lengths && allow_double) ? 1 : 0);
        };

        size_t total_steps = 0;
        if (settings.auto_select_k) {
            for (size_t k = 1; k <= settings.max_k; ++k) {
                total_steps += count_steps(k);
            }
            total_steps += count_steps(settings.max_k);
        } else {
            total_steps = count_steps(settings.num_rate_classes) * 2;
        }

        std::unique_ptr<ProgressBar> pbar;
        if (show_progress && (force_progress || ProgressBar::is_terminal())) {
            pbar = std::make_unique<ProgressBar>(
                total_steps,
                "[BUSTED] Mixture Models",
                "steps",
                force_progress,
                ProgressBar::Style::SmoothBlocks
            );
        }

        if (settings.auto_select_k) {
            auto [opt_k, null_fits] = select_optimal_k(
                settings.max_k,
                settings.refine_branch_lengths,
                settings.srv,
                settings.num_syn_rate_classes,
                settings.multiple_hits,
                pbar.get(),
                settings.error_sink,
                settings.error_sink_max_weight
            );
            res.optimal_k = opt_k;
            res.k_null_fits = null_fits;
            res.constrained = null_fits[opt_k - 1];
            res.unconstrained = fit_model(
                false,
                opt_k,
                settings.refine_branch_lengths,
                settings.srv,
                settings.num_syn_rate_classes,
                settings.multiple_hits,
                pbar.get(),
                "Unconstrained K=" + std::to_string(opt_k),
                settings.error_sink,
                settings.error_sink_max_weight
            );
        } else {
            res.optimal_k = settings.num_rate_classes;
            res.unconstrained = fit_model(
                false,
                settings.num_rate_classes,
                settings.refine_branch_lengths,
                settings.srv,
                settings.num_syn_rate_classes,
                settings.multiple_hits,
                pbar.get(),
                "Unconstrained K=" + std::to_string(settings.num_rate_classes),
                settings.error_sink,
                settings.error_sink_max_weight
            );
            res.constrained = fit_model(
                true,
                settings.num_rate_classes,
                settings.refine_branch_lengths,
                settings.srv,
                settings.num_syn_rate_classes,
                settings.multiple_hits,
                pbar.get(),
                "Constrained K=" + std::to_string(settings.num_rate_classes),
                settings.error_sink,
                settings.error_sink_max_weight
            );
        }

        // Compute LRT and p-value
        if (res.unconstrained.log_likelihood < res.constrained.log_likelihood) {
            res.unconstrained.log_likelihood = res.constrained.log_likelihood;
            res.lrt = 0.0;
            res.p_value = 0.5;
        } else {
            res.lrt = 2.0 * (res.unconstrained.log_likelihood - res.constrained.log_likelihood);
            // Asymptotic null: 0.5 * chi^2_0 + 0.5 * chi^2_2
            res.p_value = 0.5 * std::exp(-res.lrt / 2.0);
        }

        if (pbar) {
            std::ostringstream summary;
            if (res.p_value <= settings.p_value_threshold) {
                summary << "\033[1;32mEvidence of episodic diversifying selection\033[0m: LRT = "
                        << std::fixed << std::setprecision(2) << res.lrt
                        << ", p = " << std::setprecision(4) << res.p_value;
            } else {
                summary << "No evidence of diversifying selection: LRT = "
                        << std::fixed << std::setprecision(2) << res.lrt
                        << ", p = " << std::setprecision(4) << res.p_value;
            }
            pbar->finish(summary.str());
        }

        // Compute per-site Evidence Ratios (Bayes Factors for positive selection)
        size_t num_sites = aln.num_codons;
        res.evidence_ratios.assign(num_sites, 1.0);
        Scalar p_pos = res.unconstrained.test_distribution.weights.back();
        Scalar prior_odds = (p_pos > 1e-6 && p_pos < 1.0 - 1e-6) ? (p_pos / (1.0 - p_pos)) : 1e-6;

        for (size_t s_idx = 0; s_idx < num_sites; ++s_idx) {
            size_t p_idx = aln.site_to_pattern[s_idx];
            Scalar null_ll = res.constrained.site_log_likelihoods[p_idx];
            Scalar alt_ll = res.unconstrained.site_log_likelihoods[p_idx];

            if (alt_ll > null_ll) {
                Scalar delta = alt_ll - null_ll;
                Scalar post_odds = std::exp(delta) * prior_odds;
                Scalar bf = post_odds / prior_odds;
                res.evidence_ratios[s_idx] = bf;
            } else {
                res.evidence_ratios[s_idx] = 1.0;
            }
        }

        if (settings.error_sink) {
            compute_error_sink_diagnostics(
                res.unconstrained,
                settings.error_sink_bf_threshold,
                res.site_error_bfs,
                res.flagged_error_events
            );
        } else {
            res.site_error_bfs.assign(num_sites, 0.0);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        res.runtime_seconds = std::chrono::duration<double>(t1 - t0).count();
        return res;
    }

    nlohmann::json to_legacy_json(const BUSTEDResult& res) const {
        nlohmann::json j;

        j["analysis"] = {
            {"authors", "Sergei L Kosakovsky Pond"},
            {"citation", "*Gene-wide identification of episodic selection*, Mol Biol Evol. 32(5):1365-71"},
            {"contact", "spond@temple.edu"},
            {"version", "4.7-next"},
            {"settings", {
                {"error-sink", res.settings.error_sink ? 1 : 0},
                {"mss", "No"},
                {"multiple-hit", res.settings.multiple_hits},
                {"srv", (!res.unconstrained.test_distribution.syn_rates.empty() && res.unconstrained.test_distribution.syn_rates.size() > 1) ? "Yes" : "No"},
                {"optimal_k", res.optimal_k}
            }}
        };

        j["test results"] = {
            {"LRT", res.lrt},
            {"p-value", res.p_value},
            {"optimal_k", res.optimal_k}
        };

        // Fits
        nlohmann::json fits;
        if (has_gtr_fit) {
            fits["Nucleotide GTR"] = {
                {"AIC-c", gtr_aicc},
                {"Log Likelihood", gtr_log_l},
                {"Rate Distributions", {
                    {"Substitution rate from nucleotide A to nucleotide C", gtr_rates.theta_AC},
                    {"Substitution rate from nucleotide A to nucleotide G", 1.0},
                    {"Substitution rate from nucleotide A to nucleotide T", gtr_rates.theta_AT},
                    {"Substitution rate from nucleotide C to nucleotide G", gtr_rates.theta_CG},
                    {"Substitution rate from nucleotide C to nucleotide T", gtr_rates.theta_CT},
                    {"Substitution rate from nucleotide G to nucleotide T", gtr_rates.theta_GT}
                }}
            };
        }

        fits["MG94xREV with separate rates for branch sets"] = {
            {"Log Likelihood", mg94_log_l},
            {"Rate Distributions", {
                {"non-synonymous/synonymous rate ratio for *test*", {{mg94_omega, 1.0}}}
            }}
        };

        auto format_distro = [&](const BUSTEDRateDistribution& d) {
            nlohmann::json distro;
            Scalar bio_scale = res.settings.error_sink ? (1.0 - d.error_sink_weight) : 1.0;
            for (size_t k = 0; k < d.omegas.size(); ++k) {
                nlohmann::json entry = {
                    {"omega", d.omegas[k]},
                    {"proportion", bio_scale * d.weights[k]}
                };
                if (!d.annotations.empty() && k < d.annotations.size()) {
                    entry["annotation"] = d.annotations[k];
                }
                distro[std::to_string(k)] = entry;
            }
            if (res.settings.error_sink) {
                distro["error_sink"] = {
                    {"type", "saturation_replacement"},
                    {"proportion", d.error_sink_weight}
                };
            }
            return distro;
        };

        auto format_srv_distro = [](const BUSTEDRateDistribution& d) {
            nlohmann::json distro;
            for (size_t m = 0; m < d.syn_rates.size(); ++m) {
                distro[std::to_string(m)] = {
                    {"rate", d.syn_rates[m]},
                    {"proportion", d.syn_weights[m]}
                };
            }
            return distro;
        };

        nlohmann::json unc_rates = {{"Test", format_distro(res.unconstrained.test_distribution)}};
        if (!res.unconstrained.test_distribution.syn_rates.empty() && res.unconstrained.test_distribution.syn_rates.size() > 1) {
            unc_rates["Synonymous site-to-site rates"] = format_srv_distro(res.unconstrained.test_distribution);
        }
        nlohmann::json unc_fit = {
            {"AIC-c", res.unconstrained.aicc},
            {"Log Likelihood", res.unconstrained.log_likelihood},
            {"Rate Distributions", unc_rates}
        };
        if (res.settings.multiple_hits != "None") {
            unc_fit["rate at which 2 nucleotides are changed instantly within a single codon"] = res.unconstrained.delta;
            unc_fit["Fraction of subs rate at which 2 nucleotides are changed instantly within a single codon"] = res.unconstrained.frac_delta;
            if (res.settings.multiple_hits == "Double+Triple") {
                unc_fit["rate at which 3 nucleotides are changed instantly within a single codon"] = res.unconstrained.psi;
                unc_fit["Fraction of subs rate at which 3 nucleotides are changed instantly within a single codon"] = res.unconstrained.frac_psi;
            }
        }
        fits["Unconstrained model"] = unc_fit;

        nlohmann::json con_rates = {{"Test", format_distro(res.constrained.test_distribution)}};
        if (!res.constrained.test_distribution.syn_rates.empty() && res.constrained.test_distribution.syn_rates.size() > 1) {
            con_rates["Synonymous site-to-site rates"] = format_srv_distro(res.constrained.test_distribution);
        }
        nlohmann::json con_fit = {
            {"AIC-c", res.constrained.aicc},
            {"Log Likelihood", res.constrained.log_likelihood},
            {"Rate Distributions", con_rates}
        };
        if (res.settings.multiple_hits != "None") {
            con_fit["rate at which 2 nucleotides are changed instantly within a single codon"] = res.constrained.delta;
            con_fit["Fraction of subs rate at which 2 nucleotides are changed instantly within a single codon"] = res.constrained.frac_delta;
            if (res.settings.multiple_hits == "Double+Triple") {
                con_fit["rate at which 3 nucleotides are changed instantly within a single codon"] = res.constrained.psi;
                con_fit["Fraction of subs rate at which 3 nucleotides are changed instantly within a single codon"] = res.constrained.frac_psi;
            }
        }
        fits["Constrained model"] = con_fit;

        j["fits"] = fits;

        // Branch attributes
        nlohmann::json branch_attr;
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                nlohmann::json b_entry;
                if (!res.unconstrained.branch_lengths.empty() && node.id >= 0 && static_cast<size_t>(node.id) < res.unconstrained.branch_lengths.size()) {
                    b_entry["unconstrained"] = res.unconstrained.branch_lengths[node.id];
                }
                if (!res.constrained.branch_lengths.empty() && node.id >= 0 && static_cast<size_t>(node.id) < res.constrained.branch_lengths.size()) {
                    b_entry["constrained"] = res.constrained.branch_lengths[node.id];
                }
                branch_attr[node.name] = b_entry;
            }
        }
        j["branch attributes"] = {{"0", branch_attr}};

        j["Evidence Ratios"] = {
            {"constrained", {res.evidence_ratios}},
            {"optimized null", {res.evidence_ratios}}
        };
        if (res.settings.error_sink) {
            j["Evidence Ratios"]["error_sink_max_bf"] = {res.site_error_bfs};
        }

        // Site Log Likelihood
        std::vector<Scalar> unc_site_ll(aln.num_codons), con_site_ll(aln.num_codons);
        for (size_t s = 0; s < aln.num_codons; ++s) {
            size_t p = aln.site_to_pattern[s];
            unc_site_ll[s] = res.unconstrained.site_log_likelihoods[p];
            con_site_ll[s] = res.constrained.site_log_likelihoods[p];
        }
        j["Site Log Likelihood"] = {
            {"unconstrained", {unc_site_ll}},
            {"constrained", {con_site_ll}}
        };

        j["runtime"] = res.runtime_seconds;
        return j;
    }

    nlohmann::json to_modern_json(const BUSTEDResult& res, const Provenance& prov = {}) const {
        nlohmann::json j;

        j["$schema"] = "https://raw.githubusercontent.com/veg/hyphy-3/main/schemas/v3/busted.v3.schema.json";
        j["schema_version"] = "3.0.0";

        // Analysis block
        j["analysis"]["id"] = "busted";
        j["analysis"]["name"] = "Branch-Site Unrestricted Statistical Test for Episodic Diversification";
        j["analysis"]["version"] = "3.0.0";
        j["analysis"]["category"] = "gene_selection";
        j["analysis"]["description"] = "BUSTED uses a random effects likelihood approach to test whether a gene has experienced positive selection at any site on specified (or all) branches of a phylogeny.";
        j["analysis"]["citations"] = nlohmann::json::array({
            {
                {"citation", "Murrell B, Weaver S, Smith MD, Wertheim JO, Murrell S, Aylward A, Eren K, Pollner T, Martin DP, Frost SDW, Kosakovsky Pond SL (2015). Gene-Wide Identification of Episodic Selection. Mol Biol Evol 32(5): 1365-1371."},
                {"doi", "10.1093/molbev/msv035"},
                {"pmid", "25701167"}
            }
        });

        j["analysis"]["settings"]["rates"] = res.settings.num_rate_classes;
        j["analysis"]["settings"]["srv"] = res.settings.srv;
        j["analysis"]["settings"]["syn_rates"] = res.settings.num_syn_rate_classes;
        j["analysis"]["settings"]["auto_k"] = res.settings.auto_select_k;
        j["analysis"]["settings"]["multiple_hits"] = res.settings.multiple_hits;
        j["analysis"]["settings"]["error_sink"] = res.settings.error_sink;
        if (res.settings.error_sink) {
            j["analysis"]["settings"]["error_sink_max_weight"] = res.settings.error_sink_max_weight;
            j["analysis"]["settings"]["error_sink_bf_threshold"] = res.settings.error_sink_bf_threshold;
        }
        j["analysis"]["settings"]["refine_branch_lengths"] = res.settings.refine_branch_lengths;
        j["analysis"]["settings"]["pvalue_threshold"] = res.settings.p_value_threshold;
        j["analysis"]["settings"]["optimal_k"] = res.optimal_k;

        // Software block
        j["software"]["name"] = "hyphy";
        j["software"]["version"] = "3.0.0";
        j["software"]["git_commit"] = "6e22db3";
        j["software"]["build_type"] = "Release";
        j["software"]["compiler"] = Provenance::detect_compiler();

        // Provenance block
        if (!prov.invocation.cli_command.empty() || !prov.inputs.empty()) {
            j["provenance"] = prov.to_json();
        } else {
            Provenance auto_prov;
            auto_prov.invocation.cli_command = "hyphy3 busted";
            auto_prov.invocation.working_directory = Provenance::get_cwd();
            auto_prov.execution.start_time = Provenance::current_iso8601();
            auto_prov.execution.end_time = Provenance::current_iso8601();
            auto_prov.execution.wall_time_seconds = res.runtime_seconds;
            auto_prov.execution.cpu_threads = 1;
            auto_prov.execution.hostname = Provenance::get_hostname();
            auto_prov.execution.os = Provenance::detect_os();
            auto_prov.execution.compiler = Provenance::detect_compiler();
            j["provenance"] = auto_prov.to_json();
        }

        // Dataset block
        j["dataset"]["taxa_count"] = aln.num_taxa;
        j["dataset"]["codon_sites"] = aln.num_codons;
        j["dataset"]["nucleotide_sites"] = aln.num_codons * 3;
        j["dataset"]["unique_patterns"] = aln.patterns.size();
        j["dataset"]["taxa"] = aln.taxon_names;

        nlohmann::json gcode;
        gcode["id"] = code ? code->name : "Universal";
        gcode["name"] = code ? code->name : "Universal";
        gcode["sense_codons"] = code ? static_cast<int>(code->sense_codons.size()) : 61;
        if (code) {
            gcode["stop_codons"] = code->stop_codons;
        } else {
            gcode["stop_codons"] = {"TAA", "TAG", "TGA"};
        }
        j["dataset"]["genetic_code"] = gcode;

        j["dataset"]["partitions"] = nlohmann::json::array({
            {
                {"id", "default"},
                {"name", "Full Alignment"},
                {"span", nlohmann::json::array({nlohmann::json::array({1, aln.num_codons})})},
                {"sites_count", aln.num_codons},
                {"patterns_count", aln.patterns.size()}
            }
        });

        // Phylogeny block
        j["phylogeny"]["newick"] = tree.to_newick();
        nlohmann::json nodes_json;
        double sum_bl_unc = 0.0;
        double sum_bl_con = 0.0;
        for (const auto& node : tree.nodes) {
            nlohmann::json node_entry;
            node_entry["type"] = node.children.empty() ? "leaf" : "internal";

            if (!res.unconstrained.branch_lengths.empty() && node.id >= 0 && static_cast<size_t>(node.id) < res.unconstrained.branch_lengths.size()) {
                node_entry["branch_lengths"]["unconstrained"] = res.unconstrained.branch_lengths[node.id];
                if (node.id != tree.root_id) sum_bl_unc += res.unconstrained.branch_lengths[node.id];
            }
            if (!res.constrained.branch_lengths.empty() && node.id >= 0 && static_cast<size_t>(node.id) < res.constrained.branch_lengths.size()) {
                node_entry["branch_lengths"]["constrained"] = res.constrained.branch_lengths[node.id];
                if (node.id != tree.root_id) sum_bl_con += res.constrained.branch_lengths[node.id];
            }

            nodes_json[node.name] = node_entry;
        }
        j["phylogeny"]["nodes"] = nodes_json;
        j["phylogeny"]["tree_lengths"]["unconstrained"] = sum_bl_unc;
        j["phylogeny"]["tree_lengths"]["constrained"] = sum_bl_con;

        // Model fits block
        auto format_fit_modern = [&](const BUSTEDFit& fit, const std::string& desc) {
            nlohmann::json f;
            f["description"] = desc;
            f["log_likelihood"] = fit.log_likelihood;
            f["aicc"] = fit.aicc;
            f["tree_scale"] = fit.tree_scale;
            f["rate_classes"] = fit.num_rate_classes;

            Scalar bio_scale = res.settings.error_sink ? (1.0 - fit.error_sink_weight) : 1.0;
            nlohmann::json test_dist = nlohmann::json::array();
            for (size_t k = 0; k < fit.test_distribution.omegas.size(); ++k) {
                test_dist.push_back({
                    {"class", static_cast<int>(k + 1)},
                    {"omega", fit.test_distribution.omegas[k]},
                    {"proportion", bio_scale * fit.test_distribution.weights[k]}
                });
            }
            f["rate_distributions"]["test"] = test_dist;

            if (res.settings.error_sink) {
                f["error_sink"] = {
                    {"type", "saturation_replacement"},
                    {"kernel", "P_sink(i,j) = pi_j / (1 - pi_i) for i != j, 0 for i == j"},
                    {"proportion", fit.error_sink_weight},
                    {"max_weight_bound", res.settings.error_sink_max_weight}
                };
            }

            if (!fit.test_distribution.syn_rates.empty() && fit.test_distribution.syn_rates.size() > 1) {
                nlohmann::json syn_dist = nlohmann::json::array();
                for (size_t m = 0; m < fit.test_distribution.syn_rates.size(); ++m) {
                    syn_dist.push_back({
                        {"class", static_cast<int>(m + 1)},
                        {"rate", fit.test_distribution.syn_rates[m]},
                        {"proportion", fit.test_distribution.syn_weights[m]}
                    });
                }
                f["rate_distributions"]["synonymous"] = syn_dist;
            }

            if (res.settings.multiple_hits != "None") {
                f["multiple_hits"]["double_hit_rate"] = fit.delta;
                f["multiple_hits"]["double_hit_fraction"] = fit.frac_delta;
                if (res.settings.multiple_hits == "Double+Triple") {
                    f["multiple_hits"]["triple_hit_rate"] = fit.psi;
                    f["multiple_hits"]["triple_hit_fraction"] = fit.frac_psi;
                }
            }
            return f;
        };

        j["model_fits"]["unconstrained"] = format_fit_modern(res.unconstrained, "Unconstrained BUSTED mixture model allowing omega_k > 1");
        j["model_fits"]["constrained"] = format_fit_modern(res.constrained, "Constrained null BUSTED model fixing omega_k <= 1");

        if (has_gtr_fit) {
            j["model_fits"]["nucleotide_gtr"]["description"] = "General Time Reversible nucleotide model with empirical frequencies";
            j["model_fits"]["nucleotide_gtr"]["log_likelihood"] = gtr_log_l;
            j["model_fits"]["nucleotide_gtr"]["aicc"] = gtr_aicc;
            j["model_fits"]["nucleotide_gtr"]["equilibrium_frequencies"] = {
                {"A", aln.nuc_frequencies(0)},
                {"C", aln.nuc_frequencies(1)},
                {"G", aln.nuc_frequencies(2)},
                {"T", aln.nuc_frequencies(3)}
            };
            j["model_fits"]["nucleotide_gtr"]["substitution_rates"] = {
                {"AC", gtr_rates.theta_AC},
                {"AG", 1.0},
                {"AT", gtr_rates.theta_AT},
                {"CG", gtr_rates.theta_CG},
                {"CT", gtr_rates.theta_CT},
                {"GT", gtr_rates.theta_GT}
            };
        }

        j["model_fits"]["codon_mg94"]["description"] = "Muse-Gaut 1994 x GTR codon model with shared global omega";
        j["model_fits"]["codon_mg94"]["log_likelihood"] = mg94_log_l;
        j["model_fits"]["codon_mg94"]["global_parameters"] = {
            {"omega", mg94_omega}
        };

        // Statistical tests block
        j["statistical_tests"]["gene_level"] = {
            {"method", "Likelihood Ratio Test (LRT)"},
            {"test_statistic", "LRT"},
            {"statistic_value", res.lrt},
            {"distribution", "0.5 * chi^2_0 + 0.5 * chi^2_2"},
            {"degrees_of_freedom", 2},
            {"null_hypothesis", "omega_k <= 1 for all k on test branches (Constrained model)"},
            {"alternative_hypothesis", "omega_k > 1 for at least one k on test branches (Unconstrained model)"},
            {"p_value", res.p_value},
            {"decision", (res.p_value <= res.settings.p_value_threshold) ? "Episodic diversifying selection detected" : "No evidence of episodic diversifying selection"},
            {"optimal_k", static_cast<int>(res.optimal_k)}
        };

        if (res.auto_diagnostics.enabled) {
            j["auto_model_selection"] = {
                {"error_sink_gate", {
                    {"max_surprisal_bits", res.auto_diagnostics.max_sink_surprisal_bits},
                    {"max_baseline_bf", res.auto_diagnostics.max_sink_bf},
                    {"candidate_sites_gt_8bits", static_cast<int>(res.auto_diagnostics.sink_candidate_sites)},
                    {"adjacent_clusters", static_cast<int>(res.auto_diagnostics.sink_adjacent_clusters)},
                    {"delta_log_l", res.auto_diagnostics.sink_delta_ll},
                    {"activated", res.auto_diagnostics.activated_error_sink}
                }},
                {"multiple_hits_gate", {
                    {"quarantined_double_hit_score", res.auto_diagnostics.double_hit_score},
                    {"quarantined_triple_hit_score", res.auto_diagnostics.triple_hit_score},
                    {"double_hit_delta_log_l", res.auto_diagnostics.double_hit_delta_ll},
                    {"triple_hit_delta_log_l", res.auto_diagnostics.triple_hit_delta_ll},
                    {"selected", res.auto_diagnostics.selected_multiple_hits}
                }},
                {"srv_gate", {
                    {"overdispersion_z", res.auto_diagnostics.srv_overdispersion_z},
                    {"delta_log_l", res.auto_diagnostics.srv_delta_ll},
                    {"activated", res.auto_diagnostics.activated_srv},
                    {"selected_syn_rates", static_cast<int>(res.auto_diagnostics.selected_syn_rates)}
                }},
                {"omega_classes_gate", {
                    {"selected_k", static_cast<int>(res.auto_diagnostics.selected_k)},
                    {"rationale", res.auto_diagnostics.k_rationale}
                }}
            };
        }

        if (res.settings.error_sink) {
            nlohmann::json flagged = nlohmann::json::array();
            for (const auto& ev : res.flagged_error_events) {
                flagged.push_back({
                    {"site", static_cast<int>(ev.site)},
                    {"branch", ev.branch_name},
                    {"posterior", ev.posterior},
                    {"bayes_factor", ev.bayes_factor}
                });
            }
            j["error_sink_diagnostics"] = {
                {"unconstrained_epsilon", res.unconstrained.error_sink_weight},
                {"constrained_epsilon", res.constrained.error_sink_weight},
                {"bf_threshold", res.settings.error_sink_bf_threshold},
                {"flagged_branch_sites_count", static_cast<int>(res.flagged_error_events.size())},
                {"flagged_branch_sites", flagged}
            };
        }

        // Site results (Columnar format)
        j["site_results"]["columns"] = {
            {"site", {{"type", "integer"}, {"unit", "1-based codon position"}, {"description", "Alignment codon site index (1-based)"}}},
            {"evidence_ratio", {{"type", "float"}, {"description", "Per-site Evidence Ratio (Bayes Factor) favoring positive selection (omega > 1)"}}},
            {"unconstrained_log_l", {{"type", "float"}, {"description", "Site log-likelihood under unconstrained model"}}},
            {"constrained_log_l", {{"type", "float"}, {"description", "Site log-likelihood under constrained null model"}}}
        };
        if (res.settings.error_sink) {
            j["site_results"]["columns"]["error_sink_max_bf"] = {
                {"type", "float"},
                {"description", "Maximum branch-site Empirical Bayes Factor for the saturation error sink across branches at this site"}
            };
        }

        std::vector<int> col_site;
        std::vector<double> col_er, col_unc_ll, col_con_ll, col_err_bf;
        col_site.reserve(aln.num_codons);
        col_er.reserve(aln.num_codons);
        col_unc_ll.reserve(aln.num_codons);
        col_con_ll.reserve(aln.num_codons);
        if (res.settings.error_sink) col_err_bf.reserve(aln.num_codons);

        for (size_t s = 0; s < aln.num_codons; ++s) {
            col_site.push_back(static_cast<int>(s + 1));
            col_er.push_back(res.evidence_ratios[s]);
            size_t p = aln.site_to_pattern[s];
            col_unc_ll.push_back(res.unconstrained.site_log_likelihoods[p]);
            col_con_ll.push_back(res.constrained.site_log_likelihoods[p]);
            if (res.settings.error_sink && s < res.site_error_bfs.size()) {
                col_err_bf.push_back(res.site_error_bfs[s]);
            }
        }

        j["site_results"]["data"]["site"] = col_site;
        j["site_results"]["data"]["evidence_ratio"] = col_er;
        j["site_results"]["data"]["unconstrained_log_l"] = col_unc_ll;
        j["site_results"]["data"]["constrained_log_l"] = col_con_ll;
        if (res.settings.error_sink) {
            j["site_results"]["data"]["error_sink_max_bf"] = col_err_bf;
        }

        return j;
    }

    nlohmann::json to_json(
        const BUSTEDResult& res,
        JSONFormat format = JSONFormat::ModernV3,
        const Provenance& prov = {}
    ) const {
        if (format == JSONFormat::Legacy) {
            return to_legacy_json(res);
        }
        return to_modern_json(res, prov);
    }

    void save_json(
        const std::string& filepath,
        const BUSTEDResult& res,
        JSONFormat format = JSONFormat::ModernV3,
        const Provenance& prov = {}
    ) const {
        std::ofstream out(filepath);
        if (!out.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filepath);
        }
        nlohmann::json j = to_json(res, format, prov);
        out << j.dump(2) << "\n";
    }

};

} // namespace hyphy::analyses
