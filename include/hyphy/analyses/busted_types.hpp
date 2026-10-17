#pragma once

#include "hyphy/core/types.hpp"
#include "hyphy/core/stats.hpp"
#include <string>
#include <vector>

namespace hyphy::analyses {

using namespace hyphy::core;

inline Scalar busted_regularized_gamma_p(Scalar a, Scalar z) {
    return hyphy::core::stats::regularized_gamma_p(a, z);
}

inline Scalar busted_chi2_p_value(Scalar chi2, Scalar df) {
    return hyphy::core::stats::chi2_p_value(chi2, df);
}

struct BUSTEDErrorEvent {
    size_t site = 0;           // 1-based codon site
    std::string branch_name;   // Node/branch name
    Scalar posterior = 0.0;    // Posterior probability P(sink | b, s)
    Scalar bayes_factor = 0.0; // Empirical Bayes factor for error sink at (b, s)
};

struct BUSTEDRateDistribution {
    std::vector<Scalar> omegas;
    std::vector<Scalar> weights;
    std::vector<Scalar> syn_rates;
    std::vector<Scalar> syn_weights;
    Scalar error_sink_weight = 0.0;       // Contamination weight epsilon in [0, error_sink_max]
    std::vector<std::string> annotations; // per-class, parallel to omegas/weights; "" = unannotated
};

struct BUSTEDFit {
    Scalar log_likelihood = 0.0;
    Scalar aicc = 0.0;
    Scalar tree_scale = 1.0;
    size_t num_rate_classes = 3;
    BUSTEDRateDistribution test_distribution;
    bool has_background_distribution = false;
    BUSTEDRateDistribution background_distribution;
    std::vector<Scalar> branch_lengths; // Branch lengths per node id
    std::vector<Scalar> site_log_likelihoods;
    Scalar delta = 0.0;             // Double-hit rate parameter
    Scalar psi = 0.0;               // Triple-hit rate parameter
    Scalar frac_delta = 0.0;        // Fraction of 2-hit substitutions
    Scalar frac_psi = 0.0;          // Fraction of 3-hit substitutions
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
    bool full_auto = false;                // Enable 4-gate surprisal & score automatic model selector (BUSTED-Auto)
    size_t num_rate_classes = 3;           // Default K = 3
    bool srv = false;                      // Enable synonymous rate variation (BUSTED-S)
    size_t num_syn_rate_classes = 3;       // Default M = 3 for alpha
    bool auto_select_k = false;            // Automatically select K in {1, ..., max_k} via AICc
    size_t max_k = 3;                      // Max K to test in auto_select_k
    bool refine_branch_lengths = true;     // Re-optimize individual branch lengths
    Scalar p_value_threshold = 0.05;
    std::string multiple_hits = "None";    // "None", "Double", "Double+Triple"
    bool error_sink = false;               // Enable Option A saturation/replacement error sink
    Scalar error_sink_max_weight = 0.01;   // Upper bound on contamination proportion epsilon
    Scalar error_sink_bf_threshold = 100.0; // Bayes factor cutoff to flag branch-site errors
    std::string test_branches = "All";     // "All", "Internal", "Leaves", "Unlabeled", or tag/comma-separated names
    bool phenotype_mode = false;           // Run BUSTED-PH (Test selection, Background selection, and Shared distribution LRTs)
    bool relax_mode = false;               // Fit coupled-k RELAX model (omega_FG = omega_BG^k, shared weights) vs shared_distributions (k = 1)
    bool include_relax_in_ph = true;       // Automatically run coupled-k RELAX fit inside BUSTED-PH workflow
    std::string comparison_branches = "";  // Optional comparison/background branch set for BUSTED-PH (default: complement of test_branches)
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
    // BUSTED-PH fields
    bool has_background_test = false;
    Scalar background_lrt = 0.0;
    Scalar background_p_value = 1.0;
    Scalar shared_lrt = 0.0;
    Scalar shared_p_value = 1.0;
    size_t shared_df = 0;
    Scalar union_intersection_p_value = 1.0;  // max(p_FG, p_diff)
    Scalar background_delta_aicc = 0.0;       // AICc(Constrained BG) - AICc(Unconstrained)
    bool background_neutral_supported = true; // Delta AICc <= 2.0 (or p_BG > 0.068)
    Scalar ranking_score = 0.0;               // -log10(max(1e-8, p_FG)) - log10(max(1e-8, p_diff)) + log10(max(1e-8, p_BG))
    bool phenotype_associated = false;
    std::string phenotype_verdict;
    BUSTEDFit constrained_background;
    BUSTEDFit shared_distributions;
    // Coupled-k RELAX test fields (M_shared [k=1] vs M_RELAX [omega_FG = omega_BG^k, shared weights])
    bool has_relax_test = false;
    Scalar relax_k = 1.0;
    Scalar relax_lrt = 0.0;
    Scalar relax_p_value = 1.0;
    std::string relax_direction = "None"; // "Intensification", "Relaxation", or "None"
    BUSTEDFit relax_alternative;

    std::vector<Scalar> evidence_ratios; // Bayes factors per site for omega_K > 1
    std::vector<Scalar> site_error_bfs;  // Max error sink Bayes factor across branches per site
    std::vector<BUSTEDErrorEvent> flagged_error_events; // Branch-site pairs with BF_sink >= threshold
    double runtime_seconds = 0.0;
    bool is_failed = false;
    std::string error_message = "";
};

} // namespace hyphy::analyses
