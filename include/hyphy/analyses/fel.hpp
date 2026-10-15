#pragma once

#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/core/progress_bar.hpp"
#include "hyphy/core/provenance.hpp"
#include "hyphy/core/json_output.hpp"
#include "hyphy/analyses/site_visualizer.hpp"
#include "LBFGSB.h"
#include "hyphy/opt/nelder_mead.hpp"
#include "hyphy/opt/optimizer.hpp"
#include "nlohmann/json.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <regex>
#include <unordered_set>
#include <set>
#include <sstream>

namespace hyphy::analyses {

using namespace hyphy::core;
using namespace hyphy::opt;

struct SubstitutionEvent {
    int32_t node_id = -1;
    std::string branch_name;
    bool is_tested = true;
    Scalar branch_length = 0.0;

    int8_t parent_codon_idx = -1;
    int8_t child_codon_idx = -1;
    std::string parent_codon;
    std::string child_codon;

    char parent_aa = '?';
    char child_aa = '?';

    int nt_distance = 0;       // 1, 2, or 3
    bool is_synonymous = false; // true if parent_aa == child_aa

    Scalar transition_prob = 0.0;
    Scalar surprise_bits = 0.0; // Shannon self-information: -log2(transition_prob)
};

struct SiteSubstitutions {
    size_t site_index = 0;
    std::string root_codon;
    char root_aa = '?';
    std::vector<SubstitutionEvent> events;

    size_t syn_test = 0;
    size_t nonsyn_test = 0;
    size_t syn_bg = 0;
    size_t nonsyn_bg = 0;

    size_t subs_1nt = 0;
    size_t subs_2nt = 0;
    size_t subs_3nt = 0;

    Scalar mean_surprise_bits = 0.0;
    Scalar total_surprise_bits = 0.0;
    Scalar mean_branch_length = 0.0;
    Scalar site_perplexity = 1.0;

    std::string summary_string = "Conserved";
    std::string summary_string_partitioned = "Conserved";
};

struct SiteResult {
    size_t site_index = 0;
    Scalar alpha = 0.0;
    Scalar beta = 0.0;             // on tested branches
    Scalar beta_nuisance = 0.0;    // on background branches (nuisance parameter)
    Scalar alpha_null = 0.0;
    Scalar lrt = 0.0;
    Scalar p_value = 1.0;
    Scalar total_branch_length = 0.0; // tested branches length
    Scalar log_l_alt = 0.0;
    Scalar log_l_null = 0.0;
    SiteSubstitutions subs;
};

struct FELSettings {
    Scalar p_value_threshold = 0.1;
    bool full_model = true;
    bool srv = true;
    std::string test_branches = "All"; // "All", "Internal", "Leaves", "Unlabeled", or tag/comma-separated names
    std::string test_branch_regex = ""; // Optional regex to match branch names
    std::vector<std::string> test_branch_names = {}; // Optional explicit branch names
};

class FELAnalyzer {
public:
    Tree tree;
    Alignment aln;
    MG94Parameters base_params;
    FELSettings settings;

    Scalar p_value_threshold = 0.1;
    Scalar baseline_tree_length = 0.0;
    Scalar baseline_tested_tree_length = 0.0;
    Scalar baseline_bg_tree_length = 0.0;
    Scalar conversion_factor = 1.0;
    Scalar global_log_l = 0.0;
    Scalar global_aicc = 0.0;
    std::vector<SiteResult> site_results;

    bool has_gtr_fit = false;
    Scalar gtr_log_l = 0.0;
    Scalar gtr_aicc = 0.0;
    GTRParameters gtr_rates;
    std::unordered_map<std::string, Scalar> gtr_branch_lengths;
    bool full_model = true;

    // Branch selection partition:
    std::vector<bool> is_test_branch;
    std::vector<std::string> test_branch_names;
    std::vector<std::string> bg_branch_names;
    size_t num_tested = 0;
    size_t num_background = 0;

    MG94Matrix base_mg94_matrix;
    std::vector<Matrix> branch_transition_matrices;

    FELAnalyzer(Tree t, Alignment a, MG94Parameters params = MG94Parameters{}, FELSettings s = FELSettings{})
        : tree(std::move(t)), aln(std::move(a)), base_params(params), settings(std::move(s)) {
        p_value_threshold = settings.p_value_threshold;
        full_model = settings.full_model;
        prepare_branch_lengths();
    }

    // Static factory that performs Phase 1 GTR and Phase 2 MG94 fitting automatically
    static FELAnalyzer create_and_fit(
        Tree input_tree,
        Alignment aln,
        FELSettings settings = FELSettings{},
        std::function<void(const std::string&, double)> progress_cb = nullptr
    ) {
        if (progress_cb) progress_cb("Phase 1: Fitting Nucleotide GTR Model", 0.1);
        GTRFitter gtr_fitter(input_tree, aln);
        auto gtr_res = gtr_fitter.fit();

        if (progress_cb) progress_cb("Phase 2: Refining under Global MG94 Model", 0.3);
        MG94Parameters base_p;
        base_p.theta_AC = gtr_res.params.theta_AC;
        base_p.theta_AT = gtr_res.params.theta_AT;
        base_p.theta_CG = gtr_res.params.theta_CG;
        base_p.theta_CT = gtr_res.params.theta_CT;
        base_p.theta_GT = gtr_res.params.theta_GT;

        MG94Fitter mg_fitter(gtr_res.tree, aln);
        FitResult mg_res;
        if (settings.full_model) {
            mg_res = mg_fitter.fit_full_model(1.0, base_p, progress_cb);
            base_p = mg_res.params;
        } else {
            mg_res = mg_fitter.fit_omega_and_scale(1.0, base_p);
            base_p.alpha = 1.0;
            base_p.beta = mg_res.x_opt(0);
        }

        FELAnalyzer analyzer(mg_res.tree, std::move(aln), base_p, settings);
        analyzer.global_log_l = mg_res.log_likelihood;
        size_t K_mg = 5 + 1 + (mg_res.tree.num_nodes() - 1);
        size_t N_codons = analyzer.aln.num_codons;
        analyzer.global_aicc = -2.0 * mg_res.log_likelihood + 2.0 * K_mg * N_codons / (N_codons - K_mg - 1.0);

        analyzer.has_gtr_fit = true;
        analyzer.gtr_log_l = gtr_res.log_likelihood;
        analyzer.gtr_aicc = gtr_res.aicc;
        analyzer.gtr_rates = gtr_res.params;
        for (const auto& node : gtr_res.tree.nodes) {
            if (node.id != gtr_res.tree.root_id) {
                analyzer.gtr_branch_lengths[node.name] = node.branch_length;
            }
        }
        return analyzer;
    }

    // Overload for backward compatibility
    static FELAnalyzer create_and_fit(
        Tree input_tree,
        Alignment aln,
        Scalar pvalue_threshold,
        std::function<void(const std::string&, double)> progress_cb = nullptr,
        bool full_model = true
    ) {
        FELSettings s;
        s.p_value_threshold = pvalue_threshold;
        s.full_model = full_model;
        return create_and_fit(std::move(input_tree), std::move(aln), s, progress_cb);
    }

    void partition_branches() {
        size_t n = tree.num_nodes();
        is_test_branch.assign(n, false);
        test_branch_names.clear();
        bg_branch_names.clear();
        num_tested = 0;
        num_background = 0;

        bool has_tagged_test = false;
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                if (node.model_tag == "T" || node.model_tag == "Test" || node.model_tag == "test" ||
                    node.model_tag == "Foreground" || node.model_tag == "fg") {
                    has_tagged_test = true;
                    break;
                }
            }
        }

        std::regex test_regex;
        bool use_test_regex = false;
        if (!settings.test_branch_regex.empty()) {
            try {
                test_regex = std::regex(settings.test_branch_regex, std::regex::icase);
                use_test_regex = true;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid test branch regex: " << e.what() << "\n";
            }
        }

        std::unordered_set<std::string> explicit_test_set(
            settings.test_branch_names.begin(), settings.test_branch_names.end()
        );

        if (settings.test_branches.find(',') != std::string::npos) {
            std::stringstream ss(settings.test_branches);
            std::string item;
            while (std::getline(ss, item, ',')) {
                size_t start = item.find_first_not_of(" \t\r\n");
                size_t end = item.find_last_not_of(" \t\r\n");
                if (start != std::string::npos && end != std::string::npos) {
                    explicit_test_set.insert(item.substr(start, end - start + 1));
                }
            }
        }

        std::string mode = settings.test_branches;
        std::string mode_lower = mode;
        std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(), ::tolower);

        for (const auto& node : tree.nodes) {
            if (node.id == tree.root_id) continue;

            bool is_test = false;
            if (use_test_regex) {
                is_test = std::regex_search(node.name, test_regex);
            } else if (!explicit_test_set.empty()) {
                is_test = explicit_test_set.count(node.name) > 0;
            } else if (mode_lower == "all" || mode.empty()) {
                if (has_tagged_test) {
                    is_test = (node.model_tag == "T" || node.model_tag == "Test" || node.model_tag == "test" ||
                               node.model_tag == "Foreground" || node.model_tag == "fg");
                } else {
                    is_test = true;
                }
            } else if (mode_lower == "all-branches" || mode_lower == "all_branches") {
                is_test = true;
            } else if (mode_lower == "internal") {
                is_test = !node.is_leaf;
            } else if (mode_lower == "leaves" || mode_lower == "terminal") {
                is_test = node.is_leaf;
            } else if (mode_lower == "unlabeled") {
                is_test = node.model_tag.empty();
            } else if (mode_lower == "t" || mode_lower == "test" || mode_lower == "foreground" || mode_lower == "fg") {
                is_test = (node.model_tag == "T" || node.model_tag == "Test" || node.model_tag == "test" ||
                           node.model_tag == "Foreground" || node.model_tag == "fg");
            } else {
                std::string tag_lower = node.model_tag;
                std::transform(tag_lower.begin(), tag_lower.end(), tag_lower.begin(), ::tolower);
                std::string name_lower = node.name;
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

                if (!tag_lower.empty() && tag_lower == mode_lower) {
                    is_test = true;
                } else if (name_lower == mode_lower) {
                    is_test = true;
                }
            }

            is_test_branch[node.id] = is_test;
            if (is_test) {
                test_branch_names.push_back(node.name);
                num_tested++;
            } else {
                bg_branch_names.push_back(node.name);
                num_background++;
            }
        }
    }

    void prepare_branch_lengths() {
        partition_branches();

        base_mg94_matrix.update(base_params, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, *(aln.code ? aln.code : GeneticCode::universal()));
        Scalar Q_scale = base_mg94_matrix.scale_factor;
        conversion_factor = (Q_scale > 1e-12) ? (3.0 / Q_scale) : 1.0;

        baseline_tree_length = 0.0;
        baseline_tested_tree_length = 0.0;
        baseline_bg_tree_length = 0.0;
        for (auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                baseline_tree_length += node.branch_length;
                if (is_test_branch[node.id]) {
                    baseline_tested_tree_length += node.branch_length;
                } else {
                    baseline_bg_tree_length += node.branch_length;
                }
                node.branch_length *= conversion_factor;
            }
        }
        tree.scale_bridge_lengths(conversion_factor);

        branch_transition_matrices.resize(tree.num_nodes());
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                branch_transition_matrices[node.id] = base_mg94_matrix.transition_matrix(node.branch_length);
            }
        }
    }

    // Evaluate single pattern log-likelihood with underflow protection
    static Scalar compute_pattern_log_likelihood(
        const Tree& tree,
        const SitePattern& pattern,
        const std::vector<size_t>& leaf_to_taxon,
        const MG94Matrix& mg94_model,
        const std::vector<Matrix>& P_branches,
        std::vector<Vector>& node_L,
        std::vector<double>& node_log_scale
    ) {
        int S = mg94_model.pi.size();
        const double SCALER_THRESH = 1e-150;
        const double SCALER_UP = 1e150;
        const double LOG_SCALER = std::log(SCALER_UP);

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
                tree.eval_internal_node(node_id, P_branches, node_L, node_log_scale, node_L[node_id], child_scale_sum);

                double max_val = node_L[node_id].maxCoeff();
                int boost_count = 0;
                while (max_val < SCALER_THRESH && max_val > 0.0) {
                    node_L[node_id] *= SCALER_UP;
                    max_val *= SCALER_UP;
                    boost_count++;
                }
                node_log_scale[node_id] = child_scale_sum - boost_count * LOG_SCALER;
            }
        }

        Scalar pat_l = mg94_model.pi.dot(node_L[tree.root_id]);
        if (pat_l > 0.0) {
            return std::log(pat_l) + node_log_scale[tree.root_id];
        } else {
            return -1e20;
        }
    }

    // Backward-compatible unscaled pattern likelihood
    static Scalar compute_pattern_likelihood(
        const Tree& tree,
        const SitePattern& pattern,
        const std::vector<size_t>& leaf_to_taxon,
        const MG94Matrix& mg94_model,
        const std::vector<Matrix>& P_branches,
        std::vector<Vector>& node_L
    ) {
        std::vector<double> node_log_scale(tree.num_nodes(), 0.0);
        Scalar log_l = compute_pattern_log_likelihood(tree, pattern, leaf_to_taxon, mg94_model, P_branches, node_L, node_log_scale);
        return (log_l > -1e10) ? std::exp(log_l) : 0.0;
    }

    // Optimize single pattern
    SiteResult analyze_pattern(size_t p_idx, const std::vector<size_t>& leaf_to_taxon) const {
        const auto& pattern = aln.patterns[p_idx];
        const auto& gcode = *(aln.code ? aln.code : GeneticCode::universal());
        int S = gcode.num_sense_codons;

        SiteResult res;
        res.site_index = p_idx;

        // Check if invariant (all non-gap states identical)
        int8_t first_state = -1;
        bool is_invariant = true;
        for (int8_t s : pattern.states) {
            if (s >= 0) {
                if (first_state == -1) {
                    first_state = s;
                } else if (s != first_state) {
                    is_invariant = false;
                    break;
                }
            }
        }

        if (is_invariant) {
            res.alpha = 0.0;
            res.beta = 0.0;
            res.beta_nuisance = 0.0;
            res.alpha_null = 0.0;
            res.lrt = 0.0;
            res.p_value = 1.0;
            res.total_branch_length = 0.0;
            return res;
        }

        std::vector<Vector> thread_node_L(tree.num_nodes(), Vector::Zero(S));
        std::vector<double> thread_node_scale(tree.num_nodes(), 0.0);
        std::vector<Matrix> P_branches(tree.num_nodes());

        if (num_background == 0) {
            // Fast path: All branches tested (2-parameter Alternative vs 1-parameter Null)
            auto eval_pattern_lnl = [&](Scalar a, Scalar b) -> Scalar {
                MG94Parameters p = base_params;
                p.alpha = a;
                p.beta = b;

                MG94Matrix mg;
                mg.update(p, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, gcode);

                for (const auto& node : tree.nodes) {
                    if (node.id != tree.root_id) {
                        P_branches[node.id] = mg.transition_matrix(node.branch_length);
                    }
                }

                return compute_pattern_log_likelihood(tree, pattern, leaf_to_taxon, mg, P_branches, thread_node_L, thread_node_scale);
            };

            const std::vector<std::pair<Scalar, Scalar>> start_grid = {
                {0.01, 0.1}, {1.0, 0.1}, {1.0, 0.5}, {1.0, 1.0},
                {1.0, 5.0}, {10.0, 0.1}, {0.01, 0.5}, {0.01, 5.0},
                {10.0, 0.5}, {10.0, 1.0}, {10.0, 50.0}, {100.0, 1.0}
            };

            Scalar best_grid_lnl = -1e20;
            Scalar best_a = 1.0, best_b = 1.0;
            for (const auto& pt : start_grid) {
                Scalar lnl = eval_pattern_lnl(pt.first, pt.second);
                if (lnl > best_grid_lnl) {
                    best_grid_lnl = lnl;
                    best_a = pt.first;
                    best_b = pt.second;
                }
            }

            auto obj_alt = [&](Scalar a, Scalar b) -> Scalar {
                return -eval_pattern_lnl(a, b);
            };
            auto pt_alt = NelderMead2D::minimize(obj_alt, best_a, best_b, 1e-4, 1000.0, 1e-4, 100);
            res.alpha = pt_alt.a;
            res.beta = pt_alt.b;
            res.beta_nuisance = 0.0;
            res.log_l_alt = -pt_alt.f;

            auto obj_null = [&](Scalar a) -> Scalar {
                return -eval_pattern_lnl(a, a);
            };

            Scalar init_null = (std::min(res.alpha, 100.0) + 3.0 * std::min(res.beta, 100.0)) / 4.0;
            Scalar best_null_lnl = eval_pattern_lnl(init_null, init_null);
            Scalar best_null_a = init_null;
            for (Scalar cand : {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
                Scalar lnl = eval_pattern_lnl(cand, cand);
                if (lnl > best_null_lnl) {
                    best_null_lnl = lnl;
                    best_null_a = cand;
                }
            }

            Scalar ax = 1e-4;
            Scalar bx = std::clamp(best_null_a, 0.01, 45.0);
            Scalar cx = 50.0;
            auto [opt_null_a, opt_null_nll] = Brent1D::minimize(obj_null, ax, bx, cx, 1e-4, 60);

            res.alpha_null = opt_null_a;
            res.log_l_null = -opt_null_nll;

            Scalar lrt = 0.0;
            if (res.log_l_alt > -1e10 && res.log_l_null > -1e10) {
                lrt = 2.0 * (res.log_l_alt - res.log_l_null);
                if (lrt < 0.0) lrt = 0.0;
            }
            res.lrt = lrt;
            res.p_value = std::erfc(std::sqrt(lrt / 2.0));
            res.total_branch_length = res.alpha_null * baseline_tested_tree_length;

            return res;
        }

        // Branch subset path: Test vs Background (3-parameter Alternative vs 2-parameter Null)
        MG94Matrix mg_test, mg_bg;

        auto eval_pattern_lnl_subset = [&](Scalar a, Scalar b_test, Scalar b_bg) -> Scalar {
            MG94Parameters p_t = base_params;
            p_t.alpha = a;
            p_t.beta = b_test;
            mg_test.update(p_t, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, gcode);

            if (std::abs(b_test - b_bg) < 1e-9) {
                for (const auto& node : tree.nodes) {
                    if (node.id != tree.root_id) {
                        P_branches[node.id] = mg_test.transition_matrix(node.branch_length);
                    }
                }
            } else {
                MG94Parameters p_b = base_params;
                p_b.alpha = a;
                p_b.beta = b_bg;
                mg_bg.update(p_b, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, gcode);

                for (const auto& node : tree.nodes) {
                    if (node.id != tree.root_id) {
                        if (is_test_branch[node.id]) {
                            P_branches[node.id] = mg_test.transition_matrix(node.branch_length);
                        } else {
                            P_branches[node.id] = mg_bg.transition_matrix(node.branch_length);
                        }
                    }
                }
            }

            return compute_pattern_log_likelihood(tree, pattern, leaf_to_taxon, mg_test, P_branches, thread_node_L, thread_node_scale);
        };

        const std::vector<std::array<Scalar, 3>> start_grid_3d = {
            {0.01, 0.1, 0.1}, {1.0, 0.1, 0.1}, {1.0, 0.5, 0.5}, {1.0, 1.0, 1.0},
            {1.0, 5.0, 5.0}, {10.0, 0.1, 0.1}, {0.01, 0.5, 0.5}, {0.01, 5.0, 5.0},
            {10.0, 0.5, 0.5}, {10.0, 1.0, 1.0}, {10.0, 50.0, 50.0}, {100.0, 1.0, 1.0}
        };

        Scalar best_grid_lnl = -1e20;
        Scalar best_a = 1.0, best_b_test = 1.0, best_b_bg = 1.0;
        for (const auto& pt : start_grid_3d) {
            Scalar lnl = eval_pattern_lnl_subset(pt[0], pt[1], pt[2]);
            if (lnl > best_grid_lnl) {
                best_grid_lnl = lnl;
                best_a = pt[0];
                best_b_test = pt[1];
                best_b_bg = pt[2];
            }
        }

        // Alternative Model: Nelder-Mead 3D on (alpha, beta_test, beta_bg)
        auto obj_alt = [&](const std::array<Scalar, 3>& x) -> Scalar {
            return -eval_pattern_lnl_subset(x[0], x[1], x[2]);
        };
        std::array<Scalar, 3> init_alt = {best_a, best_b_test, best_b_bg};
        std::array<Scalar, 3> lb_alt = {1e-4, 1e-4, 1e-4};
        std::array<Scalar, 3> ub_alt = {1000.0, 1000.0, 1000.0};
        auto pt_alt = NelderMeadND<3>::minimize(obj_alt, init_alt, lb_alt, ub_alt, 1e-4, 100);

        res.alpha = pt_alt.x[0];
        res.beta = pt_alt.x[1];
        res.beta_nuisance = pt_alt.x[2];
        res.log_l_alt = -pt_alt.f;

        // Null Model: beta_test = alpha, beta_bg free. Nelder-Mead 2D on (alpha, beta_bg)
        auto obj_null = [&](const std::array<Scalar, 2>& x) -> Scalar {
            return -eval_pattern_lnl_subset(x[0], x[0], x[1]);
        };

        Scalar init_null_a = (std::min(res.alpha, 100.0) + 3.0 * std::min(res.beta, 100.0)) / 4.0;
        Scalar init_null_bg = res.beta_nuisance;

        Scalar best_null_lnl = -obj_null({init_null_a, init_null_bg});
        Scalar best_null_a = init_null_a;
        Scalar best_null_bg = init_null_bg;

        for (Scalar cand_a : {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
            for (Scalar cand_bg : {init_null_bg, 0.1, 1.0, 5.0}) {
                Scalar lnl = -obj_null({cand_a, cand_bg});
                if (lnl > best_null_lnl) {
                    best_null_lnl = lnl;
                    best_null_a = cand_a;
                    best_null_bg = cand_bg;
                }
            }
        }

        std::array<Scalar, 2> init_null = {best_null_a, best_null_bg};
        std::array<Scalar, 2> lb_null = {1e-4, 1e-4};
        std::array<Scalar, 2> ub_null = {1000.0, 1000.0};
        auto pt_null = NelderMeadND<2>::minimize(obj_null, init_null, lb_null, ub_null, 1e-4, 80);

        res.alpha_null = pt_null.x[0];
        res.log_l_null = -pt_null.f;

        // LRT & p-value
        Scalar lrt = 0.0;
        if (res.log_l_alt > -1e10 && res.log_l_null > -1e10) {
            lrt = 2.0 * (res.log_l_alt - res.log_l_null);
            if (lrt < 0.0) lrt = 0.0;
        }
        res.lrt = lrt;
        res.p_value = std::erfc(std::sqrt(lrt / 2.0));
        res.total_branch_length = res.alpha_null * baseline_tested_tree_length;

        return res;
    }

    static std::string format_undirected_summary(const std::vector<SubstitutionEvent>& evs, size_t s_count, size_t n_count) {
        if (n_count == 0 && s_count == 0) {
            return "Conserved";
        }
        if (n_count == 0) {
            bool syn_high_perpl = false;
            for (const auto& ev : evs) {
                if (ev.is_synonymous && ev.surprise_bits >= 10.0) {
                    syn_high_perpl = true;
                    break;
                }
            }
            std::string s = "Conserved (+" + std::to_string(s_count) + " syn";
            if (syn_high_perpl) s += "⚡";
            s += ")";
            return s;
        }

        struct PairInfo {
            char a1;
            char a2;
            int max_nt = 1;
            size_t count = 0;
            bool high_perpl = false;
        };

        // Canonical undirected pair (min_aa, max_aa)
        std::map<std::pair<char, char>, PairInfo> pairs;
        bool any_high_perpl = false;
        for (const auto& ev : evs) {
            if (ev.is_synonymous) continue;
            char u = std::min(ev.parent_aa, ev.child_aa);
            char v = std::max(ev.parent_aa, ev.child_aa);
            auto& info = pairs[{u, v}];
            info.a1 = u;
            info.a2 = v;
            info.count++;
            if (ev.nt_distance > info.max_nt) {
                info.max_nt = ev.nt_distance;
            }
            if (ev.surprise_bits >= 10.0) {
                info.high_perpl = true;
                any_high_perpl = true;
            }
        }

        // If there are more than 3 distinct pairs, use Option C compact archetype:
        // e.g. "8 NS [5 multi-hit]⚡ across 8 AAs (+1 syn)"
        if (pairs.size() > 3) {
            std::set<char> unique_aas;
            size_t multi_hit_events = 0;
            for (const auto& ev : evs) {
                if (ev.is_synonymous) continue;
                unique_aas.insert(ev.parent_aa);
                unique_aas.insert(ev.child_aa);
                if (ev.nt_distance > 1) {
                    multi_hit_events++;
                }
            }

            std::ostringstream ss;
            ss << n_count << " NS";
            if (multi_hit_events > 0) {
                ss << " [" << multi_hit_events << " multi-hit]";
            }
            if (any_high_perpl) {
                ss << "⚡";
            }
            ss << " across " << unique_aas.size() << " AAs";
            if (s_count > 0) {
                ss << " (+" << s_count << " syn)";
            }
            return ss.str();
        }

        std::ostringstream ss;
        bool first = true;
        for (const auto& [pair, info] : pairs) {
            if (!first) ss << ", ";
            ss << info.a1 << " ↔ " << info.a2;
            if (info.max_nt > 1) {
                ss << " [" << info.max_nt << "nt]";
            }
            if (info.count > 1) {
                ss << " (" << info.count << "×)";
            }
            if (info.high_perpl) {
                ss << "⚡";
            }
            first = false;
        }

        if (s_count > 0) {
            ss << " (+" << s_count << " syn)";
        }

        return ss.str();
    }

    static void format_substitution_strings(SiteSubstitutions& subs, bool has_background) {
        if (subs.events.empty()) {
            subs.summary_string = "Conserved";
            subs.summary_string_partitioned = "Conserved";
            return;
        }

        size_t total_s = subs.syn_test + subs.syn_bg;
        size_t total_n = subs.nonsyn_test + subs.nonsyn_bg;
        subs.summary_string = format_undirected_summary(subs.events, total_s, total_n);

        if (has_background) {
            std::vector<SubstitutionEvent> test_evs;
            for (const auto& ev : subs.events) {
                if (ev.is_tested) test_evs.push_back(ev);
            }
            subs.summary_string_partitioned = format_undirected_summary(test_evs, subs.syn_test, subs.nonsyn_test);
        } else {
            subs.summary_string_partitioned = subs.summary_string;
        }
    }

    SiteSubstitutions reconstruct_ancestral_pattern(size_t p, const std::vector<size_t>& leaf_to_taxon) const {
        SiteSubstitutions subs;
        const auto& pattern = aln.patterns[p];
        const auto& code = *(aln.code ? aln.code : GeneticCode::universal());
        int S = code.num_sense_codons;
        size_t num_nodes = tree.num_nodes();

        // 1. Check if pattern is constant across all observed taxa
        int8_t const_codon = -1;
        bool is_constant = true;
        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            if (tree.nodes[node_id].is_leaf) {
                size_t t_idx = leaf_to_taxon[node_id];
                if (t_idx != static_cast<size_t>(-1)) {
                    int8_t st = pattern.states[t_idx];
                    if (st >= 0 && st < S) {
                        if (const_codon == -1) {
                            const_codon = st;
                        } else if (const_codon != st) {
                            is_constant = false;
                            break;
                        }
                    }
                }
            }
        }

        if (is_constant && const_codon >= 0) {
            subs.root_codon = code.sense_codons[const_codon];
            subs.root_aa = code.sense_codon_aa[const_codon];
            subs.summary_string = "Conserved";
            subs.summary_string_partitioned = "Conserved";
            return subs;
        }

        // 2. Upward max-product pass
        std::vector<std::vector<Scalar>> M_node(num_nodes, std::vector<Scalar>(S, 1.0));
        std::vector<std::vector<int8_t>> backpointer(num_nodes, std::vector<int8_t>(S, 0));
        std::vector<std::vector<Scalar>> val_node(num_nodes, std::vector<Scalar>(S, 1.0));

        for (int32_t node_id : tree.post_order) {
            const auto& node = tree.nodes[node_id];
            if (node.is_leaf) {
                size_t t_idx = leaf_to_taxon[node_id];
                int8_t st = (t_idx != static_cast<size_t>(-1)) ? pattern.states[t_idx] : -1;
                if (st >= 0 && st < S) {
                    for (int c = 0; c < S; ++c) {
                        M_node[node_id][c] = (c == st) ? 1.0 : 0.0;
                    }
                } else {
                    for (int c = 0; c < S; ++c) {
                        M_node[node_id][c] = 1.0;
                    }
                }
            }

            if (node_id != tree.root_id) {
                const Matrix& P = branch_transition_matrices[node_id];
                Scalar max_val = 0.0;
                for (int p_state = 0; p_state < S; ++p_state) {
                    Scalar best_score = -1.0;
                    int8_t best_c = 0;
                    for (int c = 0; c < S; ++c) {
                        Scalar score = P(p_state, c) * M_node[node_id][c];
                        if (score > best_score) {
                            best_score = score;
                            best_c = static_cast<int8_t>(c);
                        }
                    }
                    backpointer[node_id][p_state] = best_c;
                    val_node[node_id][p_state] = std::max(best_score, 0.0);
                    if (val_node[node_id][p_state] > max_val) {
                        max_val = val_node[node_id][p_state];
                    }
                }

                // Underflow scaling
                if (max_val > 0.0) {
                    Scalar scale = 1.0 / max_val;
                    for (int p_state = 0; p_state < S; ++p_state) {
                        val_node[node_id][p_state] *= scale;
                    }
                }

                // Multiply into parent's M_node
                int32_t parent_id = node.parent_id;
                for (int p_state = 0; p_state < S; ++p_state) {
                    M_node[parent_id][p_state] *= val_node[node_id][p_state];
                }
            }
        }

        // 3. Optimal root state
        int8_t root_state = 0;
        Scalar best_root_score = -1.0;
        for (int c = 0; c < S; ++c) {
            Scalar score = base_mg94_matrix.pi(c) * M_node[tree.root_id][c];
            if (score > best_root_score) {
                best_root_score = score;
                root_state = static_cast<int8_t>(c);
            }
        }

        // 4. Downward traceback
        std::vector<int8_t> reconstructed_states(num_nodes, -1);
        reconstructed_states[tree.root_id] = root_state;

        subs.root_codon = code.sense_codons[root_state];
        subs.root_aa = code.sense_codon_aa[root_state];

        std::vector<int32_t> stack;
        stack.push_back(tree.root_id);
        while (!stack.empty()) {
            int32_t curr = stack.back();
            stack.pop_back();

            for (int32_t child_id : tree.nodes[curr].children) {
                int8_t parent_st = reconstructed_states[curr];
                reconstructed_states[child_id] = backpointer[child_id][parent_st];
                stack.push_back(child_id);
            }
        }

        // 5. Collect substitution events
        Scalar total_surprise = 0.0;
        Scalar total_mut_branch_length = 0.0;

        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            if (static_cast<int32_t>(node_id) == tree.root_id) continue;
            const auto& node = tree.nodes[node_id];
            int32_t parent_id = node.parent_id;
            if (parent_id < 0) continue;
            int8_t parent_st = reconstructed_states[parent_id];
            int8_t child_st = reconstructed_states[node_id];

            if (parent_st != child_st && parent_st >= 0 && child_st >= 0) {
                SubstitutionEvent ev;
                ev.node_id = static_cast<int32_t>(node_id);
                ev.branch_name = node.name;
                ev.is_tested = is_test_branch[node_id];
                ev.branch_length = (conversion_factor > 0.0) ? (node.branch_length / conversion_factor) : node.branch_length;
                ev.parent_codon_idx = parent_st;
                ev.child_codon_idx = child_st;
                ev.parent_codon = code.sense_codons[parent_st];
                ev.child_codon = code.sense_codons[child_st];
                ev.parent_aa = code.sense_codon_aa[parent_st];
                ev.child_aa = code.sense_codon_aa[child_st];

                const auto& diff = code.diff_matrix[parent_st][child_st];
                ev.nt_distance = diff.num_diffs;
                ev.is_synonymous = (ev.parent_aa == ev.child_aa);

                const Matrix& P = branch_transition_matrices[node_id];
                ev.transition_prob = std::max(P(parent_st, child_st), 1e-15);
                ev.surprise_bits = -std::log2(ev.transition_prob);

                total_surprise += ev.surprise_bits;
                total_mut_branch_length += ev.branch_length;

                if (ev.is_tested) {
                    if (ev.is_synonymous) subs.syn_test++;
                    else subs.nonsyn_test++;
                } else {
                    if (ev.is_synonymous) subs.syn_bg++;
                    else subs.nonsyn_bg++;
                }

                if (ev.nt_distance == 1) subs.subs_1nt++;
                else if (ev.nt_distance == 2) subs.subs_2nt++;
                else if (ev.nt_distance == 3) subs.subs_3nt++;

                subs.events.push_back(std::move(ev));
            }
        }

        if (!subs.events.empty()) {
            subs.total_surprise_bits = total_surprise;
            subs.mean_surprise_bits = total_surprise / subs.events.size();
            subs.mean_branch_length = total_mut_branch_length / subs.events.size();
            subs.site_perplexity = std::pow(2.0, subs.mean_surprise_bits);
        }

        format_substitution_strings(subs, num_background > 0);

        return subs;
    }

    std::vector<SiteResult> run(bool show_progress = false, bool force_progress = false) {
        std::vector<size_t> leaf_to_taxon(tree.num_nodes(), static_cast<size_t>(-1));
        for (const auto& node : tree.nodes) {
            if (node.is_leaf) {
                leaf_to_taxon[node.id] = LikelihoodEngine::find_taxon_index(aln, node.name);
            }
        }

        size_t num_patterns = aln.patterns.size();
        std::vector<SiteResult> pattern_results(num_patterns);

        SiteVisualizer viz(aln.num_codons, num_patterns, SiteVisualizerConfig{
            .method_name = "FEL",
            .p_value_threshold = p_value_threshold,
            .pos_label = "positive",
            .neg_label = "negative",
            .sig_label = "significant",
            .quiet = !show_progress,
            .force_tty = force_progress,
            .enable_chart = true
        });

        #pragma omp parallel for schedule(dynamic)
        for (size_t p = 0; p < num_patterns; ++p) {
            pattern_results[p] = analyze_pattern(p, leaf_to_taxon);
            pattern_results[p].subs = reconstruct_ancestral_pattern(p, leaf_to_taxon);

            const auto& r = pattern_results[p];
            SiteSelectionType type = SiteSelectionType::Neutral;
            Scalar metric = 0.0;
            std::string detail;
            if (r.p_value <= p_value_threshold) {
                if (r.beta > r.alpha) {
                    type = SiteSelectionType::Positive;
                    metric = r.beta / std::max(r.alpha, 1e-4);
                    detail = "▲ dN/dS = " + SiteVisualizer::format_scalar(metric, 2) + ", p = " + SiteVisualizer::format_pvalue(r.p_value);
                } else if (r.alpha > r.beta) {
                    type = SiteSelectionType::Negative;
                    metric = r.beta / std::max(r.alpha, 1e-4);
                    detail = "▼ dN/dS = " + SiteVisualizer::format_scalar(metric, 2) + ", p = " + SiteVisualizer::format_pvalue(r.p_value);
                }
            }
            viz.record_pattern(p, aln.patterns[p].sites, type, r.p_value, metric, detail);
        }

        viz.finish();

        site_results.resize(aln.num_codons);
        for (size_t s = 0; s < aln.num_codons; ++s) {
            size_t p_idx = aln.site_to_pattern[s];
            site_results[s] = pattern_results[p_idx];
            site_results[s].site_index = s;
            site_results[s].subs.site_index = s;
        }
        return site_results;
    }

    nlohmann::json to_legacy_json(const std::string& input_filepath = "", const std::string& tree_string = "") const {
        nlohmann::json j;

        // Analysis block
        j["analysis"]["authors"] = "Sergei L Kosakovsky Pond and Simon DW Frost";
        j["analysis"]["citation"] = "Not So Different After All: A Comparison of Methods for Detecting Amino Acid Sites Under Selection (2005). _Mol Biol Evol_ 22 (5): 1208-1222";
        j["analysis"]["contact"] = "spond@temple.edu";
        j["analysis"]["info"] = "FEL (Fixed Effects Likelihood) estimates site-wise synonymous (alpha) and non-synonymous (beta) rates, and uses a likelihood ratio test to determine if beta != alpha at a site.";
        j["analysis"]["requirements"] = "in-frame codon alignment and a phylogenetic tree";
        j["analysis"]["settings"]["ci"] = 0;
        j["analysis"]["settings"]["multihit"] = "None";
        j["analysis"]["settings"]["pvalue"] = p_value_threshold;
        j["analysis"]["settings"]["resample"] = 0;
        j["analysis"]["settings"]["srv"] = 1;
        j["analysis"]["version"] = "3.0-next";

        // MLE block
        j["MLE"]["headers"] = nlohmann::json::array({
            nlohmann::json::array({"alpha", "Synonymous substitution rate at a site"}),
            nlohmann::json::array({"beta", "Non-synonymous substitution rate at a site"}),
            nlohmann::json::array({"alpha=beta", "The rate estimate under the neutral model"}),
            nlohmann::json::array({"LRT", "Likelihood ratio test statistic for beta = alpha, versus beta &neq; alpha"}),
            nlohmann::json::array({"p-value", "Asymptotic p-value for evidence of selection, i.e. beta &neq; alpha"}),
            nlohmann::json::array({"Total branch length", "The total length of branches contributing to inference at this site, and used to scale dN-dS"})
        });

        nlohmann::json content = nlohmann::json::array();
        for (const auto& r : site_results) {
            content.push_back({
                r.alpha,
                r.beta,
                r.alpha_null,
                r.lrt,
                r.p_value,
                r.total_branch_length
            });
        }
        j["MLE"]["content"]["0"] = content;

        // Input block
        j["input"]["file name"] = input_filepath;
        j["input"]["genetic code"] = aln.code ? aln.code->name : "Universal";
        j["input"]["number of sequences"] = aln.num_taxa;
        j["input"]["number of sites"] = aln.num_codons;
        j["input"]["partition count"] = 1;
        if (!tree_string.empty()) {
            j["input"]["trees"]["0"] = tree_string;
        }

        // Branch attributes
        nlohmann::json branch_attr;
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                Scalar orig_bl = (conversion_factor > 0.0) ? (node.branch_length / conversion_factor) : node.branch_length;
                branch_attr[node.name]["Global MG94xREV"] = orig_bl;
                if (has_gtr_fit) {
                    branch_attr[node.name]["Nucleotide GTR"] = gtr_branch_lengths.count(node.name) ? gtr_branch_lengths.at(node.name) : orig_bl;
                }
                branch_attr[node.name]["original name"] = node.name;
            }
        }
        j["branch attributes"]["0"] = branch_attr;
        j["branch attributes"]["attributes"]["Global MG94xREV"]["attribute type"] = "branch length";
        j["branch attributes"]["attributes"]["Global MG94xREV"]["display order"] = 1;
        if (has_gtr_fit) {
            j["branch attributes"]["attributes"]["Nucleotide GTR"]["attribute type"] = "branch length";
            j["branch attributes"]["attributes"]["Nucleotide GTR"]["display order"] = 0;
        }

        // Fits
        if (has_gtr_fit) {
            j["fits"]["Nucleotide GTR"]["Log-likelihood"] = gtr_log_l;
            j["fits"]["Nucleotide GTR"]["AIC-c"] = gtr_aicc;
            j["fits"]["Nucleotide GTR"]["display order"] = 0;
            nlohmann::json gtr_ef = nlohmann::json::array();
            for (int i = 0; i < 4; ++i) {
                gtr_ef.push_back({aln.nuc_frequencies(i)});
            }
            j["fits"]["Nucleotide GTR"]["Equilibrium frequencies"] = gtr_ef;
            j["fits"]["Nucleotide GTR"]["Rate Distributions"] = {
                {"Substitution rate from nucleotide A to nucleotide C", gtr_rates.theta_AC},
                {"Substitution rate from nucleotide A to nucleotide G", 1.0},
                {"Substitution rate from nucleotide A to nucleotide T", gtr_rates.theta_AT},
                {"Substitution rate from nucleotide C to nucleotide G", gtr_rates.theta_CG},
                {"Substitution rate from nucleotide C to nucleotide T", gtr_rates.theta_CT},
                {"Substitution rate from nucleotide G to nucleotide T", gtr_rates.theta_GT}
            };
        }

        j["fits"]["Global MG94xREV"]["Log-likelihood"] = global_log_l;
        j["fits"]["Global MG94xREV"]["AIC-c"] = global_aicc;
        j["fits"]["Global MG94xREV"]["display order"] = 1;
        nlohmann::json mg_ef = nlohmann::json::array();
        for (int i = 0; i < aln.codon_frequencies_f3x4.size(); ++i) {
            mg_ef.push_back({aln.codon_frequencies_f3x4(i)});
        }
        j["fits"]["Global MG94xREV"]["Equilibrium frequencies"] = mg_ef;
        Scalar omega_mle = (base_params.alpha > 0.0) ? (base_params.beta / base_params.alpha) : 1.0;
        if (num_background > 0) {
            j["fits"]["Global MG94xREV"]["Rate Distributions"]["non-synonymous/synonymous rate ratio for *test*"] = {
                {omega_mle, 1.0}
            };
            j["fits"]["Global MG94xREV"]["Rate Distributions"]["non-synonymous/synonymous rate ratio for *background*"] = {
                {omega_mle, 1.0}
            };
        } else {
            j["fits"]["Global MG94xREV"]["Rate Distributions"]["non-synonymous/synonymous rate ratio for *test*"] = {
                {omega_mle, 1.0}
            };
        }

        // Tested branches (HyPhy 2.5 datamonkey compatibility)
        nlohmann::json tested_branches = nlohmann::json::object();
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                tested_branches[node.name] = is_test_branch[node.id] ? "test" : "background";
            }
        }
        j["tested"]["0"] = tested_branches;

        // Substitutions block (HyPhy 2.5 datamonkey / hyphy-vision compatibility)
        nlohmann::json subs_json = nlohmann::json::object();
        for (size_t s = 0; s < site_results.size(); ++s) {
            const auto& r = site_results[s];
            std::string site_str = std::to_string(s);
            if (r.subs.events.empty()) {
                subs_json[site_str] = nullptr;
            } else {
                nlohmann::json site_obj = nlohmann::json::object();
                site_obj["root"] = r.subs.root_codon;
                for (const auto& ev : r.subs.events) {
                    site_obj[ev.branch_name] = ev.child_codon;
                }
                subs_json[site_str] = site_obj;
            }
        }
        j["substitutions"]["0"] = subs_json;

        return j;
    }

    nlohmann::json to_modern_json(const std::string& input_filepath = "",
                                  const std::string& tree_string = "",
                                  const Provenance& prov = {}) const {
        auto json_settings = nlohmann::json{
            {"pvalue_threshold", p_value_threshold},
            {"srv", true},
            {"ci", false},
            {"full_model", full_model},
            {"test_branches", settings.test_branches}
        };
        if (!settings.test_branch_regex.empty()) {
            json_settings["test_branch_regex"] = settings.test_branch_regex;
        }

        AnalysisMetadata meta{
            "fel",
            "Fixed Effects Likelihood",
            "3.0.0",
            "site_selection",
            "FEL (Fixed Effects Likelihood) estimates site-wise synonymous (alpha) and non-synonymous (beta) rates, and uses a likelihood ratio test to determine if beta != alpha at a site.",
            {citation("Kosakovsky Pond SL, Frost SDW (2005). Not So Different After All: A Comparison of Methods for Detecting Amino Acid Sites Under Selection. Mol Biol Evol 22(5): 1208-1222.",
                      "10.1093/molbev/msi105", "15703277")},
            std::move(json_settings)
        };

        nlohmann::json j = new_analysis_json(meta, prov, input_filepath);
        j["dataset"] = dataset_block(aln);
        j["phylogeny"] = phylogeny_block(
            tree, tree_string,
            mg94_branch_lengths_fn(conversion_factor, has_gtr_fit, gtr_branch_lengths),
            [&](const TreeNode& node) -> nlohmann::json {
                if (node.id == tree.root_id) return nlohmann::json::object();
                return {{"tested", is_test_branch[node.id]}};
            });

        // Model fits
        if (has_gtr_fit) {
            j["model_fits"]["nucleotide_gtr"] = nucleotide_gtr_fit(
                aln, gtr_rates, gtr_log_l, gtr_aicc,
                5 + static_cast<int>(tree.nodes.size()) - 1);
        }
        j["model_fits"]["codon_mg94"] = fit_entry(
            "Muse-Gaut 1994 x GTR codon model with shared global omega",
            global_log_l, global_aicc,
            6 + static_cast<int>(tree.nodes.size()) - 1);
        j["model_fits"]["codon_mg94"]["global_parameters"] = {
            {"omega", (base_params.alpha > 0.0) ? (base_params.beta / base_params.alpha) : 1.0},
            {"tree_scale", 1.0}
        };

        // Statistical tests
        auto summary = get_summary();
        j["statistical_tests"]["site_level_summary"] = {
            {"method", "Fixed Effects Likelihood (FEL)"},
            {"test_statistic", "Likelihood Ratio Test (LRT)"},
            {"distribution", "Asymptotic Chi-squared"},
            {"degrees_of_freedom", 1},
            {"null_hypothesis", "beta = alpha (Neutral evolution)"},
            {"alternative_hypothesis", "beta != alpha (Positive or negative selection)"},
            {"threshold", p_value_threshold},
            {"total_sites", static_cast<int>(summary.tested_sites)},
            {"diversifying_count", static_cast<int>(summary.positively_selected)},
            {"purifying_count", static_cast<int>(summary.negatively_selected)},
            {"neutral_count", static_cast<int>(summary.tested_sites - summary.positively_selected - summary.negatively_selected)},
            {"tested_branches_count", static_cast<int>(num_tested)},
            {"background_branches_count", static_cast<int>(num_background)},
            {"total_tested_tree_length", (conversion_factor > 0.0) ? (baseline_tested_tree_length / conversion_factor) : baseline_tested_tree_length}
        };

        // Site results (columnar; ColumnarResults enforces equal-length)
        ColumnarResults table;
        const size_t n = site_results.size();
        const bool has_bg = num_background > 0;
        std::vector<int> col_site, col_syn, col_nonsyn, col_syn_bg, col_nonsyn_bg,
                         col_1nt, col_2nt, col_3nt;
        std::vector<double> col_alpha, col_beta, col_beta_bg, col_alpha_null,
                            col_lrt, col_pval, col_tbl, col_surprise, col_mut_bl;
        std::vector<std::string> col_class, col_history;

        col_site.reserve(n); col_alpha.reserve(n); col_beta.reserve(n);
        if (has_bg) col_beta_bg.reserve(n);
        col_alpha_null.reserve(n); col_lrt.reserve(n); col_pval.reserve(n);
        col_tbl.reserve(n); col_class.reserve(n);
        col_syn.reserve(n); col_nonsyn.reserve(n);
        if (has_bg) { col_syn_bg.reserve(n); col_nonsyn_bg.reserve(n); }
        col_1nt.reserve(n); col_2nt.reserve(n); col_3nt.reserve(n);
        col_surprise.reserve(n); col_mut_bl.reserve(n); col_history.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            const auto& r = site_results[i];
            col_site.push_back(static_cast<int>(i + 1));
            col_alpha.push_back(r.alpha);
            col_beta.push_back(r.beta);
            if (has_bg) col_beta_bg.push_back(r.beta_nuisance);
            col_alpha_null.push_back(r.alpha_null);
            col_lrt.push_back(r.lrt);
            col_pval.push_back(r.p_value);
            col_tbl.push_back(r.total_branch_length);

            if (r.p_value <= p_value_threshold) {
                if (r.beta > r.alpha) {
                    col_class.push_back("diversifying");
                } else if (r.alpha > r.beta) {
                    col_class.push_back("purifying");
                } else {
                    col_class.push_back("neutral");
                }
            } else {
                col_class.push_back("neutral");
            }

            col_syn.push_back(static_cast<int>(r.subs.syn_test));
            col_nonsyn.push_back(static_cast<int>(r.subs.nonsyn_test));
            if (has_bg) {
                col_syn_bg.push_back(static_cast<int>(r.subs.syn_bg));
                col_nonsyn_bg.push_back(static_cast<int>(r.subs.nonsyn_bg));
            }
            col_1nt.push_back(static_cast<int>(r.subs.subs_1nt));
            col_2nt.push_back(static_cast<int>(r.subs.subs_2nt));
            col_3nt.push_back(static_cast<int>(r.subs.subs_3nt));
            col_surprise.push_back(r.subs.mean_surprise_bits);
            col_mut_bl.push_back(r.subs.mean_branch_length);
            col_history.push_back(has_bg ? r.subs.summary_string_partitioned : r.subs.summary_string);
        }

        table.add_column("site", {"integer", "Alignment codon site index (1-based)", "1-based codon position"}, col_site);
        table.add_column("alpha", {"number", "Synonymous substitution rate estimate (alpha)", "substitutions/site"}, col_alpha);
        table.add_column("beta", {"number", "Non-synonymous substitution rate estimate on tested branches (beta)", "substitutions/site"}, col_beta);
        if (has_bg) {
            table.add_column("beta_background", {"number", "Non-synonymous substitution rate estimate on background branches (beta_background)", "substitutions/site"}, col_beta_bg);
        }
        table.add_column("alpha_null", {"number", "Constrained neutral rate (alpha = beta)", "substitutions/site"}, col_alpha_null);
        table.add_column("lrt", {"number", "Likelihood ratio test statistic for beta != alpha"}, col_lrt);
        table.add_column("p_value", {"number", "Asymptotic p-value from chi^2_1 distribution"}, col_pval);
        table.add_column("total_branch_length", {"number", "Total length of tested branches contributing to inference at this site", "substitutions/site"}, col_tbl);
        table.add_column("classification", {"string", "Selection classification at threshold (diversifying, purifying, neutral)"}, col_class);
        table.add_column("synonymous_substitutions", {"integer", "Count of synonymous substitutions on tested branches"}, col_syn);
        table.add_column("nonsynonymous_substitutions", {"integer", "Count of non-synonymous substitutions on tested branches"}, col_nonsyn);
        if (has_bg) {
            table.add_column("synonymous_substitutions_background", {"integer", "Count of synonymous substitutions on background branches"}, col_syn_bg);
            table.add_column("nonsynonymous_substitutions_background", {"integer", "Count of non-synonymous substitutions on background branches"}, col_nonsyn_bg);
        }
        table.add_column("substitutions_1nt", {"integer", "Count of single-nucleotide codon substitutions"}, col_1nt);
        table.add_column("substitutions_2nt", {"integer", "Count of double-nucleotide codon substitutions"}, col_2nt);
        table.add_column("substitutions_3nt", {"integer", "Count of triple-nucleotide codon substitutions"}, col_3nt);
        table.add_column("mean_substitution_surprise", {"number", "Mean Shannon self-information (-log2 P) of substitutions at this site", "bits"}, col_surprise);
        table.add_column("mean_substitution_branch_length", {"number", "Mean length of branches where substitutions occurred at this site", "substitutions/site"}, col_mut_bl);
        table.add_column("substitution_history", {"string", "Formatted substitution history summary (AA changes, hit counts, surprise)"}, col_history);
        j["site_results"] = table.to_json();

        // Compressed substitutions mapping (HyPhy 2.5 datamonkey / hyphy-vision parity)
        nlohmann::json subs_comp = nlohmann::json::object();
        for (size_t s = 0; s < site_results.size(); ++s) {
            const auto& r = site_results[s];
            std::string site_str = std::to_string(s);
            if (r.subs.events.empty()) {
                subs_comp[site_str] = nullptr;
            } else {
                nlohmann::json site_obj = nlohmann::json::object();
                site_obj["root"] = r.subs.root_codon;
                for (const auto& ev : r.subs.events) {
                    site_obj[ev.branch_name] = ev.child_codon;
                }
                subs_comp[site_str] = site_obj;
            }
        }
        j["substitutions"]["0"] = subs_comp;

        // Detailed substitution events per site
        nlohmann::json sub_events_json = nlohmann::json::array();
        for (size_t s = 0; s < site_results.size(); ++s) {
            const auto& r = site_results[s];
            if (!r.subs.events.empty()) {
                nlohmann::json site_ev_obj;
                site_ev_obj["site"] = static_cast<int>(s + 1);
                site_ev_obj["root_codon"] = r.subs.root_codon;
                site_ev_obj["root_aa"] = std::string(1, r.subs.root_aa);
                nlohmann::json ev_arr = nlohmann::json::array();
                for (const auto& ev : r.subs.events) {
                    nlohmann::json ev_json = {
                        {"branch", ev.branch_name},
                        {"tested", ev.is_tested},
                        {"branch_length", ev.branch_length},
                        {"parent_codon", ev.parent_codon},
                        {"child_codon", ev.child_codon},
                        {"parent_aa", std::string(1, ev.parent_aa)},
                        {"child_aa", std::string(1, ev.child_aa)},
                        {"nt_distance", ev.nt_distance},
                        {"is_synonymous", ev.is_synonymous},
                        {"transition_probability", ev.transition_prob},
                        {"surprise_bits", ev.surprise_bits}
                    };
                    ev_arr.push_back(ev_json);
                }
                site_ev_obj["events"] = ev_arr;
                sub_events_json.push_back(site_ev_obj);
            }
        }
        j["substitution_events"] = sub_events_json;

        return j;
    }
    nlohmann::json to_json(const std::string& input_filepath = "",
                           const std::string& tree_string = "",
                           JSONFormat format = JSONFormat::ModernV3,
                           const Provenance& prov = {}) const {
        if (format == JSONFormat::Legacy) {
            return to_legacy_json(input_filepath, tree_string);
        }
        return to_modern_json(input_filepath, tree_string, prov);
    }

    void save_json(const std::string& output_filepath,
                   const std::string& input_filepath = "",
                   const std::string& tree_string = "",
                   JSONFormat format = JSONFormat::ModernV3,
                   const Provenance& prov = {}) const {
        auto j = to_json(input_filepath, tree_string, format, prov);
        std::ofstream out(output_filepath);
        if (!out.is_open()) {
            throw std::runtime_error("Could not open output JSON file: " + output_filepath);
        }
        out << j.dump(2) << std::endl;
    }

    struct Summary {
        size_t tested_sites = 0;
        size_t positively_selected = 0;
        size_t negatively_selected = 0;
    };

    Summary get_summary() const {
        Summary s;
        s.tested_sites = site_results.size();
        for (const auto& r : site_results) {
            if (r.p_value <= p_value_threshold) {
                if (r.beta > r.alpha) {
                    s.positively_selected++;
                } else if (r.alpha > r.beta) {
                    s.negatively_selected++;
                }
            }
        }
        return s;
    }
};

} // namespace hyphy::analyses
