#pragma once

#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/core/ohmic_network.hpp"
#include "hyphy/core/wheatstone_likelihood.hpp"
#include "hyphy/opt/optimizer.hpp"
#include "LBFGSB.h"
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

namespace hyphy::analyses {

using namespace hyphy::core;

struct OhmicAnalysisSettings {
    OhmicSettings network_settings;
    bool use_codon_model = false;          // true: MG94xREV codon model; false: GTR nucleotide model
    bool optimize_bridge_weights = true;   // Refine shunt weights w_{k,r} during ML optimization
    bool optimize_branch_lengths = true;   // Refine backbone & bridge branch lengths
    bool quiet = false;
    std::string input_file;
    std::string output_file;
};

struct OhmicBridgeSummary {
    size_t bridge_index = 0;
    std::string hub_name;
    std::string bridge_name;
    std::array<std::string, 3> child_names;
    std::array<std::vector<std::string>, 4> quartet_clades; // [A, B, C, D] leaf taxon names
    std::array<Scalar, 3> initial_weights{1.0, 0.0, 0.0};
    std::array<Scalar, 3> fitted_weights{1.0, 0.0, 0.0};
    std::array<Scalar, 3> fitted_branch_lengths{0.01, 0.01, 0.01};
    Scalar ohmic_support = 1.0;
    Scalar shunt_stress = 0.0;
};

struct OhmicResult {
    OhmicAnalysisSettings settings;
    OhmicNetwork network;
    std::vector<std::string> taxon_names;
    size_t num_taxa = 0;
    size_t num_sites = 0;
    size_t num_patterns = 0;
    size_t num_super_nodes = 0;
    size_t num_bridges = 0;
    size_t num_collapsed_polytomies = 0;
    Scalar foster_sum = 0.0;

    Scalar single_tree_log_l = 0.0;
    Scalar single_tree_aicc = 0.0;
    Scalar ohmic_log_l = 0.0;
    Scalar ohmic_aicc = 0.0;
    Scalar delta_log_l = 0.0;
    Scalar delta_aicc = 0.0;

    GTRParameters gtr_params;
    MG94Parameters mg94_params;
    std::vector<OhmicBridgeSummary> bridge_summaries;
    std::vector<Scalar> site_single_tree_log_l;
    std::vector<Scalar> site_ohmic_log_l;
    std::vector<Scalar> site_delta_log_l;
    double runtime_seconds = 0.0;

    nlohmann::json to_modern_json(const std::string& filepath = "") const {
        nlohmann::json j;
        j["$schema"] = "https://raw.githubusercontent.com/veg/hyphy-3/main/schemas/v3/ohmic.v3.schema.json";
        j["schema_version"] = "3.0.0";
        j["analysis"] = {
            {"id", "ohmic"},
            {"name", "OHMIC"},
            {"description", "Ohmic Phylogenetic Networks & Linear-Time Wheatstone Pruning"},
            {"version", "3.0.0"},
            {"authors", "Sergei L Kosakovsky Pond"},
            {"contact", "spond@temple.edu"}
        };
        j["dataset"] = {
            {"file", filepath.empty() ? settings.input_file : filepath},
            {"datatype", settings.use_codon_model ? "codon" : "nucleotide"},
            {"taxa_count", static_cast<int>(num_taxa)},
            {"sites_count", static_cast<int>(num_sites)},
            {"patterns_count", static_cast<int>(num_patterns)},
            {"super_nodes_count", static_cast<int>(num_super_nodes)},
            {"taxa", taxon_names}
        };
        std::string mode_str = (network.active_mode == OhmicScaleMode::ExactDense)
            ? "ExactDense" : "ScalableWoodbury";
        j["electrical_network"] = {
            {"scale_mode", mode_str},
            {"used_hyphaeon_embeddings", network.used_hyphaeon_embeddings},
            {"embedding_dim", network.used_hyphaeon_embeddings ? static_cast<int>(network.hyphaeon_embeddings.cols()) : 0},
            {"foster_sum", foster_sum},
            {"num_bridges", static_cast<int>(num_bridges)},
            {"num_collapsed_polytomies", static_cast<int>(num_collapsed_polytomies)},
            {"backbone_tree", network.backbone_tree.to_newick(true)},
            {"collapsed_tree", network.collapsed_tree.to_newick(true)}
        };

        // Export pairwise matrices in ExactDense mode for interactive visualization (when N <= 128)
        if (network.P_edge.rows() > 0 && network.P_edge.rows() <= 128) {
            auto mat_to_json = [](const Matrix& m) {
                nlohmann::json rows = nlohmann::json::array();
                for (Eigen::Index r = 0; r < m.rows(); ++r) {
                    nlohmann::json row = nlohmann::json::array();
                    for (Eigen::Index c = 0; c < m.cols(); ++c) {
                        row.push_back(m(r, c));
                    }
                    rows.push_back(std::move(row));
                }
                return rows;
            };
            j["electrical_network"]["pairwise_foster_probabilities"] = mat_to_json(network.P_edge);
            j["electrical_network"]["pairwise_effective_resistances"] = mat_to_json(network.Omega);
            j["electrical_network"]["pairwise_tn93_distances"] = mat_to_json(network.D_tn93);
        }

        j["fits"] = {
            {"single_tree", {
                {"log_likelihood", single_tree_log_l},
                {"aicc", single_tree_aicc}
            }},
            {"ohmic_network", {
                {"log_likelihood", ohmic_log_l},
                {"aicc", ohmic_aicc},
                {"delta_log_likelihood", delta_log_l},
                {"delta_aicc", delta_aicc}
            }}
        };
        if (settings.use_codon_model) {
            j["fits"]["omega"] = mg94_params.beta;
        }
        nlohmann::json bridges_arr = nlohmann::json::array();
        for (const auto& b : bridge_summaries) {
            bridges_arr.push_back({
                {"bridge_index", b.bridge_index},
                {"hub_node", b.hub_name},
                {"bridge_node", b.bridge_name},
                {"children", {b.child_names[0], b.child_names[1], b.child_names[2]}},
                {"quartet_clades", {b.quartet_clades[0], b.quartet_clades[1], b.quartet_clades[2], b.quartet_clades[3]}},
                {"ohmic_support", b.ohmic_support},
                {"shunt_stress", b.shunt_stress},
                {"initial_weights", {b.initial_weights[0], b.initial_weights[1], b.initial_weights[2]}},
                {"fitted_weights", {b.fitted_weights[0], b.fitted_weights[1], b.fitted_weights[2]}},
                {"fitted_branch_lengths", {b.fitted_branch_lengths[0], b.fitted_branch_lengths[1], b.fitted_branch_lengths[2]}}
            });
        }
        j["wheatstone_bridges"] = bridges_arr;
        if (!site_delta_log_l.empty()) {
            j["site_results"] = {
                {"delta_log_l", site_delta_log_l},
                {"single_tree_log_l", site_single_tree_log_l},
                {"ohmic_log_l", site_ohmic_log_l}
            };
        }
        j["timers"] = {
            {"total_seconds", runtime_seconds}
        };
        return j;
    }

    nlohmann::json to_legacy_json(const std::string& filepath = "") const {
        nlohmann::json j = to_modern_json(filepath);
        j["input"] = {
            {"file name", filepath.empty() ? settings.input_file : filepath},
            {"number of sequences", static_cast<int>(num_taxa)},
            {"number of sites", static_cast<int>(num_sites)}
        };
        return j;
    }
};

class OhmicAnalyzer {
public:
    Alignment aln;
    OhmicAnalysisSettings settings;

    OhmicAnalyzer(Alignment a, OhmicAnalysisSettings s = {})
        : aln(std::move(a)), settings(std::move(s)) {}

    OhmicResult run(const Matrix* external_embeddings = nullptr) {
        auto t_start = std::chrono::steady_clock::now();

        OhmicResult result;
        result.settings = settings;
        result.taxon_names = aln.taxon_names;
        result.num_taxa = aln.num_taxa;
        result.num_sites = settings.use_codon_model ? aln.num_codons : aln.num_nucleotides;
        if (result.num_sites == 0) {
            result.num_sites = aln.num_nucleotides > 0 ? aln.num_nucleotides : aln.num_codons;
        }
        result.num_patterns = aln.patterns.size();

        // 1. Infer Ohmic electrical network and Wheatstone bridges
        result.network = OhmicNetwork::infer(aln, settings.network_settings, external_embeddings);
        result.num_super_nodes = result.network.num_super_nodes;
        result.num_bridges = result.network.bridges.size();
        result.num_collapsed_polytomies = result.network.num_collapsed_polytomies;
        result.foster_sum = result.network.foster_sum;

        // Helper to collect leaf taxon names descending from a subtree root in collapsed_tree
        auto collect_clade_leaves = [&](int32_t root_nid) -> std::vector<std::string> {
            std::vector<std::string> leaves;
            if (root_nid < 0 || static_cast<size_t>(root_nid) >= result.network.collapsed_tree.nodes.size()) return leaves;
            std::vector<int32_t> stack = {root_nid};
            while (!stack.empty()) {
                int32_t curr = stack.back();
                stack.pop_back();
                const auto& node = result.network.collapsed_tree.nodes[curr];
                if (node.is_leaf) {
                    leaves.push_back(node.name);
                } else {
                    for (int32_t c : node.children) stack.push_back(c);
                }
            }
            return leaves;
        };

        // Record initial bridge weights before ML refinement
        result.bridge_summaries.resize(result.num_bridges);
        for (size_t k = 0; k < result.num_bridges; ++k) {
            const auto& wb = result.network.bridges[k];
            auto& bs = result.bridge_summaries[k];
            bs.bridge_index = k;
            bs.hub_name = (wb.hub_node_id >= 0) ? result.network.collapsed_tree.nodes[wb.hub_node_id].name : "";
            bs.bridge_name = (wb.bridge_node_id >= 0) ? result.network.backbone_tree.nodes[wb.bridge_node_id].name : "";
            std::unordered_set<std::string> in_abc;
            for (int r = 0; r < 3; ++r) {
                int32_t cid = wb.children[r];
                bs.child_names[r] = (cid >= 0) ? result.network.collapsed_tree.nodes[cid].name : "";
                bs.quartet_clades[r] = collect_clade_leaves(cid);
                for (const auto& nm : bs.quartet_clades[r]) in_abc.insert(nm);
            }
            for (const auto& nm : aln.taxon_names) {
                if (in_abc.find(nm) == in_abc.end()) {
                    bs.quartet_clades[3].push_back(nm);
                }
            }
            bs.initial_weights = wb.weights;
            bs.fitted_weights = wb.weights;
            bs.fitted_branch_lengths = wb.branch_lengths;
            bs.ohmic_support = wb.ohmic_support;
            bs.shunt_stress = wb.shunt_stress;
        }

        if (settings.use_codon_model && aln.num_codons > 0) {
            fit_codon_network(result);
        } else {
            fit_nucleotide_network(result);
        }

        for (size_t k = 0; k < result.num_bridges; ++k) {
            result.bridge_summaries[k].fitted_weights = result.network.bridges[k].weights;
            result.bridge_summaries[k].fitted_branch_lengths = result.network.bridges[k].branch_lengths;
        }
        result.network.collapsed_tree.bridges = result.network.bridges;
        result.network.collapsed_tree.node_to_bridge_idx = result.network.node_to_bridge_idx;

        auto t_end = std::chrono::steady_clock::now();
        result.runtime_seconds = std::chrono::duration<double>(t_end - t_start).count();
        return result;
    }

private:
    static Scalar compute_aicc(Scalar log_l, size_t k_params, size_t n_obs) {
        Scalar k = static_cast<Scalar>(k_params);
        Scalar n = std::max<Scalar>(static_cast<Scalar>(n_obs), k + 2.0);
        return -2.0 * log_l + 2.0 * k + (2.0 * k * (k + 1.0)) / (n - k - 1.0);
    }

    void fit_nucleotide_network(OhmicResult& result) {
        Alignment nuc_aln = (aln.num_nucleotides > 0 && aln.num_codons == 0)
            ? aln
            : aln.to_nucleotide_alignment();

        // 1. Fit baseline single tree (GTR)
        opt::GTRFitter gtr_fitter(result.network.backbone_tree, nuc_aln);
        auto base_fit = gtr_fitter.fit();
        result.single_tree_log_l = base_fit.log_likelihood;
        result.gtr_params = base_fit.params;
        result.network.backbone_tree = base_fit.tree;

        // Transfer optimized backbone branch lengths onto collapsed_tree and bridges
        for (size_t nid = 0; nid < result.network.collapsed_tree.nodes.size(); ++nid) {
            if (nid < base_fit.tree.nodes.size() &&
                (result.network.collapsed_tree.nodes[nid].is_leaf ||
                 !result.network.collapsed_tree.nodes[nid].children.empty())) {
                // Keep 0.0 on root-child bridge hub if hub_node_id == bridge_node_id
                int32_t b_idx = result.network.node_to_bridge_idx[nid];
                if (b_idx >= 0 && result.network.bridges[b_idx].hub_node_id == result.network.bridges[b_idx].bridge_node_id) {
                    result.network.collapsed_tree.nodes[nid].branch_length = 0.0;
                } else {
                    result.network.collapsed_tree.nodes[nid].branch_length = base_fit.tree.nodes[nid].branch_length;
                }
            }
        }
        for (auto& wb : result.network.bridges) {
            if (wb.bridge_node_id >= 0 && static_cast<size_t>(wb.bridge_node_id) < base_fit.tree.nodes.size()) {
                wb.branch_lengths[0] = base_fit.tree.nodes[wb.bridge_node_id].branch_length;
                wb.branch_lengths[1] = std::max<Scalar>(1e-5, wb.branch_lengths[0] * wb.shunt_stress);
            }
        }

        GTRMatrix gtr;
        gtr.update(result.gtr_params, nuc_aln.nuc_frequencies);
        auto leaf_map = LikelihoodEngine::build_leaf_to_taxon_map(result.network.collapsed_tree, nuc_aln);

        // 2. Refine Wheatstone bridge branch lengths and shunt weights under GTR
        if (!result.network.bridges.empty() && (settings.optimize_branch_lengths || settings.optimize_bridge_weights)) {
            for (int pass = 0; pass < 2; ++pass) {
                for (size_t k = 0; k < result.network.bridges.size(); ++k) {
                    // Optimize secondary bridge length t_{k,2}
                    if (settings.optimize_branch_lengths && result.network.bridges[k].weights[1] > 0.0) {
                        Scalar init_t2 = result.network.bridges[k].branch_lengths[1];
                        auto t2_obj = [&](Scalar t2) -> Scalar {
                            Scalar old_t2 = result.network.bridges[k].branch_lengths[1];
                            result.network.bridges[k].branch_lengths[1] = t2;
                            Scalar lnl = WheatstoneLikelihood::compute_gtr_log_likelihood(
                                result.network, nuc_aln, gtr, 1.0, &leaf_map
                            );
                            result.network.bridges[k].branch_lengths[1] = old_t2;
                            return -lnl;
                        };
                        auto [best_t2, _] = opt::Brent1D::minimize(t2_obj, 1e-6, std::max<Scalar>(init_t2, 1e-4), 2.0, 1e-4, 20);
                        result.network.bridges[k].branch_lengths[1] = best_t2;
                    }
                    // Optimize primary bridge length t_{k,1}
                    if (settings.optimize_branch_lengths) {
                        Scalar init_t1 = result.network.bridges[k].branch_lengths[0];
                        auto t1_obj = [&](Scalar t1) -> Scalar {
                            Scalar old_t1 = result.network.bridges[k].branch_lengths[0];
                            result.network.bridges[k].branch_lengths[0] = t1;
                            Scalar lnl = WheatstoneLikelihood::compute_gtr_log_likelihood(
                                result.network, nuc_aln, gtr, 1.0, &leaf_map
                            );
                            result.network.bridges[k].branch_lengths[0] = old_t1;
                            return -lnl;
                        };
                        auto [best_t1, _] = opt::Brent1D::minimize(t1_obj, 1e-6, std::max<Scalar>(init_t1, 1e-4), 2.0, 1e-4, 20);
                        result.network.bridges[k].branch_lengths[0] = best_t1;
                    }
                    // Optimize shunt weight w_{k,2} in [0, 0.99]
                    if (settings.optimize_bridge_weights) {
                        Scalar init_w2 = result.network.bridges[k].weights[1];
                        auto w2_obj = [&](Scalar w2) -> Scalar {
                            auto old_w = result.network.bridges[k].weights;
                            result.network.bridges[k].weights = {1.0 - w2, w2, 0.0};
                            Scalar lnl = WheatstoneLikelihood::compute_gtr_log_likelihood(
                                result.network, nuc_aln, gtr, 1.0, &leaf_map
                            );
                            result.network.bridges[k].weights = old_w;
                            return -lnl;
                        };
                        auto [best_w2, _] = opt::Brent1D::minimize(w2_obj, 0.0, std::clamp<Scalar>(init_w2, 0.01, 0.95), 0.99, 1e-4, 20);
                        result.network.bridges[k].weights = {1.0 - best_w2, best_w2, 0.0};
                    }
                    // Also refine the 3 incident child branches {id_A, id_B, id_C} so terminal branches
                    // can relax once homoplasy is routed through the shunt bridge
                    if (settings.optimize_branch_lengths && result.network.bridges[k].weights[1] > 1e-4) {
                        for (int c_idx = 0; c_idx < 3; ++c_idx) {
                            int32_t cid = result.network.bridges[k].children[c_idx];
                            if (cid < 0) continue;
                            Scalar init_bl = result.network.collapsed_tree.nodes[cid].branch_length;
                            auto c_obj = [&](Scalar bl) -> Scalar {
                                Scalar old_bl = result.network.collapsed_tree.nodes[cid].branch_length;
                                result.network.collapsed_tree.nodes[cid].branch_length = bl;
                                Scalar lnl = WheatstoneLikelihood::compute_gtr_log_likelihood(
                                    result.network, nuc_aln, gtr, 1.0, &leaf_map
                                );
                                result.network.collapsed_tree.nodes[cid].branch_length = old_bl;
                                return -lnl;
                            };
                            auto [best_c, _] = opt::Brent1D::minimize(c_obj, 1e-6, std::max<Scalar>(init_bl, 1e-4), 2.0, 1e-4, 15);
                            result.network.collapsed_tree.nodes[cid].branch_length = best_c;
                        }
                    }
                }
            }
        }

        result.ohmic_log_l = WheatstoneLikelihood::compute_gtr_log_likelihood(
            result.network, nuc_aln, gtr, 1.0, &leaf_map
        );
        // Ensure Ohmic network is at least as good as the nested single-tree special case (w2 = 0)
        if (result.ohmic_log_l < result.single_tree_log_l) {
            for (auto& wb : result.network.bridges) {
                wb.weights = {1.0, 0.0, 0.0};
            }
            result.ohmic_log_l = WheatstoneLikelihood::compute_gtr_log_likelihood(
                result.network, nuc_aln, gtr, 1.0, &leaf_map
            );
        }

        size_t n_obs = nuc_aln.num_nucleotides * nuc_aln.num_taxa;
        size_t k_single = 5 + (2 * nuc_aln.num_taxa - 3);
        size_t k_active_bridges = 0;
        for (const auto& wb : result.network.bridges) {
            if (wb.weights[1] > 1e-4) k_active_bridges++;
        }
        size_t k_ohmic = k_single + k_active_bridges - result.num_collapsed_polytomies;

        result.single_tree_aicc = compute_aicc(result.single_tree_log_l, k_single, n_obs);
        result.ohmic_aicc = compute_aicc(result.ohmic_log_l, k_ohmic, n_obs);
        result.delta_log_l = result.ohmic_log_l - result.single_tree_log_l;
        result.delta_aicc = result.single_tree_aicc - result.ohmic_aicc;

        // Compute per-site log-likelihood profile (Ohmic vs Single-Tree)
        std::vector<Scalar> pat_ohm, pat_single;
        WheatstoneLikelihood::compute_gtr_log_likelihood(result.network, nuc_aln, gtr, 1.0, &leaf_map, &pat_ohm);
        OhmicNetwork single_net = result.network;
        for (auto& wb : single_net.bridges) wb.weights = {1.0, 0.0, 0.0};
        WheatstoneLikelihood::compute_gtr_log_likelihood(single_net, nuc_aln, gtr, 1.0, &leaf_map, &pat_single);

        size_t S_sites = nuc_aln.site_to_pattern.size();
        result.site_single_tree_log_l.resize(S_sites, 0.0);
        result.site_ohmic_log_l.resize(S_sites, 0.0);
        result.site_delta_log_l.resize(S_sites, 0.0);
        for (size_t s = 0; s < S_sites; ++s) {
            size_t p = nuc_aln.site_to_pattern[s];
            result.site_single_tree_log_l[s] = pat_single[p];
            result.site_ohmic_log_l[s] = pat_ohm[p];
            result.site_delta_log_l[s] = pat_ohm[p] - pat_single[p];
        }
    }

    void fit_codon_network(OhmicResult& result) {
        // 1. Fit baseline single tree under MG94xREV
        opt::MG94Fitter mg_fitter(result.network.backbone_tree, aln);
        auto base_fit = mg_fitter.fit_full_model(1.0, MG94Parameters{}, nullptr, 40, 1e-4);
        result.single_tree_log_l = base_fit.log_likelihood;
        result.mg94_params = base_fit.params;
        result.network.backbone_tree = base_fit.tree;

        for (size_t nid = 0; nid < result.network.collapsed_tree.nodes.size(); ++nid) {
            if (nid < base_fit.tree.nodes.size() &&
                (result.network.collapsed_tree.nodes[nid].is_leaf ||
                 !result.network.collapsed_tree.nodes[nid].children.empty())) {
                int32_t b_idx = result.network.node_to_bridge_idx[nid];
                if (b_idx >= 0 && result.network.bridges[b_idx].hub_node_id == result.network.bridges[b_idx].bridge_node_id) {
                    result.network.collapsed_tree.nodes[nid].branch_length = 0.0;
                } else {
                    result.network.collapsed_tree.nodes[nid].branch_length = base_fit.tree.nodes[nid].branch_length;
                }
            }
        }
        for (auto& wb : result.network.bridges) {
            if (wb.bridge_node_id >= 0 && static_cast<size_t>(wb.bridge_node_id) < base_fit.tree.nodes.size()) {
                wb.branch_lengths[0] = base_fit.tree.nodes[wb.bridge_node_id].branch_length;
                wb.branch_lengths[1] = std::max<Scalar>(1e-5, wb.branch_lengths[0] * wb.shunt_stress);
            }
        }

        // 2. Joint Fréchet quasi-Newton optimization over Wheatstone network
        auto leaf_map = LikelihoodEngine::build_leaf_to_taxon_map(result.network.collapsed_tree, aln);
        size_t K = result.network.bridges.size();
        if (K > 0 && (settings.optimize_branch_lengths || settings.optimize_bridge_weights)) {
            // Optimize bridge lengths t_{k,1}, t_{k,2} and logit weights w_{k,2} + omega via L-BFGS-B
            size_t dim = 3 * K + 1;
            Vector x(dim), lb(dim), ub(dim);
            for (size_t k = 0; k < K; ++k) {
                x(3 * k + 0) = std::log(std::clamp<Scalar>(result.network.bridges[k].branch_lengths[0], 1e-5, 5.0));
                lb(3 * k + 0) = -12.0; ub(3 * k + 0) = 2.0;
                x(3 * k + 1) = std::log(std::clamp<Scalar>(result.network.bridges[k].branch_lengths[1], 1e-5, 5.0));
                lb(3 * k + 1) = -12.0; ub(3 * k + 1) = 2.0;
                Scalar w2 = std::clamp<Scalar>(result.network.bridges[k].weights[1], 1e-4, 0.99);
                x(3 * k + 2) = std::log(w2 / (1.0 - w2));
                lb(3 * k + 2) = -10.0; ub(3 * k + 2) = 4.0;
            }
            x(3 * K) = std::log(std::clamp<Scalar>(result.mg94_params.beta, 1e-3, 10.0));
            lb(3 * K) = -8.0; ub(3 * K) = 4.0;

            auto obj = [&](const Vector& y, Vector& grad) -> Scalar {
                for (size_t k = 0; k < K; ++k) {
                    result.network.bridges[k].branch_lengths[0] = std::exp(y(3 * k + 0));
                    result.network.bridges[k].branch_lengths[1] = std::exp(y(3 * k + 1));
                    Scalar w2 = 1.0 / (1.0 + std::exp(-y(3 * k + 2)));
                    result.network.bridges[k].weights = {1.0 - w2, w2, 0.0};
                }
                result.mg94_params.beta = std::exp(y(3 * K));

                auto gres = WheatstoneLikelihood::compute_joint_frechet_gradients(
                    result.network, aln, result.mg94_params, &leaf_map
                );

                grad.resize(dim);
                for (size_t k = 0; k < K; ++k) {
                    Scalar t1 = result.network.bridges[k].branch_lengths[0];
                    Scalar t2 = result.network.bridges[k].branch_lengths[1];
                    Scalar w2 = result.network.bridges[k].weights[1];
                    grad(3 * k + 0) = -t1 * gres.grad_bridge_lengths[k][0];
                    grad(3 * k + 1) = -t2 * gres.grad_bridge_lengths[k][1];
                    Scalar dL_dw2 = gres.grad_bridge_weights[k][1] - gres.grad_bridge_weights[k][0];
                    grad(3 * k + 2) = -w2 * (1.0 - w2) * dL_dw2;
                }
                grad(3 * K) = -result.mg94_params.beta * gres.grad_omega;
                return -gres.log_likelihood;
            };

            LBFGSpp::LBFGSBParam<Scalar> opt_param;
            opt_param.m = 6;
            opt_param.epsilon = 1e-4;
            opt_param.max_iterations = 30;
            LBFGSpp::LBFGSBSolver<Scalar> solver(opt_param);
            Scalar fx = 0.0;
            try {
                solver.minimize(obj, x, fx, lb, ub);
            } catch (...) {}

            for (size_t k = 0; k < K; ++k) {
                result.network.bridges[k].branch_lengths[0] = std::exp(x(3 * k + 0));
                result.network.bridges[k].branch_lengths[1] = std::exp(x(3 * k + 1));
                Scalar w2 = 1.0 / (1.0 + std::exp(-x(3 * k + 2)));
                result.network.bridges[k].weights = {1.0 - w2, w2, 0.0};
            }
            result.mg94_params.beta = std::exp(x(3 * K));
        }

        auto final_g = WheatstoneLikelihood::compute_joint_frechet_gradients(
            result.network, aln, result.mg94_params, &leaf_map
        );
        result.ohmic_log_l = final_g.log_likelihood;

        size_t n_obs = aln.num_codons * aln.num_taxa;
        size_t k_single = 6 + (2 * aln.num_taxa - 3);
        size_t k_active_bridges = 0;
        for (const auto& wb : result.network.bridges) {
            if (wb.weights[1] > 1e-4) k_active_bridges++;
        }
        size_t k_ohmic = k_single + k_active_bridges - result.num_collapsed_polytomies;

        result.single_tree_aicc = compute_aicc(result.single_tree_log_l, k_single, n_obs);
        result.ohmic_aicc = compute_aicc(result.ohmic_log_l, k_ohmic, n_obs);
        result.delta_log_l = result.ohmic_log_l - result.single_tree_log_l;
        result.delta_aicc = result.single_tree_aicc - result.ohmic_aicc;

        // Compute per-site log-likelihood profile (Ohmic vs Single-Tree)
        const GeneticCode& gcode = *(aln.code ? aln.code : GeneticCode::universal());
        MG94Matrix mg94;
        mg94.update(result.mg94_params, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, gcode);
        std::vector<Scalar> pat_ohm, pat_single;
        WheatstoneLikelihood::compute_mg94_log_likelihood(result.network, aln, mg94, 1.0, &leaf_map, &pat_ohm);
        OhmicNetwork single_net = result.network;
        for (auto& wb : single_net.bridges) wb.weights = {1.0, 0.0, 0.0};
        WheatstoneLikelihood::compute_mg94_log_likelihood(single_net, aln, mg94, 1.0, &leaf_map, &pat_single);

        size_t S_sites = aln.site_to_pattern.size();
        result.site_single_tree_log_l.resize(S_sites, 0.0);
        result.site_ohmic_log_l.resize(S_sites, 0.0);
        result.site_delta_log_l.resize(S_sites, 0.0);
        for (size_t s = 0; s < S_sites; ++s) {
            size_t p = aln.site_to_pattern[s];
            result.site_single_tree_log_l[s] = pat_single[p];
            result.site_ohmic_log_l[s] = pat_ohm[p];
            result.site_delta_log_l[s] = pat_ohm[p] - pat_single[p];
        }
    }
};

} // namespace hyphy::analyses
