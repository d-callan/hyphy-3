#pragma once

#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/core/neighbor_joining.hpp"
#include "hyphy/core/progress_bar.hpp"
#include "hyphy/core/console.hpp"
#include "hyphy/opt/optimizer.hpp"
#include "hyphy/analyses/gard_visualizer.hpp"
#include <nlohmann/json.hpp>

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace hyphy::analyses {

using namespace hyphy::core;

struct GARDSettings {
    std::string input_file;
    std::string output_file;
    std::string run_mode = "Normal"; // "Normal" or "Faster"
    size_t max_breakpoints = 10000;
    size_t population_size = 32;
    Scalar mutation_rate = 0.20;
    Scalar rate_small_shifts = 0.80;
    Scalar caic_diversity_threshold = 0.01;
    size_t max_stagnant_generations = 100;
    Scalar caic_improvement_threshold = 0.01;
    bool quiet = false;
    bool dynamic_chart = true;
    bool force_tty = false;
    bool legacy_json = false;
};

struct GARDPartition {
    size_t start_site = 1; // 1-based start
    size_t end_site = 1;   // 1-based end
    core::Tree tree;
    Scalar log_likelihood = 0.0;
};

struct GARDModel {
    std::vector<size_t> breakpoints; // 0-based site coordinates
    Scalar log_likelihood = 0.0;
    Scalar aicc = 0.0;
    std::vector<GARDPartition> partitions;

    std::string to_key() const {
        if (breakpoints.empty()) return "{}";
        std::ostringstream ss;
        ss << "{";
        for (size_t i = 0; i < breakpoints.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << breakpoints[i];
        }
        ss << "}";
        return ss.str();
    }

    static std::vector<size_t> parse_key(const std::string& key) {
        std::vector<size_t> res;
        if (key.size() <= 2) return res;
        std::string inner = key.substr(1, key.size() - 2);
        std::stringstream ss(inner);
        std::string token;
        while (std::getline(ss, token, ',')) {
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) token.erase(token.begin());
            while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) token.pop_back();
            if (!token.empty()) {
                res.push_back(std::stoull(token));
            }
        }
        std::sort(res.begin(), res.end());
        return res;
    }
};

struct GARDResult {
    Scalar baseline_aicc = 0.0;
    Scalar baseline_log_l = 0.0;
    core::Tree baseline_tree;
    core::GTRParameters baseline_rates;
    Vector4 baseline_frequencies = Vector4::Zero();

    size_t num_sites = 0;
    size_t num_taxa = 0;
    size_t potential_breakpoints = 0;
    size_t min_partition_size = 0;
    std::vector<size_t> variable_sites;

    GARDModel best_model;
    Scalar single_tree_aicc = 0.0;
    bool topological_incongruence = false;

    std::map<std::string, Scalar> site_breakpoint_support;
    std::vector<std::pair<size_t, Scalar>> improvements; // (num_bps, deltaAICc)
    size_t total_models_evaluated = 0;
    double time_elapsed = 0.0;
    std::string input_file;

    nlohmann::json to_json() const {
        nlohmann::json j;
        j["$schema"] = "https://raw.githubusercontent.com/veg/hyphy-3/main/schemas/v3/gard.v3.schema.json";
        j["schema_version"] = "3.0.0";
        j["analysis"] = {
            {"id", "gard"},
            {"name", "GARD"},
            {"version", "3.0.0"},
            {"citation", "Mol Biol Evol 23(10), 1891–1901"},
            {"authors", "Sergei L Kosakovsky Pond"},
            {"contact", "spond@temple.edu"}
        };
        j["software"] = {
            {"name", "hyphy"},
            {"version", "3.0.0"},
            {"homepage", "https://hyphy.org"}
        };
        j["dataset"] = {
            {"file", input_file},
            {"taxa_count", static_cast<int>(num_taxa)},
            {"sites_count", static_cast<int>(num_sites)},
            {"variable_sites_count", static_cast<int>(variable_sites.size())},
            {"potential_breakpoints", static_cast<int>(potential_breakpoints)},
            {"minimum_partition_size", static_cast<int>(min_partition_size)}
        };
        j["baseline"] = {
            {"log_likelihood", baseline_log_l},
            {"aicc", baseline_aicc},
            {"tree", baseline_tree.to_newick(true)},
            {"rates", {
                {"AC", baseline_rates.theta_AC},
                {"AT", baseline_rates.theta_AT},
                {"CG", baseline_rates.theta_CG},
                {"CT", baseline_rates.theta_CT},
                {"GT", baseline_rates.theta_GT}
            }},
            {"frequencies", {
                {"A", baseline_frequencies(0)},
                {"C", baseline_frequencies(1)},
                {"G", baseline_frequencies(2)},
                {"T", baseline_frequencies(3)}
            }}
        };

        nlohmann::json improv_arr = nlohmann::json::array();
        for (const auto& [bps, delta] : improvements) {
            improv_arr.push_back({
                {"breakpoints_count", bps},
                {"delta_aicc", delta}
            });
        }

        j["model_selection"] = {
            {"best_model_breakpoints", best_model.breakpoints},
            {"best_model_aicc", best_model.aicc},
            {"best_model_log_likelihood", best_model.log_likelihood},
            {"improvements", improv_arr}
        };

        nlohmann::json part_arr = nlohmann::json::array();
        for (size_t p = 0; p < best_model.partitions.size(); ++p) {
            const auto& part = best_model.partitions[p];
            part_arr.push_back({
                {"partition_index", p},
                {"start_site", part.start_site},
                {"end_site", part.end_site},
                {"log_likelihood", part.log_likelihood},
                {"tree", part.tree.to_newick(true)}
            });
        }
        j["partitions"] = part_arr;

        j["topology_test"] = {
            {"single_tree_aicc", single_tree_aicc},
            {"topological_incongruence_supported", topological_incongruence},
            {"delta_aicc_topology", single_tree_aicc - best_model.aicc}
        };

        j["site_breakpoint_support"] = site_breakpoint_support;
        j["total_models_evaluated"] = total_models_evaluated;
        j["time_elapsed"] = time_elapsed;

        return j;
    }

    nlohmann::json to_legacy_json() const {
        nlohmann::json j;
        j["analysis"] = {
            {"authors", "Sergei L Kosakovsky Pond"},
            {"citation", "**Automated Phylogenetic Detection of Recombination Using a Genetic Algorithm**, _Mol Biol Evol 23(10), 1891–1901"},
            {"contact", "spond@temple.edu"},
            {"info", "GARD : Genetic Algorithms for Recombination Detection. Implements a heuristic approach to screening alignments of sequences for recombination, by using the CHC genetic algorithm to search for phylogenetic incongruence among different partitions of the data."},
            {"requirements", "A sequence alignment."},
            {"version", "0.2"}
        };

        j["baselineScore"] = baseline_aicc;
        j["bestModelAICc"] = best_model.aicc;

        nlohmann::json bp_data = nlohmann::json::object();
        nlohmann::json trees_obj = nlohmann::json::object();

        if (best_model.breakpoints.empty()) {
            bp_data["0"] = {
                {"bps", nlohmann::json::array({{std::to_string(1), std::to_string(num_sites)}})},
                {"tree", baseline_tree.to_newick(true)}
            };
            trees_obj["newickString"] = baseline_tree.to_newick(true);
        } else {
            for (size_t p = 0; p < best_model.partitions.size(); ++p) {
                const auto& part = best_model.partitions[p];
                bp_data[std::to_string(p)] = {
                    {"bps", nlohmann::json::array({{std::to_string(part.start_site), std::to_string(part.end_site)}})},
                    {"tree", part.tree.to_newick(true)}
                };
                trees_obj[std::to_string(p)] = {
                    {"newickString", part.tree.to_newick(true)}
                };
            }
        }
        j["breakpointData"] = bp_data;
        j["trees"] = trees_obj;

        nlohmann::json improv_obj = nlohmann::json::object();
        if (improvements.empty() || best_model.breakpoints.empty()) {
            improv_obj["0"] = {
                {"breakpoints", nullptr},
                {"deltaAICc", 0.0}
            };
        } else {
            for (size_t i = 0; i < improvements.size(); ++i) {
                improv_obj[std::to_string(i)] = {
                    {"breakpoints", best_model.breakpoints},
                    {"deltaAICc", improvements[i].second}
                };
            }
        }
        j["improvements"] = improv_obj;

        j["input"] = {
            {"file name", input_file},
            {"number of sequences", num_taxa},
            {"number of sites", num_sites},
            {"partition count", best_model.partitions.size()}
        };
        j["potentialBreakpoints"] = potential_breakpoints;
        j["singleTreeAICc"] = single_tree_aicc;
        j["siteBreakPointSupport"] = site_breakpoint_support;
        j["timeElapsed"] = static_cast<size_t>(time_elapsed + 0.5);
        j["totalModelCount"] = total_models_evaluated;

        return j;
    }
};

class GARDAnalyzer {
public:
    GARDSettings settings;
    const core::Alignment& alignment;

    struct PairHash {
        size_t operator()(const std::pair<size_t, size_t>& p) const {
            return std::hash<size_t>()(p.first) ^ (std::hash<size_t>()(p.second) << 16);
        }
    };

    struct PartitionFit {
        core::Tree tree;
        Scalar log_likelihood = 0.0;
    };

    GARDAnalyzer(const core::Alignment& aln, GARDSettings s = GARDSettings())
        : settings(s), alignment(aln) {
        if (settings.run_mode == "Faster") {
            settings.max_stagnant_generations = std::min(settings.population_size, static_cast<size_t>(40));
            settings.caic_improvement_threshold = 2.0;
        }
    }

    GARDResult run() {
        auto start_time = std::chrono::high_resolution_clock::now();
        GARDResult result;
        result.input_file = settings.input_file;
        result.num_sites = alignment.num_nucleotides;
        result.num_taxa = alignment.num_taxa;

        if (result.num_taxa < 3) {
            throw std::runtime_error("GARD requires at least 3 sequences for phylogenetic inference");
        }

        // 1. Identify variable sites
        result.variable_sites.clear();
        for (size_t site = 0; site < result.num_sites; ++site) {
            int8_t first_state = -1;
            bool is_variable = false;
            for (size_t t = 0; t < result.num_taxa; ++t) {
                int8_t s = core::char_to_nuc(alignment.sequences[t].raw_seq[site]);
                if (s >= 0 && s < 4) {
                    if (first_state == -1) {
                        first_state = s;
                    } else if (s != first_state) {
                        is_variable = true;
                        break;
                    }
                }
            }
            if (is_variable) {
                result.variable_sites.push_back(site);
            }
        }

        result.potential_breakpoints = result.variable_sites.empty() ? 0 : (result.variable_sites.size() - 1);
        result.min_partition_size = 2 * result.num_taxa - 3;

        size_t min_expected_sites = 5 + 2 * result.min_partition_size + 2;
        if (result.num_sites < min_expected_sites) {
            throw std::runtime_error("Alignment is too short to permit c-AIC based model comparison. Need at least " +
                                     std::to_string(min_expected_sites) + " sites for " +
                                     std::to_string(result.num_taxa) + " sequences.");
        }

        if (!settings.quiet) {
            std::cout << "[GARD] Loaded alignment with " << result.num_taxa << " sequences, "
                      << result.num_sites << " sites (" << result.variable_sites.size()
                      << " variable sites, " << result.potential_breakpoints << " potential breakpoints).\n";
            std::cout << "[GARD] Minimum partition size: " << result.min_partition_size << " sites.\n";
        }

        // 2. Baseline fit on entire alignment
        if (!settings.quiet) {
            std::cout << "[GARD] Fitting baseline (single-partition; no breakpoints) model...\n";
        }

        core::Tree baseline_nj = core::NeighborJoining::infer(alignment);
        opt::GTRFitter baseline_fitter(baseline_nj, alignment);
        auto base_fit = baseline_fitter.fit();

        result.baseline_tree = base_fit.tree;
        result.baseline_rates = base_fit.params;
        result.baseline_frequencies = alignment.nuc_frequencies;
        result.baseline_log_l = base_fit.log_likelihood;

        size_t K_base = 5 + result.min_partition_size;
        result.baseline_aicc = -2.0 * result.baseline_log_l +
            (2.0 * result.num_sites * K_base) / (result.num_sites - K_base - 1.0);

        if (!settings.quiet) {
            std::cout << "[GARD] Baseline fit: Log-L = " << std::fixed << std::setprecision(4)
                      << result.baseline_log_l << ", c-AIC = " << result.baseline_aicc << "\n";
        }

        // Set baseline as initial best model
        GARDModel best_overall_model;
        best_overall_model.breakpoints.clear();
        best_overall_model.log_likelihood = result.baseline_log_l;
        best_overall_model.aicc = result.baseline_aicc;
        GARDPartition base_part;
        base_part.start_site = 1;
        base_part.end_site = result.num_sites;
        base_part.tree = result.baseline_tree;
        base_part.log_likelihood = result.baseline_log_l;
        best_overall_model.partitions.push_back(std::move(base_part));

        result.best_model = best_overall_model;

        // Thread-safe partition cache
        std::unordered_map<std::pair<size_t, size_t>, PartitionFit, PairHash> partition_cache;
        std::mutex cache_mutex;

        // Pre-populate whole alignment in cache
        partition_cache[{0, result.num_sites}] = {result.baseline_tree, result.baseline_log_l};

        auto fit_segment = [&](size_t start, size_t end) -> PartitionFit {
            std::pair<size_t, size_t> key = {start, end};
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                auto it = partition_cache.find(key);
                if (it != partition_cache.end()) {
                    return it->second;
                }
            }

            core::Alignment slice = alignment.slice_nucleotide(start, end);
            core::Tree nj_tree = core::NeighborJoining::infer(slice);
            int passes = (settings.run_mode == "Faster") ? 1 : 2;
            auto [opt_tree, lnl] = opt::GTRBranchLengthFitter::fit(
                std::move(nj_tree), slice, result.baseline_rates, result.baseline_frequencies, passes
            );

            PartitionFit fit{std::move(opt_tree), lnl};
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                partition_cache[key] = fit;
            }
            return fit;
        };

        // Master list of evaluated models: canonical key -> c-AIC
        std::unordered_map<std::string, Scalar> master_list;
        std::mutex master_mutex;
        master_list[best_overall_model.to_key()] = result.baseline_aicc;

        auto evaluate_candidate_model = [&](const std::vector<size_t>& bps) -> GARDModel {
            GARDModel model;
            model.breakpoints = bps;
            std::sort(model.breakpoints.begin(), model.breakpoints.end());

            std::string key = model.to_key();
            {
                std::lock_guard<std::mutex> lock(master_mutex);
                auto it = master_list.find(key);
                if (it != master_list.end()) {
                    model.aicc = it->second;
                    // If cached aicc already known, return stub model with aicc
                    return model;
                }
            }

            size_t num_bp = model.breakpoints.size();
            std::vector<std::pair<size_t, size_t>> segments;
            segments.reserve(num_bp + 1);

            size_t curr_start = 0;
            for (size_t b = 0; b < num_bp; ++b) {
                size_t curr_end = model.breakpoints[b] + 1;
                segments.push_back({curr_start, curr_end});
                curr_start = curr_end;
            }
            segments.push_back({curr_start, result.num_sites});

            Scalar total_lnl = 0.0;
            model.partitions.reserve(segments.size());
            for (const auto& [s, e] : segments) {
                PartitionFit pf = fit_segment(s, e);
                total_lnl += pf.log_likelihood;
                GARDPartition part;
                part.start_site = s + 1;
                part.end_site = e;
                part.tree = pf.tree;
                part.log_likelihood = pf.log_likelihood;
                model.partitions.push_back(std::move(part));
            }

            model.log_likelihood = total_lnl;
            size_t K = 5 + (num_bp + 1) * result.min_partition_size;
            if (result.num_sites <= K + 1) {
                model.aicc = std::numeric_limits<Scalar>::infinity();
            } else {
                model.aicc = -2.0 * total_lnl + (2.0 * result.num_sites * K) / (result.num_sites - K - 1.0);
            }

            {
                std::lock_guard<std::mutex> lock(master_mutex);
                master_list[key] = model.aicc;
            }

            return model;
        };

        auto validate_breakpoints = [&](const std::vector<size_t>& bps) -> bool {
            size_t last_bp = 0;
            for (size_t bp : bps) {
                if (bp < last_bp || (bp - last_bp + 1) < result.min_partition_size) {
                    return false;
                }
                last_bp = bp + 1;
            }
            if (result.num_sites < last_bp || (result.num_sites - last_bp) < result.min_partition_size) {
                return false;
            }
            return true;
        };

        // 3. Exhaustive Single Breakpoint Analysis (B = 1)
        if (!settings.quiet) {
            std::cout << "[GARD] Performing exhaustive single breakpoint analysis ("
                      << result.potential_breakpoints << " candidate positions)...\n";
        }

        std::vector<size_t> candidate_single_bps;
        for (size_t idx = 0; idx < result.potential_breakpoints; ++idx) {
            size_t bp = result.variable_sites[idx];
            if (validate_breakpoints({bp})) {
                candidate_single_bps.push_back(bp);
            }
        }

        GARDVisualizer visualizer(result.num_sites, result.num_taxa, settings.quiet, settings.force_tty, settings.dynamic_chart);

        GARDModel best_single_bp_model;
        best_single_bp_model.aicc = std::numeric_limits<Scalar>::infinity();

        visualizer.start_single_bp(candidate_single_bps.size(), result.baseline_aicc);

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < candidate_single_bps.size(); ++i) {
            size_t bp = candidate_single_bps[i];
            GARDModel m = evaluate_candidate_model({bp});

            visualizer.record_candidate({bp}, m.aicc);
            #pragma omp critical
            {
                if (m.aicc < best_single_bp_model.aicc) {
                    best_single_bp_model = m;
                    visualizer.update_best({bp}, m.aicc);
                }
            }
            visualizer.tick(1);
        }

        visualizer.finish_single_bp(best_single_bp_model.breakpoints, best_single_bp_model.aicc, result.baseline_aicc);

        bool single_bp_improves = (best_single_bp_model.aicc < result.baseline_aicc);
        if (single_bp_improves) {
            Scalar delta = result.baseline_aicc - best_single_bp_model.aicc;
            result.improvements.push_back({1, delta});
            best_overall_model = best_single_bp_model;
            result.best_model = best_overall_model;
        }

        // 4. Multi-Breakpoint CHC Genetic Algorithm (B >= 2)
        bool adding_improves = single_bp_improves;
        size_t current_B = 1;

        std::mt19937 rng(42);

        while (adding_improves && current_B < settings.max_breakpoints &&
               (current_B + 2) * result.min_partition_size <= result.num_sites) {
            current_B += 1;

            visualizer.start_ga(current_B, settings.population_size, best_overall_model.aicc, best_overall_model.breakpoints);

            // Initialize population of models
            std::vector<std::vector<size_t>> population;
            population.reserve(settings.population_size);

            // Half seeded with best overall model + 1 new random variable site
            size_t num_seeded = settings.population_size / 2;
            for (size_t s = 0; s < num_seeded; ++s) {
                int attempts = 0;
                while (attempts < 100) {
                    std::vector<size_t> cand = best_overall_model.breakpoints;
                    size_t rand_idx = rng() % result.potential_breakpoints;
                    cand.push_back(result.variable_sites[rand_idx]);
                    std::sort(cand.begin(), cand.end());
                    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                    if (cand.size() == current_B && validate_breakpoints(cand)) {
                        population.push_back(cand);
                        break;
                    }
                    attempts++;
                }
            }

            // Other half purely random
            while (population.size() < settings.population_size) {
                int attempts = 0;
                while (attempts < 100) {
                    std::vector<size_t> cand;
                    cand.reserve(current_B);
                    while (cand.size() < current_B) {
                        size_t rand_idx = rng() % result.potential_breakpoints;
                        size_t bp = result.variable_sites[rand_idx];
                        if (std::find(cand.begin(), cand.end(), bp) == cand.end()) {
                            cand.push_back(bp);
                        }
                    }
                    std::sort(cand.begin(), cand.end());
                    if (validate_breakpoints(cand)) {
                        population.push_back(cand);
                        break;
                    }
                    attempts++;
                }
            }

            // Generational GA loop
            bool ga_terminated = false;
            size_t generation = 0;
            size_t generations_stagnant = 0;
            size_t generations_no_new_models = 0;
            Scalar previous_best_caic = std::numeric_limits<Scalar>::infinity();
            GARDModel current_best_model_for_B;
            current_best_model_for_B.aicc = std::numeric_limits<Scalar>::infinity();

            while (!ga_terminated) {
                generation++;
                (void)generation;

                // Produce child models via recombination
                std::vector<std::vector<size_t>> child_models;
                child_models.reserve(settings.population_size);

                for (size_t c = 0; c < settings.population_size; ++c) {
                    int attempts = 0;
                    while (attempts < 7) {
                        size_t p1 = rng() % population.size();
                        size_t p2 = rng() % population.size();
                        while (p2 == p1 && population.size() > 1) {
                            p2 = rng() % population.size();
                        }

                        std::vector<size_t> child_bp(current_B);
                        for (size_t b = 0; b < current_B; ++b) {
                            child_bp[b] = (rng() % 2 == 0) ? population[p1][b] : population[p2][b];
                        }
                        std::sort(child_bp.begin(), child_bp.end());
                        child_bp.erase(std::unique(child_bp.begin(), child_bp.end()), child_bp.end());

                        if (child_bp.size() == current_B && validate_breakpoints(child_bp)) {
                            child_models.push_back(std::move(child_bp));
                            break;
                        }
                        attempts++;
                    }
                }

                // Intergenerational pool = parentModels + childModels
                std::vector<std::vector<size_t>> pool = population;
                for (auto& cm : child_models) {
                    pool.push_back(std::move(cm));
                }

                // Unique models
                std::sort(pool.begin(), pool.end());
                pool.erase(std::unique(pool.begin(), pool.end()), pool.end());

                // Evaluate unevaluated models
                std::vector<std::pair<std::vector<size_t>, Scalar>> evaluated_pool(pool.size());
                #pragma omp parallel for schedule(dynamic)
                for (size_t idx = 0; idx < pool.size(); ++idx) {
                    GARDModel m = evaluate_candidate_model(pool[idx]);
                    evaluated_pool[idx] = {pool[idx], m.aicc};
                    visualizer.record_candidate(pool[idx], m.aicc);
                    visualizer.tick(1);
                }

                // Elitist selection: top population_size models
                std::sort(evaluated_pool.begin(), evaluated_pool.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });

                std::vector<std::vector<size_t>> selected;
                size_t num_keep = std::min(settings.population_size, evaluated_pool.size());
                selected.reserve(num_keep);
                for (size_t k = 0; k < num_keep; ++k) {
                    selected.push_back(evaluated_pool[k].first);
                }

                Scalar current_best_caic = evaluated_pool[0].second;
                if (current_best_caic < current_best_model_for_B.aicc) {
                    current_best_model_for_B = evaluate_candidate_model(evaluated_pool[0].first);
                }

                // Check convergence & diversity
                if (selected == population) {
                    generations_no_new_models++;
                } else {
                    generations_no_new_models = 0;
                }

                Scalar max_caic = evaluated_pool[num_keep - 1].second;
                Scalar min_caic = evaluated_pool[0].second;
                Scalar norm_range = (max_caic > 1e-12) ? ((max_caic - min_caic) / max_caic) : 0.0;

                size_t max_allowed_no_new = std::max(static_cast<size_t>(2), settings.max_stagnant_generations / 4);
                if (norm_range < settings.caic_diversity_threshold || generations_no_new_models > max_allowed_no_new) {
                    // Cataclysmic mutation: keep best, mutate others
                    std::vector<std::vector<size_t>> mutated_pop;
                    mutated_pop.push_back(selected[0]);

                    Scalar local_mut_rate = 1.0 - std::pow(1.0 - settings.mutation_rate, 1.0 / current_B);
                    std::poisson_distribution<int> poiss(1);

                    for (size_t i = 1; i < selected.size(); ++i) {
                        int attempts = 0;
                        while (attempts < 7) {
                            std::vector<size_t> mut_bp = selected[i];
                            for (size_t b = 0; b < current_B; ++b) {
                                Scalar r_val = static_cast<Scalar>(rng() % 10000) / 10000.0;
                                if (r_val >= local_mut_rate) {
                                    Scalar shift_r = static_cast<Scalar>(rng() % 10000) / 10000.0;
                                    if (shift_r < settings.rate_small_shifts) {
                                        int step = std::max(1, poiss(rng));
                                        if (rng() % 2 == 0) step = -step;

                                        // Find index in variable_sites
                                        auto it = std::lower_bound(result.variable_sites.begin(), result.variable_sites.end(), mut_bp[b]);
                                        if (it != result.variable_sites.end()) {
                                            int orig_v_idx = static_cast<int>(std::distance(result.variable_sites.begin(), it));
                                            int new_v_idx = orig_v_idx + step;
                                            if (new_v_idx >= 0 && new_v_idx < static_cast<int>(result.variable_sites.size())) {
                                                mut_bp[b] = result.variable_sites[new_v_idx];
                                            }
                                        }
                                    } else {
                                        size_t rand_v = rng() % result.potential_breakpoints;
                                        mut_bp[b] = result.variable_sites[rand_v];
                                    }
                                }
                            }
                            std::sort(mut_bp.begin(), mut_bp.end());
                            mut_bp.erase(std::unique(mut_bp.begin(), mut_bp.end()), mut_bp.end());
                            if (mut_bp.size() == current_B && validate_breakpoints(mut_bp)) {
                                mutated_pop.push_back(std::move(mut_bp));
                                break;
                            }
                            attempts++;
                        }
                    }

                    if (mutated_pop.size() <= 1) {
                        // Reinitialize with seeds
                        mutated_pop.clear();
                        mutated_pop.push_back(selected[0]);
                        while (mutated_pop.size() < settings.population_size) {
                            std::vector<size_t> cand = best_overall_model.breakpoints;
                            size_t rand_idx = rng() % result.potential_breakpoints;
                            cand.push_back(result.variable_sites[rand_idx]);
                            std::sort(cand.begin(), cand.end());
                            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                            if (cand.size() == current_B && validate_breakpoints(cand)) {
                                mutated_pop.push_back(cand);
                            }
                        }
                    }
                    population = std::move(mutated_pop);
                } else {
                    population = std::move(selected);
                }

                // Check stagnation
                if (previous_best_caic - current_best_caic < settings.caic_improvement_threshold) {
                    generations_stagnant++;
                    if (generations_stagnant >= settings.max_stagnant_generations) {
                        ga_terminated = true;
                    }
                } else {
                    generations_stagnant = 0;
                }
                previous_best_caic = current_best_caic;

                visualizer.update_ga_generation(
                    generation,
                    settings.max_stagnant_generations,
                    generations_stagnant,
                    norm_range,
                    current_best_model_for_B.breakpoints,
                    current_best_model_for_B.aicc
                );
            } // end GA loop for current_B

            // Evaluate if this B improved over best overall
            bool improved = (current_best_model_for_B.aicc < best_overall_model.aicc);
            visualizer.finish_ga(current_B, current_best_model_for_B.breakpoints, current_best_model_for_B.aicc, best_overall_model.aicc, improved);

            if (improved) {
                Scalar delta = best_overall_model.aicc - current_best_model_for_B.aicc;
                result.improvements.push_back({current_B, delta});
                best_overall_model = current_best_model_for_B;
                result.best_model = best_overall_model;
            } else {
                adding_improves = false;
            }
        } // end multi-breakpoint loop

        // 5. Post-hoc Topological Incongruence Test
        // Fit best breakpoint placement constrained to baseline single tree topology
        if (!settings.quiet) {
            std::cout << "[GARD] Running post-hoc topological incongruence test...\n";
        }

        if (best_overall_model.breakpoints.empty()) {
            result.single_tree_aicc = result.baseline_aicc;
            result.topological_incongruence = false;
        } else {
            size_t num_bp = best_overall_model.breakpoints.size();
            std::vector<std::pair<size_t, size_t>> segments;
            size_t curr_start = 0;
            for (size_t b = 0; b < num_bp; ++b) {
                size_t curr_end = best_overall_model.breakpoints[b] + 1;
                segments.push_back({curr_start, curr_end});
                curr_start = curr_end;
            }
            segments.push_back({curr_start, result.num_sites});

            Scalar single_tree_log_l = 0.0;
            for (const auto& [s, e] : segments) {
                core::Alignment slice = alignment.slice_nucleotide(s, e);
                int passes = (settings.run_mode == "Faster") ? 1 : 2;
                auto [opt_t, lnl] = opt::GTRBranchLengthFitter::fit(
                    result.baseline_tree, slice, result.baseline_rates, result.baseline_frequencies, passes
                );
                single_tree_log_l += lnl;
            }

            size_t K = 5 + (num_bp + 1) * result.min_partition_size;
            result.single_tree_aicc = -2.0 * single_tree_log_l +
                (2.0 * result.num_sites * K) / (result.num_sites - K - 1.0);

            result.topological_incongruence = (best_overall_model.aicc < result.single_tree_aicc);
        }

        if (!settings.quiet) {
            std::cout << "[GARD] Single-tree c-AIC = " << std::fixed << std::setprecision(4)
                      << result.single_tree_aicc << " vs Partitioned c-AIC = " << best_overall_model.aicc << "\n";
            std::cout << "[GARD] Topological incongruence (recombination): "
                      << (result.topological_incongruence ? "SUPPORTED" : "NOT supported") << "\n";
        }

        // 6. Compute Akaike weights for siteBreakPointSupport
        result.total_models_evaluated = master_list.size();
        Scalar min_master_caic = best_overall_model.aicc;

        std::unordered_map<size_t, Scalar> site_weights;
        Scalar total_aic_weight = 0.0;

        for (const auto& [key, score] : master_list) {
            if (std::isinf(score) || std::isnan(score)) continue;
            Scalar aicw = std::exp(-0.5 * (score - min_master_caic));
            if (aicw > 0.0) {
                total_aic_weight += aicw;
                std::vector<size_t> bps = GARDModel::parse_key(key);
                for (size_t bp : bps) {
                    site_weights[bp] += aicw;
                }
            }
        }

        result.site_breakpoint_support.clear();
        if (total_aic_weight > 0.0) {
            for (const auto& [bp, w] : site_weights) {
                result.site_breakpoint_support[std::to_string(bp)] = w / total_aic_weight;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.time_elapsed = std::chrono::duration<double>(end_time - start_time).count();

        if (!settings.quiet) {
            std::cout << "[GARD] Total models evaluated: " << result.total_models_evaluated
                      << " in " << std::fixed << std::setprecision(2) << result.time_elapsed << " seconds.\n";
        }

        return result;
    }
};

} // namespace hyphy::analyses
