#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/console.hpp"
#include "hyphy/analyses/ohmic.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace hyphy::core;
using namespace hyphy::analyses;

static void print_ohmic_banner() {
    Panel::print_banner(
        "OHMIC",
        "Ohmic Phylogenetic Networks & Linear-Time Wheatstone Pruning",
        "Foster Effective Resistance • Series-Parallel Wheatstone Bridges • Scalable Woodbury Engine",
        "Citation: Kosakovsky Pond et al. (2026) • v3.0.0"
    );
}

static void print_ohmic_usage(const char* prog) {
    print_ohmic_banner();
    std::cout << Console::bold("Usage:") << " " << prog << " [OPTIONS]\n\n"
              << Console::bold("Required arguments:") << "\n"
              << "  " << Console::brand("--alignment, -a") << " <file>         Path to sequence alignment (NEXUS or FASTA)\n\n"
              << Console::bold("Optional arguments:") << "\n"
              << "  " << Console::brand("--datatype, -d") << " <type>          Model datatype: 'nucleotide' (GTR, default) or 'codon' (MG94xREV)\n"
              << "  " << Console::brand("--hyphaeon") << "                     Use HyphAeon 384D multi-scale transformer embeddings (default)\n"
              << "  " << Console::brand("--no-hyphaeon") << "                  Disable HyphAeon transformer (use empirical conductances only)\n"
              << "  " << Console::brand("--embeddings") << " <file>            Path to precomputed [N x D] taxon embeddings TSV/CSV\n"
              << "  " << Console::brand("--scale-mode, -s") << " <mode>        Solver mode: 'auto' (default), 'dense', or 'woodbury'\n"
              << "  " << Console::brand("--min-stress") << " <tau>             Minimum shunt stress tau_Ohm to promote a bridge (default: 0.05)\n"
              << "  " << Console::brand("--max-bridges") << " <K>              Maximum disjoint Wheatstone bridges (default: 64)\n"
              << "  " << Console::brand("--woodbury-rank") << " <p>            Rank p for ScalableWoodbury landmarks (default: 256)\n"
              << "  " << Console::brand("--threads, -t") << " <N>              Number of OpenMP worker threads\n"
              << "  " << Console::brand("--output, -o") << " <file>            Path to output JSON file (default: <alignment>.OHMIC.json)\n"
              << "  " << Console::brand("--format") << " <fmt>                 Output format: 'modern' (default) or 'legacy'\n"
              << "  " << Console::brand("--quiet, -q") << "                    Suppress progress output\n"
              << "  " << Console::brand("--help, -h") << "                     Show this help message\n\n"
              << Console::bold("Examples:") << "\n"
              << "  " << prog << " --alignment tests/data/cd2.fna\n"
              << "  " << prog << " --alignment tests/data/bglobin.nex --datatype codon --threads 8\n\n";
}

int run_ohmic(int argc, char* argv[]) {
    std::string alignment_file;
    std::string output_file;
    std::string datatype = "nucleotide";
    std::string scale_mode_str = "auto";
    std::string output_format = "modern";
    std::string embeddings_file;
    Scalar min_stress = 0.05;
    size_t max_bridges = 64;
    size_t woodbury_rank = 256;
    int num_threads = 0;
    bool quiet = false;
    bool use_hyphaeon = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--alignment" || arg == "-a") && i + 1 < argc) {
            alignment_file = argv[++i];
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "--datatype" || arg == "-d") && i + 1 < argc) {
            datatype = argv[++i];
        } else if (arg == "--hyphaeon") {
            use_hyphaeon = true;
        } else if (arg == "--no-hyphaeon") {
            use_hyphaeon = false;
        } else if (arg == "--embeddings" && i + 1 < argc) {
            embeddings_file = argv[++i];
        } else if ((arg == "--scale-mode" || arg == "-s") && i + 1 < argc) {
            scale_mode_str = argv[++i];
        } else if (arg == "--min-stress" && i + 1 < argc) {
            min_stress = std::stod(argv[++i]);
        } else if (arg == "--max-bridges" && i + 1 < argc) {
            max_bridges = std::stoull(argv[++i]);
        } else if (arg == "--woodbury-rank" && i + 1 < argc) {
            woodbury_rank = std::stoull(argv[++i]);
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--format" && i + 1 < argc) {
            output_format = argv[++i];
        } else if (arg == "--legacy-json") {
            output_format = "legacy";
        } else if (arg == "--quiet" || arg == "-q") {
            quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            print_ohmic_usage(argv[0]);
            return 0;
        }
    }

    if (alignment_file.empty()) {
        std::cerr << Console::danger("Error: --alignment argument is required.\n\n");
        print_ohmic_usage(argv[0]);
        return 1;
    }

    if (num_threads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
#endif
    }

    if (output_file.empty()) {
        output_file = alignment_file + ".OHMIC.json";
    }

    if (!quiet) {
        print_ohmic_banner();
    }

    bool use_codon = (datatype == "codon" || datatype == "mg94" || datatype == "MG94");

    Alignment aln;
    try {
        if (use_codon) {
            aln = Alignment::load(alignment_file);
        } else {
            aln = Alignment::load_nucleotide(alignment_file);
        }
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Failed to load alignment: " + std::string(e.what())) << "\n";
        return 1;
    }

    OhmicAnalysisSettings settings;
    settings.input_file = alignment_file;
    settings.output_file = output_file;
    settings.use_codon_model = use_codon;
    settings.quiet = quiet;
    settings.network_settings.min_shunt_stress = min_stress;
    settings.network_settings.max_bridges = max_bridges;
    settings.network_settings.woodbury_rank = woodbury_rank;
    settings.network_settings.use_hyphaeon_transformer = use_hyphaeon;
    settings.network_settings.embeddings_filepath = embeddings_file;

    if (scale_mode_str == "dense" || scale_mode_str == "ExactDense") {
        settings.network_settings.scale_mode = OhmicScaleMode::ExactDense;
    } else if (scale_mode_str == "woodbury" || scale_mode_str == "ScalableWoodbury") {
        settings.network_settings.scale_mode = OhmicScaleMode::ScalableWoodbury;
    } else {
        settings.network_settings.scale_mode = OhmicScaleMode::Auto;
    }

    try {
        OhmicAnalyzer analyzer(aln, settings);
        OhmicResult result = analyzer.run();

        nlohmann::json json_out = (output_format == "legacy")
            ? result.to_legacy_json(alignment_file)
            : result.to_modern_json(alignment_file);

        std::ofstream out(output_file);
        if (!out.is_open()) {
            std::cerr << Console::danger("Failed to open output file: " + output_file) << "\n";
            return 1;
        }
        out << json_out.dump(2) << "\n";
        out.close();

        if (!quiet) {
            std::ostringstream lnl_s, lnl_o, dlnl, aicc_s, aicc_o, daicc, foster_ss, time_ss;
            lnl_s << std::fixed << std::setprecision(4) << result.single_tree_log_l;
            lnl_o << std::fixed << std::setprecision(4) << result.ohmic_log_l;
            dlnl  << std::fixed << std::setprecision(4) << (result.delta_log_l >= 0 ? "+" : "") << result.delta_log_l;
            aicc_s << std::fixed << std::setprecision(2) << result.single_tree_aicc;
            aicc_o << std::fixed << std::setprecision(2) << result.ohmic_aicc;
            daicc  << std::fixed << std::setprecision(2) << (result.delta_aicc >= 0 ? "+" : "") << result.delta_aicc;
            foster_ss << std::fixed << std::setprecision(5) << result.foster_sum
                      << " (Expected N-1 = " << (result.num_taxa - 1) << ")";
            time_ss << std::fixed << std::setprecision(3) << result.runtime_seconds << "s";

            std::string mode_label = (result.network.active_mode == OhmicScaleMode::ExactDense)
                ? "ExactDense (Full Laplacian Pseudoinverse)"
                : "ScalableWoodbury (O(N log N) Low-Rank Resistance)";

            bool improved = (result.delta_log_l > 0.01);
            std::string status_conclusion = improved
                ? Console::success("✔ Wheatstone pruning resolved topological conflict and improved likelihood over a single tree.")
                : Console::muted("ℹ Alignment is well-represented by a single Consensus Backbone tree.");

            std::vector<std::pair<std::string, std::string>> summary_items = {
                {"Alignment", alignment_file + " (" + std::to_string(result.num_taxa) + " taxa, " + std::to_string(result.num_sites) + " sites)"},
                {"Electrical Solver", mode_label + " [" + std::to_string(result.num_super_nodes) + " super-nodes]"},
                {"Foster Sum Invariant", (result.network.active_mode == OhmicScaleMode::ExactDense) ? foster_ss.str() : "N/A (Woodbury Mode)"},
                {"Wheatstone Bridges (K)", std::to_string(result.num_bridges) + " active 4-way bridges, " + std::to_string(result.num_collapsed_polytomies) + " collapsed polytomies"},
                {"Single-Tree Log(L) / AICc", lnl_s.str() + "  /  " + aicc_s.str()},
                {"Ohmic Network Log(L) / AICc", lnl_o.str() + "  /  " + aicc_o.str()},
                {"Improvement (Delta LogL / AICc)", dlnl.str() + " nats  /  " + daicc.str() + " AICc"},
                {"Total Runtime", time_ss.str()},
                {"Output Results File", Console::brand(output_file)}
            };

            Panel::print_summary_card(
                "OHMIC Network Analysis Results",
                summary_items,
                status_conclusion,
                improved
            );
        }
    } catch (const std::exception& e) {
        std::cerr << Console::danger("OHMIC analysis error: " + std::string(e.what())) << "\n";
        return 1;
    }

    return 0;
}

#ifndef HYPHY3_COMBINED_DRIVER
int main(int argc, char* argv[]) {
    return run_ohmic(argc, argv);
}
#endif
