#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/console.hpp"
#include "hyphy/analyses/gard.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace hyphy::core;
using namespace hyphy::analyses;

static void print_gard_banner() {
    Panel::print_banner(
        "GARD",
        "Genetic Algorithm for Recombination Detection",
        "Screening for Recombination Breakpoints & Phylogenetic Incongruence",
        "Citation: Kosakovsky Pond et al. Mol Biol Evol 23(10), 1891–1901 • v3.0.0"
    );
}

static void print_gard_usage(const char* prog) {
    print_gard_banner();
    std::cout << Console::bold("Usage:") << " " << prog << " [OPTIONS]\n\n"
              << Console::bold("Required arguments:") << "\n"
              << "  " << Console::brand("--alignment, -a") << " <file>         Path to sequence alignment (NEXUS or FASTA)\n\n"
              << Console::bold("Optional arguments:") << "\n"
              << "  " << Console::brand("--mode, -m") << " <mode>              Run mode: 'Normal' (default) or 'Faster'\n"
              << "  " << Console::brand("--max-breakpoints, -b") << " <N>     Maximum number of breakpoints (default: 10000)\n"
              << "  " << Console::brand("--threads, -t") << " <N>             Number of OpenMP worker threads\n"
              << "  " << Console::brand("--output, -o") << " <file>           Path to output JSON file (default: <alignment>.GARD.json)\n"
              << "  " << Console::brand("--format") << " <fmt>                Output format: 'modern' (default) or 'legacy'\n"
              << "  " << Console::brand("--legacy-json") << "                  Output HyPhy 2.5 compatible JSON\n"
              << "  " << Console::brand("--quiet, -q") << "                    Suppress progress output\n"
              << "  " << Console::brand("--help, -h") << "                    Show this help message\n\n"
              << Console::bold("Examples:") << "\n"
              << "  " << prog << " --alignment data/adh.nex\n"
              << "  " << prog << " --alignment data/HRVI.nex --mode Faster --threads 8\n\n";
}

int run_gard(int argc, char* argv[]) {
    std::string alignment_file;
    std::string output_file;
    std::string run_mode = "Normal";
    std::string output_format = "modern";
    size_t max_breakpoints = 10000;
    int num_threads = 0;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--alignment" || arg == "-a") && i + 1 < argc) {
            alignment_file = argv[++i];
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "--mode" || arg == "-m") && i + 1 < argc) {
            run_mode = argv[++i];
        } else if ((arg == "--max-breakpoints" || arg == "-b") && i + 1 < argc) {
            max_breakpoints = std::stoull(argv[++i]);
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--format" && i + 1 < argc) {
            output_format = argv[++i];
        } else if (arg == "--legacy-json") {
            output_format = "legacy";
        } else if (arg == "--quiet" || arg == "-q") {
            quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            print_gard_usage(argv[0]);
            return 0;
        }
    }

    if (alignment_file.empty()) {
        std::cerr << Console::danger("Error: --alignment argument is required.\n\n");
        print_gard_usage(argv[0]);
        return 1;
    }

    if (num_threads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
#endif
    }

    if (output_file.empty()) {
        output_file = alignment_file + ".GARD.json";
    }

    if (!quiet) {
        print_gard_banner();
    }

    Alignment aln;
    try {
        aln = Alignment::load_nucleotide(alignment_file);
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Failed to load alignment: " + std::string(e.what())) << "\n";
        return 1;
    }

    GARDSettings settings;
    settings.input_file = alignment_file;
    settings.output_file = output_file;
    settings.run_mode = run_mode;
    settings.max_breakpoints = max_breakpoints;
    settings.quiet = quiet;
    settings.legacy_json = (output_format == "legacy");

    try {
        GARDAnalyzer analyzer(aln, settings);
        GARDResult result = analyzer.run();

        nlohmann::json json_out;
        if (settings.legacy_json) {
            json_out = result.to_legacy_json();
        } else {
            json_out = result.to_json();
        }

        std::ofstream out(output_file);
        if (!out.is_open()) {
            std::cerr << Console::danger("Failed to open output file: " + output_file) << "\n";
            return 1;
        }
        out << json_out.dump(2) << "\n";
        out.close();

        if (!quiet) {
            std::cout << "\n" << Console::bold("=== GARD Results Summary ===") << "\n";
            std::cout << "  Alignment: " << result.num_taxa << " taxa, " << result.num_sites << " sites\n";
            std::cout << "  Baseline c-AIC: " << std::fixed << std::setprecision(4) << result.baseline_aicc << "\n";
            std::cout << "  Best model c-AIC: " << result.best_model.aicc << " ("
                      << result.best_model.breakpoints.size() << " breakpoints)\n";
            if (!result.best_model.breakpoints.empty()) {
                std::cout << "  Breakpoints (1-based): ";
                for (size_t b = 0; b < result.best_model.breakpoints.size(); ++b) {
                    if (b > 0) std::cout << ", ";
                    std::cout << (result.best_model.breakpoints[b] + 1);
                }
                std::cout << "\n";
            }
            std::cout << "  Single-tree constrained c-AIC: " << result.single_tree_aicc << "\n";
            std::cout << "  Topological incongruence (recombination): "
                      << (result.topological_incongruence ? Console::success("EVIDENCE FOUND") : Console::muted("None detected")) << "\n";
            std::cout << "  Results written to: " << Console::brand(output_file) << "\n\n";
        }

    } catch (const std::exception& e) {
        std::cerr << Console::danger("GARD analysis error: " + std::string(e.what())) << "\n";
        return 1;
    }

    return 0;
}

#ifndef HYPHY3_COMBINED_DRIVER
int main(int argc, char* argv[]) {
    return run_gard(argc, argv);
}
#endif
