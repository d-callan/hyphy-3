#include "hyphy/core/types.hpp"
#include "hyphy/core/genetic_code.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/ohmic_network.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/core/likelihood.hpp"
#include "hyphy/core/console.hpp"
#include "hyphy/opt/optimizer.hpp"
#include "hyphy/analyses/busted.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace hyphy::core;
using namespace hyphy::analyses;
using namespace hyphy::opt;

static void print_banner(bool ph_mode = false) {
    if (ph_mode) {
        Panel::print_banner(
            "BUSTED-PH",
            "Branch-site Unrestricted Statistical Test • Phenotype",
            "Testing whether episodic positive selection is associated with a phenotype/trait",
            "Citation: Mol Biol Evol. 32: 1365-1371 (2015) • v3.0.0"
        );
    } else {
        Panel::print_banner(
            "BUSTED",
            "Branch-site Unrestricted Statistical Test",
            "Gene-wide identification of episodic diversifying positive selection",
            "Citation: Mol Biol Evol. 32: 1365-1371 (2015) • v3.0.0"
        );
    }
}

static void print_usage(const char* prog, bool ph_mode = false) {
    print_banner(ph_mode);
    std::cout << Console::bold("Usage:") << " " << prog << " [OPTIONS]\n\n"
              << Console::bold("Required arguments:") << "\n"
              << "  " << Console::brand("--alignment") << " <file>   Path to codon alignment (FASTA or NEXUS)\n\n"
              << Console::bold("Optional arguments:") << "\n"
              << "  " << Console::brand("--tree") << " <file>        Path to Newick tree file (optional; auto-infers Ohmic Network if omitted)\n"
              << "  " << Console::brand("--ohmic") << "              Build Ohmic Phylogenetic Network with Wheatstone bridges\n"
              << "  " << Console::brand("--hyphaeon") << "           Opt-in: HyphAeon 384D transformer conductances (needs Python aeon-core)\n"
              << "  " << Console::brand("--no-hyphaeon") << "        Use empirical codon conductances only (default)\n"
              << "  " << Console::brand("--embeddings") << " <file>  Path to precomputed [N x 384] HyphAeon embeddings (.bin/.csv/.tsv)\n"
              << "  " << Console::brand("--code") << " <name>        Genetic code (default: Universal)\n"
              << "  " << Console::brand("--branches") << " <spec>    Branches to test (foreground/phenotype): All, Internal, Leaves, {TAG}, or names\n"
              << "  " << Console::brand("--phenotype") << "          Enable BUSTED-PH (3-test phenotype-associated selection + coupled-k RELAX)\n"
              << "  " << Console::brand("--relax") << "              Fit coupled-k RELAX model (omega_FG = omega_BG^k, shared weights) vs k=1 Null\n"
              << "  " << Console::brand("--no-relax") << "           Skip coupled-k RELAX fit when running BUSTED-PH\n"
              << "  " << Console::brand("--comparison") << " <spec>  Background/comparison branch set for BUSTED-PH / RELAX (default: non-test branches)\n"
              << "  " << Console::brand("--auto") << "                Full-Auto BUSTED: 4-gate surprisal & score selector for SRV, MH, Error Sink & K\n"
              << "  " << Console::brand("--rates") << " <N>          Number of omega rate categories (default: 3)\n"
              << "  " << Console::brand("--srv") << "                Enable synonymous rate variation across sites (BUSTED-S)\n"
              << "  " << Console::brand("--syn-rates") << " <N>      Number of synonymous rate categories (default: 3)\n"
              << "  " << Console::brand("--auto-k") << "             Automatically select optimal K via AICc step-up\n"
              << "  " << Console::brand("--multiple-hits") << " <M>  Multi-nucleotide substitutions: None (default), Double, Double+Triple\n"
              << "  " << Console::brand("--error-sink") << "          Enable non-evolutionary saturation error sink class (BUSTED-E)\n"
              << "  " << Console::brand("--error-sink-weight") << " <W> Maximum allowed error sink mixture weight (default: 0.05)\n"
              << "  " << Console::brand("--error-sink-bf") << " <BF>  Empirical Bayes Factor threshold to flag artifact branch-sites (default: 100)\n"
              << "  " << Console::brand("--no-branch-opt") << "      Disable individual branch length refinement (use proportional scaling)\n"
              << "  " << Console::brand("--threads") << " <N>        Number of OpenMP worker threads\n"
              << "  " << Console::brand("--output") << " <file>      Path to output JSON file (default: <alignment>.BUSTED[-PH].json)\n"
              << "  " << Console::brand("--progress") << "           Force interactive progress bar\n"
              << "  " << Console::brand("--no-progress") << "        Disable progress bar\n"
              << "  " << Console::brand("--help, -h") << "           Show this help message\n\n"
              << Console::bold("Examples:") << "\n"
              << "  " << prog << " --alignment data/cd2.fna --tree data/cd2.nwk --auto\n"
              << "  " << prog << " --alignment data/cd2.fna --tree data/cd2.nwk --phenotype --branches Human,Chimp,Baboon\n"
              << "  " << prog << " --alignment tests/data/adh.nex --auto --threads 8\n\n";
}

int run_busted(int argc, char* argv[]) {
    std::string alignment_file;
    std::string tree_file;
    std::string output_file;
    std::string code_name = "Universal";
    std::string embeddings_file = "";
    bool force_ohmic = false;
    bool use_hyphaeon = false;
    int num_threads = 0;
    JSONFormat json_format = JSONFormat::ModernV3;
    BUSTEDSettings settings;
    bool show_progress = ProgressBar::is_terminal();
    bool force_progress = false;

    std::string prog_str = (argc > 0 && argv[0]) ? argv[0] : "hyphy3 busted";
    if (prog_str.find("busted-ph") != std::string::npos || prog_str.find("busted_ph") != std::string::npos) {
        settings.phenotype_mode = true;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--alignment" && i + 1 < argc) {
            alignment_file = argv[++i];
        } else if (arg == "--tree" && i + 1 < argc) {
            tree_file = argv[++i];
        } else if (arg == "--ohmic" || arg == "--omhic") {
            force_ohmic = true;
        } else if (arg == "--hyphaeon") {
            force_ohmic = true;
            use_hyphaeon = true;
        } else if (arg == "--no-hyphaeon") {
            use_hyphaeon = false;
        } else if (arg == "--embeddings" && i + 1 < argc) {
            force_ohmic = true;
            embeddings_file = argv[++i];
        } else if (arg == "--code" && i + 1 < argc) {
            code_name = argv[++i];
        } else if ((arg == "--branches" || arg == "--test" || arg == "--foreground") && i + 1 < argc) {
            settings.test_branches = argv[++i];
        } else if (arg == "--phenotype" || arg == "--busted-ph") {
            settings.phenotype_mode = true;
        } else if (arg == "--relax") {
            settings.relax_mode = true;
            settings.include_relax_in_ph = true;
        } else if (arg == "--no-relax") {
            settings.relax_mode = false;
            settings.include_relax_in_ph = false;
        } else if ((arg == "--comparison" || arg == "--background" || arg == "--reference") && i + 1 < argc) {
            settings.comparison_branches = argv[++i];
        } else if (arg == "--auto" || arg == "--full-auto") {
            settings.full_auto = true;
        } else if (arg == "--rates" && i + 1 < argc) {
            settings.num_rate_classes = std::stoul(argv[++i]);
        } else if (arg == "--srv") {
            settings.srv = true;
        } else if (arg == "--syn-rates" && i + 1 < argc) {
            settings.srv = true;
            settings.num_syn_rate_classes = std::stoul(argv[++i]);
        } else if (arg == "--auto-k") {
            settings.auto_select_k = true;
        } else if (arg == "--multiple-hits" && i + 1 < argc) {
            settings.multiple_hits = argv[++i];
        } else if (arg == "--error-sink") {
            settings.error_sink = true;
        } else if (arg == "--error-sink-weight" && i + 1 < argc) {
            settings.error_sink = true;
            settings.error_sink_max_weight = std::stod(argv[++i]);
        } else if (arg == "--error-sink-bf" && i + 1 < argc) {
            settings.error_sink = true;
            settings.error_sink_bf_threshold = std::stod(argv[++i]);
        } else if (arg == "--no-branch-opt") {
            settings.refine_branch_lengths = false;
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--json-format" && i + 1 < argc) {
            std::string fmt = argv[++i];
            if (fmt == "legacy" || fmt == "datamonkey") {
                json_format = JSONFormat::Legacy;
            } else {
                json_format = JSONFormat::ModernV3;
            }
        } else if (arg == "--legacy-json") {
            json_format = JSONFormat::Legacy;
        } else if (arg == "--modern-json") {
            json_format = JSONFormat::ModernV3;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--progress") {
            show_progress = true;
            force_progress = true;
        } else if (arg == "--no-progress") {
            show_progress = false;
            force_progress = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0], settings.phenotype_mode);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0], settings.phenotype_mode);
            return 1;
        }
    }

    if (alignment_file.empty()) {
        std::cerr << "Error: --alignment is required.\n\n";
        print_usage(argv[0], settings.phenotype_mode);
        return 1;
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif

    if (output_file.empty()) {
        output_file = alignment_file + (settings.phenotype_mode ? ".BUSTED-PH.json" : ".BUSTED.json");
    }

    print_banner(settings.phenotype_mode);

    auto start_time = std::chrono::high_resolution_clock::now();
    std::string start_iso = Provenance::current_iso8601();
    std::string cli_cmd = argv[0];
    for (int i = 1; i < argc; ++i) {
        cli_cmd += " ";
        cli_cmd += argv[i];
    }

    // 1. Load Genetic Code
    std::shared_ptr<const GeneticCode> code;
    try {
        code = GeneticCode::from_name(code_name);
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Error loading genetic code '") << code_name << "': " << e.what() << "\n";
        return 1;
    }

    // 2. Load Alignment
    Alignment aln;
    try {
        aln = Alignment::load(alignment_file, code);
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Error loading alignment: ") << e.what() << "\n";
        return 1;
    }

    // 3. Resolve Tree or Ohmic Phylogenetic Network
    Tree tree;
    OhmicNetwork ohmic_net;
    std::string tree_source;
    try {
        tree = OhmicNetwork::resolve_tree(aln, tree_file, force_ohmic, use_hyphaeon, embeddings_file, &ohmic_net, &tree_source);
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Error resolving tree/network: ") << e.what() << "\n";
        return 1;
    }

    // Validate branch selection upfront before fitting GTR/MG94
    Tree::BranchSelection branch_sel;
    try {
        branch_sel = tree.select_branches(settings.test_branches, "", {}, settings.phenotype_mode || settings.relax_mode);
        if ((settings.phenotype_mode || settings.relax_mode) && branch_sel.num_unselected == 0) {
            throw std::runtime_error(
                "BUSTED-PH / RELAX requires a proper subset of foreground/test branches and a non-empty background/reference set. "
                "Specify foreground branches via --branches <spec> or label branches with {TEST}/{Foreground} in the tree."
            );
        }
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Error: ") << e.what() << "\n";
        return 1;
    }

    Panel::print_card("Dataset & Model Settings", {
        {"Alignment File", alignment_file},
        {"Sequences / Taxa", std::to_string(aln.num_taxa)},
        {"Codon Sites", std::to_string(aln.num_codons) + " (" + std::to_string(aln.num_codons * 3) + " nt)"},
        {"Unique Patterns", std::to_string(aln.patterns.size())},
        {"Tree Source", tree_source + " (" + std::to_string(tree.num_nodes()) + " nodes, " + std::to_string(tree.num_leaves()) + " leaves)"},
        {"Tested Branches", settings.test_branches + " (" + std::to_string(branch_sel.num_selected) + " test, " + std::to_string(branch_sel.num_unselected) + " background)"},
        {"Analysis Mode", settings.phenotype_mode ? (settings.include_relax_in_ph ? "BUSTED-PH + RELAX (UIT on FG/Diff + AIC-c BG + Coupled-k RELAX)" : "BUSTED-PH (UIT on FG/Diff + AIC-c BG Screen)") : (settings.relax_mode ? "BUSTED + Coupled-k RELAX" : (settings.full_auto ? "Full-Auto (4-Gate Surprisal & Score Screen)" : "Standard BUSTED"))},
        {"Rate Classes (K)", settings.full_auto ? "Auto (2..3)" : (std::to_string(settings.num_rate_classes) + (settings.auto_select_k ? " (Auto-K enabled)" : ""))},
        {"Synonymous Variation", settings.full_auto ? "Auto" : (settings.srv ? "Enabled (" + std::to_string(settings.num_syn_rate_classes) + " classes)" : "Disabled")},
        {"Multiple Hits", settings.full_auto ? "Auto" : settings.multiple_hits},
        {"Error Sink (BUSTED-E)", settings.full_auto ? "Auto" : (settings.error_sink ? "Enabled (Saturation Replacement Kernel)" : "Disabled")},
        {"Genetic Code", code->name}
    });

    // 4. Initialize and run BUSTED / BUSTED-PH
    Panel::print_step(1, 2, "Fitting Baseline GTR & Global MG94 Models", "Exchangeability rates & branch lengths");
    BUSTEDAnalyzer analyzer = BUSTEDAnalyzer::create_and_fit(tree, aln);
    analyzer.select_branch_partitions(settings.test_branches, settings.comparison_branches, settings.phenotype_mode || settings.relax_mode);

    std::cout << "  " << Console::muted("Nucleotide GTR Log-L : ") << Console::bold(std::to_string(analyzer.gtr_log_l)) << "\n"
              << "  " << Console::muted("Global MG94 Log-L    : ") << Console::bold(std::to_string(analyzer.mg94_log_l))
              << Console::muted("  |  omega = ") << Console::brand(std::to_string(analyzer.mg94_omega)) << "\n\n";

    Panel::print_step(2, 2, settings.phenotype_mode ? "Running BUSTED-PH 4-Model Workflow" : "Running BUSTED Mixture Models",
                      settings.phenotype_mode ? "Unconstrained, Constrained-Test, Constrained-BG & Shared Distributions"
                                              : (settings.full_auto ? "4-Gate Auto-Diagnostics + ECM/SQUAREM" : "ECM + SQUAREM acceleration"));
    if (settings.auto_select_k && !settings.full_auto && !settings.phenotype_mode) {
        std::cout << "  " << Console::muted("↳ Automatic model selection enabled (testing K = 1.." + std::to_string(settings.max_k) + ")...\n");
    }
    BUSTEDResult res = analyzer.run(settings, show_progress, force_progress);

    if (res.auto_diagnostics.enabled) {
        const auto& d = res.auto_diagnostics;
        std::ostringstream g1, g2, g3, g4;
        g1 << (d.activated_error_sink ? "ACTIVATED" : "Skipped")
           << " (max surprisal = " << std::fixed << std::setprecision(1) << d.max_sink_surprisal_bits
           << " bits, clusters = " << d.sink_adjacent_clusters
           << ", ΔlnL = +" << std::setprecision(2) << d.sink_delta_ll << ")";
        g2 << d.selected_multiple_hits
           << " (quarantined 2-hit score = " << std::fixed << std::setprecision(2) << d.double_hit_score
           << ", ΔlnL = +" << d.double_hit_delta_ll << ")";
        g3 << (d.activated_srv ? ("ACTIVATED (M = " + std::to_string(d.selected_syn_rates) + ")") : "Skipped (M = 1)")
           << " (Z_disp = " << std::fixed << std::setprecision(2) << d.srv_overdispersion_z
           << ", ΔlnL = +" << d.srv_delta_ll << ")";
        g4 << d.k_rationale;
        std::cout << "\n";
        Panel::print_card("Full-Auto 4-Gate Model Selection Diagnostics", {
            {"Gate 1: Error Sink (BUSTED-E)", g1.str()},
            {"Gate 2: Multi-Hits (BUSTED-MH)", g2.str()},
            {"Gate 3: Synonymous Rates (SRV)", g3.str()},
            {"Gate 4: Omega Classes (K)", g4.str()}
        });
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_runtime = std::chrono::duration<double>(end_time - start_time).count();

    // Summary table
    Table rate_table;
    rate_table.add_column("Category", Table::Align::Center, 8);
    if (res.has_background_test) {
        rate_table.add_column("Test omega (Unc)", Table::Align::Right, 18);
        rate_table.add_column("Test Weight", Table::Align::Right, 14);
        rate_table.add_column("BG omega (Unc)", Table::Align::Right, 18);
        rate_table.add_column("BG Weight", Table::Align::Right, 14);
        rate_table.add_column("Shared omega", Table::Align::Right, 16);
        rate_table.add_column("Shared Weight", Table::Align::Right, 14);

        size_t num_k = res.unconstrained.test_distribution.omegas.size();
        for (size_t k = 0; k < num_k; ++k) {
            std::ostringstream t_w, t_p, b_w, b_p, s_w, s_p;
            t_w << std::fixed << std::setprecision(4) << res.unconstrained.test_distribution.omegas[k];
            t_p << std::fixed << std::setprecision(4) << res.unconstrained.test_distribution.weights[k];
            b_w << std::fixed << std::setprecision(4) << res.unconstrained.background_distribution.omegas[k];
            b_p << std::fixed << std::setprecision(4) << res.unconstrained.background_distribution.weights[k];
            s_w << std::fixed << std::setprecision(4) << res.shared_distributions.test_distribution.omegas[k];
            s_p << std::fixed << std::setprecision(4) << res.shared_distributions.test_distribution.weights[k];

            std::string row_color = "";
            if (k + 1 == num_k && (res.unconstrained.test_distribution.omegas[k] > 1.0 || res.unconstrained.background_distribution.omegas[k] > 1.0)) {
                row_color = "\033[38;5;48m";
            }
            rate_table.add_row({
                "omega_" + std::to_string(k + 1),
                t_w.str(), t_p.str(), b_w.str(), b_p.str(), s_w.str(), s_p.str()
            }, row_color);
        }
    } else {
        rate_table.add_column("Unconstrained omega", Table::Align::Right, 20);
        rate_table.add_column("Unconstrained Weight", Table::Align::Right, 20);
        rate_table.add_column("Constrained omega", Table::Align::Right, 18);
        rate_table.add_column("Constrained Weight", Table::Align::Right, 18);

        size_t num_k = res.unconstrained.test_distribution.omegas.size();
        for (size_t k = 0; k < num_k; ++k) {
            std::ostringstream u_w_ss, u_p_ss, c_w_ss, c_p_ss;
            u_w_ss << std::fixed << std::setprecision(4) << res.unconstrained.test_distribution.omegas[k];
            u_p_ss << std::fixed << std::setprecision(4) << res.unconstrained.test_distribution.weights[k];
            c_w_ss << std::fixed << std::setprecision(4) << res.constrained.test_distribution.omegas[k];
            c_p_ss << std::fixed << std::setprecision(4) << res.constrained.test_distribution.weights[k];

            std::string row_color = "";
            if (k + 1 == num_k && res.unconstrained.test_distribution.omegas[k] > 1.0) {
                row_color = "\033[38;5;48m"; // Highlight positive selection class in emerald
            }

            rate_table.add_row({
                "omega_" + std::to_string(k + 1),
                u_w_ss.str(),
                u_p_ss.str(),
                c_w_ss.str(),
                c_p_ss.str()
            }, row_color);
        }
    }

    std::cout << "\n" << Console::bold("  Inferred Rate Distributions:") << "\n\n";
    rate_table.print();
    std::cout << "\n";

    if (!res.unconstrained.test_distribution.syn_rates.empty() && res.unconstrained.test_distribution.syn_rates.size() > 1) {
        std::cout << Console::bold("  Synonymous Site-to-Site Rates (SRV):") << "\n";
        for (size_t m = 0; m < res.unconstrained.test_distribution.syn_rates.size(); ++m) {
            std::cout << "    alpha_" << (m + 1) << " = " << std::fixed << std::setprecision(4)
                      << res.unconstrained.test_distribution.syn_rates[m]
                      << " (weight = " << res.unconstrained.test_distribution.syn_weights[m] << ")\n";
        }
        std::cout << "\n";
    }

    if (res.settings.multiple_hits != "None") {
        std::cout << Console::bold("  Multi-hit Substitution Rates:") << "\n";
        std::cout << "    delta (double-hit rate) : " << std::fixed << std::setprecision(4) << res.unconstrained.delta
                  << " (fraction: " << res.unconstrained.frac_delta * 100.0 << "%)\n";
        if (res.settings.multiple_hits == "Double+Triple") {
            std::cout << "    psi (triple-hit rate)   : " << std::fixed << std::setprecision(4) << res.unconstrained.psi
                      << " (fraction: " << res.unconstrained.frac_psi * 100.0 << "%)\n";
        }
        std::cout << "\n";
    }

    if (res.settings.error_sink) {
        std::cout << Console::bold("  Saturation Error Sink (BUSTED-E):") << "\n";
        std::cout << "    Unconstrained sink weight (eps) : " << std::scientific << std::setprecision(4) << res.unconstrained.error_sink_weight
                  << "  |  Constrained sink weight : " << res.constrained.error_sink_weight << "\n";
        std::cout << "    Flagged artifact branch-sites   : " << res.flagged_error_events.size()
                  << " (BF >= " << std::fixed << std::setprecision(1) << res.settings.error_sink_bf_threshold << ")\n";
        for (size_t f = 0; f < std::min<size_t>(5, res.flagged_error_events.size()); ++f) {
            const auto& ev = res.flagged_error_events[f];
            std::cout << "      • Codon " << ev.site << " on branch '" << ev.branch_name
                      << "' (BF_sink = " << std::fixed << std::setprecision(1) << ev.bayes_factor
                      << ", P_post = " << std::setprecision(3) << ev.posterior << ")\n";
        }
        std::cout << "\n";
    }

    std::ostringstream lrt_ss, p_ss, u_ss, c_ss, time_ss;
    lrt_ss << std::fixed << std::setprecision(4) << res.lrt;
    p_ss << std::scientific << std::setprecision(6) << res.p_value;
    u_ss << std::fixed << std::setprecision(2) << res.unconstrained.log_likelihood << " (AICc: " << res.unconstrained.aicc << ", K = " << res.optimal_k << ")";
    c_ss << std::fixed << std::setprecision(2) << res.constrained.log_likelihood << " (AICc: " << res.constrained.aicc << ", omega_" << res.optimal_k << " = 1.0)";
    time_ss << std::fixed << std::setprecision(2) << total_runtime << " s";

    std::string json_desc = (json_format == JSONFormat::Legacy) ? " [Legacy]" : " [Modern v3.0]";
    if (res.has_background_test) {
        std::ostringstream uit_ss, t_test_ss, s_test_ss, b_test_ss;
        uit_ss << "p_UIT = max(p_FG, p_diff) = " << std::scientific << std::setprecision(4) << res.union_intersection_p_value
               << (res.union_intersection_p_value <= res.settings.p_value_threshold ? " [REJECTED H0]" : " [not rejected]");
        t_test_ss << "LRT = " << std::fixed << std::setprecision(3) << res.lrt
                  << ", p_FG = " << std::scientific << std::setprecision(4) << res.p_value;
        s_test_ss << "LRT = " << std::fixed << std::setprecision(3) << res.shared_lrt
                  << " (df=" << res.shared_df << "), p_diff = " << std::scientific << std::setprecision(4) << res.shared_p_value;
        b_test_ss << "ΔAICc = " << std::fixed << std::setprecision(2) << res.background_delta_aicc
                  << " (LRT = " << std::setprecision(3) << res.background_lrt
                  << ", p_BG = " << std::scientific << std::setprecision(4) << res.background_p_value << ") "
                  << (res.background_neutral_supported ? "[BG Neutral Supported]" : "[BG Selection Present]");

        std::vector<std::pair<std::string, std::string>> sum_items = {
            {"1. Union-Intersection (FG & Diff)", uit_ss.str()},
            {"   ├─ Foreground Selection (p_FG)", t_test_ss.str()},
            {"   └─ Regime Difference (p_diff)", s_test_ss.str()},
            {"2. Background Neutrality (AIC-c)", b_test_ss.str()}
        };
        if (res.has_relax_test) {
            std::ostringstream r_ss;
            r_ss << "k = " << std::fixed << std::setprecision(3) << res.relax_k
                 << " (LRT = " << std::setprecision(3) << res.relax_lrt
                 << ", p_RELAX = " << std::scientific << std::setprecision(4) << res.relax_p_value << ") "
                 << ((res.relax_p_value <= res.settings.p_value_threshold)
                     ? ((res.relax_k > 1.0) ? "[Intensification (k > 1)]" : "[Relaxation (k < 1)]")
                     : "[No sig. k != 1 shift]");
            sum_items.push_back({"3. Coupled-k RELAX Test (df=1)", r_ss.str()});
        }
        sum_items.push_back({"Unconstrained Two-Set Fit", u_ss.str()});
        sum_items.push_back({"JSON Output File", output_file + json_desc});
        sum_items.push_back({"Total Execution Time", time_ss.str()});
        Panel::print_summary_card("BUSTED-PH Phenotype Association Summary", sum_items, res.phenotype_verdict, res.phenotype_associated);
    } else {
        std::string conclusion;
        bool is_sig = (res.p_value < 0.05);
        if (is_sig) {
            conclusion = "STATISTICALLY SIGNIFICANT evidence of episodic diversifying selection (LRT = " + lrt_ss.str() + ", p = " + p_ss.str() + " < 0.05)!";
        } else {
            conclusion = "No statistically significant evidence of episodic diversifying selection (p = " + p_ss.str() + " >= 0.05).";
        }

        std::vector<std::pair<std::string, std::string>> sum_items = {
            {"Likelihood Ratio Test (LRT)", lrt_ss.str()},
            {"p-value (mixture distribution)", p_ss.str() + " [threshold: 0.05]"},
            {"Unconstrained Model Fit", u_ss.str()},
            {"Constrained Null Model Fit", c_ss.str()},
            {"Optimal Model Selection", "K = " + std::to_string(res.optimal_k) + " rate classes"}
        };
        if (res.has_relax_test) {
            std::ostringstream r_ss;
            r_ss << "k = " << std::fixed << std::setprecision(3) << res.relax_k
                 << " (LRT = " << std::setprecision(3) << res.relax_lrt
                 << ", p_RELAX = " << std::scientific << std::setprecision(4) << res.relax_p_value << ") "
                 << ((res.relax_p_value <= res.settings.p_value_threshold)
                     ? ((res.relax_k > 1.0) ? "[Intensification (k > 1)]" : "[Relaxation (k < 1)]")
                     : "[No sig. k != 1 shift]");
            sum_items.push_back({"Coupled-k RELAX Test (df=1)", r_ss.str()});
        }
        sum_items.push_back({"JSON Output File", output_file + json_desc});
        sum_items.push_back({"Total Execution Time", time_ss.str()});

        Panel::print_summary_card("BUSTED Episodic Selection Analysis Summary", sum_items, conclusion, is_sig);
    }

    // Build Provenance
    Provenance prov;
    prov.invocation.cli_command = cli_cmd;
    prov.invocation.working_directory = Provenance::get_cwd();
    prov.invocation.arguments["alignment"] = alignment_file;
    if (!tree_file.empty()) prov.invocation.arguments["tree"] = tree_file;
    prov.invocation.arguments["code"] = code_name;
    prov.invocation.arguments["rates"] = std::to_string(settings.num_rate_classes);
    prov.invocation.arguments["srv"] = settings.srv ? "true" : "false";
    prov.invocation.arguments["syn_rates"] = std::to_string(settings.num_syn_rate_classes);
    prov.invocation.arguments["auto_k"] = settings.auto_select_k ? "true" : "false";
    prov.invocation.arguments["multiple_hits"] = settings.multiple_hits;
    prov.invocation.arguments["error_sink"] = settings.error_sink ? "true" : "false";
    prov.invocation.arguments["phenotype_mode"] = settings.phenotype_mode ? "true" : "false";
    if (!settings.comparison_branches.empty()) prov.invocation.arguments["comparison"] = settings.comparison_branches;
    prov.invocation.arguments["refine_branch_lengths"] = settings.refine_branch_lengths ? "true" : "false";
    prov.invocation.arguments["json_format"] = (json_format == JSONFormat::Legacy) ? "legacy" : "modern_v3";

    size_t aln_sz = 0;
    std::string aln_hash = crypto::SHA256::hash_file(alignment_file, &aln_sz);
    prov.inputs["alignment"] = {alignment_file, "FASTA/NEXUS", aln_hash, aln_sz};

    if (!tree_file.empty()) {
        size_t tree_sz = 0;
        std::string tree_hash = crypto::SHA256::hash_file(tree_file, &tree_sz);
        prov.inputs["tree"] = {tree_file, "Newick", tree_hash, tree_sz};
    }

    prov.execution.start_time = start_iso;
    prov.execution.end_time = Provenance::current_iso8601();
    prov.execution.wall_time_seconds = total_runtime;
#ifdef _OPENMP
    prov.execution.cpu_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    prov.execution.cpu_threads = 1;
#endif
    prov.execution.hostname = Provenance::get_hostname();
    prov.execution.os = Provenance::detect_os();
    prov.execution.compiler = Provenance::detect_compiler();

    // Save JSON output
    try {
        analyzer.save_json(output_file, res, json_format, prov);
        std::string fmt_desc = (json_format == JSONFormat::Legacy) ? "Legacy Datamonkey JSON" : "Modern HyPhy v3.0 JSON";
        std::cout << Console::success("✔") << " " << Console::bold("Saved " + fmt_desc + " report to: ")
                  << Console::brand(output_file) << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Warning: Failed to write JSON output: ") << e.what() << "\n";
    }

    return 0;
}

int run_busted_ph(int argc, char* argv[]) {
    std::vector<char*> args;
    std::string prog = (argc > 0 && argv[0]) ? std::string(argv[0]) : "hyphy3 busted-ph";
    if (prog.find("busted-ph") == std::string::npos && prog.find("busted_ph") == std::string::npos) {
        prog += " --phenotype";
    }
    args.push_back(const_cast<char*>(prog.c_str()));
    for (int i = 1; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    return run_busted(static_cast<int>(args.size()), args.data());
}

#ifndef HYPHY3_COMBINED_DRIVER
int main(int argc, char* argv[]) {
    return run_busted(argc, argv);
}
#endif
