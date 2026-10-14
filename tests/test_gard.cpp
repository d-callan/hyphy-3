#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/neighbor_joining.hpp"
#include "hyphy/analyses/gard.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

using namespace hyphy::core;
using namespace hyphy::analyses;

inline std::string resolve_test_path(const std::string& path) {
    if (std::ifstream(path).good()) return path;
    if (std::ifstream("../" + path).good()) return "../" + path;
    if (std::ifstream("../../" + path).good()) return "../../" + path;
    return path;
}

TEST_CASE("GARD: Nucleotide Alignment Loading & Slicing") {
    std::string adh_path = resolve_test_path("tests/data/adh.nex");
    auto aln = Alignment::load_nucleotide(adh_path);

    CHECK(aln.num_taxa == 23);
    CHECK(aln.num_nucleotides == 762);
    CHECK(aln.num_codons == 0);
    CHECK(aln.patterns.size() > 0);

    // Verify slicing
    auto slice1 = aln.slice_nucleotide(0, 460);
    CHECK(slice1.num_taxa == 23);
    CHECK(slice1.num_nucleotides == 460);

    auto slice2 = aln.slice_nucleotide(460, 762);
    CHECK(slice2.num_taxa == 23);
    CHECK(slice2.num_nucleotides == 302);
}

TEST_CASE("GARD: Neighbor-Joining Tree Inference") {
    std::string adh_path = resolve_test_path("tests/data/adh.nex");
    auto aln = Alignment::load_nucleotide(adh_path);

    Matrix D = NeighborJoining::compute_tn93_distances(aln);
    CHECK(D.rows() == 23);
    CHECK(D.cols() == 23);

    // Verify distance matrix symmetry and positive off-diagonals
    for (size_t i = 0; i < 23; ++i) {
        CHECK(D(i, i) == doctest::Approx(0.0));
        for (size_t j = i + 1; j < 23; ++j) {
            CHECK(D(i, j) == doctest::Approx(D(j, i)));
            CHECK(D(i, j) > 0.0);
        }
    }

    Tree tree = NeighborJoining::infer(aln);
    CHECK(tree.num_leaves() == 23);
    CHECK(tree.num_nodes() == 44); // 23 leaves + 21 internal nodes (including trifurcating root)
    CHECK(tree.root_id != INVALID_INDEX);

    // Verify all branch lengths are strictly positive
    for (const auto& node : tree.nodes) {
        if (node.id != tree.root_id) {
            CHECK(node.branch_length > 0.0);
        }
    }
}

TEST_CASE("GARD: Negative Control on HRVI.nex (0 Breakpoints Expected)") {
    std::string hrvi_path = resolve_test_path("tests/data/HRVI.nex");
    auto aln = Alignment::load_nucleotide(hrvi_path);

    CHECK(aln.num_taxa == 12);
    CHECK(aln.num_nucleotides == 428);

    GARDSettings settings;
    settings.input_file = hrvi_path;
    settings.run_mode = "Faster";
    settings.quiet = true;

    GARDAnalyzer analyzer(aln, settings);
    GARDResult result = analyzer.run();

    // Baseline c-AIC should be close to ~1450.97
    CHECK(result.baseline_aicc > 1400.0);
    CHECK(result.baseline_aicc < 1500.0);

    // No breakpoint should improve upon baseline
    CHECK(result.best_model.breakpoints.empty());
    CHECK(result.best_model.aicc == doctest::Approx(result.baseline_aicc));
    CHECK(result.topological_incongruence == false);

    // Check JSON serialization
    auto modern_json = result.to_json();
    CHECK(modern_json["analysis"]["id"] == "gard");
    CHECK(modern_json["dataset"]["taxa_count"] == 12);
    CHECK(modern_json["dataset"]["sites_count"] == 428);
    CHECK(modern_json["model_selection"]["best_model_breakpoints"].empty());

    auto legacy_json = result.to_legacy_json();
    CHECK(legacy_json["breakpointData"]["0"]["bps"][0][0] == "1");
    CHECK(legacy_json["breakpointData"]["0"]["bps"][0][1] == "428");
}

TEST_CASE("GARD: Positive Control on adh.nex (Breakpoint at Site 460)") {
    std::string adh_path = resolve_test_path("tests/data/adh.nex");
    auto aln = Alignment::load_nucleotide(adh_path);

    GARDSettings settings;
    settings.input_file = adh_path;
    settings.run_mode = "Faster";
    settings.max_breakpoints = 2;
    settings.quiet = true;

    GARDAnalyzer analyzer(aln, settings);
    GARDResult result = analyzer.run();

    CHECK(result.num_taxa == 23);
    CHECK(result.num_sites == 762);
    CHECK(result.baseline_aicc > 0.0);

    // Recombination should be detected
    CHECK(!result.best_model.breakpoints.empty());

    // The primary breakpoint on adh is known to be site 460 (0-based 459, or within small neighborhood)
    size_t primary_bp = result.best_model.breakpoints[0];
    CHECK(primary_bp >= 450);
    CHECK(primary_bp <= 470);

    // Significant improvement in c-AIC
    CHECK(result.best_model.aicc < result.baseline_aicc);
    CHECK(result.improvements.size() >= 1);
    CHECK(result.improvements[0].second > 20.0); // Substantial delta-AICc (empirically ~22.92)

    // Topological incongruence should be supported
    CHECK(result.topological_incongruence == true);
    CHECK(result.best_model.aicc < result.single_tree_aicc);

    // Verify partitions
    CHECK(result.best_model.partitions.size() >= 2);
    CHECK(result.best_model.partitions[0].start_site == 1);
    CHECK(result.best_model.partitions[0].end_site == primary_bp + 1);
    CHECK(result.best_model.partitions[1].start_site == primary_bp + 2);

    // Issue #5: Verify legacy JSON has all partitions, trees, and improvements breakpoints
    auto legacy_json = result.to_legacy_json();
    CHECK(legacy_json["breakpointData"].size() == result.best_model.partitions.size());
    CHECK(legacy_json["trees"].size() == result.best_model.partitions.size());
    CHECK(legacy_json["improvements"].size() == result.improvements.size());
    for (auto& [k, imp] : legacy_json["improvements"].items()) {
        CHECK(imp.contains("breakpoints"));
        CHECK(imp["breakpoints"].is_array());
        CHECK(!imp["breakpoints"].empty());
    }
}

TEST_CASE("GARD: Dynamic Breakpoint Chart and Visualizer") {
    // 1. Test single-breakpoint dashboard generation
    GARDVisualizer viz(762, 23, false, true, true);
    viz.chart_width = 65;
    viz.start_single_bp(315, 10378.22);

    viz.record_candidate({459}, 10353.93);
    viz.update_best({459}, 10353.93);
    viz.tick(50);

    auto lines = viz.format_dashboard_lines();
    CHECK(lines.size() == 5);

    // Line 0: Status
    CHECK(lines[0].find("Status:") != std::string::npos);
    CHECK(lines[0].find("10353.93") != std::string::npos);
    CHECK(lines[0].find("460") != std::string::npos);

    // Line 1: Ruler
    CHECK(lines[1].find("Ruler:") != std::string::npos);
    CHECK(lines[1].find("1") != std::string::npos);
    CHECK(lines[1].find("460") != std::string::npos);
    CHECK(lines[1].find("762 bp") != std::string::npos);

    // Line 2: Track
    CHECK(lines[2].find("Track:") != std::string::npos);
    CHECK(lines[2].find("├") != std::string::npos);
    CHECK(lines[2].find("┤") != std::string::npos);
    CHECK(lines[2].find("▲") != std::string::npos);

    // Line 3: Density
    CHECK(lines[3].find("Density:") != std::string::npos);

    // Line 4: Partitions
    CHECK(lines[4].find("Parts:") != std::string::npos);
    CHECK(lines[4].find("[1..460] (460bp)") != std::string::npos);
    CHECK(lines[4].find("[461..762] (302bp)") != std::string::npos);

    // 2. Test multi-breakpoint dashboard generation
    viz.start_ga(2, 50, 10353.93, {459});
    viz.record_candidate({239, 519}, 10340.12);
    viz.update_best({239, 519}, 10340.12);
    viz.tick(100);

    auto ga_lines = viz.format_dashboard_lines();
    CHECK(ga_lines.size() == 5);
    CHECK(ga_lines[1].find("240") != std::string::npos);
    CHECK(ga_lines[1].find("520") != std::string::npos);
    CHECK(ga_lines[4].find("[1..240] (240bp)") != std::string::npos);
    CHECK(ga_lines[4].find("[241..520] (280bp)") != std::string::npos);
    CHECK(ga_lines[4].find("[521..762] (242bp)") != std::string::npos);
}
