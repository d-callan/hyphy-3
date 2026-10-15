#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "hyphy/core/types.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/core/rate_matrix.hpp"
#include "hyphy/analyses/contrast_fel.hpp"

#include <iostream>
#include <fstream>

using namespace hyphy::core;
using namespace hyphy::analyses;

inline std::string resolve_test_path(const std::string& path) {
    if (std::ifstream(path).good()) return path;
    if (std::ifstream("../" + path).good()) return "../" + path;
    if (std::ifstream("../../" + path).good()) return "../../" + path;
    return path;
}

TEST_CASE("Contrast-FEL: Multi-group differential selection on cd2.fna with tree tags") {
    std::string aln_path = resolve_test_path("benchmarks/data/cd2.fna");
    std::string tree_path = resolve_test_path("benchmarks/data/cd2.nwk");

    auto aln = Alignment::from_fasta(aln_path);
    auto tree = Tree::from_newick_file(tree_path);

    ContrastFELSettings settings;
    settings.p_value_threshold = 0.10;
    settings.q_value_threshold = 0.20;
    settings.shrinkage_lambda = 1.0;
    settings.full_model = false; // Fast fitting for unit test

    auto analyzer = ContrastFELAnalyzer::create_and_fit(tree, aln, settings);

    // Verify branch partitioning
    CHECK(analyzer.testable_groups.size() >= 2);
    std::cout << "Detected " << analyzer.testable_groups.size() << " testable branch sets:\n";
    for (const auto& grp : analyzer.testable_groups) {
        std::cout << "  - " << grp.name << ": " << grp.branch_count << " branches\n";
    }

    auto results = analyzer.run(false, false);
    CHECK(results.size() == aln.num_codons);

    // Check invariants
    size_t invariant_count = 0;
    for (const auto& r : results) {
        if (r.contrast_tag == "Invariant") {
            CHECK(r.alpha == 0.0);
            CHECK(r.p_value == 1.0);
            invariant_count++;
        }
    }
    CHECK(invariant_count > 0);
    std::cout << "Invariant sites correctly identified: " << invariant_count << "\n";

    // Verify Ridge Shrinkage effect
    size_t penalized_sites = 0;
    for (const auto& r : results) {
        if (r.shrinkage_penalty > 0.0) {
            penalized_sites++;
            // Penalized LRT must be <= raw LRT
            CHECK(r.penalized_lrt <= r.raw_lrt + 1e-4);
        }
    }
    std::cout << "Sites regularized by Ridge Shrinkage: " << penalized_sites << "\n";
    CHECK(penalized_sites > 0);

    // Verify summary and JSON serialization
    auto summary = analyzer.get_summary();
    CHECK(summary.total_sites == aln.num_codons);

    auto json = analyzer.to_modern_json(aln_path, tree_path);
    CHECK(json["$schema"] == "https://raw.githubusercontent.com/veg/hyphy-3/main/schemas/v3/contrast_fel.v3.schema.json");
    CHECK(json["schema_version"] == "3.0.0");
    CHECK(json["analysis"]["id"] == "contrast-fel");
    CHECK(json["analysis"]["shrinkage_lambda"] == 1.0);
    CHECK(json["software"]["name"] == "hyphy");
    CHECK(json.contains("provenance"));
    CHECK(json["site_results"].contains("columns"));
    CHECK(json["site_results"]["row_count"] == aln.num_codons);
    CHECK(json["site_results"]["data"]["site"].size() == aln.num_codons);
    // Dynamic per-group beta columns must exist for every testable group
    for (const auto& grp : analyzer.testable_groups) {
        CHECK(json["site_results"]["data"].contains("beta_" + grp.name));
    }
}

TEST_CASE("Contrast-FEL: Internal vs Leaves 2-group contrast") {
    std::string aln_path = resolve_test_path("benchmarks/data/cd2.fna");
    std::string tree_path = resolve_test_path("benchmarks/data/cd2.nwk");

    auto aln = Alignment::from_fasta(aln_path);
    auto tree = Tree::from_newick_file(tree_path);

    ContrastFELSettings settings;
    settings.p_value_threshold = 0.05;
    settings.q_value_threshold = 0.20;
    settings.shrinkage_lambda = 0.5;
    settings.branch_specs = "Internal,Leaves";
    settings.full_model = false;

    auto analyzer = ContrastFELAnalyzer::create_and_fit(tree, aln, settings);

    CHECK(analyzer.testable_groups.size() == 2);
    CHECK(analyzer.testable_groups[0].name == "Internal");
    CHECK(analyzer.testable_groups[1].name == "Leaves");

    auto results = analyzer.run(false, false);
    CHECK(results.size() == aln.num_codons);

    // Verify that every site has exactly 2 group betas
    for (const auto& r : results) {
        CHECK(r.beta.size() == 2);
        CHECK(r.q_value >= 0.0);
        CHECK(r.q_value <= 1.0);
    }
}
