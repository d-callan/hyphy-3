#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "hyphy/core/types.hpp"
#include "hyphy/core/genetic_code.hpp"
#include "hyphy/core/alignment.hpp"
#include "hyphy/core/tree.hpp"
#include "hyphy/analyses/relax.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using namespace hyphy::core;
using namespace hyphy::analyses;

TEST_CASE("RELAX: Wertheim et al. 2015 Fig 4E Benchmark Parity (33 taxa, 286 codons)") {
    std::vector<std::string> candidates = {
        "tests/data/Fig4E.nex",
        "../tests/data/Fig4E.nex",
        "/Users/sergei/Development/hyphy-3/tests/data/Fig4E.nex",
        "/Users/sergei/Development/hyphy/tests/data/Fig4E.nex"
    };
    std::string path;
    for (const auto& c : candidates) {
        std::ifstream f(c);
        if (f.good()) { path = c; break; }
    }
    REQUIRE(!path.empty());
    auto aln = Alignment::load(path);
    REQUIRE(!aln.embedded_tree_newick.empty());
    auto tree = Tree::from_newick(aln.embedded_tree_newick);

    RELAXSettings settings;
    settings.p_value_threshold = 0.05;

    auto relax = RELAXAnalyzer::create(tree, aln, settings);

    // 1. Check branch classification from Newick {T} tags
    CHECK(relax.test_branch_names.size() == 17);
    CHECK(relax.ref_branch_names.size() == 47);

    // 2. Run RELAX analysis
    auto res = relax.run();

    std::cout << "\n=======================================================\n"
              << "   RELAX Fig4E Benchmark Validation                     \n"
              << "=======================================================\n"
              << "  GTR ln L              : " << res.gtr_log_likelihood << "\n"
              << "  MG94 Sep Rates ln L   : " << res.mg94_log_likelihood << "\n"
              << "    omega (Reference)   : " << res.mg94_omega_R << "\n"
              << "    omega (Test)        : " << res.mg94_omega_T << "\n"
              << "  RELAX Alternative ln L: " << res.alternative_fit.log_likelihood << "\n"
              << "  RELAX Null ln L       : " << res.null_fit.log_likelihood << "\n"
              << "  K parameter           : " << res.k << "\n"
              << "  LRT statistic         : " << res.lrt << "\n"
              << "  p-value               : " << res.p_value << "\n"
              << "  Runtime               : " << res.runtime_seconds << "s\n"
              << "=======================================================\n";

    // 3. Mathematical validation against HyPhy 2.5 ground truth
    // Ground truth: GTR = -5317.18
    CHECK(res.gtr_log_likelihood < -5300.0);
    CHECK(res.gtr_log_likelihood > -5330.0);

    // Ground truth: MG94 separate rates: omega_R ~ 0.11, omega_T ~ 0.68
    CHECK(res.mg94_omega_R < 0.25);
    CHECK(res.mg94_omega_T > 0.40);
    CHECK(res.mg94_omega_T > res.mg94_omega_R);

    // Ground truth: RELAX Alternative ln L ~ -4982.69
    CHECK(res.alternative_fit.log_likelihood > -4990.0);

    // Ground truth: RELAX Null ln L ~ -5044.32
    CHECK(res.null_fit.log_likelihood < -5000.0);

    // Selection relaxation: K < 0.25 (HyPhy 2.5 infers K near 0.0)
    CHECK(res.k < 0.25);
    CHECK(res.is_relaxed);
    CHECK(res.is_significant);

    // Test statistic: LRT ~ 123.27, p < 1e-15
    CHECK(res.lrt > 100.0);
    CHECK(res.p_value < 1e-15);

    // 4. Test Legacy JSON export
    auto legacy_out = res.to_legacy_json(tree, aln);
    CHECK(legacy_out.contains("analysis"));
    CHECK(legacy_out.contains("fits"));
    CHECK(legacy_out["fits"].contains("Nucleotide GTR"));
    CHECK(legacy_out["fits"].contains("MG94xREV with separate rates for branch sets"));
    CHECK(legacy_out["fits"].contains("RELAX alternative"));
    CHECK(legacy_out["fits"].contains("RELAX null"));
    CHECK(legacy_out.contains("test results"));
    CHECK(legacy_out["test results"]["LRT"] == res.lrt);
    CHECK(legacy_out["test results"]["p-value"] == res.p_value);
    CHECK(legacy_out["test results"]["relaxation or intensification parameter"] == res.k);
    CHECK(legacy_out.contains("tested"));
    CHECK(legacy_out.contains("branch attributes"));

    // Test Modern JSON export
    auto modern_out = res.to_modern_json(tree, aln);
    CHECK(modern_out.contains("statistical_tests"));
    CHECK(modern_out.contains("model_fits"));
    CHECK(modern_out["model_fits"].contains("alternative"));
    CHECK(modern_out["model_fits"].contains("null"));
    CHECK(modern_out["statistical_tests"]["hypothesis_test"]["statistic_value"] == res.lrt);
    CHECK(modern_out["statistical_tests"]["hypothesis_test"]["p_value"] == res.p_value);
    CHECK(modern_out["statistical_tests"]["hypothesis_test"]["k_parameter"] == res.k);
}

TEST_CASE("RELAX: User-specified test branch regex") {
    std::vector<std::string> candidates = {
        "tests/data/Fig4E.nex",
        "../tests/data/Fig4E.nex",
        "/Users/sergei/Development/hyphy-3/tests/data/Fig4E.nex",
        "/Users/sergei/Development/hyphy/tests/data/Fig4E.nex"
    };
    std::string path;
    for (const auto& c : candidates) {
        std::ifstream f(c);
        if (f.good()) { path = c; break; }
    }
    REQUIRE(!path.empty());
    auto aln = Alignment::load(path);
    auto tree = Tree::from_newick(aln.embedded_tree_newick);

    RELAXSettings settings;
    settings.test_branch_regex = ".*PSEUDOGENE.*";

    auto relax = RELAXAnalyzer::create(tree, aln, settings);

    // All pseudogenes should be marked as test
    CHECK(relax.test_branch_names.size() >= 12);
    for (const auto& name : relax.test_branch_names) {
        CHECK(name.find("PSEUDOGENE") != std::string::npos);
    }
}

TEST_CASE("Issue #4: RELAX case-insensitive {TEST}/{REFERENCE} tags and Unclassified partition") {
    std::string aln_path = HYPHY3_SOURCE_DIR "/benchmarks/data/cd2_reduced.fna";
    // Tag Human and Chimp as {TEST}, Baboon and RhMonkey as {REFERENCE}, leave others untagged (Unclassified)
    std::string nwk = "((((Pig:0.147969,Cow:0.213430):0.085099,Horse:0.165787,Cat:0.264806):0.058611,((RhMonkey{REFERENCE}:0.002015,Baboon{REFERENCE}:0.003108):0.022733,(Human{TEST}:0.004349,Chimp{TEST}:0.000799):0.011873):0.101856):0.340802,Rat:0.050958,Mouse:0.097950);";

    auto aln = Alignment::from_fasta(aln_path);
    auto tree = Tree::from_newick(nwk);

    RELAXSettings settings;
    auto relax = RELAXAnalyzer::create(tree, aln, settings);

    CHECK(relax.test_branch_names.size() == 2);
    CHECK(relax.ref_branch_names.size() == 2);
    CHECK(!relax.unclassified_branch_names.empty());
}

