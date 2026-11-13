// R4: merged doctest binary for core/non-analysis tests.
// Each test .cpp is included verbatim; the first one's
// DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN supplies main(), later identical
// macro definitions are benign and doctest.h include guards dedupe.
// ctest selects per-file cases with: -sf=*test_<name>.cpp

#include "test_core.cpp"
#include "test_cpp_autograd.cpp"
#include "test_tree_engine.cpp"
#include "test_report.cpp"
