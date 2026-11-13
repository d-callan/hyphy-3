// R4: merged doctest binary for optimizer/phase4 tests.
// The target also compiles src/analyses/expcm_optimizer.cpp and
// persite_optimizer.cpp once, shared by all cases below.
// ctest selects per-file cases with: -sf=*test_<name>.cpp

#include "test_expcm.cpp"
#include "test_expcm_optimizer.cpp"
#include "test_gradient_validation.cpp"
#include "test_optimizer_quick.cpp"
#include "test_phase4_persite.cpp"
#include "test_phase4_optimizer.cpp"
#include "test_phase4_validation.cpp"
