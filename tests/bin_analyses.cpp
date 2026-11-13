// R4: merged doctest binary for per-analysis tests.
// ctest selects per-file cases with: -sf=*test_<name>.cpp
//
// test_mss.cpp must come FIRST: it does '#undef NDEBUG' before <cassert>,
// which only takes effect if no earlier include pulled in <cassert> under
// NDEBUG. Its file-local resolve_path helper has no cross-file clashes.

#include "test_mss.cpp"
#include "test_fel.cpp"
#include "test_fel_cd2_parity.cpp"
#include "test_fel_multi_dataset.cpp"
#include "test_meme.cpp"
#include "test_busted.cpp"
#include "test_busted_mh.cpp"
#include "test_absrel.cpp"
#include "test_relax.cpp"
#include "test_contrast_fel.cpp"
#include "test_fubar.cpp"
#include "test_prime.cpp"
#include "test_gard.cpp"
#include "test_slac.cpp"
#include "test_dfel.cpp"
#include "test_ohmic.cpp"
#include "test_joint_align.cpp"
#include "test_characterize.cpp"
