add_executable(miter_tests
    test_smoke.cpp
    test_quantizer.cpp
    test_axis_binner.cpp
    test_cat_encoder.cpp
    test_dataset.cpp
    test_histogram.cpp
    test_booster.cpp
    test_oblique.cpp
    test_categorical.cpp
    test_determinism.cpp
    test_options.cpp
    test_symmetric_tree.cpp
    test_multiclass.cpp
    test_ctr.cpp
    test_lignum_export.cpp
)

target_include_directories(miter_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(miter_tests PRIVATE
    miter
    lignum
    GTest::gtest_main
)

if(MSVC)
    target_compile_options(miter_tests PRIVATE /fp:precise /W4)
    if(MITER_WARNINGS_AS_ERRORS)
        target_compile_options(miter_tests PRIVATE /WX)
    endif()
else()
    target_compile_options(miter_tests PRIVATE -ffp-contract=off -fno-math-errno)
    if(MITER_WARNINGS_AS_ERRORS)
        target_compile_options(miter_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

miter_apply_sanitizers(miter_tests)

include(GoogleTest)
gtest_discover_tests(miter_tests)
