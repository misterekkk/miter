cmake_minimum_required(VERSION 3.20)
project(miter LANGUAGES CXX)

option(MITER_BUILD_TESTS "Build the GoogleTest suite" OFF)
option(MITER_WARNINGS_AS_ERRORS "Compile miter with strict warnings as errors" OFF)
option(MITER_SANITIZE "Build with sanitizers (tests, not releases)" OFF)
set(MITER_SANITIZERS "address,undefined" CACHE STRING "Sanitizers for MITER_SANITIZE")
option(MITER_NATIVE "Tune for the build machine (-march/-mcpu=native); never enable for distributed wheels" OFF)
set(MITER_LIGNUM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/lignum" CACHE PATH "Path to the local lignum clone")
option(MITER_BUNDLE_OPENMP "macOS: ship libomp in the wheel as a fallback for Homebrew's" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Version lives in python/miter/__init__.py as the single source of truth.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/python/miter/__init__.py" MITER_VERSION_LINE REGEX "^__version__ = ")
if(NOT MITER_VERSION_LINE)
    message(FATAL_ERROR "Could not parse __version__ from python/miter/__init__.py")
endif()
string(REGEX REPLACE "^__version__ = \"([^\"]+)\".*$" "\\1" MITER_VERSION "${MITER_VERSION_LINE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/miter/__init__.py")

# Verify the local lignum working copy exists before configuring.
if(NOT EXISTS "${MITER_LIGNUM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Local lignum clone not found at '${MITER_LIGNUM_DIR}'.\n"
        "Clone it per docs/LIGNUM_INTEGRATION.md:\n"
        "  git clone --branch v0.4.3 https://github.com/misterekkk/lignum extern/lignum\n"
        "  git -C extern/lignum checkout -b miter-integration")
endif()

# Homebrew's libomp is keg-only, so help AppleClang find it before configuring lignum and OpenMP.
if(APPLE AND NOT OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
    foreach(prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS "${prefix}/include/omp.h")
            set(OpenMP_ROOT "${prefix}")
            break()
        endif()
    endforeach()
endif()

# Build lignum as a C++ static library only: hide SKBUILD so lignum does not register a conflicting
# nanobind _core target when scikit-build-core builds miter.
set(_miter_had_skbuild OFF)
if(DEFINED SKBUILD)
    set(_miter_had_skbuild ON)
    set(_miter_saved_skbuild "${SKBUILD}")
endif()
set(SKBUILD OFF)
set(LIGNUM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIGNUM_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(LIGNUM_BUILD_C_LIBRARY OFF CACHE BOOL "" FORCE)
if(MITER_SANITIZE)
    set(LIGNUM_SANITIZE ON CACHE BOOL "" FORCE)
    set(LIGNUM_SANITIZERS "${MITER_SANITIZERS}" CACHE STRING "" FORCE)
else()
    set(LIGNUM_SANITIZE OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory("${MITER_LIGNUM_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/extern/lignum" EXCLUDE_FROM_ALL)
unset(SKBUILD)
if(_miter_had_skbuild)
    set(SKBUILD "${_miter_saved_skbuild}")
endif()

# ---------------------------------------------------------------------------
# Core C++ library
# ---------------------------------------------------------------------------
add_library(miter STATIC
    src/miter.cpp
    src/core/booster.cpp
    src/core/categorical.cpp
    src/core/histogram.cpp
    src/core/objective.cpp
    src/core/oblique.cpp
    src/core/tree_learner.cpp
    src/data/axis_binner.cpp
    src/data/cat_encoder.cpp
    src/data/dataset.cpp
    src/data/quantizer.cpp
    src/model/evaluator.cpp
    src/model/json_io.cpp
    src/model/lignum_export.cpp
    src/model/lignum_bridge.cpp
)

set_target_properties(miter PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_definitions(miter PRIVATE MITER_VERSION="${MITER_VERSION}")

target_include_directories(miter
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(miter PRIVATE lignum)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(miter PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: miter will build single-threaded.")
endif()

# Exact floating-point semantics [FROZEN]: never use -ffast-math, -ffp-contract=fast, or /fp:fast.
if(MSVC)
    target_compile_options(miter PRIVATE /fp:precise /W4)
    if(MITER_WARNINGS_AS_ERRORS)
        target_compile_options(miter PRIVATE /WX)
    endif()
else()
    target_compile_options(miter PRIVATE -ffp-contract=off -fno-math-errno)
    if(MITER_WARNINGS_AS_ERRORS)
        target_compile_options(miter PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    if(MITER_NATIVE)
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            target_compile_options(miter PRIVATE -mcpu=native)
        else()
            target_compile_options(miter PRIVATE -march=native)
        endif()
    endif()
endif()

function(miter_apply_sanitizers target_name)
    if(MITER_SANITIZE AND NOT MSVC)
        target_compile_options(${target_name} PRIVATE
            -fsanitize=${MITER_SANITIZERS}
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all)
        target_link_options(${target_name} PRIVATE
            -fsanitize=${MITER_SANITIZERS})
        if(SKBUILD AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT APPLE AND MITER_SANITIZERS MATCHES "address")
            target_compile_options(${target_name} PRIVATE -shared-libasan)
            target_link_options(${target_name} PRIVATE -shared-libasan)
        endif()
    endif()
endfunction()

miter_apply_sanitizers(miter)

# ---------------------------------------------------------------------------
# Python extension (built by scikit-build-core)
# ---------------------------------------------------------------------------
if(SKBUILD)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core STABLE_ABI python/bindings.cpp)
    target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(_core PRIVATE miter lignum)
    if(MSVC)
        target_compile_options(_core PRIVATE /fp:precise /W4)
    else()
        target_compile_options(_core PRIVATE -ffp-contract=off -fno-math-errno)
        if(MITER_WARNINGS_AS_ERRORS)
            target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()
    endif()
    miter_apply_sanitizers(_core)

    install(TARGETS _core DESTINATION miter)

    if(APPLE AND MITER_BUNDLE_OPENMP AND OpenMP_CXX_FOUND)
        set_target_properties(_core PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            INSTALL_RPATH "/opt/homebrew/opt/libomp/lib;/usr/local/opt/libomp/lib;@loader_path/.dylibs")
        find_library(MITER_LIBOMP NAMES omp PATHS "${OpenMP_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
        get_filename_component(MITER_LIBOMP_REAL "${MITER_LIBOMP}" REALPATH)
        install(FILES "${MITER_LIBOMP_REAL}" DESTINATION miter/.dylibs RENAME libomp.dylib)
        add_custom_command(TARGET _core POST_BUILD
            COMMAND install_name_tool -change "${MITER_LIBOMP_REAL}" @rpath/libomp.dylib $<TARGET_FILE:_core>
            COMMAND install_name_tool -change "${MITER_LIBOMP}" @rpath/libomp.dylib $<TARGET_FILE:_core>
            COMMAND install_name_tool -change /opt/homebrew/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:_core>
            COMMAND install_name_tool -change /usr/local/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:_core>
            VERBATIM)
    endif()
endif()

# ---------------------------------------------------------------------------
# C++ tests
# ---------------------------------------------------------------------------
if(MITER_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests/cpp)
endif()
