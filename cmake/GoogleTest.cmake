find_package(GTest CONFIG QUIET)

if(NOT TARGET GTest::gtest_main)
    include(FetchContent)

    set(BUILD_GMOCK OFF)
    set(INSTALL_GTEST OFF)
    set(gtest_build_tests OFF)

    message(STATUS "GoogleTest package not found; fetching pinned GoogleTest 1.17.0")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723
    )
    FetchContent_MakeAvailable(googletest)
endif()
