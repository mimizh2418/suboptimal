cmake_minimum_required(VERSION 3.27)
project(suboptimal)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(BUILD_EXAMPLES "Build examples" ON)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(suboptimal ${SOURCES})

target_include_directories(
    suboptimal
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_options(suboptimal PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)
include(FetchContent)

find_package(Microsoft.GSL CONFIG)
if(NOT Microsoft.GSL_FOUND)
    message(STATUS "Fetching GSL from https://github.com/microsoft/GSL.git")
    fetchcontent_declare(
        GSL
        GIT_REPOSITORY "https://github.com/microsoft/GSL.git"
        GIT_TAG "v4.0.0"
        GIT_SHALLOW ON
    )
    fetchcontent_makeavailable(GSL)
endif()
target_link_libraries(suboptimal PRIVATE Microsoft.GSL::GSL)

find_package(Eigen3 3.4 CONFIG)
if(NOT Eigen3_FOUND)
    message(STATUS "Fetching Eigen3 from https://gitlab.com/libeigen/eigen.git")
    fetchcontent_declare(
        Eigen3
        GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
        GIT_TAG "3.4"
    )
    fetchcontent_makeavailable(Eigen3)
endif()
target_link_libraries(suboptimal PUBLIC Eigen3::Eigen)

if(BUILD_TESTING)
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
    add_executable(tests ${TEST_SOURCES})
    target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    find_package(Catch2 3.7.0 CONFIG)
    if(NOT Catch2_FOUND)
        message(
            STATUS
            "Fetching Catch2 from https://github.com/catchorg/Catch2.git"
        )
        fetchcontent_declare(
            Catch2
            GIT_REPOSITORY "https://github.com/catchorg/Catch2.git"
            GIT_TAG "v3.7.0"
        )
        fetchcontent_makeavailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    endif()
    target_link_libraries(tests PRIVATE suboptimal Catch2::Catch2WithMain)

    include(Catch)
    catch_discover_tests(tests)
endif()

if(BUILD_EXAMPLES)
    file(GLOB example_subdirs "${CMAKE_CURRENT_SOURCE_DIR}/examples/*")
    foreach(example ${example_subdirs})
        message(${example})
        if(IS_DIRECTORY ${example})
            cmake_path(GET example FILENAME example_name)
            file(GLOB_RECURSE example_sources "${example}/*.cpp")
            add_executable(${example_name} ${example_sources})
            target_compile_options(
                ${example_name}
                PRIVATE -Wall -Wextra -Wpedantic -Werror
            )
            target_include_directories(${example_name} PRIVATE "${example}")
            target_link_libraries(${example_name} PRIVATE suboptimal)
        endif()
    endforeach()
endif()
