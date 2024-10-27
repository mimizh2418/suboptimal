cmake_minimum_required(VERSION 3.28)
project(suboptimal)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SOURCES "${CMAKE_HOME_DIRECTORY}/src/*.cpp")

add_executable(suboptimal ${SOURCES})

target_include_directories(suboptimal PUBLIC "${CMAKE_HOME_DIRECTORY}/include")
target_include_directories(suboptimal PRIVATE "${CMAKE_HOME_DIRECTORY}/src")

target_compile_options(suboptimal PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(FetchContent)

find_package(Microsoft.GSL CONFIG)
if(NOT Microsoft.GSL_FOUND)
    message(STATUS "GSL not found, fetching from GitHub")
    fetchcontent_declare(
        GSL
        GIT_REPOSITORY "https://github.com/microsoft/GSL"
        GIT_TAG "v4.0.0"
        GIT_SHALLOW ON
    )
    fetchcontent_makeavailable(GSL)
endif()
target_link_libraries(suboptimal PRIVATE Microsoft.GSL::GSL)

find_package(Eigen3 3.4 CONFIG)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from GitLab")
    fetchcontent_declare(
        Eigen3
        GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
        GIT_TAG "3.4"
    )
    fetchcontent_makeavailable(Eigen3)
endif()
target_link_libraries(suboptimal PUBLIC Eigen3::Eigen)
