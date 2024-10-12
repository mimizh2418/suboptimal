cmake_minimum_required(VERSION 3.29)
project(suboptimal)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
fetchcontent_declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
)
fetchcontent_makeavailable(Eigen3)

file(GLOB_RECURSE SOURCES ${CMAKE_HOME_DIRECTORY}/src/*.cpp)

add_executable(suboptimal ${SOURCES})
target_include_directories(suboptimal PUBLIC ${CMAKE_HOME_DIRECTORY}/include)
target_link_libraries(suboptimal PUBLIC Eigen3::Eigen)
