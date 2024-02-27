cmake_minimum_required(VERSION 3.20)
project(FastChess)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/catch2
  GIT_TAG v3.4.0
  )
FetchContent_MakeAvailable(Catch2)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/lib)