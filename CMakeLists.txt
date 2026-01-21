cmake_minimum_required(VERSION 3.25)
project(rd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(NOTCURSES REQUIRED notcurses)

add_executable(rd src/main.cu)

target_include_directories(rd PRIVATE
  ${NOTCURSES_INCLUDE_DIRS}
)

target_link_libraries(rd PRIVATE
  ${NOTCURSES_LIBRARIES}
  cuda
)

target_compile_options(rd PRIVATE
  ${NOTCURSES_CFLAGS_OTHER}
)

