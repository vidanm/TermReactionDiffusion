# cmake_minimum_required(VERSION 3.25)
# project(rd LANGUAGES CXX CUDA)
#
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_STANDARD 20)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CUDA_ARCHITECTURES 86)
#
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(NOTCURSES REQUIRED notcurses)
#
# add_library(rd_cuda STATIC src/rd_cuda.cu)
# set_target_properties(rd_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# add_executable(rd src/main.cpp)
#
# target_include_directories(rd PUBLIC src)
# target_include_directories(rd PRIVATE ${NOTCURSES_INCLUDE_DIRS})
#
# target_link_libraries(rd_cuda PRIVATE cuda)
# target_link_libraries(rd PRIVATE ${NOTCURSES_LIBRARIES})
#
# target_compile_options(rd PRIVATE ${NOTCURSES_CFLAGS_OTHER})

cmake_minimum_required(VERSION 3.25)
project(rd LANGUAGES CXX CUDA)

# --- Standards ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GPU cible (Ampere dans ton cas)
set(CMAKE_CUDA_ARCHITECTURES 86)

# --- Dépendances ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(NOTCURSES REQUIRED notcurses)

# --- Librairie CUDA ---
add_library(rd_cuda STATIC
    src/rd_cuda.cu
)

set_target_properties(rd_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Headers exposés par la lib CUDA
target_include_directories(rd_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link CUDA (driver/runtime via nvcc)
target_link_libraries(rd_cuda PRIVATE
    cuda
)

# --- Exécutable ---
add_executable(rd
    src/main.cpp
)

target_link_libraries(rd PRIVATE
    rd_cuda
    ${NOTCURSES_LIBRARIES}
)

target_include_directories(rd PRIVATE
    ${NOTCURSES_INCLUDE_DIRS}
)

target_compile_options(rd PRIVATE
    ${NOTCURSES_CFLAGS_OTHER}
)
