cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Check for CUDA and conditionally enable the language
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(USE_CUDA ON)
  message(STATUS "CUDA enabled")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
else()
  message(STATUS "Building without CUDA")
endif()

# Find Python, pybind11, and TBB
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
# find_package(TBB REQUIRED)

# message(STATUS "TBB include directories: ${TBB_INCLUDE_DIRS}")
# message(STATUS "TBB library directories: ${TBB_LIBRARY_DIRS}")
# message(STATUS "TBB libraries: ${TBB_LIBRARIES}")
# include_directories(${TBB_INCLUDE_DIRS})
# link_directories(${TBB_LIBRARY_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

# Determine the XLA include directory via Python
set(Python_EXECUTABLE /opt/anaconda3/envs/perm/bin/python3.12)
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

#Create the Python module using the collected sources
python_add_library(_core MODULE src/main.cpp WITH_SOABI)
# target_link_libraries(_core PUBLIC TBB::tbb)
target_link_libraries(_core PRIVATE pybind11::headers)

# Include XLA headers and pass in the version info
target_include_directories(_core PUBLIC ${XLA_DIR})
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

if(APPLE)
  # Check if Homebrew's libomp is installed
  if(EXISTS "/opt/homebrew/opt/libomp")
    message(STATUS "Found Homebrew's libomp installation")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
  elseif(EXISTS "/usr/local/opt/libomp")
    message(STATUS "Found libomp installation in /usr/local")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
  endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    target_link_libraries(_core PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found. Code will run in serial mode.")
endif()

set(CMAKE_CXX_FLAGS "-O2")
target_compile_options(_core PRIVATE -O2)

# #Install the module to the desired output directory
install(TARGETS _core DESTINATION sooki)

# Set up the list of sources: always include the CPU source, then add CUDA sources if available

if(USE_CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  pybind11_add_module(
    gpu_ops
    ${CMAKE_CURRENT_LIST_DIR}/src/kernels.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/gpu_ops.cc)
  target_include_directories(gpu_ops PRIVATE ${XLA_DIR})
  set_target_properties(gpu_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 
  set_target_properties(gpu_ops PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
  target_link_libraries(gpu_ops PRIVATE ${CUDA_DEVRT_LIBRARY})  # Add optimization flags to GPU module as well
  target_compile_options(gpu_ops PRIVATE -O2)

  install(TARGETS gpu_ops DESTINATION sooki)
endif()