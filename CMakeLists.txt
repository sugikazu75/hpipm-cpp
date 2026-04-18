cmake_minimum_required(VERSION 3.11)
project(hpipm-cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#############
## Options ##
#############
option(OPTIMIZE_FOR_NATIVE "Enable -march=native" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

#####################
## Build hpipm-cpp ##
#####################
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
# find blasfeo
find_package(blasfeo REQUIRED)
# find hpipm
find_package(hpipm REQUIRED)
# find Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(
  ${PROJECT_NAME} 
  SHARED
  src/detail/d_ocp_qp_dim_wrapper.cpp
  src/detail/d_ocp_qp_wrapper.cpp
  src/detail/d_ocp_qp_sol_wrapper.cpp
  src/detail/d_ocp_qp_ipm_arg_wrapper.cpp
  src/detail/d_ocp_qp_ipm_ws_wrapper.cpp
  src/ocp_qp_dim.cpp
  src/ocp_qp_ipm_solver_settings.cpp
  src/ocp_qp_ipm_solver_statistics.cpp
  src/ocp_qp_ipm_solver.cpp
)

# Export a stable namespaced target for downstream consumers.
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_link_libraries(
  ${PROJECT_NAME} 
  PUBLIC
  Eigen3::Eigen
  hpipm
  blasfeo
)
target_include_directories(
  ${PROJECT_NAME} 
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
if (OPTIMIZE_FOR_NATIVE)
  target_compile_options(
    ${PROJECT_NAME} 
    PUBLIC
    -march=native
  )
endif()

#############
## Testing ##
#############
# Add testing directory
if (BUILD_TESTS)
  enable_testing()
  # coverage options
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  add_library(coverage_config INTERFACE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(coverage_config INTERFACE
      -O0        
      -g         
      --coverage 
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
      target_link_options(coverage_config INTERFACE --coverage)
    else()
      target_link_libraries(coverage_config INTERFACE --coverage)
    endif()
  endif()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif() # end if (BUILD_TESTS)

##############
## Examples ##
##############
if(BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif()

#############
## Install ##
#############
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install lib files
install(
  TARGETS ${PROJECT_NAME}  
  EXPORT ${PROJECT_NAME}-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
)
# Install header files
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
# Install CMake package config
set(CONFIG_PACKAGE_LOCATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake")
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hpipm-cpp-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hpipm-cpp-config.cmake
  INSTALL_DESTINATION ${CONFIG_PACKAGE_LOCATION}
)
install(
  EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CONFIG_PACKAGE_LOCATION}
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hpipm-cpp-config.cmake
  DESTINATION ${CONFIG_PACKAGE_LOCATION}
)
