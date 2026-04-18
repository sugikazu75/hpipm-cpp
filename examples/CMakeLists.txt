find_package(Eigen3 REQUIRED NO_MODULE)

macro(add_example EXAMPLE)
  add_executable(
    ${EXAMPLE} 
    ${EXAMPLE}.cpp
  )
  target_include_directories(
    ${EXAMPLE} 
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
  )
  target_link_libraries(
    ${EXAMPLE} 
    PRIVATE
    Eigen3::Eigen
    ${PROJECT_NAME}
  )
endmacro()

add_example(example_ocp_qp)
add_example(example_mpc)