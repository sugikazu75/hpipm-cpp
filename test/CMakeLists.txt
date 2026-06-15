# enable google test
include(gtest.cmake)

# macto for testing
macro(add_hpipm_test TESTNAME)
  add_executable(
    ${TESTNAME} 
    ${TESTNAME}.cpp
  )
  target_include_directories(
    ${TESTNAME} 
    PRIVATE
    ${GTEST_INCLUDE_PATH}
    ${GMOCK_INCLUDE_PATH}
    ${GTEST_LIBRARY_PATH}
    ${PROJECT_SOURCE_DIR}/include
    ${PINOCCHIO_INCLUDE_DIRS}
  )
  target_compile_definitions(
    ${TESTNAME}
    PRIVATE
    HPIPM_CPP_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test"
  )
  target_link_libraries(
    ${TESTNAME} 
    PRIVATE
    GTest::GTest
    GTest::GMock
    ${PROJECT_NAME}
  )
  if(TARGET googletest)
    add_dependencies(
      ${TESTNAME}
      googletest
    )
  endif()
  add_test(
    NAME ${TESTNAME} 
    COMMAND $<TARGET_FILE:${TESTNAME}>
  )
endmacro()

add_hpipm_test(d_ocp_qp_dim_wrapper)
add_hpipm_test(s_ocp_qp_dim_wrapper)
add_hpipm_test(d_ocp_qp_sol_wrapper)
add_hpipm_test(d_ocp_qp_wrapper)
add_hpipm_test(d_ocp_qp_ipm_arg_wrapper)
add_hpipm_test(d_ocp_qp_ipm_ws_wrapper)
add_hpipm_test(ocp_qp_ipm_solver)
add_hpipm_test(s_ocp_qp_ipm_solver)