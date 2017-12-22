# SET(TEST_TARGET test_spo test_Single_pressure_only.cc)
# add_executable(test_spo test_Single_pressure_only.cc)
add_executable(driver test_Single_pressure_only.cc)
# DEAL_II_SETUP_TARGET(test_spo)
# SET(TEST_TARGET test_spo)
# TARGET_LINK_LIBRARIES(test_spo wings ${Boost_LIBRARIES})
add_test(NAME test_spo
  COMMAND driver --config DEBUG
  )

DEAL_II_PICKUP_TESTS()
