cmake_minimum_required(VERSION 3.8)
project(realsense_camera_ros)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_camera_config test/test_camera_config.py)
endif()

ament_package()
