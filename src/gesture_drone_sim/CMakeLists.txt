cmake_minimum_required(VERSION 3.8)
project(gesture_drone_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_ros2_control REQUIRED)
find_package(gazebo_plugins REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(controller_manager REQUIRED)
find_package(joint_state_broadcaster REQUIRED)
find_package(effort_controllers REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(xacro REQUIRED)

# Install Python modules
install(DIRECTORY
  gesture_drone_sim
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "*.py" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install URDF files
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  gesture_drone_sim/gesture_drone_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Gazebo plugins
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package() 