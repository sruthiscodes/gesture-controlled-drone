# Gesture Controlled Drone Simulation

This project simulates a real-time, gesture-controlled drone in Gazebo Fortress on ROS 2 Humble, deployed using Python. Hand gestures are captured via the webcam, processed with MediaPipe, and translated into velocity commands that drive a quadrotor model in Gazebo.

## Prerequisites
Before you begin, make sure you have the following installed and working on Ubuntu 22.04:

- **ROS 2 Humble Hawksbill**  
- **Gazebo Fortress**  
- **Colcon build tool**  
- **Python 3.10+**  
- **Python libraries**:  
  - `opencv-python` / `python3-opencv`  
  - `numpy` / `python3-numpy`  
  - `mediapipe` / `python3-mediapipe`  
  - `cv_bridge`  
- **ROS 2 packages**:  
  `rclpy`, `geometry_msgs`, `sensor_msgs`, `gazebo_ros`, `gazebo_ros2_control`,  
  `gazebo_plugins`, `gazebo_msgs`, `controller_manager`,  
  `joint_state_broadcaster`, `effort_controllers`,  
  `robot_state_publisher`, `xacro`

### Install ROS 2 and Gazebo Dependencies
```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-gazebo-plugins \
  python3-venv python3-pip python3-opencv python3-numpy
```

## Recommended: Virtual Environment Setup
To isolate Python dependencies, use a virtual environment:

1. **Create**  
   ```bash
   python3 -m venv venv
   ```
2. **Activate**  
   ```bash
   source venv/bin/activate
   ```
3. **Install**  
   ```bash
   pip install -r requirements.txt
   ```

## Building the Package
1. **Source ROS 2**  
   ```bash
   source /opt/ros/humble/setup.bash
   ```
2. **Build**  
   ```bash
   colcon build --packages-select gesture_drone_sim
   ```
3. **Source the install space**  
   ```bash
   source install/setup.bash
   ```

## Running the Simulation
Launch Gazebo and the gesture controller node in one command:
```bash
ros2 launch gesture_drone_sim gesture_drone_sim.launch.py
```
This will:
1. Kill any existing Gazebo processes  
2. Start Gazebo with ROS 2 plugins  
3. Run `robot_state_publisher`  
4. Spawn the drone URDF/XACRO in Gazebo  
5. Launch the `gesture_drone_controller` node

## Usage
- Present hand gestures (Open Palm, Closed Fist, Thumbs Up/Down, OK, Point Left/Right) in front of your webcam.  
- The controller node processes landmarks with MediaPipe and publishes `geometry_msgs/Twist` to `/ImprovedDrone/cmd_vel`.  
- Observe the drone reacting in the Gazebo GUI and terminal logs.

