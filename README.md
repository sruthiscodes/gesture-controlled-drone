
---

## 🚀 Getting Started

### Prerequisites

- Ubuntu 20.04
- ROS 2 Foxy
- Gazebo (typically installed with ROS)
- `colcon` build tool
- Python 3 (with `mediapipe`, `numpy`, etc.)

---

### Installation

```bash
# Clone the repo
git clone https://github.com/sruthiscodes/gesture-controlled-drone.git
cd gesture-controlled-drone

# Source ROS
source /opt/ros/foxy/setup.bash

# Build the workspace
colcon build
source install/setup.bash
