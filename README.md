# RealSense ROS 2 workspace

This workspace launches one RealSense D435i as an RGB-D camera on ROS 2 Jazzy.
It includes the required RealSense ROS 4.58.3 source packages so it can build
without the sibling `realsense-ros` repository.

## Camera configuration

The default config is
`src/realsense_camera_ros/config/camera_d435i.json`. It selects the first
connected device whose reported name contains `D435I` and enables synchronized
1280x720 at 30 FPS color and depth streams.

If multiple D435i cameras are connected, device ordering is not guaranteed.
Add a `serial_no` parameter to the JSON when deterministic selection is needed.

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select realsense2_camera_msgs realsense2_camera realsense_camera_ros \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug
source install/setup.bash
```

The machine must provide librealsense 2.58 or newer and the ROS dependencies
declared by the three packages.

## Launch

Use the installed default config:

```bash
ros2 launch realsense_camera_ros realsense_camera.launch.py
```

Or provide another JSON config. `config` is the launch file's only argument:

```bash
ros2 launch realsense_camera_ros realsense_camera.launch.py \
  config:=src/realsense_camera_ros/config/camera_d435i.json
```

Primary topics:

- `/realsense/camera0/color/image_raw`
- `/realsense/camera0/color/camera_info`
- `/realsense/camera0/depth/image_rect_raw`
- `/realsense/camera0/depth/camera_info`

The upstream source provenance and license notices are under
`src/realsense-ros/`.
