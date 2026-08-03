# RealSense ROS 2 workspace

This workspace launches one RealSense D435i as an RGB-D camera on ROS 2 Jazzy.
It includes the required RealSense ROS 4.58.3 source packages so it can build
without the sibling `realsense-ros` repository.

## Camera configuration

The default config is
`src/realsense_camera_ros/config/camera_d435i.json`. It selects the first
connected device whose reported name contains `D435I` and enables synchronized
1280x720 color plus 848x480 depth and infrared streams at 30 FPS.

If multiple D435i cameras are connected, device ordering is not guaranteed.
Add a `serial_no` parameter to the JSON when deterministic selection is needed.

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select realsense2_camera_msgs realsense2_camera realsense_camera_ros \
  object_scanner_interfaces object_scanner object_scanner_web \
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

Launch the camera and the point-cloud RViz layout with color, depth, and both
infrared streams:

```bash
ros2 launch realsense_camera_ros realsense_camera_rviz.launch.py
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
- `/realsense/camera0/depth/color/points`
- `/realsense/camera0/infra1/image_rect_raw`
- `/realsense/camera0/infra2/image_rect_raw`

## Object scanning

`object_scanner` filters colored points, applies a named camera-to-world
transformation, and stores each session under `scans/<session_name>/` as
`recording.sqlite3` plus per-frame transformation metadata.

```bash
ros2 launch object_scanner_web object_scanner_web.launch.py
```

See `src/object_scanner/README.md` for output files and parameters.

## Orbbec DaBai DC1 scanning

The isolated Orbbec scanner uses the first detected DaBai DC1 and does not
modify or launch the RealSense scanner packages.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select orbbec_camera_msgs orbbec_camera \
  object_scanner_interfaces object_scanner_web_orbbec \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug
source install/setup.bash
ros2 launch object_scanner_web_orbbec object_scanner_web.launch.py
```

The web interface is available on port 5000. The scanner consumes
`/camera/color/image_raw` and `/camera/depth_registered/points`.

Upstream source provenance and license notices are under `src/realsense-ros/`
and `src/orbbec-ros/`.
