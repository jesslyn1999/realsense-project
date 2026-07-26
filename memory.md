# RealSense ROS 2 feature reference

`src/realsense-ros` provides the ROS 2 driver, messages, processing filters,
launch files, and examples for Intel RealSense cameras.

## Packages

- `realsense2_camera`: camera driver and processing nodes.
- `realsense2_camera_msgs`: custom messages, services, and actions.

## Core camera

- Intel RealSense D400 and D500 support, including the D435i.
- Color, depth, and infrared image streams.
- Gyroscope and accelerometer streams on cameras with an IMU.
- Configurable resolution, FPS, format, exposure, and gain.
- Device selection by serial number, USB port, or model.
- Automatic reconnect and hardware reset.

## Depth processing

- Depth-to-color alignment.
- Colored `sensor_msgs/msg/PointCloud2` generation.
- Combined RGB-D messages.
- Depth clipping.
- Decimation, spatial, temporal, and hole-filling filters.
- Disparity transform, HDR merge, rotation, and colorizer filters.

## ROS integration

- Image, CameraInfo, metadata, IMU, point-cloud, and TF topics.
- Static camera transforms and optical frames.
- Camera extrinsics topics.
- Configurable topic QoS and `image_transport` support.
- Dynamic camera and filter parameters.
- Device information and hardware-reset services.
- Stream-frequency and temperature diagnostics.

## IMU and synchronization: gyroscope and accelerometer

Both sensors publish `sensor_msgs/msg/Imu`, but they measure different kinds of
motion. Values are expressed relative to the sensor's frame.

- Separate accelerometer and gyro topics
- Combined IMU topic
- Copy or linear-interpolation IMU synchronization
- Color/depth frame synchronization
- Multi-camera launch
- Hardware master/slave camera synchronization

### Gyroscope

- Measures **angular velocity**: how quickly the camera rotates.
- Uses `rad/s` for the X, Y, and Z axes.
- Does not directly measure orientation or angle.
- Integrating angular velocity estimates orientation, but the estimate drifts
  over time.
- Topic: `/realsense/camera0/gyro/sample`.

Example: if the camera rotates counterclockwise around its positive Z axis at
90 degrees per second, the relevant output is approximately:

```yaml
header:
  frame_id: camera0_gyro_optical_frame
angular_velocity:
  x: 0.0
  y: 0.0
  z: 1.57  # rad/s
```

When the camera is not rotating, all three angular-velocity values should be
close to zero, with some sensor noise and bias.

### Accelerometer

- Measures **linear acceleration**, including the effect of gravity.
- Uses `m/s²` for the X, Y, and Z axes.
- Does not directly measure velocity or position.
- Integrating acceleration estimates velocity and position, but small errors
  accumulate quickly.
- Topic: `/realsense/camera0/accel/sample`.

Example: if the camera is stationary with its positive Z axis pointing upward,
the accelerometer should report approximately:

```yaml
header:
  frame_id: camera0_accel_optical_frame
linear_acceleration:
  x: 0.0
  y: 0.0
  z: 9.81  # m/s²
```

In free fall, all three acceleration values approach zero.

### Using both sensors

- The gyroscope responds well to fast rotation but accumulates orientation
  drift.
- The accelerometer provides a gravity reference but is disturbed by vibration
  and translational motion.
- Sensor-fusion software combines both measurements for a more stable
  orientation estimate.
- The driver can publish separate gyro and accelerometer topics or a combined
  `/realsense/camera0/imu` topic.
- `unite_imu_method: 1` copies measurements; `unite_imu_method: 2` uses linear
  interpolation.

Enable the D435i IMU in `camera_d435i.json` with:

```json
"enable_gyro": true,
"enable_accel": true,
"unite_imu_method": 2
```

Inspect the measurements with:

```bash
ros2 topic echo /realsense/camera0/gyro/sample
ros2 topic echo /realsense/camera0/accel/sample
```

## Launch files and examples

- Single camera: `rs_launch.py`.
- Multiple cameras: `rs_multi_camera_launch.py`.
- Synchronized cameras: `rs_multi_camera_launch_sync.py`.
- Point cloud with RViz: `rs_pointcloud_launch.py`.
- D405 and D455 point-cloud examples.
- Depth-alignment and dual-camera RViz examples.
- Rosbag playback and looping.
- YAML parameter loading.
- Intra-process communication demonstration.

## Advanced and optional features

- Composable ROS nodes.
- Optional lifecycle-node support.
- Intra-process communication.
- Optional OpenGL/GLSL GPU acceleration.
- Advanced-mode JSON camera configuration.
- On-device triggered calibration.
- D500 calibration, safety, occupancy, and labeled-point-cloud features.

## Hardware limitations

Feature availability depends on the connected camera. Safety, occupancy, and
labeled-point-cloud features require compatible D500 hardware and are not
available on a D435i.