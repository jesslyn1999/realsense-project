# Object scanner

The object scanner records only points whose embedded RealSense RGB color is
close to a configured color. Each retained point is transformed from the camera
frame into the world frame before it is written to SQLite.

The package launch file starts:

- the configured D435i driver from `realsense_camera_ros`, with color, depth,
  synchronization, and colored point-cloud output enabled;
- the `object_scanner` node.

## Data flow

The scanner approximately synchronizes these inputs within 50 ms:

- `/realsense/camera0/depth/color/points`
  (`sensor_msgs/msg/PointCloud2`)
- `/realsense/camera0/color/image_raw` (`sensor_msgs/msg/Image`)
- `/object_scanner/camera_to_world`
  (`object_scanner_interfaces/msg/NamedTransform`)

Filtering uses the `rgb` field embedded in each point. The separate image is
used only to require a synchronized RealSense RGB frame and is not recorded.

The transform producer must publish one transform for each camera frame:

- `header.frame_id`: world-frame name written to the output cloud;
- `child_frame_id`: must equal the input point cloud's `header.frame_id`;
- `transformation_name`: name written to the frame metadata;
- `matrix`: exact row-major camera-to-world 4×4 matrix;
- `header.stamp`: within 50 ms of the corresponding cloud and RGB image.

The installed `resource/transformation_matrices.json` file contains the named
4×4 matrices available to the web GUI. Its default `identity` entry leaves XYZ
unchanged. Replace it or add entries using calibrated rigid camera-to-world
matrices for the capture locations used by the scanner.

One GUI click publishes the displayed matrix for exactly the next point cloud.
The message copies the corresponding cloud timestamp and child frame.
When running this package without the web GUI, another node must publish the
required transforms.

For each retained camera-frame point `p`, the scanner computes:

```text
p_world = R_camera_to_world * p + t_camera_to_world
```

Each synchronized frame is one row in the recording database. The row contains
the source timestamp, world frame, transformation name and matrix, point count,
contiguous float32 XYZ values, and uint8 RGB values. No original cloud or full
RGB image is stored.

## Parameters

- `output_directory`: SQLite recording output root, default `scans`
- `session_name`: required recording folder name
- `target_rgb`: target `[red, green, blue]`, default `[0, 255, 0]`
- `lab_threshold`: maximum CIELAB distance from the target, default `15.0`

`target_rgb` can be updated through ROS parameters only while recording is
stopped. Runtime changes last until the scanner restarts; the next startup uses
the launch value, which defaults to green.

## Build and launch

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select object_scanner_interfaces object_scanner \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug
ln -sf build/compile_commands.json compile_commands.json
source install/setup.bash
ros2 launch object_scanner object_scanner.launch.py
```

Parameters can be overridden for the scanner node:

```bash
ros2 launch object_scanner object_scanner.launch.py \
  output_directory:=/data/scans \
  target_rgb:="[0, 255, 0]" \
  lab_threshold:=15.0
```

## Recording services

The four services use `std_srvs/srv/Trigger`:

```bash
ros2 param set /object_scanner session_name green_cup_01
ros2 service call /object_scanner/start_recording std_srvs/srv/Trigger
ros2 service call /object_scanner/pause_recording std_srvs/srv/Trigger
ros2 service call /object_scanner/resume_recording std_srvs/srv/Trigger
ros2 service call /object_scanner/stop_recording std_srvs/srv/Trigger
```

`start_recording` creates
`<output_directory>/<session_name>/recording.sqlite3`. Session names must
contain only letters, numbers, `_`, or `-`, and can change only while stopped.
An existing session directory is never overwritten: Start returns
`success=false` instead. Each frame is committed immediately. Pausing leaves
the database open and drops incoming frames, so a reader can inspect everything
recorded so far. Resuming appends to the same database.

Stopping atomically adds `metadata.json` with one entry per recorded SQLite
frame: its database ID, source timestamp, transformation name, parent frame,
and exact 4×4 matrix. It then checkpoints the WAL, switches the database to
`DELETE` journal mode, and closes it so the session contains one standalone
SQLite file plus its metadata. Stop returns `success=false` if another SQLite
reader is holding the database open. Calls that do not match the current state
return `success=false`.
