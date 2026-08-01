# Object scanner web GUI

This package provides a Flask page for controlling `object_scanner` and viewing
the colored world-frame points committed to its current SQLite recording.

The combined launch starts the configured RealSense camera, object scanner, and
web server:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select object_scanner object_scanner_web \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug
ln -sf build/compile_commands.json compile_commands.json
source install/setup.bash
ros2 launch object_scanner_web object_scanner_web.launch.py
```

The scanner still requires a synchronized camera-to-world
`geometry_msgs/msg/TransformStamped` publisher on
`/object_scanner/camera_to_world`.

Open `http://<robot-ip>:5000` from a browser on the same LAN. The server binds
to all interfaces and has no authentication, so it must be used only on the
trusted isolated network selected for this project. Three.js 0.185.1 and
OrbitControls are loaded from jsDelivr, so the browser needs internet access.

## Controls

- **Start** creates a new `scan_<perf_counter_ns>.sqlite3` session.
- **Pause** commits the session, stops accepting frames, and loads the recorded
  points into Three.js.
- **Resume** appends new frames to the same SQLite file.
- **Stop** commits and closes the session, then displays its final points.

The viewer samples uniformly across the complete recorded sequence and displays
at most 250,000 points. The database retains every filtered point.

## Reference color

The reference color can be changed only while recording is stopped:

- choose a color with the native color picker and click **Apply color**; or
- click **Pick from camera** to capture the next RGB image, move over the
  lossless preview to inspect pixels in the magnifying loupe, click a pixel,
  then use its 5×5 neighborhood average.

The selected center pixel is outlined in white and its 5×5 sampled area in
yellow. Runtime selections last until the scanner restarts. The launch value is
used again on startup and defaults to green (`[0, 255, 0]`).

Launch parameters are forwarded to `object_scanner`:

```bash
ros2 launch object_scanner_web object_scanner_web.launch.py \
  output_directory:=/data/scans \
  target_rgb:="[0, 255, 0]" \
  lab_threshold:=15.0
```

## HTTP API

- `GET /api/status`
- `POST /api/reference-color`
- `GET /api/camera-frame` while stopped
- `POST /api/recording/start`
- `POST /api/recording/pause`
- `POST /api/recording/resume`
- `POST /api/recording/stop`
- `GET /api/points` while paused or stopped

The points endpoint returns a binary `PCD1` payload: a 12-byte header containing
the displayed and total point counts, followed by float32 XYZ values and uint8
RGB values.
