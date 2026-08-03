# Orbbec object scanner web GUI

This package provides an isolated DaBai DC1 scanner, Flask controls, and a
Three.js viewer for colored world-frame points stored in SQLite.

The combined launch starts the first detected DaBai DC1, the Orbbec scanner,
and the web server:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select orbbec_camera_msgs orbbec_camera \
  object_scanner_interfaces object_scanner_web_orbbec \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug
ln -sf build/compile_commands.json compile_commands.json
source install/setup.bash
ros2 launch object_scanner_web_orbbec object_scanner_web.launch.py
```

The scanner subscribes to `/camera/color/image_raw` and
`/camera/depth_registered/points`. The point cloud must contain the Gemini
driver's `x`, `y`, `z`, and packed FLOAT32 `rgb` fields.

The web server publishes synchronized camera-to-world
`object_scanner_interfaces/msg/NamedTransform` messages on
`/object_scanner_orbbec/camera_to_world` when requested from the page.

Open `http://<robot-ip>:5000` from a browser on the same LAN. The server binds
to all interfaces and has no authentication, so it must be used only on the
trusted isolated network selected for this project. Three.js 0.185.1 and
OrbitControls are loaded from jsDelivr, so the browser needs internet access.

## Controls

- Enter a session name using letters, numbers, `_`, or `-`.
- **Start** creates
  `<output_directory>/<session_name>/recording.sqlite3`. Start fails instead of
  overwriting an existing session directory.
- **Pause** commits the session, stops accepting frames, and loads the recorded
  points into Three.js.
- **Resume** appends new frames to the same SQLite file.
- **Stop** writes per-frame transformation details to `metadata.json`,
  checkpoints and closes the standalone SQLite file, removes WAL mode, then
  displays its final points. Close external database viewers before Stop so
  they cannot block finalization.

Refreshing or reopening the browser restores the active session name, recording
state, database path, and available controls from the still-running web server.
The recording continues until **Stop** succeeds.

The viewer samples uniformly across the complete recorded sequence and displays
at most 250,000 points. The database retains every filtered point.

Finalized session folders can also be selected for manual cumulative replay.
The replay dock appears at the bottom of the Three.js viewer. **Next** adds the
next frame without removing earlier points; **Previous** removes only the most
recent frame so the viewer again represents frames 1 through the selected
frame. **Exit replay** returns to the normal viewer. A global 250,000-point
display budget is shared across all replay frames. Replay is disabled until the
active recording session has been stopped.

The replay dock's camera overlay checkbox shows the current frame's camera
source, a 0.25 m optical +Z ray, and the standard red-X, green-Y, blue-Z axis
guide using that frame's recorded transformation matrix.

## Transformation matrices

The page displays the matrix that the next **Publish transformation matrix**
click will use. Each click publishes that matrix for exactly the next point
cloud, copying its timestamp and frame ID so the scanner can synchronize the
messages. After publishing, the page advances to the next entry, updates its
`Matrix i / total` indicator, and wraps to the start of the list.

**Previous** and **Next** change only the displayed selection and wrap at both
ends. They are disabled while a transformation publish is active.

Matrices are loaded at startup from
`object_scanner_web_orbbec/resource/transformation_matrices.json`. The default `identity`
entry leaves XYZ unchanged. Replace it or add entries using calibrated rigid
camera-to-world matrices for the capture locations used by the scanner. The
publish control is available in all recording states; points are written only
when the scanner is recording.

## Reference color

The reference color can be changed only while recording is stopped:

- choose a color with the native color picker and click **Apply color**; or
- click **Pick from camera** to capture the next RGB image, move over the
  lossless preview to inspect pixels in the magnifying loupe, click a pixel,
  then use its 5×5 neighborhood average.

The selected center pixel is outlined in white and its 5×5 sampled area in
yellow. Runtime selections last until the scanner restarts. The launch value is
used again on startup and defaults to green (`[0, 255, 0]`).

Launch parameters are forwarded to the isolated Orbbec scanner:

```bash
ros2 launch object_scanner_web_orbbec object_scanner_web.launch.py \
  output_directory:=/data/scans \
  target_rgb:="[0, 255, 0]" \
  lab_threshold:=15.0
```

## HTTP API

- `GET /api/status`
- `POST /api/transformation/publish`
- `POST /api/transformation/step` with JSON `{ "delta": -1 }` or
  `{ "delta": 1 }`
- `POST /api/reference-color`
- `GET /api/camera-frame` while stopped
- `POST /api/recording/start` with JSON
  `{ "session_name": "green_cup_01" }`
- `POST /api/recording/pause`
- `POST /api/recording/resume`
- `POST /api/recording/stop`
- `GET /api/points` while paused or stopped
- `GET /api/sessions`
- `GET /api/sessions/<name>/frames`
- `GET /api/sessions/<name>/frames/<id>` with optional `max_points` from
  `1` through `250000`

The points endpoint returns a binary `PCD1` payload: a 12-byte header containing
the displayed and total point counts, followed by float32 XYZ values and uint8
RGB values.
