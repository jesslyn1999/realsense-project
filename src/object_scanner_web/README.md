# Object scanner web GUI

This package provides a Flask page for controlling `object_scanner` and viewing
the colored world-frame points committed to its current SQLite recording.

The combined launch starts the configured RealSense camera, object scanner, and
web server:

```bash
sudo apt install python3.12-venv
python3 -m venv --system-site-packages .venv_scanner
source .venv_scanner/bin/activate
python -m pip install open3d-cpu==0.19.0 "scipy>=1.15.0" \
  "scikit-learn>=1.6.0"
source /opt/ros/jazzy/setup.bash
python -m colcon build --symlink-install \
  --packages-select object_scanner_interfaces object_scanner \
  object_scanner_processing object_scanner_web \
  --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug
ln -sf build/compile_commands.json compile_commands.json
source install/setup.bash
ros2 launch object_scanner_web object_scanner_web.launch.py
```

The web server publishes synchronized camera-to-world
`object_scanner_interfaces/msg/NamedTransform` messages on
`/object_scanner/camera_to_world` when requested from the page.

Open `http://<robot-ip>:5000` from a browser on the same LAN. The server binds
to all interfaces and has no authentication, so it must be used only on the
trusted isolated network selected for this project. Three.js 0.185.1 and
OrbitControls are loaded from jsDelivr, so the browser needs internet access.

## Controls

- Enter a session name using letters, numbers, `_`, or `-`.
- **Start** creates
  `<output_directory>/<session_name>/recording.sqlite3`. Start fails instead of
  overwriting an existing session directory.
- **Pause** commits the session, stops accepting frames, synchronously refreshes
  `aligned_recording.sqlite3`, and then loads its fused cloud into Three.js.
- **Resume** appends new frames to the same SQLite file.
- **Stop** writes per-frame transformation details to `metadata.json`,
  checkpoints and closes the raw SQLite file, removes WAL mode, synchronously
  refreshes the aligned database, then displays its fused cloud. Close external
  database viewers before Stop so they cannot block finalization.

Refreshing or reopening the browser restores the active session name, recording
state, database path, and available controls from the still-running web server.
The recording continues until **Stop** succeeds.

On every Pause and Stop, the recorder non-destructively cleans, registers, and
fuses all committed frames before the service call returns. JSON or ChArUco
camera-to-world matrices provide the initial poses. The shared
`object_scanner_processing` package accepts only overlapping registration
edges whose residual improves and whose correction stays within 10 mm and
2 degrees. Failure is reported instead of displaying an unsafe raw merge.

`aligned_recording.sqlite3` is transactionally refreshed in place with every
cleaned/aligned per-frame cloud, optimized pose, fused cloud, observation
counts, and diagnostics. The viewer reads and samples that saved fused result
at 250,000 points. `recording.sqlite3` retains every original filtered point
and matrix; saved-session replay continues to show the original per-frame
captures.

Finalized session folders can also be selected for manual cumulative replay.
The replay dock appears at the bottom of the Three.js viewer. **Next** adds the
next frame without removing earlier points; **Previous** removes only the most
recent frame so the viewer again represents frames 1 through the selected
frame. **Exit replay** returns to the normal viewer. A global 250,000-point
display budget is shared across all replay frames. Replay is disabled until the
active recording session has been stopped.

The same launch also serves a simplified display remote at
`http://<robot-ip>:5001/remote`. Keep one main viewer open on port 5000. The
remote starts the fixed `demo5` replay, advances its cumulative frames, and can
replace the main display with a full-screen loading message until **Stop
loading** is clicked. **Next** is disabled when the final frame is visible.

The replay dock's camera overlay checkbox shows the current frame's camera
source, a 0.25 m optical +Z ray, and the standard red-X, green-Y, blue-Z axis
guide using that frame's recorded transformation matrix.

## Transformation modes

The transformation mode can be changed only while recording is stopped. JSON
matrices remain the default.

### JSON matrices

The page displays the matrix that the next **Publish transformation matrix**
click will use. Each click publishes that matrix for exactly the next point
cloud, copying its timestamp and frame ID so the scanner can synchronize the
messages. After publishing, the page advances to the next entry, updates its
`Matrix i / total` indicator, and wraps to the start of the list.

**Previous** and **Next** change only the displayed selection and wrap at both
ends. They are disabled while a transformation publish is active.

Matrices are loaded at startup from
`object_scanner/resource/transformation_matrices.json`. The default `identity`
entry leaves XYZ unchanged. Replace it or add entries using calibrated rigid
camera-to-world matrices for the capture locations used by the scanner. The
publish control is available in all recording states; points are written only
when the scanner is recording.

### ChArUco capture

The ChArUco mode uses the fixed board generated by
`src/board_resources/ChArUco/generate_charuco_a4.py`: 10×7 squares,
20 mm square length, 15 mm marker length, and `DICT_4X4_50`. Print the PDF at
100% / Actual size.

The board is the world reference. Its origin is the active pattern's top-left
outer corner when viewed from the printed side; +X points right, +Y points up,
and +Z points outward from the board.

While ChArUco mode is selected, its panel shows the latest RealSense RGB frame
with detected marker borders and ChArUco corners drawn over it. The browser
refreshes this diagnostic preview at no more than 5 FPS in any recording state.
Preview frames are not saved and are not reused by **Capture**.

Start recording, keep the board visible, and click **Capture**. The server uses
a synchronized `/realsense/camera0/color/image_raw` image and
`/realsense/camera0/depth/color/points` cloud with intrinsics from
`/realsense/camera0/color/camera_info`. ChArUco estimates a color-camera-to-world
pose. Because the RealSense point cloud is in `camera0_depth_optical_frame`, the
server composes that pose with the camera's static depth-to-color TF before
publishing the depth-camera-to-world matrix. `publish_tf` must remain enabled
in the RealSense camera configuration.

A capture is accepted only with at least 20 ChArUco corners and reprojection
RMSE no greater than 1.5 px. An accepted click publishes one transform and the
scanner appends one color-filtered XYZ/RGB cloud plus that matrix to the current
SQLite recording. A rejected or timed-out click writes no frame. The server
never falls back to an older pose, so every click recalibrates after the camera
moves.

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
- `POST /api/transformation/mode` with JSON
  `{ "mode": "json" }` or `{ "mode": "charuco" }` while stopped
- `POST /api/transformation/publish`
- `POST /api/transformation/step` with JSON `{ "delta": -1 }` or
  `{ "delta": 1 }`
- `GET /api/charuco/preview` in ChArUco mode, in any recording state
- `POST /api/charuco/capture` while recording in ChArUco mode
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
