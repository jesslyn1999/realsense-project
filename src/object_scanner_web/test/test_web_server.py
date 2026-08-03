import json
from pathlib import Path
import threading
import time

import numpy as np
from object_scanner.sqlite_recording import SqliteRecording
from object_scanner_web.web_server import create_app, RosControlBridge
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger


TRANSFORMATION_PATH = (
    Path(__file__).parents[2]
    / "object_scanner"
    / "resource"
    / "transformation_matrices.json"
)
IDENTITY_TRANSFORMATION = {
    "name": "identity",
    "parent_frame_id": "world",
    "matrix": np.eye(4).tolist(),
}


class FakeBridge:
    def __init__(self, database_path):
        self.state = "stopped"
        self.database_path = str(database_path)
        self.output_directory = Path(database_path).parent
        self.session_name = None
        self.target_rgb = [0, 255, 0]
        self.transformation = IDENTITY_TRANSFORMATION

    def status(self):
        return {
            "state": self.state,
            "database_path": self.database_path,
            "session_name": self.session_name,
            "target_rgb": list(self.target_rgb),
            "transformation": self.transformation,
            "transformation_index": 1,
            "transformation_total": 1,
            "transform_burst_active": False,
        }

    def refresh_status(self):
        return self.status()

    def command(self, command, session_name=None):
        states = {
            "start": "recording",
            "pause": "paused",
            "resume": "recording",
            "stop": "stopped",
        }
        if command == "start":
            self.session_name = session_name
        self.state = states[command]
        return {
            **self.status(),
            "success": True,
            "message": self.database_path,
        }

    def set_reference_color(self, rgb):
        if self.state != "stopped":
            return {
                **self.status(),
                "success": False,
                "message": "Reference color can change only while stopped",
            }
        self.target_rgb = list(rgb)
        return {
            **self.status(),
            "success": True,
            "message": "Reference color updated",
        }

    def capture_camera_frame(self):
        message = Image()
        message.width = 1
        message.height = 1
        message.step = 3
        message.encoding = "rgb8"
        message.data = bytes([1, 2, 3])
        return message

    def publish_transformation(self):
        return {
            **self.status(),
            "success": True,
            "message": "Published 'identity' for one point cloud",
            "published_transformation": self.transformation,
        }

    def step_transformation(self, delta):
        if delta not in {-1, 1}:
            raise ValueError("delta must be -1 or 1")
        return self.status()


class RejectingBridge(FakeBridge):
    def command(self, command, session_name=None):
        return {
            **self.status(),
            "success": False,
            "message": f"Cannot {command}",
        }


class TransformationTimeoutBridge(FakeBridge):
    def publish_transformation(self):
        raise TimeoutError("No point clouds received")


class TransformationBusyBridge(FakeBridge):
    def publish_transformation(self):
        raise RuntimeError("A transformation burst is already active")


def test_flask_controls_and_serves_paused_points(tmp_path):
    recording = SqliteRecording(tmp_path / "scan")
    database_path = recording.path
    recording.append_frame(
        recorded_perf_counter_ns=1,
        source_sec=2,
        source_nanosec=3,
        frame_id="world",
        transformation_name="identity",
        transformation_matrix=np.eye(4),
        xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        rgb=np.array([[0, 255, 0]], dtype=np.uint8),
    )

    share_directory = Path(__file__).parents[1]
    app = create_app(
        FakeBridge(database_path),
        share_directory,
        output_directory=tmp_path,
    )
    client = app.test_client()
    try:
        page = client.get("/")
        assert page.status_code == 200
        assert b'id="theme-toggle"' in page.data
        assert b'id="reference-color"' in page.data
        assert b'id="camera-preview"' in page.data
        assert b'id="pixel-loupe"' in page.data
        assert b'id="publish-transformation-button"' in page.data
        assert b'id="previous-transformation-button"' in page.data
        assert b'id="next-transformation-button"' in page.data
        assert b'id="transformation-position"' in page.data
        assert b'id="session-name"' in page.data
        assert b'id="saved-session"' in page.data
        assert b'id="replay-dock"' in page.data
        assert b'id="replay-previous-button"' in page.data
        assert b'id="replay-next-button"' in page.data
        assert b'id="replay-exit-button"' in page.data
        assert b'id="camera-overlay-checkbox"' in page.data
        assert client.get("/static/app.js").status_code == 200
        status = client.get("/api/status").json
        assert status["state"] == "stopped"
        assert status["transformation"] == IDENTITY_TRANSFORMATION
        assert status["transformation_index"] == 1
        assert status["transformation_total"] == 1
        response = client.post("/api/transformation/publish")
        assert response.status_code == 200
        assert (
            response.json["published_transformation"]
            == IDENTITY_TRANSFORMATION
        )
        assert client.post(
            "/api/transformation/step",
            json={"delta": -1},
        ).status_code == 200
        assert client.post(
            "/api/transformation/step",
            json={"delta": 0},
        ).status_code == 400
        response = client.post(
            "/api/reference-color",
            json={"rgb": [10, 20, 30]},
        )
        assert response.status_code == 200
        assert response.json["target_rgb"] == [10, 20, 30]
        assert (
            client.post(
                "/api/reference-color",
                json={"rgb": [1, 2]},
            ).status_code
            == 400
        )
        response = client.get("/api/camera-frame")
        assert response.status_code == 200
        assert response.data[:4] == b"RGB1"

        assert client.post("/api/recording/start").status_code == 400
        assert (
            client.post(
                "/api/recording/start",
                json={"session_name": "../invalid"},
            ).status_code
            == 400
        )
        response = client.post(
            "/api/recording/start",
            json={"session_name": "green_cup-01"},
        )
        assert response.status_code == 200
        assert response.json["state"] == "recording"
        assert response.json["session_name"] == "green_cup-01"
        refreshed_status = client.get("/api/status").json
        assert refreshed_status["state"] == "recording"
        assert refreshed_status["session_name"] == "green_cup-01"
        assert client.get("/api/points").status_code == 409
        assert (
            client.post(
                "/api/reference-color",
                json={"rgb": [0, 255, 0]},
            ).status_code
            == 409
        )

        response = client.post("/api/recording/pause")
        assert response.status_code == 200
        assert response.json["state"] == "paused"
        assert response.json["session_name"] == "green_cup-01"

        response = client.get("/api/points")
        assert response.status_code == 200
        assert response.headers["X-Displayed-Points"] == "1"
        assert response.headers["X-Total-Points"] == "1"
        assert response.data[:4] == b"PCD1"

        assert client.post("/api/recording/invalid").status_code == 404
    finally:
        recording.close()


def test_flask_lists_and_replays_only_finalized_session_folders(tmp_path):
    recording = SqliteRecording(tmp_path / "saved_scan")
    database_path = recording.path
    for frame_id, source_sec in [(1, 2), (2, 1)]:
        recording.append_frame(
            recorded_perf_counter_ns=frame_id,
            source_sec=source_sec,
            source_nanosec=0,
            frame_id="world",
            transformation_name="identity",
            transformation_matrix=np.eye(4),
            xyz=np.array([[frame_id, 0.0, 1.0]], dtype=np.float32),
            rgb=np.array([[0, 255, 0]], dtype=np.uint8),
        )
    recording.close()
    (tmp_path / "legacy.sqlite3").write_bytes(b"legacy")

    bridge = FakeBridge(database_path)
    app = create_app(
        bridge,
        Path(__file__).parents[1],
        output_directory=tmp_path,
    )
    client = app.test_client()

    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert response.json["sessions"] == [{"name": "saved_scan"}]

    response = client.get("/api/sessions/saved_scan/frames")
    assert response.status_code == 200
    assert [frame["id"] for frame in response.json["frames"]] == [2, 1]
    assert response.json["frames"][0]["transformation_name"] == "identity"
    assert response.json["frames"][0]["matrix"] == np.eye(4).tolist()
    assert response.json["total_frames"] == 2
    assert response.json["total_points"] == 2

    response = client.get(
        "/api/sessions/saved_scan/frames/1?max_points=1"
    )
    assert response.status_code == 200
    assert response.data[:4] == b"PCD1"
    assert client.get(
        "/api/sessions/saved_scan/frames/1?max_points=0"
    ).status_code == 400
    assert client.get(
        "/api/sessions/saved_scan/frames/1?max_points=250001"
    ).status_code == 400
    assert client.get("/api/sessions/missing/frames").status_code == 404

    bridge.state = "recording"
    assert client.get("/api/sessions/saved_scan/frames").status_code == 409
    bridge.state = "paused"
    assert client.get("/api/sessions/saved_scan/frames").status_code == 409


def test_flask_returns_conflict_for_rejected_ros_command(tmp_path, capfd):
    share_directory = Path(__file__).parents[1]
    app = create_app(
        RejectingBridge(tmp_path / "unused.sqlite3"),
        share_directory,
    )

    response = app.test_client().post("/api/recording/pause")
    stderr = capfd.readouterr().err

    assert response.status_code == 409
    assert not response.json["success"]
    assert "Recording command 'pause' rejected: Cannot pause" in stderr
    assert "Stack (most recent call last)" in stderr


def test_transformation_endpoint_reports_busy_and_timeout(tmp_path):
    share_directory = Path(__file__).parents[1]

    timeout_app = create_app(
        TransformationTimeoutBridge(tmp_path / "unused.sqlite3"),
        share_directory,
    )
    busy_app = create_app(
        TransformationBusyBridge(tmp_path / "unused.sqlite3"),
        share_directory,
    )

    assert (
        timeout_app.test_client().post("/api/transformation/publish").status_code
        == 504
    )
    assert (
        busy_app.test_client().post("/api/transformation/publish").status_code
        == 409
    )


def test_ros_control_bridge_can_spin():
    class CompletedFuture:
        def __init__(self, response):
            self.response = response

        def add_done_callback(self, callback):
            callback(self)

        def result(self):
            return self.response

    class FakeStatusClient:
        def __init__(self, response):
            self.response = response

        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, request):
            return CompletedFuture(self.response)

    rclpy.init()
    bridge = RosControlBridge(TRANSFORMATION_PATH)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    try:
        executor.spin_once(timeout_sec=0)

        response = Trigger.Response()
        response.success = True
        response.message = json.dumps(
            {
                "state": "paused",
                "database_path": "/tmp/cup.sqlite3",
                "session_name": "cup",
                "target_rgb": [1, 2, 3],
            }
        )
        bridge._status_client = FakeStatusClient(response)
        status = bridge.refresh_status()
        assert status["state"] == "paused"
        assert status["database_path"] == "/tmp/cup.sqlite3"
        assert status["session_name"] == "cup"
        assert status["target_rgb"] == [1, 2, 3]

        callback_started = threading.Event()
        cloud = PointCloud2()

        def receive_cloud():
            callback_started.set()
            bridge._on_pointcloud(cloud)

        with bridge._lock:
            callback_thread = threading.Thread(target=receive_cloud)
            callback_thread.start()
            assert callback_started.wait(timeout=1.0)
            callback_thread.join(timeout=0.1)
            state_lock_blocked_callback = callback_thread.is_alive()
        callback_thread.join(timeout=1.0)

        assert not state_lock_blocked_callback
    finally:
        executor.shutdown()
        bridge.destroy_node()
        rclpy.shutdown()


def test_ros_bridge_publishes_and_iterates_timestamp_matched_transforms(
    tmp_path,
):
    class FakePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    offset_transformation = {
        "name": "offset",
        "parent_frame_id": "world",
        "matrix": [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    transformation_path = tmp_path / "transformations.json"
    transformation_path.write_text(
        json.dumps([IDENTITY_TRANSFORMATION, offset_transformation]),
        encoding="utf-8",
    )

    rclpy.init()
    bridge = RosControlBridge(transformation_path)
    publisher = FakePublisher()
    bridge._transform_publisher = publisher
    assert bridge.step_transformation(-1)["transformation"] == (
        offset_transformation
    )
    assert bridge.step_transformation(-1)["transformation"] == (
        IDENTITY_TRANSFORMATION
    )
    result = []
    errors = []

    def publish_burst():
        try:
            result.append(bridge.publish_transformation(timeout_s=1.0))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=publish_burst)
    thread.start()
    try:
        deadline = time.perf_counter() + 1.0
        while not bridge.status()["transform_burst_active"]:
            assert time.perf_counter() < deadline
            time.sleep(0.001)

        with pytest.raises(RuntimeError, match="already active"):
            bridge.publish_transformation(timeout_s=0.01)
        with pytest.raises(RuntimeError, match="already active"):
            bridge.step_transformation(1)

        for index in range(1):
            cloud = PointCloud2()
            cloud.header.stamp.sec = index + 1
            cloud.header.stamp.nanosec = index + 10
            cloud.header.frame_id = "camera0_depth_optical_frame"
            bridge._on_pointcloud(cloud)

        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert not errors
        assert result[0]["success"]
        assert result[0]["published_transformation"] == IDENTITY_TRANSFORMATION
        assert result[0]["transformation"] == offset_transformation
        assert result[0]["transformation_index"] == 2
        assert result[0]["transformation_total"] == 2
        assert len(publisher.messages) == 1
        for index, message in enumerate(publisher.messages):
            assert message.header.stamp.sec == index + 1
            assert message.header.stamp.nanosec == index + 10
            assert message.header.frame_id == "world"
            assert message.child_frame_id == "camera0_depth_optical_frame"
            assert message.transformation_name == "identity"
            np.testing.assert_array_equal(
                np.asarray(message.matrix).reshape(4, 4),
                np.eye(4),
            )

        with pytest.raises(TimeoutError, match="Did not receive a point cloud"):
            bridge.publish_transformation(timeout_s=0.001)
        assert not bridge.status()["transform_burst_active"]

        result.clear()
        thread = threading.Thread(target=publish_burst)
        thread.start()
        deadline = time.perf_counter() + 1.0
        while not bridge.status()["transform_burst_active"]:
            assert time.perf_counter() < deadline
            time.sleep(0.001)
        for index in range(1):
            cloud = PointCloud2()
            cloud.header.stamp.sec = index + 20
            cloud.header.frame_id = "camera0_depth_optical_frame"
            bridge._on_pointcloud(cloud)
        thread.join(timeout=1.0)

        assert not errors
        assert result[0]["published_transformation"] == offset_transformation
        assert result[0]["transformation"] == IDENTITY_TRANSFORMATION
        assert result[0]["transformation_index"] == 1
        assert result[0]["transformation_total"] == 2
        assert len(publisher.messages) == 2
        assert publisher.messages[-1].transformation_name == "offset"
        assert publisher.messages[-1].matrix[3] == 1.0
    finally:
        thread.join(timeout=1.0)
        bridge.destroy_node()
        rclpy.shutdown()
