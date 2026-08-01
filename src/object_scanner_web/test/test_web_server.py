from pathlib import Path

import numpy as np
from object_scanner.sqlite_recording import SqliteRecording
from object_scanner_web.web_server import create_app, RosControlBridge
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image


class FakeBridge:
    def __init__(self, database_path):
        self.state = "stopped"
        self.database_path = str(database_path)
        self.target_rgb = [0, 255, 0]

    def status(self):
        return {
            "state": self.state,
            "database_path": self.database_path,
            "target_rgb": list(self.target_rgb),
        }

    def command(self, command):
        states = {
            "start": "recording",
            "pause": "paused",
            "resume": "recording",
            "stop": "stopped",
        }
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


class RejectingBridge(FakeBridge):
    def command(self, command):
        return {
            **self.status(),
            "success": False,
            "message": f"Cannot {command}",
        }


def test_flask_controls_and_serves_paused_points(tmp_path):
    database_path = tmp_path / "scan.sqlite3"
    recording = SqliteRecording(database_path)
    recording.append_frame(
        recorded_perf_counter_ns=1,
        source_sec=2,
        source_nanosec=3,
        frame_id="world",
        xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        rgb=np.array([[0, 255, 0]], dtype=np.uint8),
    )

    share_directory = Path(__file__).parents[1]
    app = create_app(FakeBridge(database_path), share_directory)
    client = app.test_client()
    try:
        page = client.get("/")
        assert page.status_code == 200
        assert b'id="reference-color"' in page.data
        assert b'id="camera-preview"' in page.data
        assert b'id="pixel-loupe"' in page.data
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/status").json["state"] == "stopped"
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

        response = client.post("/api/recording/start")
        assert response.status_code == 200
        assert response.json["state"] == "recording"
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

        response = client.get("/api/points")
        assert response.status_code == 200
        assert response.headers["X-Displayed-Points"] == "1"
        assert response.headers["X-Total-Points"] == "1"
        assert response.data[:4] == b"PCD1"

        assert client.post("/api/recording/invalid").status_code == 404
    finally:
        recording.close()


def test_flask_returns_conflict_for_rejected_ros_command(tmp_path):
    share_directory = Path(__file__).parents[1]
    app = create_app(
        RejectingBridge(tmp_path / "unused.sqlite3"),
        share_directory,
    )

    response = app.test_client().post("/api/recording/pause")

    assert response.status_code == 409
    assert not response.json["success"]


def test_ros_control_bridge_can_spin():
    rclpy.init()
    bridge = RosControlBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    try:
        executor.spin_once(timeout_sec=0)
    finally:
        executor.shutdown()
        bridge.destroy_node()
        rclpy.shutdown()
