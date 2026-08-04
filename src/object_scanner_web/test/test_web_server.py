import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped
import numpy as np
from object_scanner.sqlite_recording import SqliteRecording
from object_scanner_processing.charuco_observations import (
    CharucoCalibration,
    CharucoCalibrationError,
    CharucoFrameObservation,
)
import object_scanner_web.web_server as web_server
from object_scanner_web.web_server import (
    create_app,
    create_remote_app,
    RemoteControl,
    RosControlBridge,
)
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import TransformException


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


def make_charuco_observation(corner_count=42):
    return CharucoFrameObservation(
        corner_ids=np.arange(corner_count, dtype=np.int32),
        image_points=np.zeros((corner_count, 2)),
        depth_valid=np.ones(corner_count, dtype=bool),
        child_points=np.zeros((corner_count, 3)),
        depth_valid_pixel_counts=np.full(corner_count, 25, dtype=np.uint16),
        depth_inlier_pixel_counts=np.full(corner_count, 25, dtype=np.uint16),
        depth_mad_m=np.zeros(corner_count),
        depth_invalid_reasons=tuple("" for _ in range(corner_count)),
        camera_matrix=np.eye(3),
        distortion=np.zeros(5),
        color_from_child=np.eye(4),
        initial_reprojection_errors_px=np.full(corner_count, 0.4),
        initial_reprojection_rmse_px=0.4,
    )


class FakeBridge:
    def __init__(self, database_path):
        self.state = "stopped"
        self.database_path = str(database_path)
        self.output_directory = Path(database_path).parent
        self.session_name = None
        self.target_rgb = [0, 255, 0]
        self.transformation = IDENTITY_TRANSFORMATION
        self.transformation_mode = "json"
        self.last_charuco = None

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
            "transformation_mode": self.transformation_mode,
            "last_charuco": self.last_charuco,
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
        if self.transformation_mode != "json":
            raise RuntimeError(
                "JSON transformation publishing is unavailable "
                "in ChArUco mode"
            )
        return {
            **self.status(),
            "success": True,
            "message": "Published 'identity' for one point cloud",
            "published_transformation": self.transformation,
        }

    def step_transformation(self, delta):
        if delta not in {-1, 1}:
            raise ValueError("delta must be -1 or 1")
        if self.transformation_mode != "json":
            raise RuntimeError(
                "JSON transformation selection is unavailable "
                "in ChArUco mode"
            )
        return self.status()

    def set_transformation_mode(self, mode):
        if self.state != "stopped":
            return {
                **self.status(),
                "success": False,
                "message": "Transformation mode can change only while stopped",
            }
        self.transformation_mode = mode
        return {
            **self.status(),
            "success": True,
            "message": f"Transformation mode set to {mode}",
        }

    def capture_charuco(self):
        if self.transformation_mode != "charuco":
            raise RuntimeError("ChArUco capture is unavailable in JSON mode")
        if self.state != "recording":
            raise RuntimeError(
                "ChArUco capture is available only while recording"
            )
        self.last_charuco = {
            "success": True,
            "message": "ChArUco pose accepted",
            "corner_count": 54,
            "reprojection_rmse_px": 0.25,
            "matrix": np.eye(4).tolist(),
        }
        return {
            **self.status(),
            "success": True,
            "message": "Captured one ChArUco-calibrated point cloud",
            "charuco_capture": self.last_charuco,
        }

    def build_charuco_preview(self):
        if self.transformation_mode != "charuco":
            raise RuntimeError(
                "ChArUco preview is unavailable in JSON mode"
            )
        return np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)


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


class CharucoRejectingBridge(FakeBridge):
    def capture_charuco(self):
        raise CharucoCalibrationError(
            "ChArUco capture requires at least 20 corners; detected 8",
            corner_count=8,
        )


class NoImageBridge(FakeBridge):
    def build_charuco_preview(self):
        raise TimeoutError("No RGB image received")


def test_flask_controls_and_serves_paused_points(tmp_path, monkeypatch):
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
    aligned_reads = []

    def fake_read_fused_cloud(path, revision):
        aligned_reads.append((path, revision))
        return SimpleNamespace(
            xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            rgb=np.array([[0, 255, 0]], dtype=np.uint8),
            raw_points=1,
            cleaned_points=1,
            accepted_edges=1,
            rejected_edges=0,
            charuco_frame_count=1,
            charuco_reprojection_max_px=0.4,
            cloud_overlap_fraction_3mm=0.995,
            quality_warning="Best-effort aligned output",
        )

    monkeypatch.setattr(
        web_server,
        "read_fused_cloud",
        fake_read_fused_cloud,
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
        assert b'id="transformation-mode"' in page.data
        assert b'<option value="charuco" selected>' in page.data
        assert b'id="json-transformation-panel"' in page.data
        assert b'id="charuco-transformation-panel"' in page.data
        assert b'id="charuco-preview"' in page.data
        assert b'id="charuco-preview-status"' in page.data
        assert b'id="charuco-capture-button"' in page.data
        assert b'id="previous-transformation-button"' in page.data
        assert b'id="next-transformation-button"' in page.data
        assert b'id="transformation-position"' in page.data
        assert b'id="session-name"' in page.data
        assert b'id="saved-session"' in page.data
        assert b'id="replay-dock"' in page.data
        assert b'id="analyze-button"' in page.data
        assert b'id="adjust-repair-button"' in page.data
        assert b'id="analysis-dock"' in page.data
        assert b'id="analysis-repair-visible"' in page.data
        assert b'id="analysis-segment-visible"' in page.data
        assert b'id="analysis-move-button"' in page.data
        assert b'id="analysis-rotate-button"' in page.data
        assert b'id="analysis-scale-button"' in page.data
        assert b'id="analysis-apply-button"' in page.data
        assert b'id="replay-previous-button"' in page.data
        assert b'id="replay-next-button"' in page.data
        assert b'id="replay-exit-button"' in page.data
        assert b'id="replay-raw-button"' in page.data
        assert b'id="replay-filtered-button"' in page.data
        assert b'id="replay-aligned-button"' in page.data
        assert b'id="camera-overlay-checkbox"' in page.data
        assert b'id="orientation-gizmo"' in page.data
        assert b'id="point-coordinate-tooltip"' in page.data
        assert b'id="scan-help-button"' in page.data
        assert b'id="scan-help-dialog"' in page.data
        assert b'id="remote-loading-overlay"' in page.data
        assert b"Move in small steps" in page.data
        assert b"Keep at least 60%" in page.data
        app_javascript = client.get("/static/app.js")
        assert app_javascript.status_code == 200
        assert b"camera.up.set(0, 0, 1);" in app_javascript.data
        assert b"grid.rotation.x = Math.PI / 2;" in app_javascript.data
        assert b"TransformControls" in app_javascript.data
        assert b"STLLoader" in app_javascript.data
        status = client.get("/api/status").json
        assert status["state"] == "stopped"
        assert status["transformation"] == IDENTITY_TRANSFORMATION
        assert status["transformation_index"] == 1
        assert status["transformation_total"] == 1
        assert status["transformation_mode"] == "json"
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
        assert response.headers["X-Raw-Points"] == "1"
        assert response.headers["X-Cleaned-Points"] == "1"
        assert response.headers["X-Accepted-Edges"] == "1"
        assert response.headers["X-Rejected-Edges"] == "0"
        assert response.headers["X-Charuco-Frames"] == "1"
        assert response.headers["X-Charuco-Max-Reprojection-Px"] == "0.4"
        assert response.headers["X-Cloud-3mm-Fraction"] == "0.995"
        assert (
            response.headers["X-Processing-Warning"]
            == "Best-effort aligned output"
        )
        assert response.data[:4] == b"PCD1"
        assert len(aligned_reads) == 1
        assert aligned_reads[0][0].name == "aligned_recording.sqlite3"
        assert aligned_reads[0][1].frame_count == 1
        assert client.get("/api/points").status_code == 200
        assert len(aligned_reads) == 2

        assert client.post("/api/recording/resume").status_code == 200
        recording.append_frame(
            recorded_perf_counter_ns=2,
            source_sec=3,
            source_nanosec=4,
            frame_id="world",
            transformation_name="identity",
            transformation_matrix=np.eye(4),
            xyz=np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
            rgb=np.array([[0, 255, 0]], dtype=np.uint8),
        )
        assert client.post("/api/recording/pause").status_code == 200
        assert client.get("/api/points").status_code == 200
        assert len(aligned_reads) == 3
        assert aligned_reads[-1][1].frame_count == 2

        assert client.post("/api/recording/invalid").status_code == 404
    finally:
        recording.close()


def test_remote_page_queues_commands_and_tracks_main_viewer(tmp_path):
    share_directory = Path(__file__).parents[1]
    remote_control = RemoteControl()
    main_client = create_app(
        FakeBridge(tmp_path / "unused.sqlite3"),
        share_directory,
        remote_control=remote_control,
    ).test_client()
    remote_app = create_remote_app(remote_control, share_directory)
    remote_client = remote_app.test_client()

    page = remote_client.get("/remote")
    assert page.status_code == 200
    assert b'id="remote-start-replay"' in page.data
    assert b'id="remote-next"' in page.data
    assert b'id="remote-show-loading"' in page.data
    assert b'id="remote-stop-loading"' in page.data
    assert b"Start replay demo5" in page.data
    assert remote_client.get("/static/remote.js").status_code == 200

    status = remote_client.get("/api/status").json
    assert not status["connected"]
    assert not status["can_next"]
    assert status["pending_command"] is None

    viewer_report = {
        "replay_mode": False,
        "replay_index": 0,
        "replay_total": 0,
        "can_next": False,
        "loading_visible": False,
        "busy": False,
        "stage": "raw",
    }
    assert main_client.post(
        "/api/remote/viewer",
        json=viewer_report,
    ).status_code == 200
    assert remote_client.get("/api/status").json["connected"]

    assert remote_client.post(
        "/api/command",
        json={"command": "unknown"},
    ).status_code == 400
    response = remote_client.post(
        "/api/command",
        json={"command": "start_replay"},
    )
    assert response.status_code == 202
    command = response.json["pending_command"]
    assert command["command"] == "start_replay"
    assert remote_client.post(
        "/api/command",
        json={"command": "next"},
    ).status_code == 409

    pending = main_client.get("/api/remote/command").json
    assert pending["command"] == command
    assert pending["demo_session"] == "demo5"

    completed_report = {
        **viewer_report,
        "replay_mode": True,
        "replay_total": 2,
        "can_next": True,
        "completed_command_id": command["id"],
        "error": None,
    }
    assert main_client.post(
        "/api/remote/viewer",
        json=completed_report,
    ).status_code == 200
    status = remote_client.get("/api/status").json
    assert status["pending_command"] is None
    assert status["replay_mode"]
    assert status["can_next"]

    response = remote_client.post(
        "/api/command",
        json={"command": "next"},
    )
    command = response.json["pending_command"]
    completed_report.update(
        replay_index=1,
        can_next=False,
        completed_command_id=command["id"],
    )
    assert main_client.post(
        "/api/remote/viewer",
        json=completed_report,
    ).status_code == 200
    assert not remote_client.get("/api/status").json["can_next"]

    response = remote_client.post(
        "/api/command",
        json={"command": "show_loading"},
    )
    command = response.json["pending_command"]
    completed_report.update(
        loading_visible=True,
        completed_command_id=command["id"],
    )
    assert main_client.post(
        "/api/remote/viewer",
        json=completed_report,
    ).status_code == 200
    assert main_client.get(
        "/api/remote/command"
    ).json["loading_visible"]

    response = remote_client.post(
        "/api/command",
        json={"command": "stop_loading"},
    )
    command = response.json["pending_command"]
    completed_report.update(
        loading_visible=False,
        completed_command_id=command["id"],
    )
    assert main_client.post(
        "/api/remote/viewer",
        json=completed_report,
    ).status_code == 200
    assert not remote_client.get("/api/status").json["loading_visible"]


def test_points_refuses_missing_or_stale_aligned_output(tmp_path, monkeypatch):
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

    def reject_aligned_output(_path, _revision):
        raise web_server.AlignedRecordingError(
            "aligned recording does not match the current raw recording"
        )

    monkeypatch.setattr(
        web_server,
        "read_fused_cloud",
        reject_aligned_output,
    )
    app = create_app(
        FakeBridge(database_path),
        Path(__file__).parents[1],
        output_directory=tmp_path,
    )
    client = app.test_client()
    aligned_response = client.get("/api/points")
    raw_response = client.get("/api/points?source=raw")
    invalid_response = client.get("/api/points?source=unknown")
    recording.close()

    assert aligned_response.status_code == 422
    assert not aligned_response.json["success"]
    assert "does not match" in aligned_response.json["message"]

    assert raw_response.status_code == 200
    assert raw_response.data[:4] == b"PCD1"
    assert raw_response.content_type == "application/octet-stream"
    assert raw_response.headers["Cache-Control"] == "no-store"
    assert raw_response.headers["X-Displayed-Points"] == "1"
    assert raw_response.headers["X-Total-Points"] == "1"
    assert raw_response.headers["X-Preview-Source"] == "raw"

    assert invalid_response.status_code == 400
    assert not invalid_response.json["success"]
    assert "source" in invalid_response.json["message"]


def test_points_reads_saved_output_after_raw_wal_checkpoint(
    tmp_path,
    monkeypatch,
):
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
    aligned_reads = []

    def fake_read_fused_cloud(path, revision):
        aligned_reads.append((path, revision))
        return SimpleNamespace(
            xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            rgb=np.array([[0, 255, 0]], dtype=np.uint8),
            raw_points=1,
            cleaned_points=1,
            accepted_edges=1,
            rejected_edges=0,
            charuco_frame_count=0,
            charuco_reprojection_max_px=None,
            cloud_overlap_fraction_3mm=None,
        )

    monkeypatch.setattr(
        web_server,
        "read_fused_cloud",
        fake_read_fused_cloud,
    )
    client = create_app(
        FakeBridge(database_path),
        Path(__file__).parents[1],
        output_directory=tmp_path,
    ).test_client()
    closed = False
    try:
        assert client.get("/api/points").status_code == 200
        recording.close()
        closed = True
        assert client.get("/api/points").status_code == 200
        assert len(aligned_reads) == 2
        assert aligned_reads[0][1] == aligned_reads[1][1]
    finally:
        if not closed:
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
    assert response.json["stages"] == ["raw"]

    response = client.get(
        "/api/sessions/saved_scan/frames/1?max_points=1&stage=raw"
    )
    assert response.status_code == 200
    assert response.data[:4] == b"PCD1"
    assert client.get(
        "/api/sessions/saved_scan/frames/1?stage=filtered"
    ).status_code == 404
    assert client.get(
        "/api/sessions/saved_scan/frames/1?stage=unknown"
    ).status_code == 400
    assert client.get(
        "/api/sessions/saved_scan/frames/1?max_points=0"
    ).status_code == 400
    assert client.get(
        "/api/sessions/saved_scan/frames/1?max_points=250001"
    ).status_code == 400

    aligned_path = database_path.parent / "aligned_recording.sqlite3"
    aligned_path.write_bytes(database_path.read_bytes())
    response = client.get("/api/sessions/saved_scan/frames")
    assert response.json["stages"] == ["raw", "filtered", "aligned"]
    assert response.json["frames"][0]["optimized_matrix"] == np.eye(4).tolist()
    assert client.get(
        "/api/sessions/saved_scan/frames/1?stage=filtered"
    ).status_code == 200
    assert client.get(
        "/api/sessions/saved_scan/frames/1?stage=aligned"
    ).status_code == 200

    assert client.get("/api/sessions/missing/frames").status_code == 404

    bridge.state = "recording"
    assert client.get("/api/sessions/saved_scan/frames").status_code == 409
    bridge.state = "paused"
    assert client.get("/api/sessions/saved_scan/frames").status_code == 409


def test_demo5_repair_reference_and_analysis_routes(tmp_path, monkeypatch):
    recording = SqliteRecording(tmp_path / "demo5")
    database_path = recording.path
    recording.append_frame(
        recorded_perf_counter_ns=1,
        source_sec=2,
        source_nanosec=3,
        frame_id="world",
        transformation_name="identity",
        transformation_matrix=np.eye(4),
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        rgb=np.array([[100, 100, 100]], dtype=np.uint8),
    )
    recording.close()
    aligned_path = database_path.parent / "aligned_recording.sqlite3"
    aligned_path.write_bytes(b"aligned")
    repair_path = tmp_path / "repair.stl"
    repair_path.write_bytes(b"repair mesh")
    calls = []

    def fake_segment(path, stl_path, transform):
        if transform is not None and np.asarray(transform).shape != (4, 4):
            raise ValueError("transform must be a finite 4x4 matrix")
        calls.append((path, stl_path, transform))
        return SimpleNamespace(
            xyz=np.array([[0.1, -0.2, 0.08]], dtype=np.float32),
            transform=np.eye(4) if transform is None else transform,
            scale_m_per_stl_unit=0.0057,
        )

    monkeypatch.setattr(web_server, "segment_repair", fake_segment)
    bridge = FakeBridge(database_path)
    client = create_app(
        bridge,
        Path(__file__).parents[1],
        output_directory=tmp_path,
        repair_stl_path=repair_path,
    ).test_client()

    reference = client.get("/api/repair/reference")
    assert reference.status_code == 200
    assert reference.data == b"repair mesh"
    assert reference.content_type == "model/stl"
    assert reference.headers["Cache-Control"] == "no-store"

    automatic = client.post(
        "/api/sessions/demo5/repair-analysis",
        json={"transform": None},
    )
    assert automatic.status_code == 200
    assert automatic.json["point_count"] == 1
    assert automatic.json["points"] == pytest.approx([0.1, -0.2, 0.08])
    assert automatic.json["scale_m_per_stl_unit"] == 0.0057
    assert calls[0] == (aligned_path, repair_path, None)

    refined_transform = np.eye(4)
    refined_transform[:3, 3] = [0.125, -0.2342, 0.0802]
    refined = client.post(
        "/api/sessions/demo5/repair-analysis",
        json={"transform": refined_transform.tolist()},
    )
    assert refined.status_code == 200
    np.testing.assert_allclose(refined.json["transform"], refined_transform)
    np.testing.assert_allclose(calls[1][2], refined_transform)

    assert client.post(
        "/api/sessions/demo5/repair-analysis",
        json={},
    ).status_code == 400
    assert client.post(
        "/api/sessions/demo5/repair-analysis",
        json={"transform": [[1.0]]},
    ).status_code == 400
    assert client.post(
        "/api/sessions/other/repair-analysis",
        json={"transform": None},
    ).status_code == 404

    bridge.state = "recording"
    assert client.post(
        "/api/sessions/demo5/repair-analysis",
        json={"transform": None},
    ).status_code == 409


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


def test_bridge_refreshes_state_after_alignment_failure(monkeypatch):
    class CompletedFuture:
        def add_done_callback(self, callback):
            callback(self)

        def result(self):
            response = Trigger.Response()
            response.success = False
            response.message = "Recording paused, but aligned output failed"
            return response

    class FailedPauseClient:
        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, request):
            return CompletedFuture()

    rclpy.init()
    bridge = RosControlBridge(TRANSFORMATION_PATH)
    refresh_count = 0

    def refresh_status():
        nonlocal refresh_count
        refresh_count += 1
        bridge._state = "recording" if refresh_count == 1 else "paused"
        bridge._database_path = "/tmp/scan/recording.sqlite3"
        bridge._session_name = "scan"

    monkeypatch.setattr(
        bridge,
        "_refresh_scanner_status_unlocked",
        refresh_status,
    )
    bridge._service_clients["pause"] = FailedPauseClient()
    try:
        result = bridge.command("pause")

        assert not result["success"]
        assert result["state"] == "paused"
        assert refresh_count == 2
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


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


def test_transformation_mode_and_charuco_routes(tmp_path):
    share_directory = Path(__file__).parents[1]
    bridge = FakeBridge(tmp_path / "unused.sqlite3")
    client = create_app(bridge, share_directory).test_client()

    assert (
        client.post(
            "/api/transformation/mode",
            json={"mode": "invalid"},
        ).status_code
        == 400
    )
    assert client.post("/api/charuco/capture").status_code == 409
    assert client.get("/api/charuco/preview").status_code == 409

    response = client.post(
        "/api/transformation/mode",
        json={"mode": "charuco"},
    )
    assert response.status_code == 200
    assert response.json["transformation_mode"] == "charuco"
    preview = client.get("/api/charuco/preview")
    assert preview.status_code == 200
    assert preview.data[:4] == b"RGB1"
    assert preview.headers["X-Image-Width"] == "2"
    assert preview.headers["X-Image-Height"] == "1"
    assert client.post("/api/charuco/capture").status_code == 409
    assert client.post("/api/transformation/publish").status_code == 409
    assert (
        client.post(
            "/api/transformation/step",
            json={"delta": 1},
        ).status_code
        == 409
    )

    bridge.state = "recording"
    assert client.get("/api/charuco/preview").status_code == 200
    assert (
        client.post(
            "/api/transformation/mode",
            json={"mode": "json"},
        ).status_code
        == 409
    )
    response = client.post("/api/charuco/capture")
    assert response.status_code == 200
    assert response.json["charuco_capture"]["corner_count"] == 54
    assert response.json["charuco_capture"]["reprojection_rmse_px"] == 0.25
    bridge.state = "paused"
    assert client.get("/api/charuco/preview").status_code == 200

    rejecting_bridge = CharucoRejectingBridge(
        tmp_path / "rejected.sqlite3"
    )
    rejecting_bridge.transformation_mode = "charuco"
    rejecting_bridge.state = "recording"
    rejecting_client = create_app(
        rejecting_bridge,
        share_directory,
    ).test_client()
    response = rejecting_client.post("/api/charuco/capture")
    assert response.status_code == 422
    assert response.json["corner_count"] == 8
    assert response.json["reprojection_rmse_px"] is None

    no_image_bridge = NoImageBridge(tmp_path / "no-image.sqlite3")
    no_image_bridge.transformation_mode = "charuco"
    no_image_client = create_app(
        no_image_bridge,
        share_directory,
    ).test_client()
    assert no_image_client.get("/api/charuco/preview").status_code == 503


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


def test_charuco_sensor_frame_requires_matching_color_camera_frames():
    cloud = PointCloud2()
    cloud.header.frame_id = "camera0_depth_optical_frame"
    image = Image()
    image.header.frame_id = "camera0_color_optical_frame"
    depth = Image()
    depth.header.frame_id = image.header.frame_id
    camera_info = CameraInfo()
    camera_info.header.frame_id = "wrong_frame"

    with pytest.raises(CharucoCalibrationError, match="CameraInfo frame"):
        RosControlBridge._calibrate_sensor_frame(
            cloud,
            image,
            depth,
            camera_info,
        )
    with pytest.raises(CharucoCalibrationError, match="No camera intrinsics"):
        RosControlBridge._calibrate_sensor_frame(
            cloud,
            image,
            depth,
            None,
        )


def test_ros_bridge_charuco_capture_composes_color_and_depth_frames(
    monkeypatch,
):
    class FakePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class FakeTfBuffer:
        def __init__(self):
            self.lookups = []

        def lookup_transform(self, target, source, stamp, timeout):
            self.lookups.append((target, source, stamp, timeout))
            transform = TransformStamped()
            transform.header.frame_id = target
            transform.child_frame_id = source
            transform.transform.translation.x = 0.02
            transform.transform.rotation.w = 1.0
            return transform

    world_from_color = np.eye(4)
    world_from_color[:3, 3] = [0.1, -0.2, 0.3]
    calibration = CharucoCalibration(
        camera_to_world=world_from_color,
        corner_count=42,
        reprojection_rmse_px=0.4,
    )
    monkeypatch.setattr(
        "object_scanner_web.web_server.calibrate_charuco",
        lambda *_args: calibration,
    )
    observation = make_charuco_observation()
    monkeypatch.setattr(
        "object_scanner_web.web_server.build_charuco_observation",
        lambda *_args: observation,
    )

    rclpy.init()
    bridge = RosControlBridge(TRANSFORMATION_PATH)
    bridge._refresh_scanner_status_unlocked = lambda: None
    bridge._transform_publisher = FakePublisher()
    bridge._tf_buffer = FakeTfBuffer()
    thread = None
    try:
        assert bridge.set_transformation_mode("charuco")["success"]
        with pytest.raises(TimeoutError, match="No RGB image received"):
            bridge.build_charuco_preview()
        bridge._state = "recording"

        color_frame = "camera0_color_optical_frame"
        depth_frame = "camera0_depth_optical_frame"
        camera_info = CameraInfo()
        camera_info.header.frame_id = color_frame
        camera_info.width = 1
        camera_info.height = 1
        camera_info.k = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        cloud = PointCloud2()
        cloud.header.stamp.sec = 4
        cloud.header.stamp.nanosec = 5
        cloud.header.frame_id = depth_frame
        image = Image()
        image.header.stamp = cloud.header.stamp
        image.header.frame_id = color_frame
        image.width = 1
        image.height = 1
        image.step = 3
        image.encoding = "rgb8"
        image.data = bytes([0, 255, 0])
        depth = Image()
        depth.header.stamp = cloud.header.stamp
        depth.header.frame_id = color_frame
        depth.width = 1
        depth.height = 1
        depth.step = 2
        depth.encoding = "16UC1"
        depth.data = np.array([500], dtype="<u2").tobytes()
        bridge._on_color_image(image)
        np.testing.assert_array_equal(
            bridge.build_charuco_preview(),
            [[[0, 255, 0]]],
        )

        results = []
        errors = []

        def capture():
            try:
                results.append(bridge.capture_charuco(timeout_s=1.0))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=capture)
        thread.start()
        deadline = time.perf_counter() + 1.0
        while not bridge.status()["transform_burst_active"]:
            assert time.perf_counter() < deadline
            time.sleep(0.001)

        bridge._on_synchronized_sensor_frame(
            cloud,
            image,
            depth,
            camera_info,
        )
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert not errors
        assert results[0]["charuco_capture"]["corner_count"] == 42
        assert len(bridge._transform_publisher.messages) == 1
        assert len(bridge._tf_buffer.lookups) == 1
        target, source, _stamp, _timeout = bridge._tf_buffer.lookups[0]
        assert (target, source) == (color_frame, depth_frame)

        message = bridge._transform_publisher.messages[0]
        assert message.header.stamp.sec == 4
        assert message.header.stamp.nanosec == 5
        assert message.header.frame_id == "world"
        assert message.child_frame_id == depth_frame
        assert message.transformation_name == "charuco"
        expected_world_from_depth = world_from_color.copy()
        expected_world_from_depth[0, 3] += 0.02
        np.testing.assert_allclose(
            np.asarray(message.matrix).reshape(4, 4),
            expected_world_from_depth,
        )
        np.testing.assert_allclose(
            results[0]["charuco_capture"]["matrix"],
            expected_world_from_depth,
        )
        assert list(message.charuco_observation.corner_ids) == list(range(42))
        assert results[0]["charuco_capture"][
            "valid_depth_corner_count"
        ] == 42

        bridge._on_synchronized_sensor_frame(
            cloud,
            image,
            depth,
            camera_info,
        )
        assert len(bridge._transform_publisher.messages) == 1
    finally:
        if thread is not None:
            thread.join(timeout=1.0)
        bridge.destroy_node()
        rclpy.shutdown()


def test_charuco_capture_rejects_missing_depth_to_color_tf():
    class MissingTfBuffer:
        def lookup_transform(self, target, source, stamp, timeout):
            raise TransformException("transform is unavailable")

    rclpy.init()
    bridge = RosControlBridge(TRANSFORMATION_PATH)
    bridge._tf_buffer = MissingTfBuffer()
    try:
        cloud = PointCloud2()
        cloud.header.frame_id = "camera0_depth_optical_frame"
        image = Image()
        image.header.frame_id = "camera0_color_optical_frame"

        with pytest.raises(
            CharucoCalibrationError,
            match="Cannot transform point-cloud frame",
        ):
            bridge._color_from_pointcloud(cloud, image)
    finally:
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
    assert bridge.status()["transformation_mode"] == "charuco"
    bridge._transformation_mode = "json"
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
