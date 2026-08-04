"""Serve scanner controls and SQLite point data over Flask."""

import json
from pathlib import Path
import sqlite3
import threading
import time

from ament_index_python.packages import get_package_share_directory
from flask import Flask, jsonify, render_template, request, Response, send_file
import message_filters
import numpy as np
from object_scanner.pointcloud import transform_to_matrix
from object_scanner.sqlite_recording import validated_session_name
from object_scanner.transformations import (
    load_transformation_matrices,
    transformation_to_message,
    TransformationMatrix,
)
from object_scanner_interfaces.msg import NamedTransform
from object_scanner_processing.aligned_recording import (
    aligned_database_path,
    AlignedRecordingError,
    read_fused_cloud,
    source_revision,
)
from object_scanner_processing.charuco_observations import (
    annotate_charuco,
    build_charuco_observation,
    calibrate_charuco,
    CharucoCalibrationError,
    depth_image_to_meters,
)
from object_scanner_processing.repair_segmentation import segment_repair
from object_scanner_web.camera_frame import (
    build_camera_payload,
    build_rgb_payload,
    image_to_rgb,
)
from object_scanner_web.sqlite_reader import (
    build_array_payload,
    build_point_payload,
    build_replay_frame_payload,
    list_frames,
    PAYLOAD_HEADER,
    REPLAY_STAGES,
)
import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


HOST = "0.0.0.0"
PORT = 5000
REMOTE_PORT = 5001
REMOTE_DEMO_SESSION = "demo5"
REMOTE_COMMANDS = {
    "start_replay",
    "next",
    "show_loading",
    "stop_loading",
}
REMOTE_VIEWER_TIMEOUT_S = 2.0
MAX_DISPLAY_POINTS = 250_000
REPAIR_SESSION = "demo5"
REPAIR_STL_NAME = "pipe-testing08-repair(3).stl"
SERVICE_TIMEOUT_S = 5.0
ALIGNMENT_TIMEOUT_S = 600.0
CAMERA_TIMEOUT_S = 5.0
COLOR_TOPIC = "/realsense/camera0/color/image_raw"
COLOR_CAMERA_INFO_TOPIC = "/realsense/camera0/color/camera_info"
REGISTERED_DEPTH_TOPIC = (
    "/realsense/camera0/aligned_depth_to_color/image_raw"
)
POINTCLOUD_TOPIC = "/realsense/camera0/depth/color/points"
TRANSFORM_TOPIC = "/object_scanner/camera_to_world"
TRANSFORM_BURST_COUNT = 1
TRANSFORM_TIMEOUT_S = 5.0
TF_TIMEOUT_S = 0.5
SENSOR_SYNC_QUEUE_SIZE = 30
SENSOR_SYNC_SLOP_S = 0.05
TRANSFORMATION_MODES = {"json", "charuco"}
STATUS_SERVICE = "/object_scanner/recording_status"
SERVICE_NAMES = {
    "start": "/object_scanner/start_recording",
    "pause": "/object_scanner/pause_recording",
    "resume": "/object_scanner/resume_recording",
    "stop": "/object_scanner/stop_recording",
}


def _attach_charuco_observation(message, observation) -> None:
    target = message.charuco_observation
    target.corner_ids = observation.corner_ids.tolist()
    target.image_points = observation.image_points.reshape(-1).tolist()
    target.depth_valid = observation.depth_valid.tolist()
    target.child_points = observation.child_points.reshape(-1).tolist()
    target.depth_valid_pixel_counts = (
        observation.depth_valid_pixel_counts.tolist()
    )
    target.depth_inlier_pixel_counts = (
        observation.depth_inlier_pixel_counts.tolist()
    )
    target.depth_mad_m = observation.depth_mad_m.tolist()
    target.depth_invalid_reasons = list(observation.depth_invalid_reasons)
    target.camera_matrix = observation.camera_matrix.reshape(-1).tolist()
    target.distortion = observation.distortion.tolist()
    target.color_from_child = (
        observation.color_from_child.reshape(-1).tolist()
    )
    target.initial_reprojection_errors_px = (
        observation.initial_reprojection_errors_px.tolist()
    )
    target.initial_reprojection_rmse_px = (
        observation.initial_reprojection_rmse_px
    )


def validated_rgb(value) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or channel < 0
            or channel > 255
            for channel in value
        )
    ):
        raise ValueError("rgb must contain three integer values from 0 to 255")
    return list(value)


def validated_transformation_mode(value) -> str:
    if value not in TRANSFORMATION_MODES:
        raise ValueError("mode must be 'json' or 'charuco'")
    return value


class RemoteControl:
    """Share one pending command and viewer state between both Flask apps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_command_id = 1
        self._pending_command: dict | None = None
        self._last_viewer_report_ns: int | None = None
        self._last_error: str | None = None
        self._viewer_state = {
            "replay_mode": False,
            "replay_index": 0,
            "replay_total": 0,
            "can_next": False,
            "loading_visible": False,
            "busy": False,
            "stage": "raw",
        }

    def enqueue(self, command: str) -> dict:
        if command not in REMOTE_COMMANDS:
            raise ValueError(
                f"command must be one of {', '.join(sorted(REMOTE_COMMANDS))}"
            )
        with self._lock:
            if self._pending_command is not None:
                raise RuntimeError("Another remote command is still pending")
            pending = {
                "id": self._next_command_id,
                "command": command,
            }
            self._next_command_id += 1
            self._pending_command = pending
            self._last_error = None
            return dict(pending)

    def report_viewer(self, report: dict) -> dict:
        if not isinstance(report, dict):
            raise ValueError("viewer report must be a JSON object")

        boolean_fields = (
            "replay_mode",
            "can_next",
            "loading_visible",
            "busy",
        )
        for field in boolean_fields:
            if not isinstance(report.get(field), bool):
                raise ValueError(f"{field} must be a boolean")

        integer_fields = ("replay_index", "replay_total")
        for field in integer_fields:
            value = report.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")

        stage = report.get("stage")
        if stage not in REPLAY_STAGES:
            raise ValueError(
                f"stage must be one of {', '.join(sorted(REPLAY_STAGES))}"
            )

        completed_command_id = report.get("completed_command_id")
        if completed_command_id is not None and (
            isinstance(completed_command_id, bool)
            or not isinstance(completed_command_id, int)
            or completed_command_id < 1
        ):
            raise ValueError("completed_command_id must be a positive integer")

        error = report.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string")

        viewer_state = {
            field: report[field]
            for field in (*boolean_fields, *integer_fields)
        }
        viewer_state["stage"] = stage

        now_ns = time.perf_counter_ns()
        with self._lock:
            if completed_command_id is not None:
                if (
                    self._pending_command is None
                    or self._pending_command["id"] != completed_command_id
                ):
                    raise ValueError("completed command is not pending")
                self._pending_command = None
                self._last_error = error
            self._viewer_state = viewer_state
            self._last_viewer_report_ns = now_ns
            return self._status_unlocked(now_ns)

    def status(self) -> dict:
        with self._lock:
            return self._status_unlocked(time.perf_counter_ns())

    def _status_unlocked(self, now_ns: int) -> dict:
        connected = (
            self._last_viewer_report_ns is not None
            and now_ns - self._last_viewer_report_ns
            <= int(REMOTE_VIEWER_TIMEOUT_S * 1_000_000_000)
        )
        return {
            **self._viewer_state,
            "connected": connected,
            "demo_session": REMOTE_DEMO_SESSION,
            "pending_command": (
                None
                if self._pending_command is None
                else dict(self._pending_command)
            ),
            "last_error": self._last_error,
        }


class RosControlBridge(Node):
    """Call object_scanner services from Flask request threads."""

    def __init__(self, transformation_path: Path | None = None) -> None:
        super().__init__("object_scanner_web")
        self.declare_parameter("output_directory", "scans")
        self.declare_parameter("target_rgb", [179, 192, 187])
        self.output_directory = Path(
            self.get_parameter("output_directory").value
        ).expanduser()
        if transformation_path is None:
            transformation_path = (
                Path(get_package_share_directory("object_scanner"))
                / "resource"
                / "transformation_matrices.json"
            )
        self._transformations = load_transformation_matrices(
            transformation_path
        )
        self._transformation_mode = "charuco"
        self._transformation_index = 0
        self._active_transformation = None
        self._remaining_transform_messages = 0
        self._next_charuco_capture_id = 1
        self._charuco_capture_id: int | None = None
        self._charuco_result: dict | None = None
        self._charuco_error: Exception | None = None
        self._last_charuco: dict | None = None
        self._state = "stopped"
        self._database_path: str | None = None
        self._session_name: str | None = None
        self._target_rgb = validated_rgb(
            self.get_parameter("target_rgb").value
        )
        self._lock = threading.Lock()
        self._transform_condition = threading.Condition()
        self._service_clients = {
            command: self.create_client(Trigger, service_name)
            for command, service_name in SERVICE_NAMES.items()
        }
        self._status_client = self.create_client(Trigger, STATUS_SERVICE)
        self._parameter_client = AsyncParameterClient(
            self,
            "/object_scanner",
        )
        self._image_condition = threading.Condition()
        self._capture_armed = False
        self._captured_image: Image | None = None
        self._latest_color_image: Image | None = None
        self._image_subscriber = message_filters.Subscriber(
            self,
            Image,
            COLOR_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )
        self._image_subscriber.registerCallback(self._on_color_image)
        self._pointcloud_subscriber = message_filters.Subscriber(
            self,
            PointCloud2,
            POINTCLOUD_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )
        self._pointcloud_subscriber.registerCallback(self._on_pointcloud)
        self._depth_subscriber = message_filters.Subscriber(
            self,
            Image,
            REGISTERED_DEPTH_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )
        self._camera_info_subscriber = message_filters.Subscriber(
            self,
            CameraInfo,
            COLOR_CAMERA_INFO_TOPIC,
            qos_profile=QoSProfile(depth=1),
        )
        self._sensor_synchronizer = (
            message_filters.ApproximateTimeSynchronizer(
                [
                    self._pointcloud_subscriber,
                    self._image_subscriber,
                    self._depth_subscriber,
                    self._camera_info_subscriber,
                ],
                queue_size=SENSOR_SYNC_QUEUE_SIZE,
                slop=SENSOR_SYNC_SLOP_S,
            )
        )
        self._sensor_synchronizer.registerCallback(
            self._on_synchronized_sensor_frame
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._transform_publisher = self.create_publisher(
            NamedTransform,
            TRANSFORM_TOPIC,
            qos_profile_sensor_data,
        )

    def status(self) -> dict:
        with self._lock:
            return self._status_unlocked()

    def refresh_status(self) -> dict:
        with self._lock:
            self._refresh_scanner_status_unlocked()
            return self._status_unlocked()

    def command(
        self,
        command: str,
        session_name: str | None = None,
    ) -> dict:
        if command not in self._service_clients:
            raise ValueError(f"Unknown recording command: {command}")
        if command == "start":
            session_name = validated_session_name(session_name)

        with self._lock:
            self._refresh_scanner_status_unlocked()
            if command == "start":
                if self._state != "stopped":
                    result = self._status_unlocked()
                    result.update(
                        success=False,
                        message=f"Recorder is already {self._state}",
                    )
                    return result
                parameter_result = self._set_scanner_parameter_unlocked(
                    Parameter("session_name", value=session_name)
                )
                if not parameter_result.successful:
                    result = self._status_unlocked()
                    result.update(
                        success=False,
                        message=parameter_result.reason,
                    )
                    return result
                self._session_name = session_name

            client = self._service_clients[command]
            if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
                raise TimeoutError(f"{SERVICE_NAMES[command]} is unavailable")

            future = client.call_async(Trigger.Request())
            response = self._wait_for_future(
                future,
                SERVICE_NAMES[command],
                (
                    ALIGNMENT_TIMEOUT_S
                    if command in {"pause", "stop"}
                    else SERVICE_TIMEOUT_S
                ),
            )
            if response is None:
                raise RuntimeError(
                    f"{SERVICE_NAMES[command]} returned no response"
                )

            if response.success:
                if command == "start":
                    self._state = "recording"
                    self._database_path = response.message
                elif command == "pause":
                    self._state = "paused"
                elif command == "resume":
                    self._state = "recording"
                elif command == "stop":
                    self._state = "stopped"
            elif command in {"pause", "stop"}:
                self._refresh_scanner_status_unlocked()

            result = self._status_unlocked()
            result.update(
                success=bool(response.success),
                message=response.message,
            )
            return result

    def set_reference_color(self, rgb: list[int]) -> dict:
        color = validated_rgb(rgb)
        with self._lock:
            self._refresh_scanner_status_unlocked()
            if self._state != "stopped":
                result = self._status_unlocked()
                result.update(
                    success=False,
                    message="Reference color can change only while stopped",
                )
                return result
            parameter_result = self._set_scanner_parameter_unlocked(
                Parameter("target_rgb", value=color)
            )
            if parameter_result.successful:
                self._target_rgb = color
            result = self._status_unlocked()
            result.update(
                success=bool(parameter_result.successful),
                message=parameter_result.reason or "Reference color updated",
            )
            return result

    def set_transformation_mode(self, mode: str) -> dict:
        selected_mode = validated_transformation_mode(mode)
        with self._lock:
            self._refresh_scanner_status_unlocked()
            if self._state != "stopped":
                result = self._status_unlocked()
                result.update(
                    success=False,
                    message="Transformation mode can change only while stopped",
                )
                return result
            with self._transform_condition:
                transformation_active = self._transformation_active_unlocked()
                if not transformation_active:
                    self._transformation_mode = selected_mode
            result = self._status_unlocked()
            if transformation_active:
                result.update(
                    success=False,
                    message="A transformation capture is already active",
                )
                return result
            result.update(
                success=True,
                message=f"Transformation mode set to {selected_mode}",
            )
            return result

    def _set_scanner_parameter_unlocked(
        self,
        parameter: Parameter,
    ):
        if not self._parameter_client.wait_for_services(
            timeout_sec=SERVICE_TIMEOUT_S
        ):
            raise TimeoutError(
                "/object_scanner parameter services are unavailable"
            )

        future = self._parameter_client.set_parameters([parameter])
        response = self._wait_for_future(
            future,
            "/object_scanner/set_parameters",
        )
        if response is None or not response.results:
            raise RuntimeError(
                "/object_scanner/set_parameters returned no result"
            )
        return response.results[0]

    def _refresh_scanner_status_unlocked(self) -> None:
        if not self._status_client.wait_for_service(
            timeout_sec=SERVICE_TIMEOUT_S
        ):
            raise TimeoutError(f"{STATUS_SERVICE} is unavailable")

        future = self._status_client.call_async(Trigger.Request())
        response = self._wait_for_future(future, STATUS_SERVICE)
        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            raise RuntimeError(f"{STATUS_SERVICE} failed: {message}")

        try:
            status = json.loads(response.message)
            state = status["state"]
            if state not in {"stopped", "recording", "paused"}:
                raise ValueError(f"invalid state: {state}")
            database_path = status["database_path"]
            if database_path is not None and not isinstance(database_path, str):
                raise ValueError("database_path must be a string or null")
            session_name = status["session_name"]
            if session_name is not None:
                session_name = validated_session_name(session_name)
            target_rgb = validated_rgb(status["target_rgb"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{STATUS_SERVICE} returned invalid status: {error}"
            ) from error

        self._state = state
        self._database_path = database_path
        self._session_name = session_name
        self._target_rgb = target_rgb

    def publish_transformation(
        self,
        timeout_s: float = TRANSFORM_TIMEOUT_S,
    ) -> dict:
        """Publish the displayed matrix for the next point cloud."""
        with self._transform_condition:
            if self._transformation_mode != "json":
                raise RuntimeError(
                    "JSON transformation publishing is unavailable "
                    "in ChArUco mode"
                )
            if self._transformation_active_unlocked():
                raise RuntimeError("A transformation burst is already active")

            transformation = self._current_transformation_unlocked()
            self._active_transformation = transformation
            self._remaining_transform_messages = TRANSFORM_BURST_COUNT
            deadline = time.perf_counter() + timeout_s
            while self._remaining_transform_messages:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    self._active_transformation = None
                    self._remaining_transform_messages = 0
                    raise TimeoutError(
                        f"Did not receive a point cloud from {POINTCLOUD_TOPIC}"
                    )
                self._transform_condition.wait(timeout=remaining)

            self._transformation_index = (
                self._transformation_index + 1
            ) % len(self._transformations)

        result = self.status()
        result.update(
            success=True,
            message=(
                f"Published '{transformation.name}' for one point cloud"
            ),
            published_transformation=transformation.as_dict(),
        )
        return result

    def step_transformation(self, delta: int) -> dict:
        if isinstance(delta, bool) or delta not in {-1, 1}:
            raise ValueError("delta must be -1 or 1")
        with self._transform_condition:
            if self._transformation_mode != "json":
                raise RuntimeError(
                    "JSON transformation selection is unavailable "
                    "in ChArUco mode"
                )
            if self._transformation_active_unlocked():
                raise RuntimeError("A transformation burst is already active")
            self._transformation_index = (
                self._transformation_index + delta
            ) % len(self._transformations)
        return self.status()

    def capture_camera_frame(self) -> Image:
        with self._lock:
            if self._state != "stopped":
                raise RuntimeError(
                    "Camera color capture is available only while stopped"
                )
            with self._image_condition:
                self._captured_image = None
                self._capture_armed = True
                deadline = time.perf_counter() + CAMERA_TIMEOUT_S
                while self._captured_image is None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        self._capture_armed = False
                        raise TimeoutError(
                            f"No RGB image received from {COLOR_TOPIC}"
                        )
                    self._image_condition.wait(timeout=remaining)
                return self._captured_image

    def _on_color_image(self, message: Image) -> None:
        with self._image_condition:
            self._latest_color_image = message
            if not self._capture_armed:
                return
            self._captured_image = message
            self._capture_armed = False
            self._image_condition.notify_all()

    def build_charuco_preview(self) -> np.ndarray:
        with self._transform_condition:
            if self._transformation_mode != "charuco":
                raise RuntimeError(
                    "ChArUco preview is unavailable in JSON mode"
                )
        with self._image_condition:
            message = self._latest_color_image
        if message is None:
            raise TimeoutError(f"No RGB image received from {COLOR_TOPIC}")
        return annotate_charuco(image_to_rgb(message))

    def capture_charuco(
        self,
        timeout_s: float = TRANSFORM_TIMEOUT_S,
    ) -> dict:
        with self._lock:
            self._refresh_scanner_status_unlocked()
            if self._state != "recording":
                raise RuntimeError(
                    "ChArUco capture is available only while recording"
                )

        with self._transform_condition:
            if self._transformation_mode != "charuco":
                raise RuntimeError(
                    "ChArUco capture is unavailable in JSON mode"
                )
            if self._transformation_active_unlocked():
                raise RuntimeError("A transformation capture is already active")

            capture_id = self._next_charuco_capture_id
            self._next_charuco_capture_id += 1
            self._charuco_capture_id = capture_id
            self._charuco_result = None
            self._charuco_error = None
            deadline = time.perf_counter() + timeout_s
            while (
                self._charuco_capture_id == capture_id
                and self._charuco_result is None
                and self._charuco_error is None
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    self._charuco_capture_id = None
                    raise TimeoutError(
                        "Did not receive synchronized RGB, registered depth, "
                        f"CameraInfo, and point cloud within {timeout_s:.1f} "
                        "seconds"
                    )
                self._transform_condition.wait(timeout=remaining)

            if self._charuco_error is not None:
                error = self._charuco_error
                self._charuco_error = None
                raise error
            capture = self._charuco_result
            self._charuco_result = None

        assert capture is not None
        result = self.status()
        result.update(
            success=True,
            message="Captured one ChArUco-calibrated point cloud",
            charuco_capture=capture,
        )
        return result

    def _on_pointcloud(self, message: PointCloud2) -> None:
        with self._transform_condition:
            transformation = self._active_transformation
            if transformation is None:
                return
            if not message.header.frame_id:
                self.get_logger().error(
                    "Cannot publish transformation for a point cloud "
                    "with an empty frame"
                )
                return

            transform_message = transformation_to_message(
                transformation,
                message.header.stamp,
                message.header.frame_id,
            )
            self._transform_publisher.publish(transform_message)
            self._remaining_transform_messages -= 1
            if self._remaining_transform_messages == 0:
                self._active_transformation = None
                self._transform_condition.notify_all()

    def _on_synchronized_sensor_frame(
        self,
        pointcloud: PointCloud2,
        color_image: Image,
        registered_depth: Image,
        camera_info: CameraInfo,
    ) -> None:
        with self._transform_condition:
            capture_id = self._charuco_capture_id
        if capture_id is None:
            return

        try:
            calibration, depth_m = self._calibrate_sensor_frame(
                pointcloud,
                color_image,
                registered_depth,
                camera_info,
            )
            color_from_pointcloud = self._color_from_pointcloud(
                pointcloud,
                color_image,
            )
            world_from_pointcloud = (
                calibration.camera_to_world @ color_from_pointcloud
            )
            observation = build_charuco_observation(
                calibration,
                depth_m,
                np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3),
                np.asarray(camera_info.d, dtype=np.float64),
                color_from_pointcloud,
            )
            transformation = TransformationMatrix(
                name="charuco",
                parent_frame_id="world",
                matrix=tuple(
                    tuple(float(value) for value in row)
                    for row in world_from_pointcloud
                ),
            )
            transform_message = transformation_to_message(
                transformation,
                pointcloud.header.stamp,
                pointcloud.header.frame_id,
            )
            _attach_charuco_observation(transform_message, observation)
            capture = observation.as_dict()
            capture.update(
                success=True,
                message="ChArUco pose accepted",
                matrix=world_from_pointcloud.tolist(),
            )
        except (CharucoCalibrationError, ValueError) as error:
            if isinstance(error, CharucoCalibrationError):
                capture_error = error
            else:
                capture_error = CharucoCalibrationError(str(error))
            with self._transform_condition:
                if self._charuco_capture_id != capture_id:
                    return
                self._charuco_error = capture_error
                self._last_charuco = capture_error.as_dict()
                self._charuco_capture_id = None
                self._transform_condition.notify_all()
            return

        with self._transform_condition:
            if self._charuco_capture_id != capture_id:
                return
            self._transform_publisher.publish(transform_message)
            self._charuco_result = capture
            self._last_charuco = capture
            self._charuco_capture_id = None
            self._transform_condition.notify_all()

    @staticmethod
    def _calibrate_sensor_frame(
        pointcloud: PointCloud2,
        color_image: Image,
        registered_depth: Image,
        camera_info: CameraInfo | None,
    ):
        if camera_info is None:
            raise CharucoCalibrationError(
                f"No camera intrinsics received from {COLOR_CAMERA_INFO_TOPIC}"
            )
        pointcloud_frame = pointcloud.header.frame_id
        if not pointcloud_frame:
            raise CharucoCalibrationError("Point-cloud frame is empty")
        color_frame = color_image.header.frame_id
        if not color_frame:
            raise CharucoCalibrationError("RGB image frame is empty")
        if camera_info.header.frame_id != color_frame:
            raise CharucoCalibrationError(
                "CameraInfo frame "
                f"'{camera_info.header.frame_id}' does not match RGB image "
                f"frame '{color_frame}'"
            )
        if (
            camera_info.width != color_image.width
            or camera_info.height != color_image.height
        ):
            raise CharucoCalibrationError(
                "CameraInfo dimensions do not match the RGB image"
            )
        if not registered_depth.header.frame_id:
            raise CharucoCalibrationError(
                "Registered-depth image frame is empty"
            )
        if (
            registered_depth.width != color_image.width
            or registered_depth.height != color_image.height
        ):
            raise CharucoCalibrationError(
                "Registered-depth dimensions do not match the RGB image"
            )

        calibration = calibrate_charuco(
            image_to_rgb(color_image),
            np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3),
            np.asarray(camera_info.d, dtype=np.float64),
        )
        depth_m = depth_image_to_meters(
            bytes(registered_depth.data),
            width=registered_depth.width,
            height=registered_depth.height,
            step=registered_depth.step,
            encoding=registered_depth.encoding,
            is_bigendian=registered_depth.is_bigendian,
        )
        return calibration, depth_m

    def _color_from_pointcloud(
        self,
        pointcloud: PointCloud2,
        color_image: Image,
    ) -> np.ndarray:
        pointcloud_frame = pointcloud.header.frame_id
        color_frame = color_image.header.frame_id
        if pointcloud_frame == color_frame:
            return np.eye(4, dtype=np.float64)

        try:
            color_from_pointcloud_message = self._tf_buffer.lookup_transform(
                color_frame,
                pointcloud_frame,
                Time.from_msg(pointcloud.header.stamp),
                timeout=Duration(seconds=TF_TIMEOUT_S),
            )
        except TransformException as error:
            raise CharucoCalibrationError(
                "Cannot transform point-cloud frame "
                f"'{pointcloud_frame}' into RGB frame '{color_frame}': {error}"
            ) from error

        try:
            color_from_pointcloud = transform_to_matrix(
                color_from_pointcloud_message
            )
        except ValueError as error:
            raise CharucoCalibrationError(
                f"Invalid RealSense depth-to-color transform: {error}"
            ) from error
        return color_from_pointcloud

    @staticmethod
    def _wait_for_future(
        future,
        operation: str,
        timeout_s: float = SERVICE_TIMEOUT_S,
    ):
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        deadline = time.perf_counter() + timeout_s
        while not completed.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"{operation} did not respond")
            completed.wait(timeout=min(remaining, 0.1))
        try:
            return future.result()
        except Exception as error:
            raise RuntimeError(f"{operation} failed: {error}") from error

    def _status_unlocked(self) -> dict:
        result = {
            "state": self._state,
            "database_path": self._database_path,
            "session_name": self._session_name,
            "target_rgb": list(self._target_rgb),
            "transformation_mode": self._transformation_mode,
        }
        with self._transform_condition:
            result.update(
                transformation=(
                    self._current_transformation_unlocked().as_dict()
                ),
                transformation_index=self._transformation_index + 1,
                transformation_total=len(self._transformations),
                transform_burst_active=self._transformation_active_unlocked(),
                last_charuco=self._last_charuco,
            )
        return result

    def _transformation_active_unlocked(self) -> bool:
        return bool(
            self._remaining_transform_messages
            or self._charuco_capture_id is not None
        )

    def _current_transformation_unlocked(self):
        return self._transformations[self._transformation_index]


def create_app(
    bridge,
    share_directory: Path | None = None,
    output_directory: Path | None = None,
    remote_control: RemoteControl | None = None,
    repair_stl_path: Path | None = None,
) -> Flask:
    """Create the Flask API around a ROS control bridge."""
    if share_directory is None:
        share_directory = Path(
            get_package_share_directory("object_scanner_web")
        )
    app = Flask(
        __name__,
        template_folder=str(share_directory / "templates"),
        static_folder=str(share_directory / "static"),
    )
    if output_directory is None:
        output_directory = Path(bridge.output_directory)
    if remote_control is None:
        remote_control = RemoteControl()
    if repair_stl_path is None:
        repair_stl_path = share_directory / "repair" / REPAIR_STL_NAME
    repair_stl_path = Path(repair_stl_path)

    def session_database_path(session_name: str) -> Path:
        name = validated_session_name(session_name)
        session_directory = output_directory / name
        database_path = session_directory / "recording.sqlite3"
        metadata_path = session_directory / "metadata.json"
        if not database_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Saved session does not exist: {name}")
        return database_path

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/remote/command")
    def remote_command():
        current = remote_control.status()
        return jsonify(
            command=current["pending_command"],
            demo_session=current["demo_session"],
            loading_visible=current["loading_visible"],
        )

    @app.post("/api/remote/viewer")
    def remote_viewer():
        try:
            current = remote_control.report_viewer(
                request.get_json(silent=True)
            )
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        return jsonify(current)

    @app.get("/api/status")
    def status():
        try:
            current = bridge.refresh_status()
        except TimeoutError as error:
            app.logger.exception("Cannot refresh scanner status: %s", error)
            return jsonify(success=False, message=str(error)), 503
        except RuntimeError as error:
            app.logger.exception("Cannot refresh scanner status: %s", error)
            return jsonify(success=False, message=str(error)), 502
        return jsonify(current)

    @app.get("/api/sessions")
    def sessions():
        saved_sessions = []
        if output_directory.is_dir():
            for session_directory in output_directory.iterdir():
                if not session_directory.is_dir():
                    continue
                try:
                    name = validated_session_name(session_directory.name)
                    session_database_path(name)
                except (FileNotFoundError, ValueError):
                    continue
                saved_sessions.append(
                    {
                        "name": name,
                    }
                )
        saved_sessions.sort(key=lambda session: session["name"])
        return jsonify(sessions=saved_sessions)

    @app.get("/api/repair/reference")
    def repair_reference():
        if not repair_stl_path.is_file():
            return (
                jsonify(success=False, message="Repair reference STL not found"),
                404,
            )
        response = send_file(repair_stl_path, mimetype="model/stl")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/sessions/<session_name>/repair-analysis")
    def repair_analysis(session_name):
        try:
            if session_name != REPAIR_SESSION:
                raise FileNotFoundError(
                    f"Repair analysis is available only for '{REPAIR_SESSION}'"
                )
            if bridge.status()["state"] != "stopped":
                return (
                    jsonify(
                        success=False,
                        message="Stop recording before repair analysis",
                    ),
                    409,
                )
            database_path = session_database_path(session_name)
            aligned_path = aligned_database_path(database_path)
            if not aligned_path.is_file():
                raise FileNotFoundError(
                    f"Aligned recording not found for session '{session_name}'"
                )
            body = request.get_json(silent=True)
            if not isinstance(body, dict) or "transform" not in body:
                raise ValueError("JSON body must contain transform")
            transform = body["transform"]
            result = segment_repair(
                aligned_path,
                repair_stl_path,
                None if transform is None else np.asarray(transform),
            )
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        except FileNotFoundError as error:
            return jsonify(success=False, message=str(error)), 404
        except (AlignedRecordingError, RuntimeError, sqlite3.Error) as error:
            app.logger.exception("Repair analysis failed: %s", error)
            return jsonify(success=False, message=str(error)), 500
        return jsonify(
            success=True,
            transform=result.transform.tolist(),
            points=result.xyz.reshape(-1).tolist(),
            point_count=len(result.xyz),
            scale_m_per_stl_unit=result.scale_m_per_stl_unit,
        )

    @app.get("/api/sessions/<session_name>/frames")
    def session_frames(session_name):
        try:
            current = bridge.refresh_status()
            if current["state"] != "stopped":
                return (
                    jsonify(
                        success=False,
                        message="Stop recording before replay",
                    ),
                    409,
                )
            database_path = session_database_path(session_name)
            frames = list_frames(database_path)
            stages = ["raw"]
            aligned_path = aligned_database_path(database_path)
            if aligned_path.is_file():
                aligned_frames = {
                    frame["id"]: frame for frame in list_frames(aligned_path)
                }
                if set(aligned_frames) == {frame["id"] for frame in frames}:
                    for frame in frames:
                        frame["optimized_matrix"] = aligned_frames[
                            frame["id"]
                        ]["matrix"]
                    stages.extend(("filtered", "aligned"))
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        except FileNotFoundError as error:
            return jsonify(success=False, message=str(error)), 404
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 503
        except (RuntimeError, sqlite3.Error) as error:
            return jsonify(success=False, message=str(error)), 500
        return jsonify(
            session_name=session_name,
            frames=frames,
            total_frames=len(frames),
            total_points=sum(frame["point_count"] for frame in frames),
            stages=stages,
        )

    @app.get("/api/sessions/<session_name>/frames/<int:frame_id>")
    def session_frame(session_name, frame_id):
        try:
            current = bridge.status()
            if current["state"] != "stopped":
                return (
                    jsonify(
                        success=False,
                        message="Stop recording before replay",
                    ),
                    409,
                )
            database_path = session_database_path(session_name)
            stage = request.args.get("stage", "raw")
            if stage not in REPLAY_STAGES:
                raise ValueError(
                    f"stage must be one of {', '.join(sorted(REPLAY_STAGES))}"
                )
            max_points_value = request.args.get("max_points")
            max_points = (
                MAX_DISPLAY_POINTS
                if max_points_value is None
                else int(max_points_value)
            )
            if max_points < 1 or max_points > MAX_DISPLAY_POINTS:
                raise ValueError(
                    f"max_points must be from 1 to {MAX_DISPLAY_POINTS}"
                )
            aligned_path = aligned_database_path(database_path)
            if stage != "raw" and not aligned_path.is_file():
                raise FileNotFoundError(
                    f"Aligned recording not found for session '{session_name}'"
                )
            payload = build_replay_frame_payload(
                database_path,
                aligned_path,
                frame_id,
                max_points,
                stage,
            )
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        except (FileNotFoundError, LookupError) as error:
            return jsonify(success=False, message=str(error)), 404
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 503
        except (RuntimeError, sqlite3.Error) as error:
            return jsonify(success=False, message=str(error)), 500

        displayed_points, total_points = PAYLOAD_HEADER.unpack_from(payload)[1:]
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Displayed-Points"] = str(displayed_points)
        response.headers["X-Total-Points"] = str(total_points)
        return response

    @app.post("/api/reference-color")
    def reference_color():
        body = request.get_json(silent=True)
        try:
            rgb = validated_rgb(body.get("rgb") if body else None)
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        try:
            result = bridge.set_reference_color(rgb)
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 503
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 502
        return jsonify(result), 200 if result["success"] else 409

    @app.get("/api/camera-frame")
    def camera_frame():
        try:
            message = bridge.capture_camera_frame()
            payload = build_camera_payload(message)
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 504
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 409
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 500

        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Image-Width"] = str(message.width)
        response.headers["X-Image-Height"] = str(message.height)
        return response

    @app.post("/api/transformation/publish")
    def publish_transformation():
        try:
            result = bridge.publish_transformation()
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 504
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 409
        return jsonify(result)

    @app.post("/api/transformation/mode")
    def transformation_mode():
        body = request.get_json(silent=True)
        try:
            mode = validated_transformation_mode(
                body.get("mode") if body else None
            )
            result = bridge.set_transformation_mode(mode)
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 503
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 502
        return jsonify(result), 200 if result["success"] else 409

    @app.post("/api/transformation/step")
    def step_transformation():
        body = request.get_json(silent=True)
        try:
            delta = body.get("delta") if body else None
            result = bridge.step_transformation(delta)
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 409
        return jsonify(result)

    @app.post("/api/charuco/capture")
    def capture_charuco():
        try:
            result = bridge.capture_charuco()
        except CharucoCalibrationError as error:
            return jsonify(error.as_dict()), 422
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 504
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 409
        return jsonify(result)

    @app.get("/api/charuco/preview")
    def charuco_preview():
        try:
            preview = bridge.build_charuco_preview()
            payload = build_rgb_payload(preview)
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 503
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 409
        except (CharucoCalibrationError, ValueError) as error:
            return jsonify(success=False, message=str(error)), 500

        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Image-Width"] = str(preview.shape[1])
        response.headers["X-Image-Height"] = str(preview.shape[0])
        return response

    @app.post("/api/recording/<command>")
    def recording_command(command):
        if command not in SERVICE_NAMES:
            return jsonify(success=False, message="Unknown command"), 404
        session_name = None
        if command == "start":
            body = request.get_json(silent=True)
            try:
                session_name = validated_session_name(
                    body.get("session_name") if body else None
                )
            except ValueError as error:
                return jsonify(success=False, message=str(error)), 400
        try:
            result = bridge.command(command, session_name)
        except TimeoutError as error:
            app.logger.exception(
                "Recording command '%s' timed out: %s",
                command,
                error,
            )
            return jsonify(success=False, message=str(error)), 503
        except RuntimeError as error:
            app.logger.exception(
                "Recording command '%s' failed: %s",
                command,
                error,
            )
            return jsonify(success=False, message=str(error)), 502
        if not result["success"]:
            app.logger.error(
                "Recording command '%s' rejected: %s",
                command,
                result["message"],
                stack_info=True,
            )
        return jsonify(result), 200 if result["success"] else 409

    @app.get("/api/points")
    def points():
        source = request.args.get("source", "aligned")
        if source not in {"aligned", "raw"}:
            return (
                jsonify(
                    success=False,
                    message="source must be 'aligned' or 'raw'",
                ),
                400,
            )

        current = bridge.status()
        if current["state"] not in {"paused", "stopped"}:
            return (
                jsonify(
                    success=False,
                    message="Pause recording before loading points",
                ),
                409,
            )
        if not current["database_path"]:
            return (
                jsonify(success=False, message="No recording is available"),
                404,
            )

        database_path = Path(current["database_path"])
        if not database_path.is_file():
            return (
                jsonify(success=False, message="Recording database not found"),
                404,
            )

        if source == "raw":
            try:
                payload = build_point_payload(
                    database_path,
                    MAX_DISPLAY_POINTS,
                )
            except (sqlite3.Error, ValueError) as error:
                return (
                    jsonify(
                        success=False,
                        message=f"Cannot load raw points: {error}",
                    ),
                    500,
                )

            displayed_points, total_points = PAYLOAD_HEADER.unpack_from(
                payload
            )[1:]
            response = Response(payload, mimetype="application/octet-stream")
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Displayed-Points"] = str(displayed_points)
            response.headers["X-Total-Points"] = str(total_points)
            response.headers["X-Preview-Source"] = "raw"
            return response

        try:
            revision = source_revision(database_path)
            result = read_fused_cloud(
                aligned_database_path(database_path),
                revision,
            )
            stride = max(
                1,
                (len(result.xyz) + MAX_DISPLAY_POINTS - 1)
                // MAX_DISPLAY_POINTS,
            )
            payload = build_array_payload(
                result.xyz[::stride],
                result.rgb[::stride],
                len(result.xyz),
            )
        except AlignedRecordingError as error:
            app.logger.error(
                "Cannot load aligned point cloud for '%s': %s",
                database_path,
                error,
            )
            return (
                jsonify(
                    success=False,
                    message=f"Cannot load aligned points: {error}",
                ),
                422,
            )
        except (sqlite3.Error, ValueError) as error:
            return (
                jsonify(
                    success=False,
                    message=f"Cannot process points: {error}",
                ),
                500,
            )

        displayed_points, total_points = PAYLOAD_HEADER.unpack_from(payload)[1:]
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Displayed-Points"] = str(displayed_points)
        response.headers["X-Total-Points"] = str(total_points)
        response.headers["X-Raw-Points"] = str(result.raw_points)
        response.headers["X-Cleaned-Points"] = str(result.cleaned_points)
        response.headers["X-Accepted-Edges"] = str(result.accepted_edges)
        response.headers["X-Rejected-Edges"] = str(result.rejected_edges)
        response.headers["X-Charuco-Frames"] = str(
            result.charuco_frame_count
        )
        if getattr(result, "quality_warning", None):
            response.headers["X-Processing-Warning"] = result.quality_warning
        if result.charuco_reprojection_max_px is not None:
            response.headers["X-Charuco-Max-Reprojection-Px"] = str(
                result.charuco_reprojection_max_px
            )
            response.headers["X-Cloud-3mm-Fraction"] = str(
                result.cloud_overlap_fraction_3mm
            )
        return response

    return app


def create_remote_app(
    remote_control: RemoteControl,
    share_directory: Path | None = None,
) -> Flask:
    """Create the small remote-control page served on port 5001."""
    if share_directory is None:
        share_directory = Path(
            get_package_share_directory("object_scanner_web")
        )
    app = Flask(
        f"{__name__}_remote",
        template_folder=str(share_directory / "templates"),
        static_folder=str(share_directory / "static"),
    )

    @app.get("/remote")
    def remote():
        return render_template(
            "remote.html",
            demo_session=REMOTE_DEMO_SESSION,
        )

    @app.get("/api/status")
    def remote_status():
        return jsonify(remote_control.status())

    @app.post("/api/command")
    def remote_command():
        body = request.get_json(silent=True)
        try:
            pending = remote_control.enqueue(
                body.get("command") if isinstance(body, dict) else None
            )
        except ValueError as error:
            return jsonify(success=False, message=str(error)), 400
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 409
        return jsonify(success=True, pending_command=pending), 202

    return app


def main(args=None) -> None:
    rclpy.init(args=args)
    bridge = RosControlBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor_thread = threading.Thread(
        target=executor.spin,
        name="object-scanner-web-ros",
        daemon=True,
    )
    executor_thread.start()
    remote_control = RemoteControl()
    app = create_app(bridge, remote_control=remote_control)
    remote_app = create_remote_app(remote_control)
    remote_thread = threading.Thread(
        target=lambda: remote_app.run(
            host=HOST,
            port=REMOTE_PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        ),
        name="object-scanner-web-remote",
        daemon=True,
    )
    remote_thread.start()
    try:
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    finally:
        executor.shutdown()
        executor_thread.join()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
