"""Serve scanner controls and SQLite point data over Flask."""

import json
from pathlib import Path
import sqlite3
import threading
import time

from ament_index_python.packages import get_package_share_directory
from flask import Flask, jsonify, render_template, request, Response
from object_scanner.sqlite_recording import validated_session_name
from object_scanner.transformations import (
    load_transformation_matrices,
    transformation_to_message,
)
from object_scanner_interfaces.msg import NamedTransform
from object_scanner_web.camera_frame import build_camera_payload
from object_scanner_web.sqlite_reader import (
    build_frame_payload,
    build_point_payload,
    list_frames,
    PAYLOAD_HEADER,
)
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger


HOST = "0.0.0.0"
PORT = 5000
MAX_DISPLAY_POINTS = 250_000
SERVICE_TIMEOUT_S = 5.0
CAMERA_TIMEOUT_S = 5.0
COLOR_TOPIC = "/realsense/camera0/color/image_raw"
POINTCLOUD_TOPIC = "/realsense/camera0/depth/color/points"
TRANSFORM_TOPIC = "/object_scanner/camera_to_world"
TRANSFORM_BURST_COUNT = 1
TRANSFORM_TIMEOUT_S = 5.0
STATUS_SERVICE = "/object_scanner/recording_status"
SERVICE_NAMES = {
    "start": "/object_scanner/start_recording",
    "pause": "/object_scanner/pause_recording",
    "resume": "/object_scanner/resume_recording",
    "stop": "/object_scanner/stop_recording",
}


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


class RosControlBridge(Node):
    """Call object_scanner services from Flask request threads."""

    def __init__(self, transformation_path: Path | None = None) -> None:
        super().__init__("object_scanner_web")
        self.declare_parameter("output_directory", "scans")
        self.declare_parameter("target_rgb", [0, 255, 0])
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
        self._transformation_index = 0
        self._active_transformation = None
        self._remaining_transform_messages = 0
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
        self._image_subscription = self.create_subscription(
            Image,
            COLOR_TOPIC,
            self._on_color_image,
            qos_profile_sensor_data,
        )
        self._transform_publisher = self.create_publisher(
            NamedTransform,
            TRANSFORM_TOPIC,
            qos_profile_sensor_data,
        )
        self._pointcloud_subscription = self.create_subscription(
            PointCloud2,
            POINTCLOUD_TOPIC,
            self._on_pointcloud,
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
            if self._remaining_transform_messages:
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
            if self._remaining_transform_messages:
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
            if not self._capture_armed:
                return
            self._captured_image = message
            self._capture_armed = False
            self._image_condition.notify_all()

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

    @staticmethod
    def _wait_for_future(future, operation: str):
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        deadline = time.perf_counter() + SERVICE_TIMEOUT_S
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
        }
        with self._transform_condition:
            result.update(
                transformation=(
                    self._current_transformation_unlocked().as_dict()
                ),
                transformation_index=self._transformation_index + 1,
                transformation_total=len(self._transformations),
                transform_burst_active=bool(
                    self._remaining_transform_messages
                ),
            )
        return result

    def _current_transformation_unlocked(self):
        return self._transformations[self._transformation_index]


def create_app(
    bridge,
    share_directory: Path | None = None,
    output_directory: Path | None = None,
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
            payload = build_frame_payload(
                database_path,
                frame_id,
                max_points,
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

        try:
            payload = build_point_payload(
                database_path,
                MAX_DISPLAY_POINTS,
            )
        except (sqlite3.Error, ValueError) as error:
            return (
                jsonify(success=False, message=f"Cannot read points: {error}"),
                500,
            )

        displayed_points, total_points = PAYLOAD_HEADER.unpack_from(payload)[1:]
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Displayed-Points"] = str(displayed_points)
        response.headers["X-Total-Points"] = str(total_points)
        return response

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
    app = create_app(bridge)
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
