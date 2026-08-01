"""Serve scanner controls and SQLite point data over Flask."""

from pathlib import Path
import sqlite3
import threading
import time

from ament_index_python.packages import get_package_share_directory
from flask import Flask, jsonify, render_template, request, Response
from object_scanner_web.camera_frame import build_camera_payload
from object_scanner_web.sqlite_reader import build_point_payload, PAYLOAD_HEADER
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


HOST = "0.0.0.0"
PORT = 5000
MAX_DISPLAY_POINTS = 250_000
SERVICE_TIMEOUT_S = 5.0
CAMERA_TIMEOUT_S = 5.0
COLOR_TOPIC = "/realsense/camera0/color/image_raw"
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

    def __init__(self) -> None:
        super().__init__("object_scanner_web")
        self.declare_parameter("target_rgb", [0, 255, 0])
        self._state = "stopped"
        self._database_path: str | None = None
        self._target_rgb = validated_rgb(
            self.get_parameter("target_rgb").value
        )
        self._lock = threading.Lock()
        self._service_clients = {
            command: self.create_client(Trigger, service_name)
            for command, service_name in SERVICE_NAMES.items()
        }
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

    def status(self) -> dict:
        with self._lock:
            return self._status_unlocked()

    def command(self, command: str) -> dict:
        if command not in self._service_clients:
            raise ValueError(f"Unknown recording command: {command}")

        with self._lock:
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
            if self._state != "stopped":
                result = self._status_unlocked()
                result.update(
                    success=False,
                    message="Reference color can change only while stopped",
                )
                return result
            if not self._parameter_client.wait_for_services(
                timeout_sec=SERVICE_TIMEOUT_S
            ):
                raise TimeoutError(
                    "/object_scanner parameter services are unavailable"
                )

            future = self._parameter_client.set_parameters(
                [Parameter("target_rgb", value=color)]
            )
            response = self._wait_for_future(
                future,
                "/object_scanner/set_parameters",
            )
            if response is None or not response.results:
                raise RuntimeError(
                    "/object_scanner/set_parameters returned no result"
                )

            parameter_result = response.results[0]
            if parameter_result.successful:
                self._target_rgb = color
            result = self._status_unlocked()
            result.update(
                success=bool(parameter_result.successful),
                message=parameter_result.reason or "Reference color updated",
            )
            return result

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
        return {
            "state": self._state,
            "database_path": self._database_path,
            "target_rgb": list(self._target_rgb),
        }


def create_app(bridge, share_directory: Path | None = None) -> Flask:
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

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(bridge.status())

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

    @app.post("/api/recording/<command>")
    def recording_command(command):
        if command not in SERVICE_NAMES:
            return jsonify(success=False, message="Unknown command"), 404
        try:
            result = bridge.command(command)
        except TimeoutError as error:
            return jsonify(success=False, message=str(error)), 503
        except RuntimeError as error:
            return jsonify(success=False, message=str(error)), 502
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
