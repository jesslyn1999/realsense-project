"""Filter, transform, and record synchronized Orbbec point clouds."""

from enum import Enum
import json
import math
from pathlib import Path
import sqlite3
import time

import message_filters
import numpy as np
from object_scanner_web_orbbec.pointcloud import (
    filter_colored_points,
    transform_filtered_cloud,
)
from object_scanner_web_orbbec.sqlite_recording import (
    SqliteRecording,
    validated_session_name,
)
from object_scanner_interfaces.msg import NamedTransform
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger

POINTCLOUD_TOPIC = "/camera/depth_registered/points"
COLOR_TOPIC = "/camera/color/image_raw"
TRANSFORM_TOPIC = "/object_scanner_orbbec/camera_to_world"
SYNC_QUEUE_SIZE = 30
SYNC_SLOP_S = 0.05


class RecordingState(Enum):
    STOPPED = "stopped"
    RECORDING = "recording"
    PAUSED = "paused"


class OrbbecObjectScannerNode(Node):
    """Record green-filtered colored points in a supplied world frame."""

    def __init__(self) -> None:
        super().__init__("object_scanner_orbbec")

        self.declare_parameter("output_directory", "scans")
        self.declare_parameter("target_rgb", [0, 255, 0])
        self.declare_parameter("lab_threshold", 15.0)
        self.declare_parameter("session_name", "")
        self._load_parameters()

        self._state = RecordingState.STOPPED
        self._recording: SqliteRecording | None = None
        self._database_path: Path | None = None
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._pointcloud_subscriber = message_filters.Subscriber(
            self,
            PointCloud2,
            POINTCLOUD_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )
        self._color_subscriber = message_filters.Subscriber(
            self,
            Image,
            COLOR_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )
        self._transform_subscriber = message_filters.Subscriber(
            self,
            NamedTransform,
            TRANSFORM_TOPIC,
            qos_profile=qos_profile_sensor_data,
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [
                self._pointcloud_subscriber,
                self._color_subscriber,
                self._transform_subscriber,
            ],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_S,
        )
        self._synchronizer.registerCallback(self._on_synchronized_frame)

        self.create_service(
            Trigger,
            "~/start_recording",
            self._on_start_recording,
        )
        self.create_service(
            Trigger,
            "~/pause_recording",
            self._on_pause_recording,
        )
        self.create_service(
            Trigger,
            "~/resume_recording",
            self._on_resume_recording,
        )
        self.create_service(
            Trigger,
            "~/stop_recording",
            self._on_stop_recording,
        )
        self.create_service(
            Trigger,
            "~/recording_status",
            self._on_recording_status,
        )

    def _load_parameters(self) -> None:
        self._output_directory = Path(
            self.get_parameter("output_directory").value
        ).expanduser()
        self._target_rgb = self._validated_target_rgb(
            self.get_parameter("target_rgb").value
        )
        self._lab_threshold = float(
            self.get_parameter("lab_threshold").value
        )
        self._session_name = self.get_parameter("session_name").value

        if not math.isfinite(self._lab_threshold) or self._lab_threshold < 0:
            raise ValueError("lab_threshold must be finite and non-negative")

    @staticmethod
    def _validated_target_rgb(value) -> tuple[int, int, int]:
        try:
            channels = tuple(value)
        except TypeError as error:
            raise ValueError(
                "target_rgb must contain three integer values from 0 to 255"
            ) from error
        if (
            len(channels) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or channel < 0
                or channel > 255
                for channel in channels
            )
        ):
            raise ValueError(
                "target_rgb must contain three integer values from 0 to 255"
            )
        return channels

    def _on_set_parameters(
        self,
        parameters: list[Parameter],
    ) -> SetParametersResult:
        target_rgb = self._target_rgb
        session_name = self._session_name
        for parameter in parameters:
            if parameter.name not in {"session_name", "target_rgb"}:
                continue
            if self._state is not RecordingState.STOPPED:
                reason = (
                    f"{parameter.name} can change only while "
                    "recording is stopped"
                )
                self.get_logger().error(
                    f"Rejected parameter '{parameter.name}' while recorder "
                    f"is {self._state.value}: {reason}"
                )
                return SetParametersResult(
                    successful=False,
                    reason=reason,
                )
            try:
                if parameter.name == "target_rgb":
                    target_rgb = self._validated_target_rgb(parameter.value)
                else:
                    session_name = validated_session_name(parameter.value)
            except ValueError as error:
                return SetParametersResult(
                    successful=False,
                    reason=str(error),
                )

        self._target_rgb = target_rgb
        self._session_name = session_name
        return SetParametersResult(successful=True)

    # ── Synchronized processing ────────────────────────────────────────

    def _on_synchronized_frame(
        self,
        pointcloud: PointCloud2,
        _color_image: Image,
        transform: NamedTransform,
    ) -> None:
        if self._state is not RecordingState.RECORDING:
            return
        if transform.child_frame_id != pointcloud.header.frame_id:
            self.get_logger().error(
                "Transform child frame "
                f"'{transform.child_frame_id}' does not match point-cloud frame "
                f"'{pointcloud.header.frame_id}'"
            )
            return
        if not transform.header.frame_id:
            self.get_logger().error("Transform world frame is empty")
            return
        if not transform.transformation_name:
            self.get_logger().error("Transform name is empty")
            return

        try:
            matrix = np.asarray(transform.matrix, dtype=np.float64).reshape(4, 4)
            xyz, rgb = filter_colored_points(
                pointcloud,
                self._target_rgb,
                self._lab_threshold,
            )
            world_xyz, world_rgb = transform_filtered_cloud(
                matrix,
                xyz,
                rgb,
            )
        except ValueError as error:
            self.get_logger().error(f"Cannot process point cloud: {error}")
            return

        assert self._recording is not None
        try:
            self._recording.append_frame(
                recorded_perf_counter_ns=time.perf_counter_ns(),
                source_sec=pointcloud.header.stamp.sec,
                source_nanosec=pointcloud.header.stamp.nanosec,
                frame_id=transform.header.frame_id,
                transformation_name=transform.transformation_name,
                transformation_matrix=matrix,
                xyz=world_xyz,
                rgb=world_rgb,
            )
        except (sqlite3.Error, ValueError) as error:
            self._state = RecordingState.PAUSED
            self.get_logger().error(
                f"SQLite write failed; recording paused: {error}"
            )

    # ── Recording services ─────────────────────────────────────────────

    def _on_recording_status(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        response.success = True
        response.message = json.dumps(
            {
                "state": self._state.value,
                "database_path": (
                    str(self._database_path)
                    if self._database_path is not None
                    else None
                ),
                "session_name": self._session_name or None,
                "target_rgb": list(self._target_rgb),
            },
            separators=(",", ":"),
        )
        return response

    def _on_start_recording(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        if self._state is not RecordingState.STOPPED:
            return self._set_response(
                response,
                False,
                f"Recorder is already {self._state.value}",
            )

        try:
            session_name = validated_session_name(self._session_name)
            session_directory = self._output_directory / session_name
            recording = SqliteRecording(session_directory)
            database_path = recording.path
        except (OSError, sqlite3.Error, ValueError) as error:
            return self._set_response(
                response,
                False,
                f"Cannot start recording: {error}",
            )

        self._recording = recording
        self._database_path = database_path
        self._state = RecordingState.RECORDING
        self.get_logger().info(f"Recording filtered points to {database_path}")
        return self._set_response(response, True, str(database_path))

    def _on_pause_recording(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        if self._state is not RecordingState.RECORDING:
            return self._set_response(
                response,
                False,
                f"Cannot pause while recorder is {self._state.value}",
            )

        assert self._recording is not None
        try:
            self._recording.commit()
        except sqlite3.Error as error:
            return self._set_response(
                response,
                False,
                f"Cannot pause recording: {error}",
            )

        self._state = RecordingState.PAUSED
        self.get_logger().info("Recording paused")
        return self._set_response(
            response,
            True,
            str(self._database_path),
        )

    def _on_resume_recording(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        if self._state is not RecordingState.PAUSED:
            return self._set_response(
                response,
                False,
                f"Cannot resume while recorder is {self._state.value}",
            )

        self._state = RecordingState.RECORDING
        self.get_logger().info("Recording resumed")
        return self._set_response(response, True, "Recording resumed")

    def _on_stop_recording(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        if self._state is RecordingState.STOPPED:
            return self._set_response(
                response,
                False,
                "Recorder is already stopped",
            )

        database_path = self._database_path
        try:
            self._close_recording()
        except (OSError, sqlite3.Error) as error:
            return self._set_response(
                response,
                False,
                f"Cannot stop recording: {error}",
            )

        self.get_logger().info(f"Stopped recording {database_path}")
        return self._set_response(
            response,
            True,
            str(database_path),
        )

    @staticmethod
    def _set_response(
        response: Trigger.Response,
        success: bool,
        message: str,
    ) -> Trigger.Response:
        response.success = success
        response.message = message
        return response

    def _close_recording(self) -> None:
        recording = self._recording
        if recording is None:
            return
        recording.close()
        self._recording = None
        self._database_path = None
        self._state = RecordingState.STOPPED

    def shutdown(self) -> None:
        if self._recording is None:
            return
        try:
            self._close_recording()
        except (OSError, sqlite3.Error) as error:
            self.get_logger().error(f"Cannot close recording: {error}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrbbecObjectScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
