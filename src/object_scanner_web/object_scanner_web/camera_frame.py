"""Encode a lossless ROS RGB image for browser pixel selection."""

import struct

import numpy as np
from sensor_msgs.msg import Image


CAMERA_HEADER = struct.Struct("<4sII")
CAMERA_MAGIC = b"RGB1"


def image_to_rgb(message: Image) -> np.ndarray:
    """Return a row-packed RGB8 array from an RGB8 or BGR8 ROS image."""
    encoding = message.encoding.lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"Unsupported color encoding: {message.encoding}")
    packed_row_size = message.width * 3
    if message.width < 1 or message.height < 1:
        raise ValueError("Camera image dimensions must be positive")
    if message.step < packed_row_size:
        raise ValueError("Camera image row step is too small")
    if len(message.data) < message.height * message.step:
        raise ValueError("Camera image data is incomplete")

    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height,
        message.step,
    )
    image = rows[:, :packed_row_size].reshape(
        message.height,
        message.width,
        3,
    )
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def build_camera_payload(message: Image) -> bytes:
    """Encode width, height, and exact RGB bytes for the browser."""
    image = image_to_rgb(message)
    header = CAMERA_HEADER.pack(CAMERA_MAGIC, message.width, message.height)
    return header + image.tobytes()
