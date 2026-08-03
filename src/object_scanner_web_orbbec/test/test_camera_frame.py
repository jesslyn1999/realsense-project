import numpy as np
from object_scanner_web_orbbec.camera_frame import (
    build_camera_payload,
    CAMERA_HEADER,
    CAMERA_MAGIC,
    image_to_rgb,
)
from sensor_msgs.msg import Image


def make_image(encoding, data):
    message = Image()
    message.width = 2
    message.height = 1
    message.step = 8
    message.encoding = encoding
    message.data = bytes(data)
    return message


def test_rgb_payload_removes_row_padding_without_changing_colors():
    message = make_image(
        "rgb8",
        [1, 2, 3, 4, 5, 6, 99, 99],
    )

    payload = build_camera_payload(message)

    magic, width, height = CAMERA_HEADER.unpack_from(payload)
    assert (magic, width, height) == (CAMERA_MAGIC, 2, 1)
    assert payload[CAMERA_HEADER.size:] == bytes([1, 2, 3, 4, 5, 6])


def test_bgr_image_is_converted_to_rgb():
    message = make_image(
        "bgr8",
        [3, 2, 1, 6, 5, 4, 99, 99],
    )

    np.testing.assert_array_equal(
        image_to_rgb(message),
        [[[1, 2, 3], [4, 5, 6]]],
    )
