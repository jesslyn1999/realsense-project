from dataclasses import replace
import sqlite3

import numpy as np
import object_scanner_processing.aligned_recording as aligned_recording
from object_scanner_processing.aligned_recording import (
    aligned_database_path,
    AlignedRecordingError,
    generate_aligned_recording,
    read_fused_cloud,
    source_revision,
    StaleAlignedRecordingError,
    write_aligned_recording,
)
from object_scanner_processing.pointcloud_processing import (
    AlignedFrame,
    CharucoCornerDiagnostics,
    CharucoFrameDiagnostics,
    EdgeDiagnostics,
    FrameDiagnostics,
    PointCloudProcessingError,
    ProcessingResult,
)
from object_scanner_processing.recording import read_frames, RecordedFrame
import pytest


def _create_raw_database(path):
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE frames (
                id INTEGER PRIMARY KEY,
                recorded_perf_counter_ns INTEGER NOT NULL,
                source_sec INTEGER NOT NULL,
                source_nanosec INTEGER NOT NULL,
                point_count INTEGER NOT NULL,
                frame_id TEXT NOT NULL,
                transformation_name TEXT NOT NULL,
                transformation_matrix BLOB NOT NULL,
                xyz BLOB NOT NULL,
                rgb BLOB NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def _append_raw_frame(path, frame_id):
    xyz = np.array(
        [[frame_id, 0.0, 0.5], [frame_id, 0.1, 0.5]],
        dtype="<f4",
    )
    rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            INSERT INTO frames (
                id, recorded_perf_counter_ns, source_sec, source_nanosec,
                point_count, frame_id, transformation_name,
                transformation_matrix, xyz, rgb
            ) VALUES (?, ?, ?, 0, ?, 'world', 'charuco', ?, ?, ?)
            """,
            (
                frame_id,
                frame_id * 100,
                frame_id,
                len(xyz),
                np.eye(4, dtype="<f8").tobytes(),
                xyz.tobytes(),
                rgb.tobytes(),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _recorded_frame(frame_id):
    xyz = np.array(
        [[frame_id, 0.0, 0.5], [frame_id, 0.1, 0.5]],
        dtype="<f4",
    )
    return RecordedFrame(
        id=frame_id,
        recorded_perf_counter_ns=frame_id * 100,
        source_sec=frame_id,
        source_nanosec=0,
        parent_frame_id="world",
        transformation_name="charuco",
        matrix=np.eye(4),
        xyz=xyz,
        rgb=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
    )


def _processing_result(frame_count, fused_offset=0.0):
    aligned_frames = []
    diagnostics = []
    for frame_id in range(1, frame_count + 1):
        source = _recorded_frame(frame_id)
        frame_diagnostics = FrameDiagnostics(
            frame_id=frame_id,
            input_points=2,
            voxel_points=1,
            radius_points=1,
            cleaned_points=1,
        )
        diagnostics.append(frame_diagnostics)
        aligned_frames.append(
            AlignedFrame(
                source=source,
                optimized_pose=np.eye(4),
                xyz=np.array(
                    [[frame_id + fused_offset, 0.0, 0.5]],
                    dtype="<f4",
                ),
                rgb=source.rgb[:1].copy(),
                diagnostics=frame_diagnostics,
            )
        )
    edges = tuple(
        EdgeDiagnostics(
            source_frame_id=source_id,
            target_frame_id=source_id + 1,
            accepted=True,
            overlap_before=0.8,
            overlap_after=0.9,
            rmse_before_m=0.004,
            rmse_after_m=0.002,
            correction_m=0.001,
            correction_deg=0.1,
            reason="accepted",
        )
        for source_id in range(1, frame_count)
    )
    xyz = np.concatenate([frame.xyz for frame in aligned_frames])
    rgb = np.concatenate([frame.rgb for frame in aligned_frames])
    return ProcessingResult(
        xyz=xyz,
        rgb=rgb,
        observation_counts=np.ones(len(xyz), dtype=np.uint16),
        optimized_poses=tuple(np.eye(4) for _ in aligned_frames),
        aligned_frames=tuple(aligned_frames),
        frames=tuple(diagnostics),
        edges=edges,
    )


def _raw_database(tmp_path, frame_count=2):
    path = tmp_path / "recording.sqlite3"
    _create_raw_database(path)
    for frame_id in range(1, frame_count + 1):
        _append_raw_frame(path, frame_id)
    return path


def test_writes_compatible_frames_and_reads_fused_cloud(tmp_path):
    raw_path = _raw_database(tmp_path)
    raw_before = raw_path.read_bytes()
    result = _processing_result(2)

    aligned_path = write_aligned_recording(raw_path, result)
    fused = read_fused_cloud(aligned_path, source_revision(raw_path))
    saved_frames = read_frames(aligned_path)

    assert aligned_path == aligned_database_path(raw_path)
    assert raw_path.read_bytes() == raw_before
    np.testing.assert_array_equal(fused.xyz, result.xyz)
    np.testing.assert_array_equal(fused.rgb, result.rgb)
    np.testing.assert_array_equal(
        fused.observation_counts,
        result.observation_counts,
    )
    assert fused.raw_points == 4
    assert fused.cleaned_points == 2
    assert fused.accepted_edges == 1
    assert fused.rejected_edges == 0
    assert [frame.id for frame in saved_frames] == [1, 2]
    assert saved_frames[0].recorded_perf_counter_ns == 100
    assert saved_frames[0].transformation_name == (
        "charuco_pose_graph_optimized"
    )


def test_persists_and_reads_charuco_acceptance_diagnostics(tmp_path):
    raw_path = _raw_database(tmp_path)
    result = _processing_result(2)
    frame_diagnostics = tuple(
        CharucoFrameDiagnostics(
            frame_id=frame_id,
            temporal_weight=0.5 ** (frame_id - 1),
            matched_prior_count=0 if frame_id == 1 else 20,
            corner_count=20,
            valid_depth_count=20,
            invalid_depth_count=0,
            reprojection_rmse_px=0.2,
            reprojection_max_px=0.4,
            corner_3d_rmse_m=0.001,
            corner_3d_max_m=0.002,
            cloud_overlap_fraction_3mm=0.995,
        )
        for frame_id in (1, 2)
    )
    corner_diagnostics = tuple(
        CharucoCornerDiagnostics(
            frame_id=frame_id,
            corner_id=corner_id,
            retained=True,
            depth_valid=True,
            reprojection_error_px=0.2,
            corner_3d_residual_m=0.001,
        )
        for frame_id in (1, 2)
        for corner_id in range(20)
    )
    result = replace(
        result,
        charuco_frames=frame_diagnostics,
        charuco_corners=corner_diagnostics,
        cloud_overlap_fraction_3mm=0.995,
    )

    aligned_path = write_aligned_recording(raw_path, result)
    fused = read_fused_cloud(aligned_path, source_revision(raw_path))

    assert fused.charuco_frame_count == 2
    assert fused.charuco_reprojection_max_px == pytest.approx(0.4)
    assert fused.cloud_overlap_fraction_3mm == pytest.approx(0.995)
    connection = sqlite3.connect(aligned_path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM charuco_frame_diagnostics"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM charuco_corner_diagnostics"
        ).fetchone()[0] == 40
    finally:
        connection.close()


def test_refreshes_every_row_in_the_same_database_file(tmp_path):
    raw_path = _raw_database(tmp_path)
    aligned_path = write_aligned_recording(raw_path, _processing_result(2))
    original_inode = aligned_path.stat().st_ino
    _append_raw_frame(raw_path, 3)

    refreshed_path = write_aligned_recording(
        raw_path,
        _processing_result(3, fused_offset=10.0),
    )
    fused = read_fused_cloud(refreshed_path, source_revision(raw_path))

    assert refreshed_path.stat().st_ino == original_inode
    assert fused.source_revision.frame_count == 3
    np.testing.assert_array_equal(fused.xyz[:, 0], [11.0, 12.0, 13.0])
    assert [frame.id for frame in read_frames(refreshed_path)] == [1, 2, 3]


def test_rejects_stale_aligned_recording(tmp_path):
    raw_path = _raw_database(tmp_path)
    aligned_path = write_aligned_recording(raw_path, _processing_result(2))
    _append_raw_frame(raw_path, 3)

    with pytest.raises(StaleAlignedRecordingError, match="does not match"):
        read_fused_cloud(aligned_path, source_revision(raw_path))


def test_first_write_failure_leaves_no_aligned_database(tmp_path):
    raw_path = _raw_database(tmp_path, frame_count=2)
    aligned_path = aligned_database_path(raw_path)

    with pytest.raises(AlignedRecordingError, match="frame count"):
        write_aligned_recording(raw_path, _processing_result(1))

    assert not aligned_path.exists()


def test_rejects_malformed_fused_blob(tmp_path):
    raw_path = _raw_database(tmp_path)
    aligned_path = write_aligned_recording(raw_path, _processing_result(2))
    connection = sqlite3.connect(aligned_path)
    try:
        connection.execute("UPDATE fused_cloud SET xyz = X'00'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(AlignedRecordingError, match="invalid fused point data"):
        read_fused_cloud(aligned_path)


def test_failed_refresh_rolls_back_to_previous_complete_result(tmp_path):
    raw_path = _raw_database(tmp_path)
    original_result = _processing_result(2)
    aligned_path = write_aligned_recording(raw_path, original_result)
    connection = sqlite3.connect(aligned_path)
    try:
        connection.execute(
            """
            CREATE TRIGGER reject_aligned_frames
            BEFORE INSERT ON frames
            BEGIN
                SELECT RAISE(ABORT, 'synthetic write failure');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="synthetic write failure"):
        write_aligned_recording(
            raw_path,
            _processing_result(2, fused_offset=20.0),
        )

    fused = read_fused_cloud(aligned_path, source_revision(raw_path))
    np.testing.assert_array_equal(fused.xyz, original_result.xyz)


def test_processing_failure_preserves_previous_complete_result(
    tmp_path,
    monkeypatch,
):
    raw_path = _raw_database(tmp_path)
    original_result = _processing_result(2)
    aligned_path = write_aligned_recording(raw_path, original_result)

    def reject_processing(_raw_path):
        raise PointCloudProcessingError("registration graph is weak")

    monkeypatch.setattr(
        aligned_recording,
        "process_recording",
        reject_processing,
    )
    with pytest.raises(PointCloudProcessingError, match="graph is weak"):
        generate_aligned_recording(raw_path)

    fused = read_fused_cloud(aligned_path, source_revision(raw_path))
    np.testing.assert_array_equal(fused.xyz, original_result.xyz)
