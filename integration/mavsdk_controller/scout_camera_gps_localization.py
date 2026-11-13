"""Camera + GPS victim localization for the PX4/Gazebo Scout.

This module deliberately has no LiDAR dependency.  It projects the bottom-centre
of an image detection through a calibrated camera mount and an acquisition-time
PX4 GPS/attitude pose onto an explicitly supplied horizontal terrain plane.

Coordinates:
- PX4 attitude: roll, pitch, yaw in radians; FRD body to NED navigation.
- Camera optical: +X right, +Y down, +Z forward.
- GPS altitude and terrain altitude: metres above the same vertical datum.
- Camera extrinsic: 4x4 transform mapping camera-optical coordinates to body FRD.
- Output: WGS84 latitude/longitude and local NED offset from the acquisition pose.

A calibration is never trusted just because a JSON field says "verified".  The
caller must supply a calibration record with verification evidence and an
approved calibration ID through the trusted coordinator policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

WGS84_A_M = 6378137.0
WGS84_E2 = 6.69437999014e-3
CALIBRATION_SCHEMA = "scout-camera-gps-calibration-v1"


class LocalizationError(ValueError):
    """Raised when camera-GPS localization inputs are invalid or incomplete."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LocalizationError(f"{name}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LocalizationError(f"{name}_invalid") from exc
    if not math.isfinite(result):
        raise LocalizationError(f"{name}_invalid")
    return result


def _matrix4(value: Any, name: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise LocalizationError(f"{name}_must_be_4x4")
    matrix = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise LocalizationError(f"{name}_must_be_4x4")
        matrix.append([_finite(v, name) for v in row])
    # Homogeneous transform last row.
    if any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], (0.0, 0.0, 0.0, 1.0))):
        raise LocalizationError(f"{name}_invalid_homogeneous_row")
    # Rotation columns must be approximately orthonormal and right handed.
    r = [row[:3] for row in matrix[:3]]
    cols = [[r[i][j] for i in range(3)] for j in range(3)]
    for i in range(3):
        if abs(sum(x*x for x in cols[i]) - 1.0) > 0.03:
            raise LocalizationError(f"{name}_rotation_not_orthonormal")
        for j in range(i):
            if abs(sum(cols[i][k] * cols[j][k] for k in range(3))) > 0.03:
                raise LocalizationError(f"{name}_rotation_not_orthonormal")
    det = (
        r[0][0] * (r[1][1]*r[2][2] - r[1][2]*r[2][1])
        - r[0][1] * (r[1][0]*r[2][2] - r[1][2]*r[2][0])
        + r[0][2] * (r[1][0]*r[2][1] - r[1][1]*r[2][0])
    )
    if det < 0.97 or det > 1.03:
        raise LocalizationError(f"{name}_rotation_not_proper")
    return matrix


def validate_camera_gps_calibration(
    calibration: Mapping[str, Any],
    approved_calibration_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a camera-GPS calibration contract and coordinator approval."""
    if not isinstance(calibration, Mapping):
        raise LocalizationError("camera_gps_calibration_missing")
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise LocalizationError("camera_gps_calibration_schema_invalid")
    calibration_id = calibration.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise LocalizationError("camera_gps_calibration_id_missing")
    if approved_calibration_ids is not None and calibration_id not in approved_calibration_ids:
        raise LocalizationError("camera_gps_calibration_not_approved")
    if calibration.get("verification_status") != "empirically_verified":
        raise LocalizationError("camera_gps_calibration_not_empirically_verified")
    evidence = calibration.get("verification_evidence")
    if not isinstance(evidence, Mapping) or not evidence.get("validation_dataset") or not evidence.get("validated_at_utc"):
        raise LocalizationError("camera_gps_verification_evidence_missing")

    intr = calibration.get("camera_intrinsics")
    if not isinstance(intr, Mapping):
        raise LocalizationError("camera_intrinsics_missing")
    fx = _finite(intr.get("fx_px"), "fx_px")
    fy = _finite(intr.get("fy_px"), "fy_px")
    cx = _finite(intr.get("cx_px"), "cx_px")
    cy = _finite(intr.get("cy_px"), "cy_px")
    width, height = intr.get("width_px"), intr.get("height_px")
    if fx <= 0 or fy <= 0 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise LocalizationError("camera_intrinsics_invalid")
    if not (0 <= cx <= width and 0 <= cy <= height):
        raise LocalizationError("camera_principal_point_invalid")
    if calibration.get("distortion_model", intr.get("distortion_model", "none")) not in ("none", "plumb_bob_zero"):
        raise LocalizationError("unsupported_camera_distortion_model")

    transforms = calibration.get("transforms")
    if not isinstance(transforms, Mapping):
        raise LocalizationError("camera_extrinsics_missing")
    transform = _matrix4(transforms.get("body_frd_from_camera_optical"), "body_frd_from_camera_optical")
    uncertainty = calibration.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise LocalizationError("calibration_uncertainty_missing")
    std = {}
    for key in ("intrinsic_px", "extrinsic_translation_m", "extrinsic_rotation_rad"):
        std[key] = _finite(uncertainty.get(key), f"uncertainty_{key}")
        if std[key] < 0:
            raise LocalizationError(f"uncertainty_{key}_negative")
    return {
        "calibration_id": calibration_id,
        "fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy,
        "width_px": width, "height_px": height,
        "body_frd_from_camera_optical": transform,
        "uncertainty": std,
    }


def body_frd_to_ned(roll_rad: float, pitch_rad: float, yaw_rad: float) -> list[list[float]]:
    """PX4 aerospace Euler rotation mapping body FRD vectors into local NED."""
    r, p, y = map(float, (roll_rad, pitch_rad, yaw_rad))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
        [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
        [-sp,   sr*cp,          cr*cp],
    ]


def _matvec3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(float(matrix[i][j]) * float(vector[j]) for j in range(3)) for i in range(3))


def _geodetic_offset(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    denom = math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    prime_vertical = WGS84_A_M / denom
    meridian = WGS84_A_M * (1.0 - WGS84_E2) / (denom ** 3)
    dlat = north_m / meridian
    dlon = east_m / (prime_vertical * max(abs(math.cos(lat)), 1e-8))
    return math.degrees(lat + dlat), math.degrees(math.radians(lon_deg) + dlon)


@dataclass(frozen=True)
class AcquisitionPose:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    position_sigma_m: float
    attitude_sigma_rad: float
    capture_monotonic_s: float
    pose_origin_id: str
    pose_bracket_s: float = 0.0
    clock_mapping_sigma_s: float = 0.0


def project_detection_to_gps(
    detection: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pose: AcquisitionPose,
    *,
    terrain_altitude_m: float | None,
    terrain_sigma_m: float | None,
    terrain_intersection: Callable | None = None,
    maximum_range_m: float = 40.0,
    minimum_range_m: float = 0.5,
    maximum_pose_bracket_s: float = 0.25,
    maximum_clock_sigma_s: float = 0.01,
    approved_calibration_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Project a YOLO bbox footpoint onto a known terrain altitude.

    Returns validated geometry only when all explicit evidence is available.
    This function does not itself establish that calibration/terrain evidence
    is operationally trustworthy; the coordinator must independently approve it.
    """
    cfg = validate_camera_gps_calibration(calibration, approved_calibration_ids)
    if (terrain_altitude_m is None and terrain_intersection is None) or terrain_sigma_m is None:
        raise LocalizationError("terrain_altitude_or_uncertainty_missing")
    terrain_sigma = _finite(terrain_sigma_m, "terrain_sigma_m")
    if terrain_sigma < 0:
        raise LocalizationError("terrain_sigma_negative")

    if not isinstance(detection, Mapping):
        raise LocalizationError("detection_missing")
    bbox = detection.get("xyxy_pixels")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise LocalizationError("detection_bbox_invalid")
    x1, y1, x2, y2 = [_finite(v, "bbox") for v in bbox]
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > cfg["width_px"] or y2 > cfg["height_px"]:
        raise LocalizationError("detection_bbox_outside_image")
    pixel_anchor = calibration.get("detection_pixel_anchor", "bbox_bottom_center")
    u = (x1 + x2) / 2.0
    if pixel_anchor == "bbox_center":
        v = (y1 + y2) / 2.0
    elif pixel_anchor == "bbox_bottom_center":
        v = y2
    else:
        raise LocalizationError("unsupported_detection_pixel_anchor")

    for name, value in (
        ("latitude_deg", pose.latitude_deg), ("longitude_deg", pose.longitude_deg),
        ("altitude_m", pose.altitude_m), ("roll_rad", pose.roll_rad),
        ("pitch_rad", pose.pitch_rad), ("yaw_rad", pose.yaw_rad),
        ("position_sigma_m", pose.position_sigma_m),
        ("attitude_sigma_rad", pose.attitude_sigma_rad),
        ("capture_monotonic_s", pose.capture_monotonic_s),
        ("pose_bracket_s", pose.pose_bracket_s),
        ("clock_mapping_sigma_s", pose.clock_mapping_sigma_s),
    ):
        _finite(value, name)
    if not (-90 <= pose.latitude_deg <= 90 and -180 <= pose.longitude_deg <= 180):
        raise LocalizationError("gps_fix_invalid")
    if not pose.pose_origin_id:
        raise LocalizationError("px4_origin_id_missing")
    if pose.position_sigma_m < 0 or pose.attitude_sigma_rad < 0:
        raise LocalizationError("pose_uncertainty_invalid")
    if pose.pose_bracket_s < 0 or pose.pose_bracket_s > maximum_pose_bracket_s:
        raise LocalizationError("pose_bracket_too_wide_or_invalid")
    if pose.clock_mapping_sigma_s < 0 or pose.clock_mapping_sigma_s > maximum_clock_sigma_s:
        raise LocalizationError("clock_mapping_uncertain")

    ray_camera = ((u - cfg["cx_px"]) / cfg["fx_px"], (v - cfg["cy_px"]) / cfg["fy_px"], 1.0)
    t = cfg["body_frd_from_camera_optical"]
    ray_body = _matvec3([row[:3] for row in t[:3]], ray_camera)
    camera_offset_body = (t[0][3], t[1][3], t[2][3])
    body_to_ned = body_frd_to_ned(pose.roll_rad, pose.pitch_rad, pose.yaw_rad)
    ray_ned_raw = _matvec3(body_to_ned, ray_body)
    norm = math.sqrt(sum(x*x for x in ray_ned_raw))
    ray_ned = tuple(x / norm for x in ray_ned_raw)
    camera_offset_ned = _matvec3(body_to_ned, camera_offset_body)

    camera_lat, camera_lon = pose.latitude_deg, pose.longitude_deg
    camera_alt = pose.altitude_m - camera_offset_ned[2]
    if ray_ned[2] <= 1e-6:
        raise LocalizationError("camera_ray_does_not_intersect_ground")
    if terrain_intersection is None:
        terrain_alt = _finite(terrain_altitude_m, "terrain_altitude_m")
        distance = (camera_alt - terrain_alt) / ray_ned[2]
    else:
        distance = _finite(terrain_intersection(pose, camera_offset_ned, ray_ned), "terrain_intersection_distance")
        terrain_alt = camera_alt - distance * ray_ned[2]
    if distance <= 0:
        raise LocalizationError("terrain_not_below_camera")
    if distance < minimum_range_m or distance > maximum_range_m:
        raise LocalizationError("projected_target_out_of_range")

    north = camera_offset_ned[0] + distance * ray_ned[0]
    east = camera_offset_ned[1] + distance * ray_ned[1]
    down = camera_offset_ned[2] + distance * ray_ned[2]
    lat, lon = _geodetic_offset(camera_lat, camera_lon, north, east)
    confidence = _finite(detection.get("confidence"), "detection_confidence")
    if not 0 <= confidence <= 1:
        raise LocalizationError("detection_confidence_invalid")

    # Conservative first-order uncertainty bound.  This is intentionally not
    # represented as a validated covariance; empirical validation is required.
    pixel_sigma = _finite(detection.get("footpoint_sigma_px", 3.0), "footpoint_sigma_px")
    if pixel_sigma < 0:
        raise LocalizationError("footpoint_uncertainty_invalid")
    angular_sigma = (
        pixel_sigma / min(cfg["fx_px"], cfg["fy_px"])
        + cfg["uncertainty"]["intrinsic_px"] / min(cfg["fx_px"], cfg["fy_px"])
        + cfg["uncertainty"]["extrinsic_rotation_rad"]
        + pose.attitude_sigma_rad
    )
    geometric_sigma = distance * angular_sigma
    mount_sigma = cfg["uncertainty"]["extrinsic_translation_m"]
    time_sigma_m = abs(distance) * pose.clock_mapping_sigma_s * max(0.0, abs(pose.attitude_sigma_rad))
    uncertainty_m = math.sqrt(
        pose.position_sigma_m**2 + terrain_sigma**2 + mount_sigma**2
        + geometric_sigma**2 + time_sigma_m**2
    )
    return {
        "localization_status": "validated_geometry_pending_coordinator_approval",
        "location_status": "camera_gps_projected",
        "coordinate_frame": "wgs84_geodetic",
        "latitude_deg": lat,
        "longitude_deg": lon,
        "altitude_m": terrain_alt,
        "scout_local_ned_m": {"north_m": north, "east_m": east, "down_m": down},
        "range_m": distance,
        "localization_uncertainty_m": uncertainty_m,
        "localization_confidence": confidence,
        "dispatchable": False,
        "ground_truth": False,
        "calibration_id": cfg["calibration_id"],
        "coordinate_origin_id": pose.pose_origin_id,
        "timestamp_domain": "host_monotonic_camera_acquisition",
        "camera_capture_monotonic_s": pose.capture_monotonic_s,
        "pose_bracket_s": pose.pose_bracket_s,
        "clock_mapping_sigma_s": pose.clock_mapping_sigma_s,
        "terrain_altitude_m": terrain_alt,
        "terrain_sigma_m": terrain_sigma,
        "uncertainty_components_m": {
            "gps_position": pose.position_sigma_m,
            "terrain": terrain_sigma,
            "camera_mount_translation": mount_sigma,
            "projection_geometry": geometric_sigma,
            "clock_mapping_attitude_term": time_sigma_m,
        },
        "transformation_provenance": {
            "method": "camera_optical_ray_to_body_frd_to_px4_ned_to_wgs84",
            "calibration_id": cfg["calibration_id"],
            "body_frd_from_camera_optical": "empirically_verified_4x4",
            "body_frd_to_ned": "PX4_roll_pitch_yaw_rotation",
            "timestamp_domain": "host_monotonic_camera_acquisition",
            "coordinate_origin_id": pose.pose_origin_id,
            "terrain_model": ("gazebo_enu_collision_mesh" if terrain_intersection is not None
                              else "explicit_horizontal_altitude_same_vertical_datum"),
        },
    }
