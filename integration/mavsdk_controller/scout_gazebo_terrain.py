"""Ray intersections with the original Gazebo world's collision terrain."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from scout_camera_gps_localization import WGS84_A_M, WGS84_E2, LocalizationError


class GazeboTerrain:
    """Load a static ENU terrain mesh and intersect rays from PX4 poses."""

    def __init__(self, world_path: Path):
        world = ET.parse(world_path).getroot().find("world")
        if world is None:
            raise ValueError("missing Gazebo world")
        reference = world.find("spherical_coordinates")
        if (reference is None or reference.findtext("world_frame_orientation") != "ENU"
                or reference.findtext("surface_model") != "EARTH_WGS84"
                or float(reference.findtext("heading_deg", "0")) != 0):
            raise ValueError("terrain requires a WGS84 ENU world with zero heading")
        self.latitude = float(reference.findtext("latitude_deg"))
        self.longitude = float(reference.findtext("longitude_deg"))
        self.elevation = float(reference.findtext("elevation"))
        terrain = world.find("model[@name='terrain']")
        if terrain is None:
            raise ValueError("missing terrain model")
        pose = [float(value) for value in terrain.findtext("pose").split()]
        if len(pose) != 6 or pose[3:] != [0, 0, 0]:
            raise ValueError("terrain requires an unrotated model")
        link = terrain.find("link[@name='landform']")
        collision = link.find("collision[@name='collision']")
        for element in (link, collision):
            if any(float(value) != 0 for value in element.findtext("pose", "0").split()):
                raise ValueError("terrain collision pose must be identity")
        mesh = collision.find("geometry/mesh")
        uri = mesh.findtext("uri")
        expected = "model://earthquake_landslide_assets/meshes/terrain_apron_collision.obj"
        if uri != expected:
            raise ValueError("unsupported terrain collision mesh")
        scale = [float(value) for value in mesh.findtext("scale", "1 1 1").split()]
        vertices = []
        faces = []
        with (world_path.parent / uri.removeprefix("model://")).open() as stream:
            for line in stream:
                fields = line.split("#", 1)[0].split()
                if fields and fields[0] == "v":
                    vertices.append([float(value) for value in fields[1:4]])
                if fields and fields[0] == "f":
                    indices = [int(value.split("/")[0]) for value in fields[1:]]
                    if len(indices) != 3 or any(index < 1 for index in indices):
                        raise ValueError("terrain OBJ must have positive triangle indices")
                    faces.append([index - 1 for index in indices])
        if not vertices or not faces:
            raise ValueError("empty terrain collision mesh")
        triangles = (np.asarray(vertices) * scale + pose[:3])[faces]
        if not np.isfinite(triangles).all():
            raise ValueError("nonfinite terrain mesh")
        self.a, b, c = triangles.transpose(1, 0, 2)
        self.e1 = b - self.a
        self.e2 = c - self.a

        latitude_rad = math.radians(self.latitude)
        denominator = math.sqrt(1 - WGS84_E2 * math.sin(latitude_rad) ** 2)
        self.metres_per_lat_radian = WGS84_A_M * (1 - WGS84_E2) / denominator**3
        self.metres_per_lon_radian = WGS84_A_M * math.cos(latitude_rad) / denominator

    def intersection_distance(self, pose, camera_offset_ned, ray_ned):
        north, east, down = camera_offset_ned
        origin = np.array([
            math.radians(pose.longitude_deg - self.longitude) * self.metres_per_lon_radian + east,
            math.radians(pose.latitude_deg - self.latitude) * self.metres_per_lat_radian + north,
            pose.altitude_m - self.elevation - down,
        ])
        direction = np.array([ray_ned[1], ray_ned[0], -ray_ned[2]])
        cross = np.cross(direction, self.e2)
        determinant = np.einsum("ij,ij->i", self.e1, cross)
        relative = origin - self.a
        q = np.cross(relative, self.e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.einsum("ij,ij->i", relative, cross) / determinant
            v = (q @ direction) / determinant
            distance = np.einsum("ij,ij->i", self.e2, q) / determinant
        hits = distance[(abs(determinant) > 1e-9) & (u >= -1e-9) & (v >= -1e-9)
                        & (u + v <= 1 + 1e-9) & (distance > 0)]
        if not hits.size:
            raise LocalizationError("camera_ray_misses_gazebo_terrain")
        return float(hits.min())
