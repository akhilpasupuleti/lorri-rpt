import h3
from geopy.distance import geodesic
from math import radians, degrees, atan2
import numpy as np
from utils.constants import CONSTANTS
from typing import Tuple


def compute_bearing_angle(
    origin: Tuple[float, float], destination: Tuple[float, float]
) -> float:
    """
    Compute initial bearing (forward azimuth) in degrees between two lat/lon points.
    """

    lat1, lon1, lat2, lon2 = map(
        radians, [origin[0], origin[1], destination[0], destination[1]]
    )
    dlon = lon2 - lon1

    x = atan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )
    bearing = (degrees(x) + 360) % 360
    return round(bearing, 2)


def compute_hex_ring_distance(origin_hex: str, destination_hex: str) -> int:
    """
    Compute H3 hex ring distance (number of hexes between two cells).
    Returns -1 if not computable.
    """
    try:
        ring_dist = h3.grid_distance(origin_hex, destination_hex)
        return ring_dist if ring_dist is not None else -1
    except Exception:
        return -1


def encode_lane_hex(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    h3_res: int = CONSTANTS.H3_RES,
) -> str:
    """
    Get origin and destination H3 hex IDs and combine into lane identifier.
    """
    origin_hex = h3.latlng_to_cell(origin[0], origin[1], h3_res)
    destination_hex = h3.latlng_to_cell(destination[0], destination[1], h3_res)
    return f"{origin_hex}_{destination_hex}", origin_hex, destination_hex


def compute_haversine_distance(
    origin: Tuple[float, float], destination: Tuple[float, float]
) -> float:
    """
    Compute great-circle (haversine) distance in kilometers between two (lat, lon) pairs.
    """
    return geodesic(origin, destination).km


def compute_lane_features_from_coords(
    origin: dict, destination: dict, h3_res: int = CONSTANTS.H3_RES
) -> dict:
    """
    Compute lane-level features from origin and destination lat/lng dicts.

    Args:
        origin (dict): {"lat": float, "lng": float}
        destination (dict): {"lat": float, "lng": float}
        h3_res (int): H3 resolution (default 6)

    Returns:
        dict: {
            "origin_hex", "destination_hex", "lane_hex",
            "h_distance_km", "bearing_angle_deg", "hex_ring_distance"
        }
    """
    orig_coords = (origin["lat"], origin["lng"])
    dest_coords = (destination["lat"], destination["lng"])

    lane_hex, origin_hex, destination_hex = encode_lane_hex(
        orig_coords, dest_coords, CONSTANTS.H3_RES
    )

    h_distance_km = compute_haversine_distance(orig_coords, dest_coords)

    bearing_angle_deg = compute_bearing_angle(orig_coords, dest_coords)

    hex_ring_distance = compute_hex_ring_distance(origin_hex, destination_hex)

    return {
        "origin_hex": origin_hex,
        "destination_hex": destination_hex,
        "lane_hex": lane_hex,
        "h_distance_km": round(h_distance_km, 2),
        "bearing_angle_deg": round(bearing_angle_deg, 2),
        "hex_ring_distance": hex_ring_distance,
    }


def compute_lane_features_from_row(row, h3_res=CONSTANTS.H3_RES):
    origin = {"lat": row["Origin Latitude"], "lng": row["Origin Longitude"]}
    destination = {
        "lat": row["Destination Latitude"],
        "lng": row["Destination Longitude"],
    }
    return compute_lane_features_from_coords(origin, destination, h3_res)
