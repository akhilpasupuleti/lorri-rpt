# src/utils/feature_config.py

# Common features
BASE_FEATURES = [
    "origin_hex",
    "destination_hex",
    "lane_hex",
    "bearing_angle_deg",
    "hex_ring_distance",
    "no_of_wheels",
    "capacity_mt",
    "length_ft",
    "is_multi_axle",
    "body_type",
    "day",
    "month",
    "year",
    "fuel_price_inr_per_litre",
]

DISTANCE_FEATURE_SETS = {
    "g": ["g_distance_km", "fuel_price_per_km_g", "estimated_fuel_cost_g"],
    "h": ["h_distance_km", "fuel_price_per_km_h", "estimated_fuel_cost_h"],
}
CATEGORICAL_COLS = ["origin_hex", "destination_hex", "lane_hex", "body_type"]

TARGET = "log_base_price"
