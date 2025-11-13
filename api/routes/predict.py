from api.schemas import RPTRequest, RPTResponse
from utils.validators import validate_coordinates, validate_date
from utils.constants import CONSTANTS
from utils.geo_utils import (
    compute_bearing_angle,
    compute_hex_ring_distance,
    encode_lane_hex,
    compute_haversine_distance,
)
from utils.model_utils import predict_lane_price
from utils.cost_utils import compute_estimated_fuel_cost, compute_fuel_price_per_km


def predict_rate(request: RPTRequest, model, category_map, model_config) -> RPTResponse:

    # Extract origin and destination coordinates
    origin_coords = (request.origin.location.lat, request.origin.location.lon)
    destination_coords = (
        request.destination.location.lat,
        request.destination.location.lon,
    )

    # Validate coordinates
    validate_coordinates(*origin_coords, label="Origin")
    validate_coordinates(*destination_coords, label="Destination")

    # Validate date
    parsed_date = validate_date(request.date)

    # parse date components
    day = parsed_date.day
    month = parsed_date.month
    year = parsed_date.year

    # calculate geo features
    lane_hex, origin_hex, destination_hex = encode_lane_hex(
        origin_coords, destination_coords, CONSTANTS.H3_RES
    )
    h_distance_km = compute_haversine_distance(origin_coords, destination_coords)
    bearing_angle_deg = compute_bearing_angle(origin_coords, destination_coords)
    hex_ring_distance = compute_hex_ring_distance(origin_hex, destination_hex)

    # truck type and dimensions
    no_of_wheels = request.truck.no_of_wheels
    capacity_mt = request.truck.capacity_mt
    length_ft = request.truck.length_ft
    is_multi_axle = 1 if request.truck.axle_type.upper() != "SA" else 0
    body_type = request.truck.body_type.upper()

    # fuel price metrics
    fuel_price = request.fuel_price or CONSTANTS.DEFAULT_FUEL_PRICE
    fuel_price_per_km_h = compute_fuel_price_per_km(fuel_price, h_distance_km)
    estimated_fuel_cost_h = compute_estimated_fuel_cost(
        fuel_price, h_distance_km, capacity_mt
    )
    
    # input features
    input_kwargs = {
        "origin_hex": origin_hex,
        "destination_hex": destination_hex,
        "lane_hex": lane_hex,
        "bearing_angle_deg": bearing_angle_deg,
        "hex_ring_distance": hex_ring_distance,
        "no_of_wheels": no_of_wheels,
        "capacity_mt": capacity_mt,
        "length_ft": length_ft,
        "is_multi_axle": is_multi_axle,
        "body_type": body_type,
        "day": day,
        "month": month,
        "year": year,
        "fuel_price_inr_per_litre": fuel_price,
        "h_distance_km": h_distance_km,
        "fuel_price_per_km_h": fuel_price_per_km_h,
        "estimated_fuel_cost_h": estimated_fuel_cost_h,
    }

    features = model_config["input_features"]

    # Sanity check for mismatched features
    missing = [f for f in features if f not in input_kwargs]
    if missing:
        raise ValueError(f"Missing required features for prediction: {missing}")

    predicted_price = predict_lane_price(
        model=model,
        category_map=category_map,
        features=features,
        **input_kwargs,
    )

    return RPTResponse(
        predicted_base_price=predicted_price,
        model_version=CONSTANTS.MODEL_VERSION,
        input_features=input_kwargs,
    )
