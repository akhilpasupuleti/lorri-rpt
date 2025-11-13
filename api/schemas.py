from pydantic import BaseModel
from typing import List, Literal, Optional, Any, Dict


class AutoMatchHeadersRequest(BaseModel):
    headers: List[str]
    sample_rows: List[List[Any]]


class AutoMatchHeadersResponse(BaseModel):
    mapping: Dict[str, str]
    missing_fields: List[str] = []
    error: Optional[str] = None


class TruckCleanRequest(BaseModel):
    raw_truck_name: str


class TruckCleanResponse(BaseModel):
    raw_truck_name: str
    cleaned_code: str
    dimensions: dict
    reasoning: str


class LocationDetails(BaseModel):
    suggestion: Optional[str] = ""
    district: Optional[str] = ""
    lat: float
    location: Optional[str] = ""
    lon: float
    state: Optional[str] = ""
    label: Optional[str] = ""
    score: Optional[float] = 0.0


class CompositeLocation(BaseModel):
    location: LocationDetails
    location_name: str
    coordinates: List[float]  # [lon, lat]


class TruckType(BaseModel):
    truck_type: str
    no_of_wheels: int
    capacity_mt: float
    length_ft: float
    axle_type: Literal["SA", "MA"]
    body_type: Literal["OB", "CB", "CN", "HQ", "RF"]


class RPTRequest(BaseModel):
    origin: CompositeLocation
    destination: CompositeLocation
    truck: TruckType
    date: str  # 'YYYY-MM-DD'
    fuel_price: Optional[float] = None


class InputFeatures(BaseModel):
    origin_hex: str
    destination_hex: str
    lane_hex: str
    bearing_angle_deg: float
    hex_ring_distance: int
    no_of_wheels: int
    capacity_mt: float
    length_ft: float
    is_multi_axle: int
    body_type: str
    day: int
    month: int
    year: int
    h_distance_km: float
    fuel_price_per_km_h: float
    estimated_fuel_cost_h: float


class RPTResponse(BaseModel):
    predicted_base_price: Optional[float]
    model_version: str
    input_features: InputFeatures
    error: Optional[str] = None


class BulkTruckCleanRequest(BaseModel):
    raw_truck_names: List[str]
