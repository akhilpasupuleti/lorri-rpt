from fastapi import HTTPException
from datetime import datetime


def validate_coordinates(lat: float, lon: float, label: str):
    if not (-90 <= lat <= 90):
        raise HTTPException(
            status_code=400,
            detail=f"{label} latitude {lat} is out of range (-90 to 90)",
        )
    if not (-180 <= lon <= 180):
        raise HTTPException(
            status_code=400,
            detail=f"{label} longitude {lon} is out of range (-180 to 180)",
        )


def validate_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Date must be in 'YYYY-MM-DD' format"
        )
