import numpy as np


def compute_fuel_price_per_km(fuel_price, distance):
    if distance and distance > 0:
        return round(fuel_price / distance, 2)
    return np.nan


def compute_estimated_fuel_cost(fuel_price, distance, capacity):
    if all([fuel_price, distance, capacity]) and capacity > 0:
        return round(fuel_price * distance * capacity, 2)
    return np.nan
