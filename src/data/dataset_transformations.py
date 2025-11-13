# %%
# Imports & Configuration
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import atan2, degrees, radians
from geopy.distance import geodesic
import json

# %%
# Local module imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from utils.geo_utils import compute_lane_features_from_row
from utils.cleaning_utils import iqr_filter
from utils.cost_utils import compute_estimated_fuel_cost, compute_fuel_price_per_km

# Constants
H3_RES = 6
RAW_DATA_PATH = "../../data/raw/dataset-hector-apollo-v2.xlsx"

# %%
# Load Data
df = pd.read_excel(RAW_DATA_PATH, engine="openpyxl")

# %%
# Step 1: Geospatial Features
features = [
    "origin_hex",
    "destination_hex",
    "lane_hex",
    "haversine_distance",
    "bearing_angle",
    "hex_ring_distance",
]
df[features] = df.apply(lambda row: pd.Series(compute_lane_features_from_row(row)), axis=1)

# %%
# Step 2: Truck Feature Engineering
df.rename(columns={"capacity_mt": "capacity", "length_ft": "length"}, inplace=True)

df["axle_type"] = (
    df["axle_type"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({"MULTI AXLE": "MA", "SINGLE AXLE": "SA"})
)
df["is_multi_axle"] = df["axle_type"].map({"SA": 0, "MA": 1})

df["body_type"] = (
    df["body_type"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace(
        {
            "CONTAINER": "CB",
            "CLOSED BODY": "CB",
            "OPEN BODY": "OB",
            "REEFER": "CB",
            "HIGH CUBE": "HC",
            "FLATBED": "FL",
        }
    )
)

# %%
df.describe()
# %%
# Step 3: Cost Feature Engineering
df.rename(
    columns={
        "Fuel Price - Diesel (INR Rs per liter)": "fuel_price",
        "G-Distance (km)": "g_distance",
    },
    inplace=True,
)

df["fuel_price_per_km_g"] = df.apply(
    lambda row: compute_fuel_price_per_km(row["fuel_price"], row["g_distance"]), axis=1
)
df["fuel_price_per_km_h"] = df.apply(
    lambda row: compute_fuel_price_per_km(row["fuel_price"], row["haversine_distance"]),
    axis=1,
)
df["estimated_fuel_cost_g"] = df.apply(
    lambda row: compute_estimated_fuel_cost(
        row["fuel_price"], row["g_distance"], row["capacity"]
    ),
    axis=1,
)
df["estimated_fuel_cost_h"] = (
    df["fuel_price"] * df["haversine_distance"] * df["capacity"]
)
df["fuel_cost_per_tkm_g"] = df["fuel_price_per_km_g"] / df["capacity"]
df["fuel_cost_per_tkm_h"] = df["fuel_price_per_km_h"] / df["capacity"]

# %%
# Step 4: Temporal Features
df["Date"] = pd.to_datetime(df["Date"])
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year

# %%
# Step 5: Target Variable
df.rename(columns={"Base Charge": "base_price"}, inplace=True)
df["log_base_price"] = np.log1p(df["base_price"])

# %%
# Step 6: Cleanup
columns_to_drop = [
    "Cleaned Origin Name",
    "Cleaned Origin District",
    "Cleaned Origin State",
    "Cleaned Destination Name",
    "Cleaned Destination District",
    "Destination State",
    "Mapped Truck Type",
    "Date",
    "Month",
    "Day",
    "Year",
    "Mapped OrginName",
    "Mapped DestinationName",
    "Cleaned Truck Type",
    "Company",
    "H-Distance (km)",
]

df["lane_identifier"] = (
    df["Cleaned Origin Name"].str.strip()
    + " - "
    + df["Cleaned Origin District"].str.strip()
    + ", "
    + df["Cleaned Origin State"].str.strip()
    + " → "
    + df["Cleaned Destination Name"].str.strip()
    + " - "
    + df["Cleaned Destination District"].str.strip()
    + ", "
    + df["Destination State"].str.strip()
    + " | "
    + df["capacity"].astype(str)
    + "MT_"
    + df["axle_type"]
    + "_"
    + df["body_type"]
)

df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

df = df[df["g_distance"] > 0]
df.dropna(subset=["g_distance", "fuel_price", "base_price"], inplace=True)

# %%
# Step 7: Rename and Reorder Columns
df.rename(
    columns={
        "Origin Latitude": "origin_lat",
        "Origin Longitude": "origin_lng",
        "Destination Latitude": "destination_lat",
        "Destination Longitude": "destination_lng",
        "capacity": "capacity_mt",
        "length": "length_ft",
        "fuel_price": "fuel_price_inr_per_litre",
        "g_distance": "g_distance_km",
        "haversine_distance": "h_distance_km",
        "bearing_angle": "bearing_angle_deg",
    },
    inplace=True,
)

final_columns = [
    "lane_identifier",
    "origin_lat",
    "origin_lng",
    "destination_lat",
    "destination_lng",
    "origin_hex",
    "destination_hex",
    "lane_hex",
    "no_of_wheels",
    "capacity_mt",
    "axle_type",
    "is_multi_axle",
    "body_type",
    "length_ft",
    "fuel_price_inr_per_litre",
    "g_distance_km",
    "h_distance_km",
    "hex_ring_distance",
    "bearing_angle_deg",
    "fuel_price_per_km_g",
    "fuel_price_per_km_h",
    "estimated_fuel_cost_g",
    "estimated_fuel_cost_h",
    "fuel_cost_per_tkm_g",
    "fuel_cost_per_tkm_h",
    "day",
    "month",
    "year",
    "base_price",
    "log_base_price",
]
df = df[final_columns]

# %%
# Step 8: Final Cleaning
numeric_cols = [
    "origin_lat",
    "origin_lng",
    "destination_lat",
    "destination_lng",
    "no_of_wheels",
    "capacity_mt",
    "length_ft",
    "fuel_price_inr_per_litre",
    "g_distance_km",
    "h_distance_km",
    "hex_ring_distance",
    "bearing_angle_deg",
    "fuel_price_per_km_g",
    "fuel_price_per_km_h",
    "estimated_fuel_cost_g",
    "estimated_fuel_cost_h",
    "fuel_cost_per_tkm_g",
    "fuel_cost_per_tkm_h",
    "base_price",
    "log_base_price",
    "is_multi_axle",
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# %%
# Filter suspicious data
df = df[
    (df["origin_hex"] != df["destination_hex"])
    & (df["base_price"] >= 100)
    & (df["g_distance_km"] > 1)
    & (df["h_distance_km"] > 1)
    & (df["bearing_angle_deg"] > 0)
]

# %%
# Apply additional filters
df = df[
    ~((df["bearing_angle_deg"] <= 1) & (df["h_distance_km"] < 10))
    & ~(df["g_distance_km"] < df["h_distance_km"])
]

# %%
# Remove outliers
df = iqr_filter(df, "base_price")
df["log_base_price"] = np.log1p(df["base_price"])

# %%
# Step 9: Visualizations
plt.figure(figsize=(8, 5))
sns.histplot(df["log_base_price"], bins=50, kde=True)
plt.title("Log Base Price Distribution (Filtered)")
plt.xlabel("Log(Base Price + 1) ₹")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="g_distance_km", y="base_price", data=df)
plt.title("Base Price vs Google Distance")
plt.xlabel("Google Distance (km)")
plt.ylabel("Base Price (₹)")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Step 10: Save

# Create output dirs
DATA_VERSION = "v4"
BASE_NAME = f"rpt_prepared_lightgbm_{DATA_VERSION}"
OUTPUT_DIR = f"../../data/processed/{DATA_VERSION}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = f"{OUTPUT_DIR}/{BASE_NAME}.csv"
METADATA_PATH = f"{OUTPUT_DIR}/{BASE_NAME}_metadata.json"


meta = {
    "dataset_name": "rpt_prepared_lightgbm_4.csv",
    "version": "v4",
    "rows": len(df),
    "columns": list(df.columns),
    "filters": [
        "Removed g_distance < h_distance",
        "Filtered bearing_angle <= 1 with h_distance < 10",
        "IQR filter on base_price",
    ],
    "features": {
        "fuel": [
            "fuel_price_per_km_g",
            "fuel_price_per_km_h",
            "estimated_fuel_cost_g",
            "estimated_fuel_cost_h",
        ],
        "target": ["base_price", "log_base_price"],
    },
}

df.to_csv(OUTPUT_PATH, index=False)
with open(METADATA_PATH, "w") as f:
    json.dump(meta, f, indent=4)

print("Saved dataset:", OUTPUT_PATH)
print("Saved metadata:", METADATA_PATH)


# %%
data_path = "../../data/processed/v4/rpt_prepared_lightgbm_v4.csv"
df = pd.read_csv(data_path)

df.describe()
