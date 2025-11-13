# %%
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

from utils.model_utils import (
    load_model,
    evaluate,
    prepare_categorical,
)

from src.utils.feature_config import (
    TARGET,
    CATEGORICAL_COLS,
)

# %%
# --- Configuration ---
DATA_PATH = "../../data/processed/v4/rpt_prepared_lightgbm_v4.csv"

MODELS = {
    "google": {
        "version": "v1_g",
        "label": "Google Distance",
        "distance_feature": "g_distance_km",
        "fuel_feature": "fuel_price_per_km_g",
    },
    "haversine": {
        "version": "v1_h",
        "label": "Haversine Distance",
        "distance_feature": "h_distance_km",
        "fuel_feature": "fuel_price_per_km_h",
    },
}

# %%
# --- Load data ---
df = pd.read_csv(DATA_PATH)
df_test = df[df["year"] == 2025].copy()
true_target_log = df_test[TARGET].copy()

results = {}

# %%
# --- Evaluate Each Model ---
for key, model_info in MODELS.items():
    model, category_map, config = load_model(model_info["version"])
    features = config["input_features"]

    df_features = df_test[features].copy()

    # Categorical handling
    df_features, _, _ = prepare_categorical(df_features, df_features.copy(), category_map)

    # Predict
    y_pred_log = model.predict(df_features)
    results[key] = {
        "pred_log": y_pred_log,
        "metrics": evaluate(true_target_log, y_pred_log),
        "label": model_info["label"]
    }

# %%
# --- Print Comparison Table ---
print("\n📊 RPT Model Comparison:\n")
metric_names = ["RMSE", "MAE", "MAPE", "R2", "Accuracy (1-MAPE)"]

row_fmt = "{:<22} {:>12.4f} {:>12.4f}"
print("{:<22} {:>12} {:>12}".format("Metric", "Google", "Haversine"))
print("-" * 50)
for metric in metric_names:
    g = results["google"]["metrics"][metric]
    h = results["haversine"]["metrics"][metric]
    print(row_fmt.format(metric, g, h))

# %%
# --- Optional Plots ---
sns.set(style="whitegrid")

# %%
# Error distributions
plt.figure(figsize=(10, 5))
for key, color in zip(["google", "haversine"], ["blue", "green"]):
    # y_true = true_target_log.apply(lambda x: pd.np.expm1(x))
    # y_pred = pd.np.expm1(results[key]["pred_log"])
    y_true = np.expm1(true_target_log)
    y_pred = np.expm1(results[key]["pred_log"])
    
    errors = y_true - y_pred
    sns.kdeplot(errors, label=f"{results[key]['label']} Error", shade=True)
plt.title("Prediction Error Distribution")
plt.xlabel("Actual - Predicted Freight Rate")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Scatter actual vs predicted
plt.figure(figsize=(6, 6))
# y_true = pd.np.expm1(true_target_log)
y_true = np.expm1(true_target_log)
for key, color in zip(["google", "haversine"], ["blue", "green"]):
    # y_pred = pd.np.expm1(results[key]["pred_log"])
    y_pred = np.expm1(results[key]["pred_log"])
    plt.scatter(y_true, y_pred, label=results[key]["label"], alpha=0.3, s=10)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Perfect")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Freight Rates")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# ----------------------------------------
# Residuals vs. Predicted Plot
# ----------------------------------------
plt.figure(figsize=(10, 5))
for key, color in zip(["google", "haversine"], ["blue", "green"]):
    y_pred = np.expm1(results[key]["pred_log"])
    y_true = np.expm1(true_target_log)
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals, label=results[key]["label"], alpha=0.3, s=10)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Predicted Freight Rates")
plt.xlabel("Predicted Rate")
plt.ylabel("Residual (Actual - Predicted)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ----------------------------------------
# Top 10 Absolute Errors
# ----------------------------------------
error_df = pd.DataFrame({"actual": np.expm1(true_target_log)})

for key in results:
    error_df[f"{key}_predicted"] = np.expm1(results[key]["pred_log"])
    error_df[f"{key}_abs_error"] = (error_df["actual"] - error_df[f"{key}_predicted"]).abs()

for key in results:
    print(f"\n🔎 Top 10 Absolute Errors for {results[key]['label']}:")
    display_cols = ["actual", f"{key}_predicted", f"{key}_abs_error"]
    print(error_df[display_cols].sort_values(by=f"{key}_abs_error", ascending=False).head(10))

# %%
# ----------------------------------------
# SHAP Feature Importance Comparison
# ----------------------------------------
import shap

# Summary plots for both models
for key in results:
    print(f"\n🧬 SHAP Feature Importance for {results[key]['label']}:")
    model, category_map, config = load_model(MODELS[key]["version"])
    features = config["input_features"]

    df_features = df_test[features].copy()
    df_features, _, _ = prepare_categorical(df_features, df_features.copy(), category_map)

    explainer = shap.Explainer(model)
    shap_values = explainer(df_features)

    shap.plots.beeswarm(shap_values, max_display=15, show=True)


# %%
# ----------------------------------------
# Lane-Level Error Investigation (Google and Haversine)
# ----------------------------------------

def create_error_df(model_key: str, label: str) -> pd.DataFrame:
    model, category_map, config = load_model(MODELS[model_key]["version"]   )
    features = config["input_features"]

    # Subset for required features
    df_features = df_test[features].copy()
    df_features, _, _ = prepare_categorical(df_features, df_features.copy(), category_map)

    y_true = np.expm1(true_target_log)
    y_pred = np.expm1(results[model_key]["pred_log"])

    error_df = df_features.copy()
    error_df["actual"] = y_true
    error_df["predicted"] = y_pred
    error_df["abs_error"] = (y_true - y_pred).abs()
    error_df["relative_error_%"] = (error_df["abs_error"] / y_true * 100).round(2)
    error_df["model"] = label

    # Optional: extract origin/destination hexes or raw lat/lngs if available
    cols = [
        "origin_hex", "destination_hex", "lane_hex",
        "actual", "predicted", "abs_error", "relative_error_%",
        "no_of_wheels", "capacity_mt", "axle_type", "body_type",
        "day", "month", "year"
    ]
    return error_df[cols] if all(c in error_df.columns for c in cols) else error_df

# Create error tables for both models
df_errors_google = create_error_df("google", "Google Distance")
df_errors_haversine = create_error_df("haversine", "Haversine Distance")

# Combine both for side-by-side comparison (optional)
df_errors_combined = pd.concat([df_errors_google, df_errors_haversine], axis=0, ignore_index=True)

# Show worst offending lanes
top_error_lanes = df_errors_combined.sort_values(by="abs_error", ascending=False).head(20)
print("\n🚨 Top Error Lanes (Combined View):")
print(top_error_lanes.to_string(index=False))


# %%
df_errors_combined.to_csv("lane_error_report.csv", index=False)


# %%
lane_group_stats = df_errors_combined.groupby("lane_hex").agg({
    "abs_error": ["mean", "max", "count"],
    "relative_error_%": "mean"
}).sort_values(by=("abs_error", "mean"), ascending=False)

print(lane_group_stats.head(10))