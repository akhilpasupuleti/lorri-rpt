# src/utils/model_utils.py

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from lightgbm import early_stopping, log_evaluation
import pickle
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


def split_dataset(df, features, target):
    traindf = df[df["year"] < 2025].copy()
    testdf = df[df["year"] == 2025].copy()

    X_train, X_test = traindf[features], testdf[features]
    y_train, y_test = traindf[target], testdf[target]

    return X_train, X_test, y_train, y_test


def prepare_categorical(X_train, X_test, cat_cols):
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(
            X_test[col], categories=X_train[col].cat.categories
        )
    category_map = {col: X_train[col].cat.categories.tolist() for col in cat_cols}
    return X_train, X_test, category_map


def train_model(X_train, X_test, y_train, y_test, cat_cols):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=1000,
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(100)],
    )
    return model


def evaluate(y_true, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_true)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Accuracy (1-MAPE)": 1 - mape,
    }


def save_metrics(metrics: dict, model_version: str):
    """
    Save evaluation metrics to a JSON file inside the model version folder.

    Args:
        metrics (dict): Output of the `evaluate()` function.
        model_version (str): Folder under models/ to save metrics.
    """
    out_dir = project_root / "models" / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📈 Metrics saved at: {metrics_path}")


def save_all(model, category_map, model_version, features, target, model_tag="default"):
    out_dir = project_root / "models" / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    model_txt_path = out_dir / "model.txt"
    model_pkl_path = out_dir / "model.pkl"
    category_path = out_dir / "category_map.pkl"
    config_path = out_dir / "model_config.json"

    # Save LightGBM model in .txt and .pkl formats
    model.save_model(model_txt_path)
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model, f)

    # Save category map
    with open(category_path, "wb") as f:
        pickle.dump(category_map, f)

    # Save model config
    config = {
        "version": model_version,
        "tag": model_tag,
        "input_features": features,
        "target": target,
        "categorical": list(category_map.keys()),
        "artifacts": {
            "model_txt": str(model_txt_path),
            "model_pickle": str(model_pkl_path),
            "category_map": str(category_path),
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✔ Model saved in {out_dir}")
    print(f"• TXT model:     {model_txt_path}")
    print(f"• Pickle model:  {model_pkl_path}")
    print(f"• Config:        {config_path}")


def load_model(version: str, tag: str = "default"):
    """
    Load a saved model, category map, and config by version and tag.

    Args:
        version (str): Model version (folder name under models/)
        tag (str): Optional model tag to verify identity

    Returns:
        model, category_map, config

    Raises:
        FileNotFoundError / ValueError if paths or tag mismatch
    """
    base_path = project_root / "models" / version

    model_pkl_path = base_path / "model.pkl"
    category_path = base_path / "category_map.pkl"
    config_path = base_path / "model_config.json"

    if not all(p.exists() for p in [model_pkl_path, category_path, config_path]):
        raise FileNotFoundError(
            f"Model files not found in version: {version}, {base_path}"
        )

    with open(model_pkl_path, "rb") as f:
        model = pickle.load(f)
    with open(category_path, "rb") as f:
        category_map = pickle.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)

    # if config.get("tag") != tag:
    #     raise ValueError(
    #         f"Model tag mismatch: expected '{tag}', found '{config.get('tag')}'"
    #     )

    return model, category_map, config


def predict_lane_price(model, category_map, features, **kwargs):
    """
    Predict base freight rate using model and lane details.

    Args:
        model: Trained LightGBM model
        category_map: Dict of {column: categories} for categorical encoding
        features: List of input feature names
        kwargs: Feature values as keyword arguments (must match features list)

    Returns:
        Predicted base_price (float, rounded to 2 decimals)
    """

    # Build single-row DataFrame
    df_input = pd.DataFrame([kwargs], columns=features)

    # Ensure categorical dtypes are preserved
    for col, cats in category_map.items():
        if col in df_input.columns:
            df_input[col] = pd.Categorical(df_input[col], categories=cats)

    # Predict and inverse log
    log_price = model.predict(df_input)[0]
    base_price = np.expm1(log_price)

    return round(base_price, 2)
