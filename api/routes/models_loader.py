import os
import joblib
import json

from utils.constants import CONSTANTS

MODEL_VERSION = CONSTANTS.MODEL_VERSION
MODEL_DIR = os.path.join("models", MODEL_VERSION)


class ModelLoader:
    def __init__(self):
        self.model = None
        self.category_map = None
        self.metadata = None
        self.load_model()

    def load_model(self):
        self.model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        with open(os.path.join(MODEL_DIR, "category_map.pkl"), "rb") as f:
            self.category_map = joblib.load(f)
        with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
            self.metadata = json.load(f)

    def get_model(self):
        return self.model

    def get_category_map(self):
        return self.category_map

    def get_metadata(self):
        return self.metadata


# Singleton instance
model_loader = ModelLoader()
