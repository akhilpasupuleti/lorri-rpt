from fastapi import FastAPI, APIRouter, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    RPTRequest,
    RPTResponse,
    TruckCleanRequest,
    TruckCleanResponse,
    AutoMatchHeadersRequest,
    AutoMatchHeadersResponse,
    BulkTruckCleanRequest,
)
from api.routes.predict import predict_rate
from api.routes.truck_cleaner import clean_truck_gemini, clean_truck_gemini_async
from api.routes.auto_match_headers import auto_match_headers
from typing import List
import joblib
import os
import json
import asyncio


from utils.constants import CONSTANTS

app = FastAPI(title="Freight Rate Prediction API")


origins = [
    "http://localhost",
    "http://localhost:3000",
    "*",  # REMOVE this in production!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API router with prefix
api_router = APIRouter(prefix="/api")


@app.on_event("startup")
def load_model_once():
    model_dir = os.path.join("models", CONSTANTS.MODEL_VERSION)

    # load model
    model_path = os.path.join(model_dir, "model.pkl")
    app.state.model = joblib.load(model_path)

    model_config_path = os.path.join(model_dir, "model_config.json")
    with open(model_config_path, "r") as f:
        app.state.model_config = json.load(f)

    # load category map
    category_map_path = os.path.join(model_dir, "category_map.pkl")
    app.state.category_map = joblib.load(category_map_path)

    print("Model, category map, and config loaded successfully.")


@app.get("/")
def root():
    return {"status": "RPT API running"}


@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
    }


@api_router.post("/clean_truck", response_model=TruckCleanResponse)
def clean_truck(req: TruckCleanRequest):
    result = clean_truck_gemini(req.raw_truck_name)
    return TruckCleanResponse(**result)


@api_router.post("/auto_match_headers", response_model=AutoMatchHeadersResponse)
def match_headers(req: AutoMatchHeadersRequest):
    result = auto_match_headers(req.headers, req.sample_rows)
    return AutoMatchHeadersResponse(**result)


@api_router.post("/predict", response_model=RPTResponse)
def predict(request: RPTRequest, req: Request):
    model = req.app.state.model
    category_map = req.app.state.category_map
    model_config = req.app.state.model_config

    return predict_rate(
        request,
        model=model,
        category_map=category_map,
        model_config=model_config,
    )


@api_router.post("/bulk_clean_truck")
async def bulk_clean_truck(req: BulkTruckCleanRequest) -> List[TruckCleanResponse]:
    tasks = []
    for raw in req.raw_truck_names:
        tasks.append(clean_truck_gemini_async(raw))

    cleaned_trucks = []
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for raw, result in zip(req.raw_truck_names, results):
        if isinstance(result, Exception):
            cleaned_trucks.append(
                {
                    "raw_truck_name": raw,
                    "cleaned_code": "",
                    "dimensions": {},
                    "error": str(result),
                }
            )
        else:
            cleaned_trucks.append(result)

    return cleaned_trucks


@api_router.post("/bulk_predict", response_model=List[RPTResponse])
def bulk_predict(
    request: Request,
    lanes: List[RPTRequest] = Body(...),
):
    model = request.app.state.model
    category_map = request.app.state.category_map
    model_config = request.app.state.model_config

    results = []
    for lane in lanes:
        try:
            response = predict_rate(
                request=lane,
                model=model,
                category_map=category_map,
                model_config=model_config,
            )
            results.append(response)
        except Exception as e:
            # include error in response list
            results.append(
                RPTResponse(
                    predicted_base_price=None,
                    model_version=CONSTANTS.MODEL_VERSION,
                    input_features={},
                    error=str(e),
                )
            )

    return results


app.include_router(api_router)
