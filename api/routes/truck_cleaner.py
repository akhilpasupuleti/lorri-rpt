import requests
import json
from utils.constants import CONSTANTS
from utils.llm_utils import read_prompt_from_file
import httpx
from typing import Dict


def clean_truck_gemini(raw_text: str) -> dict:

    filepath = "utils/prompts/truck_cleaning.txt"

    # Read the base prompt from the file
    base_prompt = read_prompt_from_file(filepath)

    # Construct the full prompt with the raw text
    full_prompt = f'{base_prompt}\n\nRaw Truck: "{raw_text}"\n\nOutput:'

    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        CONSTANTS.GEMINI_URL, headers=headers, data=json.dumps(payload)
    )

    if response.status_code == 200:
        candidates = response.json().get("candidates", [])

        if candidates:
            try:
                cleaned_output = candidates[0]["content"]["parts"][0]["text"]
                parsed_json = json.loads(
                    cleaned_output.strip("```json").strip("```").strip()
                )
                parsed_json["raw_truck_name"] = raw_text
                return parsed_json
            except Exception as e:
                raise ValueError(f"Gemini returned unparsable output: {e}")
    raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")


async def clean_truck_gemini_async(raw_text: str) -> Dict:
    filepath = "utils/prompts/truck_cleaning.txt"
    base_prompt = read_prompt_from_file(filepath)
    full_prompt = f'{base_prompt}\n\nRaw Truck: "{raw_text}"\n\nOutput:'

    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            CONSTANTS.GEMINI_URL, headers=headers, json=payload
        )

    if response.status_code == 200:
        candidates = response.json().get("candidates", [])
        if candidates:
            try:
                cleaned_output = candidates[0]["content"]["parts"][0]["text"]
                parsed_json = json.loads(
                    cleaned_output.strip("```json").strip("```").strip()
                )
                parsed_json["raw_truck_name"] = raw_text
                return parsed_json
            except Exception as e:
                raise ValueError(f"Gemini returned unparsable output: {e}")

    raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")
