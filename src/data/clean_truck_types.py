# %%
import requests
import json
import time
import pandas as pd
import os

# %%

BASE_PROMPT = """You are a logistics expert specializing in Indian truck classification.

Your task is to convert raw truck type descriptions into standardized truck codes using the following structure:

<WheelConfig>_<CapacityCode>_<AxleCode>_<BodyTypeCode>_<LengthCode>

You MUST return output in this format:

{
  "raw_input": "<Original input string>",
  "cleaned_code": "<Standard truck code>",
  "dimensions": {
    "no_of_wheels": <int>,
    "capacity_mt": <float>,
    "axle_type": "<decoded axle label>",
    "body_type": "<decoded body label>",
    "length_ft": <int>
  },
  "reasoning": "<One-line reasoning why this mapping was chosen>"
}

Important rules:
- The cleaned_code **must contain all 5 parts**, in this exact order: WheelConfig, CapacityCode, AxleCode, BodyTypeCode, LengthCode (e.g., `6WL_5MT_SA_CL_L14`). Never skip or omit any part.
- Dimensions must exactly match the components in `cleaned_code` (do NOT guess them from the raw input separately).
- If a truck is more than **9 MT**, it is assumed to be **Multi Axle (MA)**.
- If capacity ≤ 9 MT, it is assumed to be **Single Axle (SA)**.
- If capacity > 9 MT and multi axle, assume it is a **10-wheeler (10WL)**.
- If single axle and closer to 9 MT, assume **6WL**.
- If closer to 3 MT, assume **4WL**.
- If capacity is > 20 MT, assume **12WL** by default unless explicitly a trailer.
- If capacity is **less than 18 MT**, do NOT assign wheel configs above 10WL — e.g., do not use 12WL, 14WL, 18WL, etc. for <18MT trucks.
- Always prefer the **lowest sufficient** wheel config for a given capacity (e.g., don't over-assign).

If body type is not explicitly mentioned, assume it is a **Closed Body (CB)**.

Use the following references:

[Wheel Config]
- 4WL: 4-wheeler
- 6WL: 6-wheeler
- 10WL: 10-wheeler (≈18MT)
- 12WL: 12-wheeler
- 14WL: 14-wheeler
- 18WL: 18-wheeler
- 22WL: 22-wheeler
- TRAILER: Tractor-Trailer
- UNK: Unknown

[Capacity Code]
- 1MT → 0.5-1.5 MT
- 2MT → 1-3 MT
- 3MT → 2.5-3.5 MT
- 5MT → 4-6 MT
- 7MT → 6.5-7.5 MT
- 8MT → 7.5-8.5 MT - Ususally 10 wheelers
- 9MT → 8.5-9.5 MT
- 10MT → 9.5-11 MT
- 12MT → 11-13 MT
- 16MT → 15-17 MT
- 18MT → 17-19 MT
- 19MT → 18-20 MT
- 20MT → 19-21 MT
- 21MT → 20-22 MT
- 25MT → 24-26 MT
- 28MT → 26-29 MT
- 32MT → 31-33 MT
- 40MT+ → 40+ MT
- UNK

[Axle Type]
- SA: Single Axle (≤9MT)
- MA: Multi Axle (>9MT)
- UNK

[Body Type]
- OB: Open Body
- CL: Closed Body
- CT: Container
- HC: High Cube
- RF: Reefer (look for ° or minus signs)
- FB: Flatbed (ODC)
- TL: Tanker
- UNK

[Length Code]
- L8, L10, L14, L17, L19, L20, L22, L24, L28, L32, L36, L40, L45+, UNK

Example:
Raw Input: "2 MT closed body"

Output:
{
  "raw_input": "2 MT closed body",
  "cleaned_code": "4WL_2MT_SA_CL_L10",
  "dimensions": {
    "no_of_wheels": 4,
    "capacity_mt": 2,
    "axle_type": "Single Axle",
    "body_type": "Closed Body",
    "length_ft": 10
  },
  "reasoning": "2MT trucks are usually 4-wheelers with single axle and closed body, typically 10 ft long."
}

THINK CAREFULLY. The `dimensions` object must be derived from the `cleaned_code`, not from assumptions about the raw input.
"""

API_KEY = "AIzaSyAir8ATgQ8Bvm2lG8kzz4netXJvSm-2vwY"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# %%

# Load from Excel or CSV
INPUT_FILE = "../raw_truck_inputs.xlsx"
COLUMN_NAME = "Truck Type"

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"🚫 File not found: {INPUT_FILE}")

# Detect file type
if INPUT_FILE.endswith(".csv"):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

# Extract list of raw truck names
raw_truck_inputs = df[COLUMN_NAME].dropna().astype(str).str.strip().tolist()

length_of_raw = len(raw_truck_inputs)
print(f"✅ Loaded {length_of_raw} raw truck inputs.")


# %%
# Normalization function
def normalize_raw_truck(raw):
    return raw.replace(" ", "").upper()


# %%
# Load cache
CACHE_FILE = "cleaned_truck_outputs2.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        results = json.load(f)
else:
    results = []

# Build dict cache: norm_key → entry
cache_map = {
    entry.get("normalized_key", normalize_raw_truck(entry.get("raw", ""))): entry
    for entry in results
    if entry.get("status") != "cached"
}

print(f"🔁 Loaded {len(cache_map)} cached entries.")


# %%
# Gemini request function
def call_gemini_api(prompt):
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        candidates = response.json().get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"]
        return "No response from Gemini"
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


# %%
import random

# Run batch cleaning
for idx, raw in enumerate(raw_truck_inputs, start=1):
    norm_key = normalize_raw_truck(raw)

    # ✅ Skip if key was added during earlier iteration (in-batch duplicate)
    if norm_key in cache_map:
        print(f"✅ Skipping (already cleaned): {raw}")
        cache_entry = cache_map[norm_key].copy()
        cache_entry["status"] = "cached"
        cache_entry["raw"] = raw
        cache_entry["normalized_key"] = norm_key
        results.append(cache_entry)
        continue

    full_prompt = BASE_PROMPT + f'\n\nRaw Truck: "{raw}"\n\nOutput:'
    output_text = call_gemini_api(full_prompt)

    cleaned_output = output_text.strip().strip("```json").strip("```").strip()

    try:
        parsed_json = json.loads(cleaned_output)
        parsed_json["raw"] = raw
        parsed_json["normalized_key"] = norm_key
        parsed_json["status"] = "cleaned"
        results.append(parsed_json)
        cache_map[norm_key] = parsed_json
    except json.JSONDecodeError as e:
        print("⚠️  Failed to parse JSON from LLM response")
        print("Error:", e)
        results.append(
            {
                "raw": raw,
                "normalized_key": norm_key,
                "status": "failed",
                "output_raw": output_text,
            }
        )

    print(f"✅ Done [{idx}/{len(raw_truck_inputs)}]: {raw}")
    print("-" * 60)

    time.sleep(0.5 + random.uniform(0, 0.35))


# %%
# Save updated JSON cache
with open(CACHE_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved cleaned results to {CACHE_FILE}")


# %%

# Flatten results to write to Excel
excel_rows = []

for entry in results:
    if "cleaned_code" in entry:
        excel_rows.append(
            {
                "raw": entry.get("raw", ""),
                "normalized_key": entry.get("normalized_key", ""),
                "cleaned_code": entry.get("cleaned_code", ""),
                "no_of_wheels": entry["dimensions"].get("no_of_wheels", ""),
                "capacity_mt": entry["dimensions"].get("capacity_mt", ""),
                "axle_type": entry["dimensions"].get("axle_type", ""),
                "body_type": entry["dimensions"].get("body_type", ""),
                "length_ft": entry["dimensions"].get("length_ft", ""),
                "reasoning": entry.get("reasoning", ""),
                "status": entry.get("status", ""),
            }
        )
    else:
        excel_rows.append(
            {
                "raw": entry.get("raw", ""),
                "normalized_key": entry.get("normalized_key", ""),
                "cleaned_code": "",
                "no_of_wheels": "",
                "capacity_mt": "",
                "axle_type": "",
                "body_type": "",
                "length_ft": "",
                "reasoning": f"FAILED: {entry.get('output_raw', '')[:100]}",
                "status": entry.get("status", ""),
            }
        )

# Create DataFrame and write to Excel
EXCEL_OUTPUT = "apollo_cleaned_trucks_2.xlsx"
df_out = pd.DataFrame(excel_rows)
df_out.to_excel(EXCEL_OUTPUT, index=False)
print(f"📄 Saved cleaned Excel to {EXCEL_OUTPUT}")
