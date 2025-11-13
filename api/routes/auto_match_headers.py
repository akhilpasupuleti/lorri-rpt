import requests
import json
from utils.constants import CONSTANTS
from utils.llm_utils import read_prompt_from_file


def auto_match_headers(headers, sample_rows):
    GEMINI_URL = CONSTANTS.GEMINI_URL

    filepath = "utils/prompts/auto_header_matching.txt"
    prompt_template = read_prompt_from_file(filepath)

    # Replace placeholders safely
    prompt = prompt_template.replace("{{headers}}", json.dumps(headers)).replace(
        "{{sample_rows}}", json.dumps(sample_rows)
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers_req = {"Content-Type": "application/json"}

    response = requests.post(GEMINI_URL, headers=headers_req, data=json.dumps(payload))

    if response.status_code == 200:
        candidates = response.json().get("candidates", [])
        if candidates:
            try:
                mapping_text = candidates[0]["content"]["parts"][0]["text"]
                parsed = json.loads(mapping_text.strip("```json").strip("```").strip())

                mapping = parsed.get("mapping", {})
                missing = parsed.get("missing_fields", [])

                if missing:
                    return {
                        "mapping": mapping,
                        "error": f"Missing fields: {', '.join(missing)}",
                        "missing_fields": missing,
                    }
                else:
                    return {"mapping": mapping, "missing_fields": []}

            except Exception as e:
                raise ValueError(f"Gemini returned unparsable output: {e}")
    else:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")
