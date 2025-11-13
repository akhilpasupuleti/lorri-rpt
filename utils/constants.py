from dotenv import load_dotenv
import os

# Default to "dev" if not set
env_name = os.getenv("RPT_ENV", "dev")
load_dotenv(dotenv_path=f".env.{env_name}")


class CONSTANTS:
    MODEL_VERSION = os.getenv("MODEL_VERSION", "v1_h")
    H3_RES = int(os.getenv("H3_RESOLUTION", 6))
    DEFAULT_FUEL_PRICE = float(os.getenv("DEFAULT_FUEL_PRICE", 95))

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    BASE_PROMPT = os.getenv("GEMINI_BASE_PROMPT")
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
