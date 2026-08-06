import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Bind configurations with backward compatibility
PADDLE_BASE_URL = os.getenv("PADDLE_BASE_URL", os.getenv("JOB_URL", "")).strip()
PADDLE_API_KEY = os.getenv("PADDLE_API_KEY", os.getenv("TOKEN", "")).strip()
PADDLE_API_SECRET = os.getenv("PADDLE_API_SECRET", "").strip()

# Gemini OCR configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY.startswith("your_gemini_api_key"):
    GEMINI_API_KEY = ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "").strip()
if GEMINI_API_KEY_2.startswith("your_gemini_api_key"):
    GEMINI_API_KEY_2 = ""
GEMINI_MODEL_2 = os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash-lite").strip()

GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3", "").strip()
if GEMINI_API_KEY_3.startswith("your_gemini_api_key"):
    GEMINI_API_KEY_3 = ""
GEMINI_MODEL_3 = os.getenv("GEMINI_MODEL_3", "gemini-2.5-pro").strip()

GEMINI_API_KEY_4 = os.getenv("GEMINI_API_KEY_4", "").strip()
if GEMINI_API_KEY_4.startswith("your_gemini_api_key"):
    GEMINI_API_KEY_4 = ""
GEMINI_MODEL_4 = os.getenv("GEMINI_MODEL_4", "gemini-2.5-pro").strip()

GEMINI_API_KEY_5 = os.getenv("GEMINI_API_KEY_5", "").strip()
if GEMINI_API_KEY_5.startswith("your_gemini_api_key"):
    GEMINI_API_KEY_5 = ""
GEMINI_MODEL_5 = os.getenv("GEMINI_MODEL_5", "gemini-2.5-pro").strip()

# Google Sheets configurations
GOOGLE_SHEETS_CREDENTIALS_JSON = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON", "").strip()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_SHEETS_CREDENTIALS_PATH", 
    os.path.join(BASE_DIR, "credentials.json")
).strip()
GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME", "OCR_EQ_PARSER").strip()
GOOGLE_SHEETS_ENABLED = bool(GOOGLE_SHEETS_CREDENTIALS_JSON) or os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH)
# Set to "false" in .env to allow all rows through without duplicate filtering
GOOGLE_SHEETS_DEDUP_ENABLED = os.getenv("GOOGLE_SHEETS_DEDUP_ENABLED", "true").strip().lower() not in ("false", "0", "no")

# Bind server port (Render passes PORT environment variable dynamically)
PORT = int(os.getenv("PORT", "8000"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "0.5"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "300"))
PADDLE_MODEL = os.getenv("MODEL", "PaddleOCR-VL-1.6").strip()
IMAGE_MAX_DIMENSION = int(os.getenv("IMAGE_MAX_DIMENSION", "1600"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr-api")

def validate_config():
    """Validates required configurations on startup."""
    logger.info("Validating configuration on startup...")
    
    # At least one OCR engine must have credentials
    has_paddle = bool(PADDLE_BASE_URL and PADDLE_API_KEY)
    has_gemini = bool(GEMINI_API_KEY)
    has_gemini_2 = bool(GEMINI_API_KEY_2)
    has_gemini_3 = bool(GEMINI_API_KEY_3)
    has_gemini_4 = bool(GEMINI_API_KEY_4)
    has_gemini_5 = bool(GEMINI_API_KEY_5)
    
    if not has_paddle and not has_gemini and not has_gemini_2 and not has_gemini_3 and not has_gemini_4 and not has_gemini_5:
        error_msg = (
            "Configuration validation failed. "
            "At least one OCR engine must be configured. "
            "Please define GEMINI_API_KEY or PADDLE variables."
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
        
    logger.info("Configuration successfully validated.")
    if has_paddle:
        logger.info(f"Paddle OCR Engine: active. Base URL: {PADDLE_BASE_URL}, Model: {PADDLE_MODEL}")
    else:
        logger.warning("Paddle OCR Engine: inactive (PADDLE_BASE_URL or PADDLE_API_KEY not configured)")
        
    if has_gemini:
        logger.info(f"Gemini OCR Engine 1: active. Model: {GEMINI_MODEL}")
    else:
        logger.warning("Gemini OCR Engine 1: inactive (GEMINI_API_KEY not configured)")

    if has_gemini_2:
        logger.info(f"Gemini OCR Engine 2: active. Model: {GEMINI_MODEL_2}")
    else:
        logger.warning("Gemini OCR Engine 2: inactive (GEMINI_API_KEY_2 not configured)")

    if has_gemini_3:
        logger.info(f"Gemini OCR Engine 3: active. Model: {GEMINI_MODEL_3}")
    else:
        logger.warning("Gemini OCR Engine 3: inactive (GEMINI_API_KEY_3 not configured)")
    if has_gemini_4:
        logger.info(f"Gemini OCR Engine 4: active. Model: {GEMINI_MODEL_4}")
    else:
        logger.warning("Gemini OCR Engine 4: inactive (GEMINI_API_KEY_4 not configured)")
    if has_gemini_5:
        logger.info(f"Gemini OCR Engine 5: active. Model: {GEMINI_MODEL_5}")
    else:
        logger.warning("Gemini OCR Engine 5: inactive (GEMINI_API_KEY_5 not configured)")

    if GOOGLE_SHEETS_ENABLED:
        source = "Environment variable (JSON)" if GOOGLE_SHEETS_CREDENTIALS_JSON else f"File ({GOOGLE_SHEETS_CREDENTIALS_PATH})"
        logger.info(f"Google Sheets Integration: active. Sheet Name: {GOOGLE_SHEETS_NAME}, Credentials Source: {source}")
    else:
        logger.warning("Google Sheets Integration: inactive (Neither GOOGLE_SHEETS_CREDENTIALS_JSON env nor local credentials.json file found)")
