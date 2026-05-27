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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

# Google Sheets configurations
GOOGLE_SHEETS_CREDENTIALS_JSON = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON", "").strip()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_SHEETS_CREDENTIALS_PATH", 
    os.path.join(BASE_DIR, "credentials.json")
).strip()
GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME", "OCR_EQ_PARSER").strip()
GOOGLE_SHEETS_ENABLED = bool(GOOGLE_SHEETS_CREDENTIALS_JSON) or os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH)

# Bind server port (Render passes PORT environment variable dynamically)
PORT = int(os.getenv("PORT", "8000"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "300"))
PADDLE_MODEL = os.getenv("MODEL", "PaddleOCR-VL-1.5").strip()

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
    
    if not has_paddle and not has_gemini:
        error_msg = (
            "Configuration validation failed. "
            "At least one OCR engine must be configured. "
            "Please define GEMINI_API_KEY or (PADDLE_BASE_URL and PADDLE_API_KEY)."
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
        
    logger.info("Configuration successfully validated.")
    if has_paddle:
        logger.info(f"Paddle OCR Engine: active. Base URL: {PADDLE_BASE_URL}, Model: {PADDLE_MODEL}")
    else:
        logger.warning("Paddle OCR Engine: inactive (PADDLE_BASE_URL or PADDLE_API_KEY not configured)")
        
    if has_gemini:
        logger.info(f"Gemini OCR Engine: active. Model: {GEMINI_MODEL}")
    else:
        logger.warning("Gemini OCR Engine: inactive (GEMINI_API_KEY not configured)")

    if GOOGLE_SHEETS_ENABLED:
        source = "Environment variable (JSON)" if GOOGLE_SHEETS_CREDENTIALS_JSON else f"File ({GOOGLE_SHEETS_CREDENTIALS_PATH})"
        logger.info(f"Google Sheets Integration: active. Sheet Name: {GOOGLE_SHEETS_NAME}, Credentials Source: {source}")
    else:
        logger.warning("Google Sheets Integration: inactive (Neither GOOGLE_SHEETS_CREDENTIALS_JSON env nor local credentials.json file found)")

