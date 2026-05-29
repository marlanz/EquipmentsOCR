import os
import re
import io
import json
import unicodedata

import time
import asyncio
import httpx
import logging
from typing import Optional, Dict, List
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import gspread
from google.oauth2 import service_account

from app.config import (
    PADDLE_BASE_URL,
    PADDLE_API_KEY,
    PADDLE_MODEL,
    POLL_INTERVAL,
    MAX_WAIT_SECONDS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GOOGLE_SHEETS_CREDENTIALS_JSON,
    GOOGLE_SHEETS_CREDENTIALS_PATH,
    GOOGLE_SHEETS_NAME,
    GOOGLE_SHEETS_ENABLED,
    logger,
)
from app.schemas import OCRResult

# --- PaddleOCR Validation Limits ---
PADDLE_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB limit
PADDLE_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
PADDLE_ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "application/pdf"}

# --- Gemini Validation Limits ---
GEMINI_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
GEMINI_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
GEMINI_ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

# Initialize Gemini Client if API key is present
client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    logger.info("Initializing Google GenAI client...")
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not configured. client will be initialized as None.")


class EquipmentOCRResult(BaseModel):
    """Structured response schema returned directly from the Gemini OCR generation call."""
    markdown: str = Field(
        description=(
            "All extracted text formatted as a markdown string, "
            "preserving the label's line structure. "
            "Example: 'MÁY HÀN CO2\\n\\nMã MMTB : B001\\n\\nModel : X1'"
        )
    )
    machine_name: str = Field(description="Full machine/equipment name as it appears on the label")
    ma_mmtb: str = Field(description="Equipment ID code labelled 'Mã MMTB'")
    model: str = Field(description="Model number labelled 'Model'")
    xuong: str = Field(description="Workshop / xưởng value")
    vi_tri: str = Field(description="Location value labelled 'Vị trí'")


# ── PaddleOCR Helper Functions ──────────────────────────────────────────────

def parse_markdown_to_key_value(markdown_text: str) -> Dict[str, str]:
    """Extracts structured key-value maps from PaddleOCR's page markdown output."""
    key_value = {}
    if not markdown_text:
        return key_value

    lines = [line.strip() for line in markdown_text.split("\n") if line.strip()]
    if not lines:
        return key_value

    # 1. Heading/Title identification on first line
    first_line = lines[0]
    first_line_clean = re.sub(r"^#+\s*", "", first_line).strip()
    
    if first_line_clean and ":" not in first_line_clean and "：" not in first_line_clean:
        key_value["machine_name"] = first_line_clean

    # 2. Key-Value pairs capture
    kv_pattern = re.compile(r"^\s*(?:\*\*)?\s*([^*：:]+?)\s*(?:\*\*)?\s*[:：]\s*(.*)$")

    for line in lines:
        match = kv_pattern.match(line)
        if match:
            k = match.group(1).strip()
            v = match.group(2).strip()

            k = re.sub(r"^\*+\s*|\s*\*+$", "", k).strip()
            v = re.sub(r"^\*+\s*|\s*\*+$", "", v).strip()

            if k and v:
                key_value[k] = v

    return key_value


def validate_upload_paddle(filename: str, content_type: str, file_size: int):
    """Checks uploaded file extensions, MIME-types, and size limits for PaddleOCR."""
    import os

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has an empty filename."
        )

    ext = os.path.splitext(filename.lower())[1]
    if ext not in PADDLE_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(PADDLE_ALLOWED_EXTENSIONS)}"
        )

    if content_type not in PADDLE_ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{content_type}'. Allowed types: {', '.join(PADDLE_ALLOWED_MIME_TYPES)}"
        )

    if file_size > PADDLE_MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size. Size: {file_size} bytes, Max: {PADDLE_MAX_FILE_SIZE} bytes (20MB)."
        )


async def submit_ocr_job(filename: str, file_bytes: bytes, content_type: str) -> str:
    """Uploads file bytes directly to the Paddle OCR API in memory."""
    headers = {
        "Authorization": f"bearer {PADDLE_API_KEY}"
    }
    files = {
        "file": (filename, file_bytes, content_type)
    }
    data = {
        "model": PADDLE_MODEL,
        "optionalPayload": json.dumps({
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        })
    }

    logger.info(f"Submitting in-memory OCR job for file '{filename}'...")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                PADDLE_BASE_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=120.0
            )
        except httpx.TimeoutException:
            logger.error("Timeout occurred while submitting file to Paddle API")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Paddle API request timed out during submission."
            )
        except httpx.RequestError as exc:
            logger.error(f"HTTP request error during submission: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to communicate with Paddle API: {exc}"
            )

    if r.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Paddle API authentication failure. Please check your token."
        )
    
    if r.status_code != 200:
        logger.error(f"Paddle API submission returned status {r.status_code}: {r.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream Paddle API submission failed with status {r.status_code}."
        )

    try:
        response_json = r.json()
        job_id = response_json["data"]["jobId"]
        logger.info(f"OCR Job submitted successfully. Job ID: {job_id}")
        return job_id
    except (KeyError, ValueError) as exc:
        logger.error(f"Malformed submission response from Paddle: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Malformed OCR submission response from Paddle API."
        )


async def poll_ocr_job(job_id: str) -> str:
    """Polls the Paddle API status until 'done' or 'failed'."""
    headers = {
        "Authorization": f"bearer {PADDLE_API_KEY}"
    }
    status_url = f"{PADDLE_BASE_URL}/{job_id}"
    start_time = time.time()

    logger.info(f"Starting polling loop for Job ID {job_id}...")
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < MAX_WAIT_SECONDS:
            try:
                r = await client.get(status_url, headers=headers, timeout=30.0)
            except httpx.TimeoutException:
                logger.warning(f"Timeout checking status for Job {job_id}, retrying...")
                await asyncio.sleep(POLL_INTERVAL)
                continue
            except httpx.RequestError as exc:
                logger.error(f"Error querying status for Job {job_id}: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to communicate with Paddle status API: {exc}"
                )

            if r.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Paddle API authentication failed during status polling."
                )

            if r.status_code != 200:
                logger.error(f"Downstream Paddle status query returned status {r.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Downstream Paddle API status check failed: Status {r.status_code}."
                )

            try:
                res_data = r.json()
                data = res_data["data"]
                state = data["state"]
            except (KeyError, ValueError) as exc:
                logger.error(f"Malformed status response for Job {job_id}: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Malformed status response from Paddle API."
                )

            if state == "done":
                return data["resultUrl"]["jsonUrl"]

            if state == "failed":
                error_msg = data.get("errorMsg", "OCR job failed")
                logger.error(f"OCR Job {job_id} failed on Paddle server: {error_msg}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Downstream Paddle OCR job failed: {error_msg}"
                )

            logger.info(f"Job {job_id} is '{state}'. Sleeping {POLL_INTERVAL}s...")
            await asyncio.sleep(POLL_INTERVAL)

        logger.error(f"Polling timed out for Job ID {job_id} after {MAX_WAIT_SECONDS} seconds.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Paddle OCR job polling timed out."
        )


async def download_and_parse_jsonl(jsonl_url: str) -> List[OCRResult]:
    """Downloads JSONL layout result from Paddle and parses structured pages."""
    logger.info(f"Downloading OCR result from: {jsonl_url}")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(jsonl_url, timeout=60.0)
            r.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timeout downloading results from Paddle storage."
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to download results from Paddle: Status {exc.response.status_code}."
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Network error downloading results from Paddle: {exc}"
            )

    pages = []
    try:
        lines = r.text.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            line_data = json.loads(line)
            result = line_data.get("result", {})

            for res in result.get("layoutParsingResults", []):
                markdown_text = res.get("markdown", {}).get("text", "")
                key_value = parse_markdown_to_key_value(markdown_text)

                pages.append(
                    OCRResult(
                        markdown=markdown_text,
                        key_value=key_value
                    )
                )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error(f"Error parsing final JSONL results: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse malformed OCR JSONL result from Paddle."
        )

    return pages


async def check_paddle_connectivity() -> bool:
    """Validates external endpoint availability for PaddleOCR."""
    if not PADDLE_API_KEY or not PADDLE_BASE_URL:
        return False
    headers = {"Authorization": f"bearer {PADDLE_API_KEY}"}
    try:
        async with httpx.AsyncClient() as client:
            await client.get(PADDLE_BASE_URL, headers=headers, timeout=2.0)
            return True
    except Exception as exc:
        logger.warning(f"Paddle connection check failed: {exc}")
        return False


# ── Gemini OCR Helper Functions ─────────────────────────────────────────────

def validate_upload_gemini(filename: str, content_type: str, file_size: int):
    """Validates the uploaded file extension, MIME type, and size restrictions for Gemini OCR."""
    import os

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has an empty filename."
        )

    ext = os.path.splitext(filename.lower())[1]
    if ext not in GEMINI_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(GEMINI_ALLOWED_EXTENSIONS)}"
        )

    if content_type not in GEMINI_ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{content_type}'. Allowed types: {', '.join(GEMINI_ALLOWED_MIME_TYPES)}"
        )

    if file_size > GEMINI_MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size. Size: {file_size} bytes, Max: {GEMINI_MAX_FILE_SIZE} bytes (5MB)."
        )


def verify_image_bytes(content: bytes) -> Image.Image:
    """Attempts to parse and verify the image bytes using PIL."""
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content))
        return image
    except (UnidentifiedImageError, Exception) as img_err:
        logger.error(f"Image verification failed: {img_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image."
        )


def is_rate_limit_error(exception: Exception) -> bool:
    """Helper filter for tenacity to retry on 429/RESOURCE_EXHAUSTED API errors."""
    if isinstance(exception, APIError):
        is_429 = exception.code == 429 or exception.status == "RESOURCE_EXHAUSTED"
        if is_429:
            logger.warning("Gemini API rate limit hit (429/RESOURCE_EXHAUSTED). Triggering tenacity retry...")
            return True
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_rate_limit_error),
    reraise=True
)
def call_gemini_ocr(image: Image.Image) -> genai.types.GenerateContentResponse:
    """Synchronous Gemini model call. Wrapped in tenacity retry policy."""
    if not client:
        logger.error("Attempted to call Gemini OCR but client is not initialized.")
        raise RuntimeError("Gemini Client is not initialized due to missing API key.")

    logger.info(f"Executing Gemini {GEMINI_MODEL} structured OCR request...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            image,
            "Perform OCR on this equipment label image. "
            "Extract the following fields: markdown, machine_name, ma_mmtb, model, xuong, vi_tri. "
            "For 'markdown', reproduce all visible text preserving line breaks. "
            "Extract all other fields exactly as they appear on the label."
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": EquipmentOCRResult,
        }
    )
    return response


async def check_gemini_connectivity() -> bool:
    """Asynchronously checks Gemini credentials and network connectivity."""
    if not client:
        return False
    try:
        def _check():
            for _ in client.models.list(config={"page_size": 1}):
                return True
            return False
        return await asyncio.to_thread(_check)
    except Exception as exc:
        logger.warning(f"Gemini connection health check failed: {exc}")
        return False


# ── Google Sheets Helper Functions ───────────────────────────────────────────

_gspread_client = None


def get_gspread_client():
    """Lazily initializes and returns the authorized gspread client."""
    global _gspread_client
    if not GOOGLE_SHEETS_ENABLED:
        return None
    if _gspread_client is None:
        try:
            logger.info("Initializing Google Sheets client...")
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            
            if GOOGLE_SHEETS_CREDENTIALS_JSON:
                logger.info("Loading credentials from GOOGLE_SHEETS_CREDENTIALS_JSON environment variable...")
                creds_info = json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON)
                creds = service_account.Credentials.from_service_account_info(
                    creds_info,
                    scopes=scopes
                )
            elif os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH):
                logger.info(f"Loading credentials from file: {GOOGLE_SHEETS_CREDENTIALS_PATH}...")
                creds = service_account.Credentials.from_service_account_file(
                    GOOGLE_SHEETS_CREDENTIALS_PATH,
                    scopes=scopes
                )
            else:
                raise ValueError("No Google Sheets credentials found (neither env variable nor file).")
                
            _gspread_client = gspread.authorize(creds)
            logger.info("Google Sheets client authorized successfully.")
        except Exception as exc:
            logger.error(f"Failed to authorize Google Sheets client: {exc}")
            return None
    return _gspread_client


# ── Google Sheets Normalization Maps ─────────────────────────────────────────
KEY_MAPPING = {
    "Mã MMTB": ["mã mmtb", "mã máy", "ma mmtb", "ma may", "má"],
    "Model": ["model", "mô hình"],
    "Xưởng": ["xưởng", "xương", "xuong"],
    "Vị trí": ["vị trí", "vị tri", "vi tri", "vị tri"],
}


def _normalize(s: str) -> str:
    """Strips diacritics and lowercases for fuzzy key matching."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()


def get_standardized_value(key_value_dict: Dict[str, str], standard_key: str) -> str:
    """Finds value from a key-value dictionary using primary key or fuzzy aliases."""
    if not key_value_dict:
        return ""
        
    # Step 1: Try exact match first
    if standard_key in key_value_dict:
        return key_value_dict[standard_key]
        
    # Step 2: Try case-insensitive exact match
    for k, v in key_value_dict.items():
        if k.lower().strip() == standard_key.lower().strip():
            return v
            
    # Step 3: Try diacritic-insensitive normalization match
    norm_standard = _normalize(standard_key)
    for k, v in key_value_dict.items():
        if _normalize(k) == norm_standard:
            return v
            
    # Step 4: Try spelling variation aliases (case, whitespace, and diacritic insensitive)
    aliases = [_normalize(a) for a in KEY_MAPPING.get(standard_key, [])]
    for key, val in key_value_dict.items():
        if _normalize(key) in aliases:
            return val
            
    return ""


def extract_row_data(kv: Dict[str, str]) -> List[str]:
    """Robustly standardizes key-value mappings for Google Sheets columns."""
    if not kv:
        return ["", "", "", "", "", ""]
        
    # Extract machine name using robust standard fallbacks
    machine_name = ""
    for k in ["machine_name", "TÊN MMTB", "tên mmtb", "tên thiết bị", "machine name"]:
        if k in kv:
            machine_name = kv[k]
            break
        # Case insensitive fallback
        matched = False
        for ok in kv:
            if ok.lower().strip() == k.lower().strip():
                machine_name = kv[ok]
                matched = True
                break
        if matched:
            break

    # Standardize other fields using the key mapping Aliases
    ma_mmtb = get_standardized_value(kv, "Mã MMTB")
    model = get_standardized_value(kv, "Model")
    xuong = get_standardized_value(kv, "Xưởng")
    vi_tri = get_standardized_value(kv, "Vị trí")
    status_val = kv.get("status", "")

    # If Model is empty but is written in the machine name, extract it
    if not model and machine_name and "model" in machine_name.lower():
        parts = re.split(r'(?i)model', machine_name)
        if len(parts) > 1:
            model = parts[1].strip()
            # Clean up leading punctuation/separators
            model = re.sub(r'^[:：\-=\s]+', '', model).strip()

    return [machine_name, ma_mmtb, model, xuong, vi_tri, status_val]


def append_results_to_sheet_sync(
    results: List[OCRResult],
    source: str = "UNKNOWN",
) -> List[str]:
    """Synchronous implementation to append OCR results to Google Sheets.

    Args:
        results:  List of OCRResult objects to persist.
        source:   Human-readable caller label used in log messages
                  (e.g. 'TELEGRAM_CONFIRM', 'TELEGRAM_CORRECTED_CONFIRM').

    Returns:
        List of Mã MMTB values that were actually inserted
        (skipped duplicates are excluded).
    """
    # ── Guard: sheets integration must be enabled ──────────────────────────
    if not GOOGLE_SHEETS_ENABLED:
        logger.warning(
            f"[{source}] Google Sheets integration is disabled. Save skipped."
        )
        return []

    client = get_gspread_client()
    if not client:
        logger.warning(
            f"[{source}] gspread client unavailable. Save skipped."
        )
        return []

    inserted_ids: List[str] = []

    try:
        logger.info(f"[{source}] Opening Google Sheet: '{GOOGLE_SHEETS_NAME}'...")
        spreadsheet = client.open(GOOGLE_SHEETS_NAME)
        worksheet   = spreadsheet.get_worksheet(0)

        # ── Ensure header row exists ───────────────────────────────────────
        existing = worksheet.get_all_values()
        if not existing:
            headers = ["TÊN MMTB", "Mã MMTB", "MODEL", "XƯỞNG", "VỊ TRÍ", "TRẠNG THÁI"]
            logger.info(f"[{source}] Sheet is empty — writing headers.")
            worksheet.append_row(headers)
            existing = [headers]

        # ── Build set of existing Mã MMTB values (col index 1, 0-based) ───
        # Scan up to the last 100 data rows to catch recent duplicates.
        DEDUP_WINDOW = 100
        data_rows  = existing[1:]          # skip header
        recent_rows = data_rows[-DEDUP_WINDOW:] if len(data_rows) > DEDUP_WINDOW else data_rows
        existing_ma_mmtb = {
            row[1].strip().lower()
            for row in recent_rows
            if len(row) > 1 and row[1].strip()
        }
        logger.debug(
            f"[{source}] Dedup window contains {len(existing_ma_mmtb)} unique Mã MMTB values."
        )

        # ── Filter out duplicates and build insert batch ───────────────────
        rows_to_insert = []
        for result in results:
            row      = extract_row_data(result.key_value)
            ma_mmtb  = row[1].strip().lower() if len(row) > 1 else ""

            if ma_mmtb and ma_mmtb in existing_ma_mmtb:
                logger.warning(
                    f"[{source}] DUPLICATE SKIPPED — Mã MMTB '{row[1]}' "
                    f"already exists in the last {DEDUP_WINDOW} rows."
                )
                continue

            rows_to_insert.append(row)
            if ma_mmtb:
                existing_ma_mmtb.add(ma_mmtb)   # guard against duplicates within the same batch
                inserted_ids.append(row[1])

        # ── Batch append ───────────────────────────────────────────────────
        if rows_to_insert:
            logger.info(
                f"[{source}] Appending {len(rows_to_insert)} row(s) "
                f"to '{GOOGLE_SHEETS_NAME}'..."
            )
            worksheet.append_rows(rows_to_insert)
            logger.info(
                f"[{source}] Successfully saved {len(rows_to_insert)} row(s): "
                f"{[r[1] for r in rows_to_insert]}"
            )
        else:
            logger.info(f"[{source}] No new rows to insert (all duplicates skipped).")

    except Exception as exc:
        logger.error(
            f"[{source}] Error appending to '{GOOGLE_SHEETS_NAME}': {exc}",
            exc_info=True,
        )

    return inserted_ids


async def append_results_to_sheet(
    results: List[OCRResult],
    source: str = "UNKNOWN",
) -> List[str]:
    """Async wrapper — runs append_results_to_sheet_sync in a thread pool.

    NOTE: This function is intentionally NOT called by any route handler.
    It exists only for external callers (e.g. the Telegram bot) that need
    an async interface. Routes must NEVER call this directly.
    """
    if not GOOGLE_SHEETS_ENABLED:
        logger.debug(
            f"[{source}] Google Sheets integration disabled. Async dispatch skipped."
        )
        return []
    return await asyncio.to_thread(append_results_to_sheet_sync, results, source)

