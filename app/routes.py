import time
import asyncio
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from google.genai.errors import APIError

from app.schemas import OCRResponse, OCRResult, HealthResponse
from app.helpers import (
    validate_upload_paddle,
    submit_ocr_job,
    poll_ocr_job,
    download_and_parse_jsonl,
    check_paddle_connectivity,
    validate_upload_gemini,
    verify_image_bytes,
    call_gemini_ocr,
    check_gemini_connectivity,
    logger,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check"
)
async def health_check():
    """Performs a health check on the FastAPI application, environment loading,
    and checks connection to both external Paddle and Gemini APIs.
    """
    logger.info("Health check requested.")
    
    # Verify downstream API connectivity in parallel
    paddle_task = check_paddle_connectivity()
    gemini_task = check_gemini_connectivity()
    paddle_connected, gemini_connected = await asyncio.gather(paddle_task, gemini_task)
    
    if not paddle_connected:
        logger.warning("Baidu Paddle OCR API is currently unreachable.")
    if not gemini_connected:
        logger.warning("Google Gemini API is currently unreachable.")

    # Status is degraded if either downstream API fails connection checks
    status_val = "healthy" if (paddle_connected and gemini_connected) else "degraded"

    return HealthResponse(
        status=status_val,
        service="ocr-api",
        version="1.0.0"
    )


@router.post(
    "/parse-text",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse document or image using PaddleOCR"
)
async def parse_text(file: UploadFile = File(...)):
    """Accepts document uploads (PNG, JPG, JPEG, PDF), processes them in memory,
    submits them to Baidu PaddleOCR API, polls for completion, and returns
    structured markdown and key-value extractions.
    """
    start_time = time.time()

    # 1. Read file bytes and compute file size in-memory
    try:
        file_bytes = await file.read()
    except Exception as read_err:
        logger.error(f"Read error for '{file.filename}': {read_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {read_err}"
        )
    file_size = len(file_bytes)
    
    # 2. Run validations (Extension, MIME type, and size restrictions)
    validate_upload_paddle(file.filename, file.content_type, file_size)

    # 3. Submit OCR Job to external Paddle OCR endpoint
    job_id = await submit_ocr_job(file.filename, file_bytes, file.content_type)

    # 4. Poll Job status until completion or timeout failure
    jsonl_url = await poll_ocr_job(job_id)

    # 5. Retrieve JSONL pages results and parse them into structured structures
    results = await download_and_parse_jsonl(jsonl_url)

    processing_time = round(time.time() - start_time, 3)

    return OCRResponse(results=results, processing_time=processing_time)


@router.post(
    "/parse-text-gemini",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse document or image using Gemini OCR"
)
async def parse_text_gemini(file: UploadFile = File(...)):
    """Accepts image uploads (PNG, JPG, JPEG, WEBP, GIF), processes them in memory,
    performs OCR via Gemini 2.5 Flash-Lite, and returns structured markdown and key-value extractions.
    """
    start_time = time.time()

    # 1. Read file bytes in-memory
    try:
        file_bytes = await file.read()
    except Exception as read_err:
        logger.error(f"Read error for '{file.filename}': {read_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {read_err}"
        )
    file_size = len(file_bytes)
    
    # 2. Run validations (Extension, MIME type, and size restrictions)
    validate_upload_gemini(file.filename, file.content_type, file_size)

    # 3. Verify image bytes using PIL (offloaded to threadpool)
    image = await asyncio.to_thread(verify_image_bytes, file_bytes)

    # 4. Call Gemini OCR (offloaded to threadpool)
    try:
        response = await asyncio.to_thread(call_gemini_ocr, image)
    except APIError as api_err:
        duration_seconds = time.time() - start_time
        logger.error(f"Gemini APIError (code={api_err.code}): {api_err.message}. Duration: {duration_seconds:.4f}s")
        if api_err.code == 429 or api_err.status == "RESOURCE_EXHAUSTED":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OCR service temporarily overloaded. Please retry later."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API Error: {api_err.message}"
        )
    except Exception as ocr_err:
        duration_seconds = time.time() - start_time
        logger.error(f"Unexpected OCR error for '{file.filename}': {ocr_err}. Duration: {duration_seconds:.4f}s")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing error: {ocr_err}"
        )

    # 5. Parse and build response
    parsed = response.parsed
    if not parsed:
        duration_seconds = time.time() - start_time
        logger.error(f"Failed to parse structured OCR result for '{file.filename}'. Duration: {duration_seconds:.4f}s")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse structured OCR metadata from image response."
        )

    # Construct key-value dict matching response schema requirements
    # Correcting typo "Xương" to "Xưởng" for consistency
    kv = {
        "machine_name": parsed.machine_name,
        "Mã MMTB": parsed.ma_mmtb,
        "Model": parsed.model,
        "Xưởng": parsed.xuong,
        "Vị trí": parsed.vi_tri,
    }

    results = [
        OCRResult(
            markdown=parsed.markdown,
            key_value=kv
        )
    ]

    processing_time = round(time.time() - start_time, 3)

    return OCRResponse(results=results, processing_time=processing_time)
