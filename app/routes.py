import time
import asyncio
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from google.genai.errors import APIError

from app.schemas import OCRResponse, OCRResult, HealthResponse
from app.helpers import (
    validate_upload_paddle,
    optimize_image_bytes,
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


async def process_single_paddle(file: UploadFile) -> List[OCRResult]:
    """Asynchronous worker to process a single document upload via PaddleOCR API."""
    t_start = time.perf_counter()

    # 1. Read file bytes and compute file size in-memory
    try:
        file_bytes = await file.read()
    except Exception as read_err:
        logger.error(f"Read error for '{file.filename}': {read_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file '{file.filename}': {read_err}"
        )
    file_size = len(file_bytes)
    t_read = time.perf_counter() - t_start
    
    # 2. Run validations (Extension, MIME type, and size restrictions)
    validate_upload_paddle(file.filename, file.content_type, file_size)

    # 3. Optimize image in memory (resize high-res images if needed)
    opt_bytes, was_resized, t_preprocess = optimize_image_bytes(file_bytes)

    # 4. Submit OCR Job to external Paddle OCR endpoint
    t_sub0 = time.perf_counter()
    job_id = await submit_ocr_job(file.filename, opt_bytes, file.content_type)
    t_submit = time.perf_counter() - t_sub0

    # 5. Poll Job status until completion or timeout failure
    t_poll0 = time.perf_counter()
    jsonl_url = await poll_ocr_job(job_id)
    t_poll = time.perf_counter() - t_poll0

    # 6. Retrieve JSONL pages results and parse them into structured structures
    t_dl0 = time.perf_counter()
    pages = await download_and_parse_jsonl(jsonl_url)
    t_dl_parse = time.perf_counter() - t_dl0

    t_total = time.perf_counter() - t_start
    logger.info(
        f"Pipeline timing for '{file.filename}': "
        f"Read={t_read:.3f}s, Preprocess={t_preprocess:.3f}s (resized={was_resized}), "
        f"Submit={t_submit:.3f}s, Poll/Inference={t_poll:.3f}s, "
        f"Download&Parse={t_dl_parse:.3f}s, Total={t_total:.3f}s"
    )

    return pages


@router.post(
    "/parse-text",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse document or image using PaddleOCR"
)
async def parse_text(
    files: List[UploadFile] = File(...),
):
    """Accepts document uploads (PNG, JPG, JPEG, PDF), processes them in memory
    concurrently, submits them to Baidu PaddleOCR API, polls for completion, and returns
    structured markdown and key-value extractions for all files.
    Google Sheets saving is handled externally (e.g. Telegram bot) upon user confirmation.
    """
    start_time = time.time()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded for processing."
        )

    # Process all files concurrently
    tasks = [process_single_paddle(f) for f in files]
    results_nested = await asyncio.gather(*tasks)

    # Flatten the results list
    results = []
    for res_list in results_nested:
        results.extend(res_list)

    processing_time = round(time.time() - start_time, 3)

    return OCRResponse(results=results, processing_time=processing_time)


async def process_single_gemini(file: UploadFile, model_index: int = 1) -> OCRResult:
    """Asynchronous worker to process a single image with PIL validation and Gemini OCR."""
    # 1. Read file bytes in-memory
    try:
        file_bytes = await file.read()
    except Exception as read_err:
        logger.error(f"Read error for '{file.filename}': {read_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file '{file.filename}': {read_err}"
        )
    file_size = len(file_bytes)
    
    # 2. Run validations (Extension, MIME type, and size restrictions)
    validate_upload_gemini(file.filename, file.content_type, file_size)

    # 3. Verify image bytes using PIL (offloaded to threadpool)
    image = await asyncio.to_thread(verify_image_bytes, file_bytes)

    # 4. Call Gemini OCR (offloaded to threadpool)
    try:
        response = await asyncio.to_thread(call_gemini_ocr, image, model_index)
    except APIError as api_err:
        logger.error(f"Gemini APIError (code={api_err.code}) for '{file.filename}': {api_err.message}")
        if api_err.code == 429 or api_err.status == "RESOURCE_EXHAUSTED":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"OCR service temporarily overloaded for '{file.filename}'. Please retry later."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API Error for '{file.filename}': {api_err.message}"
        )
    except Exception as ocr_err:
        logger.error(f"Unexpected OCR error for '{file.filename}': {ocr_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing error for '{file.filename}': {ocr_err}"
        )

    # 5. Parse and build response
    parsed = response.parsed
    if not parsed:
        logger.error(f"Failed to parse structured OCR result for '{file.filename}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse structured OCR metadata from '{file.filename}' response."
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

    return OCRResult(
        markdown=parsed.markdown,
        key_value=kv
    )


@router.post(
    "/parse-text-gemini",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse document or image using Gemini OCR"
)
async def parse_text_gemini(
    files: List[UploadFile] = File(...),
    model_index: int = 1
):
    """Accepts multiple image uploads (PNG, JPG, JPEG, WEBP, GIF), processes them in memory
    concurrently, performs OCR via Gemini 2.5 Flash-Lite, and returns structured markdown and key-value extractions.
    Google Sheets saving is handled externally (e.g. Telegram bot) upon user confirmation.
    """
    start_time = time.time()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded for processing."
        )

    # Run processing tasks concurrently using asyncio.gather
    tasks = [process_single_gemini(f, model_index) for f in files]
    results = await asyncio.gather(*tasks)

    processing_time = round(time.time() - start_time, 3)

    return OCRResponse(results=results, processing_time=processing_time)
