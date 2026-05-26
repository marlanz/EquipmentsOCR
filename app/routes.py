import time
from fastapi import APIRouter, File, UploadFile, status
from app.schemas import OCRResponse, HealthResponse
from app.helpers import (
    validate_upload,
    submit_ocr_job,
    poll_ocr_job,
    download_and_parse_jsonl,
    check_paddle_connectivity,
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
    and checks connection to the external Paddle API.
    """
    logger.info("Health check requested.")
    
    # Verify downstream API connectivity with a brief 2-second timeout
    paddle_connected = await check_paddle_connectivity()
    
    if not paddle_connected:
        logger.warning("FastAPI is running, but external Paddle OCR API is currently unreachable.")

    return HealthResponse(
        status="healthy",
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

    # 1. Read file bytes and compute file size completely in-memory
    file_bytes = await file.read()
    file_size = len(file_bytes)
    
    # 2. Run validations (Extension, MIME type, and size restrictions)
    validate_upload(file.filename, file.content_type, file_size)

    # 3. Submit OCR Job to external Paddle OCR endpoint
    job_id = await submit_ocr_job(file.filename, file_bytes, file.content_type)

    # 4. Poll Job status until completion or timeout failure
    jsonl_url = await poll_ocr_job(job_id)

    # 5. Retrieve JSONL pages results and parse them into structured structures
    results = await download_and_parse_jsonl(jsonl_url)

    processing_time = round(time.time() - start_time, 3)

    return OCRResponse(results=results, processing_time=processing_time)
