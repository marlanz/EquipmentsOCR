from pydantic import BaseModel, Field
from typing import Dict, List

class OCRResult(BaseModel):
    markdown: str = Field(
        ...,
        description="Reconstructed multi-line OCR text"
    )
    key_value: Dict[str, str] = Field(
        ...,
        description="Extracted key-value structured pairs"
    )

class OCRResponse(BaseModel):
    results: List[OCRResult] = Field(
        ...,
        description="Structured OCR execution results"
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the document in seconds"
    )

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class ErrorResponse(BaseModel):
    error: str
