from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import validate_config, logger
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event manager for validating configurations on startup."""
    try:
        validate_config()
    except ValueError as exc:
        logger.critical(f"App configuration check failed on startup: {exc}")
        raise exc
    
    logger.info("OCR API Wrapper Service has started successfully.")
    yield
    logger.info("OCR API Wrapper Service is shutting down.")


app = FastAPI(
    title="PaddleOCR API Wrapper",
    description="A clean, production-ready FastAPI wrapper around PaddleOCR API.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# --- Standardized Error Handling Hook Procedures ---

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Intercepts standard HTTP exceptions to return normalized JSON error outputs."""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Intercepts Pydantic model and endpoint parameter validation failures."""
    errors = exc.errors()
    error_details = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "invalid value")
        error_details.append(f"[{loc}]: {msg}")
        
    error_message = f"Validation failed: {'; '.join(error_details)}"
    logger.error(error_message)
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": error_message}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catches unhandled server exceptions to format errors securely."""
    logger.exception(f"Unhandled system error encountered: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "OCR processing failed"}
    )


# Register endpoints router
app.include_router(router)


# --- Custom OpenAPI Schema Post-Processor ---
# Fixes visual display issues in Swagger UI (/docs) where newer FastAPI
# versions render List[UploadFile] as strings instead of binary upload buttons.
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # 1. Update components / schemas definitions
    for component in openapi_schema.get("components", {}).get("schemas", {}).values():
        for prop in component.get("properties", {}).values():
            if prop.get("contentMediaType") == "application/octet-stream":
                prop["format"] = "binary"
                prop.pop("contentMediaType", None)
            elif prop.get("type") == "array" and prop.get("items", {}).get("contentMediaType") == "application/octet-stream":
                prop["items"]["format"] = "binary"
                prop["items"].pop("contentMediaType", None)

    # 2. Update inline path requestBody schemas
    for path in openapi_schema.get("paths", {}).values():
        for operation in path.values():
            request_body = operation.get("requestBody", {})
            content = request_body.get("content", {})
            for media_type in content.values():
                schema = media_type.get("schema", {})
                for prop in schema.get("properties", {}).values():
                    if prop.get("contentMediaType") == "application/octet-stream":
                        prop["format"] = "binary"
                        prop.pop("contentMediaType", None)
                    elif prop.get("type") == "array" and prop.get("items", {}).get("contentMediaType") == "application/octet-stream":
                        prop["items"]["format"] = "binary"
                        prop["items"].pop("contentMediaType", None)

    app.openapi_schema = openapi_schema
    return openapi_schema

app.openapi = custom_openapi

