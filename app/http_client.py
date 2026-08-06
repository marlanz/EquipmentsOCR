"""
app/http_client.py
──────────────────
Manages a persistent, shared httpx.AsyncClient instance for HTTP connection pooling
and keep-alive reuse across downstream Paddle API calls and Telegram bot requests.
"""

import httpx
from typing import Optional
import logging

logger = logging.getLogger("ocr-api")

_shared_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Returns the global shared AsyncClient instance with connection pooling.
    Initializes a new client if none exists or if closed.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30.0
        )
        timeouts = httpx.Timeout(timeout=120.0, connect=10.0)
        _shared_client = httpx.AsyncClient(limits=limits, timeout=timeouts)
        logger.info("Initialized shared AsyncClient with connection pooling.")
    return _shared_client


async def close_http_client():
    """Closes the global shared AsyncClient if open."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        logger.info("Closed shared AsyncClient.")
    _shared_client = None
