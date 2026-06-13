import re
import logging

logger = logging.getLogger("ocr-api")

MMTB_PATTERN = re.compile(r"^B\d{8}$")

def validate_mmtb(code: str) -> bool:
    """
    Returns True if code matches B########.
    B followed by exactly 8 digits.
    """
    if not code:
        return False
    stripped = code.strip()
    return bool(MMTB_PATTERN.match(stripped))
