"""
app/paddle_transformer.py
─────────────────────────
Dedicated PaddleOCR post-processing transformer.

Converts PaddleOCR raw JSON (a single layoutParsingResult entry) into a
clean (markdown, key_value) pair with significantly higher accuracy than
the generic markdown-based approach.

Entry point
───────────
    from app.paddle_transformer import transform_paddleocr_result
    parsed = transform_paddleocr_result(res, debug=False)
    # parsed.markdown   → str
    # parsed.key_value  → Dict[str, str]
    # parsed.source     → "raw_blocks" | "enhanced_markdown" | "markdown_basic"

The Gemini OCR pipeline is NOT used here and must NOT be changed.
"""

from __future__ import annotations

import re
import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical field names — the only output keys the bot/sheet cares about
# ─────────────────────────────────────────────────────────────────────────────
TRACKED_FIELDS: List[str] = ["Mã MMTB", "Model", "Xưởng", "Vị trí"]

# ─────────────────────────────────────────────────────────────────────────────
# Alias map: normalized_alias → canonical key
# Covers every OCR corruption variant observed in the wild.
# ─────────────────────────────────────────────────────────────────────────────
_ALIAS_MAP: Dict[str, str] = {
    # ── Mã MMTB ──────────────────────────────────────────────────────────────
    "ma mmtb":       "Mã MMTB",
    "mammtb":        "Mã MMTB",
    "ma may":        "Mã MMTB",
    "mamay":         "Mã MMTB",
    "mmtb":          "Mã MMTB",
    "ma mmtb":       "Mã MMTB",
    "ma mtb":        "Mã MMTB",
    "m mmtb":        "Mã MMTB",
    "ma":            "Mã MMTB",
    # ── Model ─────────────────────────────────────────────────────────────────
    "model":         "Model",
    "mo hinh":       "Model",
    "mohinh":        "Model",
    "model so":      "Model",
    # ── Xưởng ─────────────────────────────────────────────────────────────────
    "xuong":         "Xưởng",
    "xuong san xuat":"Xưởng",
    "nha may":       "Xưởng",
    "phan xuong":    "Xưởng",
    "px":            "Xưởng",      # very short — used as last resort only
    # ── Vị trí ────────────────────────────────────────────────────────────────
    "vi tri":        "Vị trí",
    "vitri":         "Vị trí",
    "v tri":         "Vị trí",     # normalized from "V: Trí"
    "vtri":          "Vị trí",
    "tri":           "Vị trí",     # corrupted "Vị trí" (image split)
    "i tri":         "Vị trí",     # corruption "(i trí"
    "(i tri":        "Vị trí",     # full "(i trí" OCR artifact
    "vi tri ":       "Vị trí",
    "vitri ":        "Vị trí",
    "location":      "Vị trí",
    "vi":            "Vị trí",     # very short — last resort only
}

# Short aliases (len < 4) are only used as last resort to avoid false positives
_SHORT_ALIASES = {"px", "vi", "ma"}

# Compiled regexes — defined once at module level for performance
_RE_KV_STANDARD = re.compile(
    r"^\s*(?:\*\*)?\s*([^*：:\|]+?)\s*(?:\*\*)?\s*[:：]\s*(.*)$"
)
_RE_KV_TABLE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
)
_RE_HTML_BLOCK = re.compile(
    r"<div[^>]*>.*?</div>", re.IGNORECASE | re.DOTALL
)
_RE_STRIP_BOLD = re.compile(r"^\*+\s*|\s*\*+$")
_RE_ALT_SEP = re.compile(r"^[:.：\-\.\s]+")


# ─────────────────────────────────────────────────────────────────────────────
# Branding-detection constants (used by Stage 5 machine_name extractor)
# ─────────────────────────────────────────────────────────────────────────────

# Company branding / logo text that must never be the machine_name.
# Values are diacritic-stripped, lowercase.
_BRANDING_BLOCKLIST: frozenset = frozenset({
    "daidung",
    "dridung",
    "daidong",
    "dai dung",
    "dai oung",
    "dai dung",   # ĐẠI DŨNG normalized
    "viet hung",
    "viet nam",
})

# If a line contains ANY of these (normalized), it is equipment text, not branding.
_EQUIPMENT_KW_NORM: Tuple[str, ...] = (
    "may",          # MÁY / Máy  (machine in Vietnamese)
    "machine", "equipment", "thiet bi",
    "han",          # hàn  (welding)
    "tien",         # tiện (lathe)
    "phay",         # phay (milling)
    "khoan",        # khoan(drilling)
    "mai",          # mài  (grinding)
    "cat",          # cắt  (cutting)
    "gouging", "welding", "cutting", "drilling",
    "milling", "grinding", "laser", "cnc", "robot",
    "bom",          # bơm  (pump)
    "nen",          # nén  (compress)
    "uon",          # uốn  (bending)
    "ghep",         # ghép (joining)
    "ep",           # ép   (pressing)
    "cuon",         # cuốn (winding)
    "keo",          # kéo  (drawing)
)

# Normalized strings that signal the START of the metadata section.
# Walking backwards from the first line that contains one of these gives
# the equipment name candidates.
_ANCHOR_PATTERNS_NORM: Tuple[str, ...] = (
    "ma mmtb",
    "mammtb",
    "code mmtb",
    "ma may",
    "mamay",
    "ma mtb",
    "ma",
)

# Helpers for machine_name extraction
_RE_HEADING_PREFIX = re.compile(r"^#+\s*")
_RE_LEADING_BULLET = re.compile(r"^[-\*•\s]+")
_RE_VIET_LOWER     = re.compile(
    r"[\xe0\xe1\xe2\xe3\xe8\xe9\xea\xec\xed\xf2\xf3\xf4\xf5\xf9\xfa\xfd"
    r"\u0103\u0111\u01a1\u01b0"
    r"\u1ea1\u1ea3\u1ea5\u1ea7\u1ea9\u1eab\u1ead\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7"
    r"\u1eb9\u1ebb\u1ebd\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7\u1ec9\u1ecb"
    r"\u1ecd\u1ecf\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u1edb\u1edd\u1edf\u1ee1\u1ee3"
    r"\u1ee5\u1ee7\u1ee9\u1eeb\u1eed\u1eef\u1ef1\u1ef3\u1ef7\u1ef9]"
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Strip diacritics, lowercase, collapse whitespace."""
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).lower().strip()


def normalize_key(raw_key: str) -> Optional[str]:
    """
    Maps a raw OCR key string to its canonical field name.
    Returns None if no confident match is found.

    Resolution order
    ────────────────
    1. Direct alias map lookup (exact, after normalization)
    2. Canonical field diacritic-normalized match
    3. Contains match (alias inside text, or text inside alias) — min 4 chars
    4. Startswith fuzzy match — min 4 chars
    """
    if not raw_key:
        return None

    s    = raw_key.strip()
    norm = _normalize(s)
    # Clean leading/trailing symbols commonly used for bullets/formatting
    norm = re.sub(r"^[\s\-\*•\.\:_]+|[\s\-\*•\.\:_]+$", "", norm)

    # 1. Exact alias
    if norm in _ALIAS_MAP:
        return _ALIAS_MAP[norm]

    # 2. Matches a canonical field directly
    for canonical in TRACKED_FIELDS:
        if _normalize(canonical) == norm:
            return canonical

    # 3. Contains match — skip very short aliases
    for alias, canonical in _ALIAS_MAP.items():
        if alias in _SHORT_ALIASES:
            continue
        if len(alias) >= 4 and (alias in norm or norm in alias):
            return canonical

    # 4. Startswith fuzzy
    for alias, canonical in _ALIAS_MAP.items():
        if alias in _SHORT_ALIASES:
            continue
        if len(alias) >= 4 and (norm.startswith(alias) or alias.startswith(norm)):
            return canonical

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Block:
    """A single OCR text detection with optional bounding box."""
    text: str
    bbox: List[float] = field(default_factory=list)  # [x1, y1, x2, y2]
    conf: float = 1.0

    @property
    def x1(self) -> float:
        return self.bbox[0] if len(self.bbox) >= 4 else 0.0

    @property
    def y1(self) -> float:
        return self.bbox[1] if len(self.bbox) >= 4 else 0.0

    @property
    def x2(self) -> float:
        return self.bbox[2] if len(self.bbox) >= 4 else 0.0

    @property
    def y2(self) -> float:
        return self.bbox[3] if len(self.bbox) >= 4 else 0.0

    @property
    def cy(self) -> float:
        """Y centre."""
        return (self.y1 + self.y2) / 2 if self.bbox else 0.0

    @property
    def cx(self) -> float:
        """X centre."""
        return (self.x1 + self.x2) / 2 if self.bbox else 0.0

    @property
    def height(self) -> float:
        return max(self.y2 - self.y1, 1.0)


@dataclass
class ParsedResult:
    """Output of transform_paddleocr_result."""
    markdown:   str                      = ""
    key_value:  Dict[str, str]           = field(default_factory=dict)
    source:     str                      = "unknown"   # see constants below
    debug_info: Optional[Dict[str, Any]] = None


# source constants
SRC_RAW_BLOCKS         = "raw_blocks"
SRC_ENHANCED_MARKDOWN  = "enhanced_markdown"
SRC_MARKDOWN_BASIC     = "markdown_basic"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Block Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_to_xyxy(bbox: Any) -> List[float]:
    """
    Normalise various bbox formats to [x1, y1, x2, y2].
    Handles:
      • [x1, y1, x2, y2]
      • [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  (polygon)
    """
    if not bbox:
        return []
    if isinstance(bbox[0], (list, tuple)):
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        return [min(xs), min(ys), max(xs), max(ys)]
    return [float(v) for v in bbox[:4]]


def extract_text_blocks(res: dict) -> List[_Block]:
    """
    Tries three JSON paths to extract OCR text blocks from a
    layoutParsingResult entry.  Returns an empty list if the raw data is
    unavailable (caller then falls back to markdown parsing).

    JSON paths tried (in order):
    ─────────────────────────────
    1. res["blocks"][].content / text  +  sub_blocks
    2. res["ocr_results"][].text / rec_res
    3. res["rec_texts"] + res["dt_boxes"]
    """
    blocks: List[_Block] = []
    seen: set = set()

    def _add(text: str, bbox: Any, conf: float = 1.0) -> None:
        text = text.strip()
        if not text or text in seen:
            return
        seen.add(text)
        blocks.append(_Block(text=text, bbox=_bbox_to_xyxy(bbox), conf=conf))

    # ── Path 1: structured layout blocks ─────────────────────────────────────
    for blk in res.get("blocks", []):
        text = blk.get("content", blk.get("text", ""))
        bbox = blk.get("bbox", blk.get("box", []))
        if text:
            _add(text, bbox)
        for sub in blk.get("sub_blocks", []):
            sub_text = sub.get("content", sub.get("text", ""))
            sub_bbox = sub.get("bbox", sub.get("box", []))
            if sub_text:
                _add(sub_text, sub_bbox)

    # ── Path 2: ocr_results list ──────────────────────────────────────────────
    for item in res.get("ocr_results", []):
        text = item.get("text", "")
        if not text:
            rec = item.get("rec_res")
            if isinstance(rec, (list, tuple)) and rec:
                text = str(rec[0])
        bbox = item.get("bbox", item.get("box", item.get("dt_boxes", [])))
        conf = float(item.get("confidence", item.get("score", 1.0)))
        if text:
            _add(text, bbox, conf)

    # ── Path 3: rec_texts + dt_boxes (standard PaddleOCR format) ─────────────
    rec_texts  = res.get("rec_texts",  [])
    dt_boxes   = res.get("dt_boxes",   [])
    rec_scores = res.get("rec_scores", [])
    for i, text in enumerate(rec_texts):
        bbox  = dt_boxes[i]   if i < len(dt_boxes)   else []
        conf  = float(rec_scores[i]) if i < len(rec_scores) else 1.0
        _add(text, bbox, conf)

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2a — Spatial KV Parsing (from raw blocks)
# ─────────────────────────────────────────────────────────────────────────────

def _group_into_lines(blocks: List[_Block]) -> List[List[_Block]]:
    """
    Groups blocks into horizontal lines.
    Two blocks belong to the same line if their Y-centre difference is less
    than 60% of the taller block's height.
    Each line is sorted left-to-right (ascending X).
    """
    if not blocks:
        return []

    sorted_blks = sorted(blocks, key=lambda b: (b.cy, b.cx))
    lines: List[List[_Block]] = [[sorted_blks[0]]]

    for blk in sorted_blks[1:]:
        prev = lines[-1][-1]
        threshold = 0.6 * max(prev.height, blk.height)
        if abs(blk.cy - prev.cy) <= threshold:
            lines[-1].append(blk)
        else:
            lines.append([blk])

    # Sort each line L→R
    for ln in lines:
        ln.sort(key=lambda b: b.cx)

    return lines


def _parse_kv_from_text(text: str) -> Optional[Tuple[str, str]]:
    """
    Tries to parse a single text string as a 'key : value' pair.
    Returns (raw_key, value) or None.  Key normalization is NOT applied here.
    """
    text = text.strip()
    if not text:
        return None

    # Check if there is a second colon and combined they form a known key (e.g. "V: Trí : B16")
    parts = [p.strip() for p in re.split(r"[:：]", text)]
    if len(parts) >= 3:
        combined = f"{parts[0]} {parts[1]}"
        combined_clean = _RE_STRIP_BOLD.sub("", combined).strip()
        if normalize_key(combined_clean):
            val = ":".join(parts[2:]).strip()
            val = _RE_STRIP_BOLD.sub("", val).strip()
            return (combined_clean, val)

    # Standard: "key : value" or "**key** : value"
    m = _RE_KV_STANDARD.match(text)
    if m:
        k = _RE_STRIP_BOLD.sub("", m.group(1).strip()).strip()
        # Also strip leading bullet/dash characters from the key
        k = re.sub(r"^[-\*•\s]+", "", k).strip()
        v = _RE_STRIP_BOLD.sub("", m.group(2).strip()).strip()
        if k:
            return (k, v)   # v may be empty (key-only line)

    # Table row: "| key | value |"
    m = _RE_KV_TABLE.match(text)
    if m:
        k = m.group(1).strip()
        v = m.group(2).strip()
        if k and v:
            return (k, v)

    # Known field with alternative separator (handles ". " or " - ")
    # Only for long-enough known aliases to avoid false positives
    norm_text = _normalize(text)
    for alias, canonical in _ALIAS_MAP.items():
        if len(alias) < 4:
            continue
        if norm_text.startswith(alias):
            rest = text[len(alias):].strip()
            rest = _RE_ALT_SEP.sub("", rest).strip()
            if rest:
                return (canonical, rest)

    return None


def _parse_line(line_blocks: List[_Block]) -> Optional[Tuple[str, str]]:
    """
    Tries to extract a KV pair from a group of spatially co-linear blocks.
    Returns (raw_key, value) or None.  Key is NOT yet normalised here.
    """
    if not line_blocks:
        return None

    # 1. Merged text of entire line
    merged = " ".join(b.text for b in line_blocks)
    result = _parse_kv_from_text(merged)
    if result and result[1]:           # require non-empty value
        return result

    # 2. If first block is a known field key, treat rest as value
    if len(line_blocks) >= 2:
        first = line_blocks[0].text.strip()
        canonical = normalize_key(first)
        if canonical:
            rest_text = " ".join(
                b.text.lstrip(":").strip() for b in line_blocks[1:]
            ).strip()
            if rest_text:
                return (canonical, rest_text)

    # 3. Key-only line (value expected on next line)
    if result and not result[1]:
        return result   # (key, "")

    return None


def parse_kv_from_blocks(
    blocks: List[_Block],
    debug: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Main spatial KV parser.  Groups blocks into lines then extracts pairs.

    Returns
    ───────
    (kv_dict, debug_lines)
    """
    dbg: List[str] = []
    kv: Dict[str, str] = {}

    if not blocks:
        return kv, dbg

    lines = _group_into_lines(blocks)
    if debug:
        dbg.append(f"  Grouped into {len(lines)} line(s):")
        for i, ln in enumerate(lines):
            dbg.append(f"    line {i}: {[b.text for b in ln]}")

    machine_name_candidate: Optional[str] = None
    pending_key: Optional[str]            = None   # key with no value yet

    for line_blocks in lines:
        merged_text = " ".join(b.text for b in line_blocks).strip()
        parsed = _parse_line(line_blocks)

        if parsed:
            raw_key, value = parsed
            canonical = normalize_key(raw_key) or raw_key

            if value:
                kv[canonical] = value
                if debug:
                    dbg.append(f"  ✔ line parse: {raw_key!r} → [{canonical}] = {value!r}")
                pending_key = None
            else:
                # Key-only line — wait for value on next iteration
                pending_key = canonical
                if debug:
                    dbg.append(f"  ⏳ pending key: {canonical!r}")

        elif pending_key:
            # This line is the value for the pending key
            value = merged_text.lstrip(":").strip()
            if value and normalize_key(value) is None:
                # Make sure we're not mapping a key onto another key
                kv[pending_key] = value
                if debug:
                    dbg.append(f"  ✔ multi-line: [{pending_key}] = {value!r}")
            pending_key = None

        else:
            # Not a KV pair — first candidate becomes machine_name
            candidate_key = normalize_key(merged_text)
            if candidate_key:
                # It's a key with no value; mark pending
                pending_key = candidate_key
            elif machine_name_candidate is None and merged_text:
                machine_name_candidate = merged_text
                if debug:
                    dbg.append(f"  📋 machine_name candidate: {merged_text!r}")
            pending_key = candidate_key  # may be None

    if machine_name_candidate and "machine_name" not in kv:
        kv["machine_name"] = machine_name_candidate

    return kv, dbg


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b — Enhanced Markdown KV Parser (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _enhanced_markdown_kv(
    markdown_text: str,
    debug: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Improved markdown-to-KV parser.  Superset of the original
    parse_markdown_to_key_value with the following additions:

    • Table rows: | key | value |
    • Multi-line KV: orphan ": value" lines merged with previous key
    • Alternative separators (. –) for known fields
    • "trí" → "Vị trí" key promotion
    • Normalises all extracted keys through normalize_key()
    • Strips HTML <div> image blocks before scanning
    """
    dbg: List[str] = []
    kv:  Dict[str, str] = {}

    if not markdown_text:
        return kv, dbg

    # Strip HTML image/div blocks — they interrupt KV pair continuity
    clean_md = _RE_HTML_BLOCK.sub("", markdown_text)

    raw_lines = [ln.strip() for ln in clean_md.split("\n")]
    lines     = [ln for ln in raw_lines if ln]

    if not lines:
        return kv, dbg

    # 1. First line heading / machine name
    first = re.sub(r"^#+\s*", "", lines[0]).strip()
    if first and ":" not in first and "：" not in first and "|" not in first:
        kv["machine_name"] = first
        if debug:
            dbg.append(f"  📋 machine_name: {first!r}")

    pending_key: Optional[str] = None

    for line in lines:
        # ── Table row ─────────────────────────────────────────────────────────
        tm = _RE_KV_TABLE.match(line)
        if tm:
            k = tm.group(1).strip()
            v = tm.group(2).strip()
            if k and v:
                canonical = normalize_key(k) or k
                kv[canonical] = v
                if debug:
                    dbg.append(f"  ✔ table: {k!r} → [{canonical}] = {v!r}")
            pending_key = None
            continue

        # ── Orphan value line (": value") ─────────────────────────────────────
        if line.startswith(":") or line.startswith("："):
            if pending_key:
                value = line.lstrip(":：").strip()
                if value:
                    kv[pending_key] = value
                    if debug:
                        dbg.append(f"  ✔ orphan-val: [{pending_key}] = {value!r}")
                pending_key = None
            continue

        # ── Standard KV ───────────────────────────────────────────────────────
        parsed = _parse_kv_from_text(line)
        if parsed:
            k, v = parsed
            canonical = normalize_key(k) or k
            if v:
                kv[canonical] = v
                if debug:
                    dbg.append(f"  ✔ standard: {k!r} → [{canonical}] = {v!r}")
                pending_key = None
            else:
                # Key-only; value may come on next line
                pending_key = canonical
                if debug:
                    dbg.append(f"  ⏳ pending key: {canonical!r}")
            continue

        # ── Known field with alt separator ────────────────────────────────────
        norm_line = _normalize(line)
        matched = False
        for alias, canonical in _ALIAS_MAP.items():
            if len(alias) < 4:
                continue
            if norm_line.startswith(alias):
                rest = line[len(alias):].strip()
                rest = _RE_ALT_SEP.sub("", rest).strip()
                if rest:
                    kv[canonical] = rest
                    if debug:
                        dbg.append(f"  ✔ alt-sep: {line!r} → [{canonical}] = {rest!r}")
                    matched = True
                    pending_key = None
                    break
        if matched:
            continue

        # ── Standalone known-field name line (no colon, no value) ────────────
        # e.g. "Model" appears alone — value will come on the next line
        standalone_canonical = normalize_key(line)
        if standalone_canonical:
            pending_key = standalone_canonical
            if debug:
                dbg.append(f"  ⏳ standalone key: {standalone_canonical!r}")
            continue

        # ── Nothing matched ────────────────────────────────────────────────────
        if pending_key and line:
            # Non-matching line after a pending key: treat as implicit value
            # but only if it doesn't look like another known field
            if normalize_key(line) is None:
                kv[pending_key] = line.lstrip(":").strip()
                if debug:
                    dbg.append(f"  ✔ implicit-val: [{pending_key}] = {line!r}")
            pending_key = None

    return kv, dbg


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Key Normalisation post-pass
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_kv_keys(kv: Dict[str, str]) -> Dict[str, str]:
    """
    Runs every key in the kv dict through normalize_key and re-maps it
    to the canonical name.  Does not overwrite already-canonical values.
    Also strips leading bullet/dash chars from any unmapped raw keys.
    """
    result: Dict[str, str] = {}
    for k, v in kv.items():
        canonical = normalize_key(k)
        if canonical:
            final_key = canonical
        else:
            # Clean up bullet/dash prefix from unrecognised raw keys
            clean_k = re.sub(r"^[-\*•\s]+", "", k).strip()
            final_key = clean_k if clean_k else k
        # Don't overwrite an already-present canonical key
        if final_key not in result:
            result[final_key] = v
        elif not result[final_key] and v:
            result[final_key] = v   # prefer non-empty value
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Missing Field Recovery
# ─────────────────────────────────────────────────────────────────────────────

def _recover_missing_fields(
    kv: Dict[str, str],
    blocks: List[_Block],
    markdown_text: str,
    debug: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    """
    If any of the 4 tracked fields are still absent, performs a secondary
    search through all available text sources.

    Recovery strategies (in order):
    1. "trí" or "(i trí" key promotion → "Vị trí"
    2. Re-scan every block text with flexible separator matching
    3. Re-scan markdown lines with flexible separator matching
    """
    dbg: List[str] = []
    missing = [f for f in TRACKED_FIELDS if f not in kv or not kv[f]]

    if not missing:
        return kv, dbg
    if debug:
        dbg.append(f"  🔍 Missing fields: {missing}")

    # ── Strategy 1: key promotion ─────────────────────────────────────────────
    # "trí" is stored when "Vị" and "trí" were in separate text boxes
    for bad_key in ("trí", "tri", "(i trí", "(i tri", "i tri"):
        if bad_key in kv and "Vị trí" not in kv:
            kv["Vị trí"] = kv.pop(bad_key)
            if debug:
                dbg.append(f"  ✔ promoted {bad_key!r} → 'Vị trí'")

    # ── Strategy 2: re-scan block texts ──────────────────────────────────────
    missing = [f for f in TRACKED_FIELDS if f not in kv or not kv[f]]
    all_texts = [b.text for b in blocks]

    for text in all_texts:
        parsed = _parse_kv_from_text(text)
        if parsed:
            raw_k, v = parsed
            canonical = normalize_key(raw_k)
            if canonical and canonical in missing and v:
                kv[canonical] = v
                missing.remove(canonical)
                if debug:
                    dbg.append(f"  ✔ block-recovery: [{canonical}] = {v!r}")
        if not missing:
            break

    # ── Strategy 3: re-scan markdown lines ───────────────────────────────────
    missing = [f for f in TRACKED_FIELDS if f not in kv or not kv[f]]
    if missing and markdown_text:
        clean = _RE_HTML_BLOCK.sub("", markdown_text)
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                continue
            parsed = _parse_kv_from_text(line)
            if parsed:
                raw_k, v = parsed
                canonical = normalize_key(raw_k)
                if canonical and canonical in missing and v:
                    kv[canonical] = v
                    missing.remove(canonical)
                    if debug:
                        dbg.append(f"  ✔ md-recovery: [{canonical}] = {v!r}")
            if not missing:
                break

    return kv, dbg


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Anchor-walk machine_name extractor
# ─────────────────────────────────────────────────────────────────────────────

def _is_branding_line(text: str) -> bool:
    """
    Returns True if the text is company branding/section header that should
    never be treated as the equipment machine_name.

    Detection strategy (in order):
    1. Empty / markdown-heading-only lines
    2. Lines ending with ':'  → section headers (e.g. "## HÀN:")
    3. Direct blocklist match (after normalization)
    4. Heuristic: ALL CAPS + short + no equipment keywords
    """
    if not text:
        return True

    # Strip markdown heading markers for analysis
    clean = _RE_HEADING_PREFIX.sub("", text).strip()
    if not clean:
        return True     # e.g. "##" with nothing after

    # Section headers end with a colon
    if clean.rstrip().endswith(":") or clean.rstrip().endswith("\uff1a"):
        return True

    norm = _normalize(clean)

    # Direct blocklist match
    if norm in _BRANDING_BLOCKLIST:
        return True

    # Fuzzy blocklist: normalized text starts with a known brand prefix
    for brand in _BRANDING_BLOCKLIST:
        if len(brand) >= 5 and norm.startswith(brand):
            return True

    # Digit check: model numbers / specifications (containing digits) are not branding
    c = re.sub(r"^[\s\-\*•\.\:_]+", "", clean).strip()
    if any(char.isdigit() for char in c):
        return False

    # Heuristic: ALL-CAPS + short + no equipment keywords.
    # ALL-CAPS check: no ASCII lowercase letters AND no lowercase Vietnamese chars.
    has_ascii_lower = bool(re.search(r"[a-z]", clean))
    has_viet_lower  = bool(_RE_VIET_LOWER.search(clean))
    is_all_caps     = not has_ascii_lower and not has_viet_lower
    is_short        = len(re.sub(r"\s+", "", clean)) <= 20
    has_equip_kw    = any(kw in norm for kw in _EQUIPMENT_KW_NORM)

    if is_all_caps and is_short and not has_equip_kw:
        return True

    return False


def _is_metadata_kv_line(text: str) -> bool:
    """
    Returns True if the line is a metadata KV pair for a known field
    (e.g. "- Model: MZ-1000", "-Mä MMTB/Code MMTB: B22300035").
    Used to skip metadata lines while walking backwards for machine_name.
    """
    if not text:
        return False
    # Strip leading bullet/dash chars
    clean = _RE_LEADING_BULLET.sub("", text).strip()
    parsed = _parse_kv_from_text(clean)
    if parsed:
        key, _ = parsed
        key_norm = _normalize(key.split("/")[0])
        # Check against alias map
        if normalize_key(key_norm):
            return True
    return False


def _machine_name_score(text: str) -> int:
    """
    Scores a candidate machine_name line.
    Higher score = more likely to be the true equipment name.
    """
    norm  = _normalize(text)
    score = 0
    for kw in _EQUIPMENT_KW_NORM:
        if kw in norm:
            score += 3
    # Prefer lines that are not purely supplementary/parenthetical
    if not text.startswith("("):
        score += 1
    # Prefer longer names (more descriptive)
    if len(text) > 10:
        score += 1
    return score


def _extract_machine_name_from_lines(text_lines: List[str]) -> Optional[str]:
    """
    Core anchor-walk algorithm.

    Steps
    ─────
    1. Find the first line containing a metadata anchor (Mã MMTB / Mã máy…).
    2. Walk backwards from that anchor position.
    3. Skip: HTML blocks, empty lines, pure-numeric lines, metadata KV pairs.
    4. STOP when a branding line is encountered.
    5. Score all collected candidates by equipment-keyword presence.
    6. Return the highest-scored candidate, or the closest one to the anchor
       if no candidate has equipment keywords.
    """
    if not text_lines:
        return None

    # 1. Find anchor
    anchor_idx: Optional[int] = None
    for i, line in enumerate(text_lines):
        stripped = _RE_LEADING_BULLET.sub("", line).strip()
        norm_stripped = _normalize(stripped.split("/")[0])  # handle "Mã MMTB/Code MMTB"
        norm_full     = _normalize(line)
        for pat in _ANCHOR_PATTERNS_NORM:
            if pat == "ma":
                if re.search(r"\bma\b", norm_stripped):
                    anchor_idx = i
                    break
            else:
                if norm_stripped.startswith(pat) or pat in norm_full:
                    anchor_idx = i
                    break
        if anchor_idx is not None:
            break

    if anchor_idx is None:
        return None     # no anchor; caller uses existing machine_name

    # 2. Walk backwards collecting candidates
    candidates: List[str] = []

    for i in range(anchor_idx - 1, -1, -1):
        raw_line = text_lines[i].strip()

        # Skip empty lines
        if not raw_line:
            continue

        # Skip HTML div/img blocks
        if raw_line.startswith("<") or _RE_HTML_BLOCK.search(raw_line):
            continue

        # Strip heading markers for analysis
        clean = _RE_HEADING_PREFIX.sub("", raw_line).strip()
        if not clean:
            continue

        # 3. STOP on branding / section header
        if _is_branding_line(clean):
            break

        # Skip pure-numeric lines (e.g. equipment IDs like "19")
        if re.match(r"^\d[\d\s,\.]*$", clean):
            continue

        # Skip lines that are metadata KV pairs for known fields
        if _is_metadata_kv_line(raw_line):
            continue

        # Valid candidate
        candidates.append(clean)

    if not candidates:
        return None

    # 5. Score and pick best candidate
    scored = [(c, _machine_name_score(c)) for c in candidates]
    best_score = max(s for _, s in scored)

    if best_score > 0:
        # Among candidates with the highest score, take the first one
        # (closest to anchor = most specific equipment description)
        best = next(c for c, s in scored if s == best_score)
    else:
        # No equipment keywords in any candidate; fall back to closest to anchor
        best = candidates[0]

    return best


def extract_machine_name_from_paddle_blocks(
    blocks: List[_Block],
    markdown_text: str = "",
) -> Optional[str]:
    """
    Public API for extracting machine_name using the anchor-walk strategy.

    Tries two sources in order:
    1. Block texts (sorted by Y position, top-to-bottom reading order)
    2. Markdown text lines (fallback when no raw blocks available)

    This is a PaddleOCR-specific function.  The Gemini pipeline is not affected.

    Parameters
    ──────────
    blocks        : List of _Block objects (may be empty).
    markdown_text : Raw Paddle markdown string.

    Returns
    ───────
    The extracted machine_name string, or None if no anchor is found.
    """
    # Try blocks first (sorted by Y, most reliable)
    if blocks:
        sorted_blks = sorted(blocks, key=lambda b: (b.cy, b.cx))
        block_lines = [b.text for b in sorted_blks]
        result = _extract_machine_name_from_lines(block_lines)
        if result:
            return result

    # Fallback: markdown lines (HTML blocks already handled inside the algorithm)
    if markdown_text:
        clean_md = _RE_HTML_BLOCK.sub("", markdown_text)
        md_lines = [ln.strip() for ln in clean_md.split("\n")]
        md_lines = [ln for ln in md_lines if ln]
        return _extract_machine_name_from_lines(md_lines)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def transform_paddleocr_result(
    res: dict,
    debug: bool = False,
) -> ParsedResult:
    """
    Converts a single PaddleOCR layoutParsingResult dict into a
    ParsedResult with clean markdown and key_value fields.

    Parameters
    ──────────
    res   : A single entry from result["layoutParsingResults"]
            (contains "markdown.text" and optionally "blocks", "ocr_results", etc.)
    debug : When True, attach a debug_info dict with per-stage logs.

    Returns
    ───────
    ParsedResult with:
      .markdown    — the original Paddle markdown text (unmodified)
      .key_value   — extracted and normalised key-value dict
      .source      — which extraction strategy was used
      .debug_info  — optional dict with per-stage logs
    """
    markdown_text = res.get("markdown", {}).get("text", "")
    dbg_all: Dict[str, Any] = {} if debug else {}

    # ── Stage 1: try to get raw text blocks ──────────────────────────────────
    if debug:
        dbg_all["stage"] = {}

    blocks = extract_text_blocks(res)

    if debug:
        dbg_all["stage"]["1_blocks_extracted"] = len(blocks)
        dbg_all["stage"]["1_block_texts"] = [b.text for b in blocks]
        logger.debug(
            "[PaddleTransformer] Stage 1: extracted %d block(s)", len(blocks)
        )

    # ── Stage 2: KV extraction ───────────────────────────────────────────────
    if blocks:
        kv, dbg2 = parse_kv_from_blocks(blocks, debug=debug)
        source = SRC_RAW_BLOCKS
        if debug:
            dbg_all["stage"]["2_source"] = source
            dbg_all["stage"]["2_log"] = dbg2
            logger.debug("[PaddleTransformer] Stage 2 (raw blocks): %s", dbg2)
    else:
        kv, dbg2 = _enhanced_markdown_kv(markdown_text, debug=debug)
        source = SRC_ENHANCED_MARKDOWN
        if debug:
            dbg_all["stage"]["2_source"] = source
            dbg_all["stage"]["2_log"] = dbg2
            logger.debug("[PaddleTransformer] Stage 2 (enhanced markdown): %s", dbg2)

    # ── Stage 3: normalise all keys ──────────────────────────────────────────
    kv = _normalise_kv_keys(kv)
    if debug:
        dbg_all["stage"]["3_after_normalise"] = dict(kv)

    # ── Stage 4: missing field recovery ─────────────────────────────────────
    kv, dbg4 = _recover_missing_fields(kv, blocks, markdown_text, debug=debug)
    if debug:
        dbg_all["stage"]["4_after_recovery"] = dict(kv)
        dbg_all["stage"]["4_log"] = dbg4
        logger.debug("[PaddleTransformer] Stage 4 (recovery): %s", dbg4)

    # ── Stage 5: anchor-walk machine_name extraction ─────────────────────────
    # Overrides whatever machine_name Stage 2 produced (which is the
    # naive first-line heuristic).  This properly handles company branding
    # like DAIDUNG / DAI DUNG appearing before the real equipment name.
    better_name = extract_machine_name_from_paddle_blocks(blocks, markdown_text)
    if better_name:
        kv["machine_name"] = better_name
        if debug:
            dbg_all["stage"]["5_machine_name"] = better_name
            logger.debug("[PaddleTransformer] Stage 5 machine_name: %s", better_name)

    if debug:
        missing_final = [f for f in TRACKED_FIELDS if f not in kv or not kv[f]]
        dbg_all["missing_fields"] = missing_final
        dbg_all["final_kv"]       = dict(kv)
        logger.debug(
            "[PaddleTransformer] Final KV: %s | missing: %s", kv, missing_final
        )

    return ParsedResult(
        markdown=markdown_text,
        key_value=kv,
        source=source,
        debug_info=dbg_all if debug else None,
    )
