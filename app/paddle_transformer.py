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
    "tri":           "Vị trí",     # corrupted "Vị trí" (image split)
    "i tri":         "Vị trí",     # corruption "(i trí"
    "(i tri":        "Vị trí",     # full "(i trí" OCR artifact
    "vi tri ":       "Vị trí",
    "vitri ":        "Vị trí",
    "location":      "Vị trí",
    "vi":            "Vị trí",     # very short — last resort only
}

# Short aliases (len < 4) are only used as last resort to avoid false positives
_SHORT_ALIASES = {"px", "vi"}

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

    # Standard: "key : value" or "**key** : value"
    m = _RE_KV_STANDARD.match(text)
    if m:
        k = _RE_STRIP_BOLD.sub("", m.group(1).strip()).strip()
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
        sm = _RE_KV_STANDARD.match(line)
        if sm:
            k = _RE_STRIP_BOLD.sub("", sm.group(1).strip()).strip()
            v = _RE_STRIP_BOLD.sub("", sm.group(2).strip()).strip()
            if k:
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
    """
    result: Dict[str, str] = {}
    for k, v in kv.items():
        canonical = normalize_key(k)
        final_key = canonical if canonical else k
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
