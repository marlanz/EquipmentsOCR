"""
diagnose_pipeline.py
────────────────────
Runs the full PaddleOCR text → markdown → key-value pipeline on a set of
synthetic and real-world test cases and reports exactly where information
is lost.

Run with:   python diagnose_pipeline.py
Output:     printed to stdout  (also saved to diagnose_output.txt)
"""

import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Inline copy of the production parser (no FastAPI/gspread dependencies needed)
# ─────────────────────────────────────────────────────────────────────────────

def parse_markdown_to_key_value(markdown_text: str) -> Dict[str, str]:
    """Exact copy of app/helpers.py:parse_markdown_to_key_value (production)."""
    key_value: Dict[str, str] = {}
    if not markdown_text:
        return key_value

    lines = [line.strip() for line in markdown_text.split("\n") if line.strip()]
    if not lines:
        return key_value

    # 1. Heading/Title on first line
    first_line = lines[0]
    first_line_clean = re.sub(r"^#+\s*", "", first_line).strip()
    if first_line_clean and ":" not in first_line_clean and "：" not in first_line_clean:
        key_value["machine_name"] = first_line_clean

    # 2. Key-Value pairs
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


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helper (same as production)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()


FIELD_NORM_KEYS = {
    "Mã MMTB":  ["ma mmtb", "ma may", "ma mmtb"],
    "Model":    ["model", "mo hinh"],
    "Xưởng":   ["xuong", "xuong san xuat", "nha may"],
    "Vị trí":  ["vi tri", "vitri", "tri"],
}


def _has_field(kv: Dict[str, str], canonical: str) -> Tuple[bool, str]:
    """Returns (found, matched_key). Checks exact, case-insensitive, alias."""
    if canonical in kv:
        return True, canonical
    for k in kv:
        if k.lower().strip() == canonical.lower().strip():
            return True, k
    norm_c = _norm(canonical)
    for k in kv:
        if _norm(k) == norm_c:
            return True, k
    aliases = [_norm(a) for a in FIELD_NORM_KEYS.get(canonical, [])]
    for k in kv:
        if _norm(k) in aliases:
            return True, k
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
#   Each case has:
#     name        – human label
#     raw_blocks  – what PaddleOCR text-detection actually produces
#                   (simulates the "source of truth" OCR text list)
#     markdown    – what the Paddle layout-parser puts in markdown.text
#                   (the string fed to parse_markdown_to_key_value)
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [

    # ── TC-1: Real sample from output/doc_0.md ──────────────────────────────
    {
        "name": "TC-1  Real sample – output/doc_0.md (MÁY HÀN CO2)",
        "raw_blocks": [
            "MÁY HÀN CO2 TÂN THÀNH - TTC-500T",
            "Mã MMTB : B22401469",
            "Model : TTC-500T",
            "Xương : AH6",
            "Vị",          # split across two text boxes
            "trí : RHT1",
        ],
        "markdown": (
            "MÁY HÀN CO2 TÂN THÀNH - TTC-500T\n"
            "\n"
            "Mã MMTB : B22401469\n"
            "\n"
            "Model : TTC-500T\n"
            "\n"
            "Xương : AH6\n"
            "\n"
            '<div style="text-align: center;"><img src="imgs/img_in_image_box_0_309_47_355.jpg" '
            'alt="Image" width="7%" /></div>\n'
            "\n"
            "\n"
            "trí : RHT1\n"
        ),
    },

    # ── TC-2: "Vị trí" value on NEXT line (common label layout) ────────────
    {
        "name": "TC-2  Value on next line (multi-line KV)",
        "raw_blocks": [
            "MÁY TIỆN CNC",
            "Mã MMTB : A001",
            "Model",
            ": HAAS ST-10",
            "Xưởng : CNC",
            "Vị trí",
            ": C12",
        ],
        "markdown": (
            "MÁY TIỆN CNC\n"
            "\n"
            "Mã MMTB : A001\n"
            "\n"
            "Model\n"
            "\n"
            ": HAAS ST-10\n"
            "\n"
            "Xưởng : CNC\n"
            "\n"
            "Vị trí\n"
            "\n"
            ": C12\n"
        ),
    },

    # ── TC-3: Table layout – Paddle renders as markdown table ───────────────
    {
        "name": "TC-3  Table layout (Paddle renders | separators)",
        "raw_blocks": [
            "MÁY PHAY CNC",
            "| Mã MMTB | C003 |",
            "| Model | VMC-850 |",
            "| Xưởng | Phay |",
            "| Vị trí | D5 |",
        ],
        "markdown": (
            "MÁY PHAY CNC\n"
            "\n"
            "| Mã MMTB | C003 |\n"
            "| Model | VMC-850 |\n"
            "| Xưởng | Phay |\n"
            "| Vị trí | D5 |\n"
        ),
    },

    # ── TC-4: Inline image element splits a KV pair ─────────────────────────
    {
        "name": "TC-4  Inline image between key and value",
        "raw_blocks": [
            "MÁY HÀN MIG",
            "Mã MMTB : D005",
            "[logo image]",
            "Model : MIG-350",
            "Xưởng : HÀN",
            "Vị trí : E2",
        ],
        "markdown": (
            "MÁY HÀN MIG\n"
            "\n"
            "Mã MMTB : D005\n"
            "\n"
            '<div style="text-align: center;"><img src="imgs/logo.jpg" /></div>\n'
            "\n"
            "Model : MIG-350\n"
            "\n"
            "Xưởng : HÀN\n"
            "\n"
            "Vị trí : E2\n"
        ),
    },

    # ── TC-5: Full colon dropped / OCR mis-reads separator ──────────────────
    {
        "name": "TC-5  OCR mis-reads colon as period or dash",
        "raw_blocks": [
            "MÁY MÀI",
            "Mã MMTB . E009",
            "Model - GR-200",
            "Xưởng : MÀI",
            "Vị trí : F3",
        ],
        "markdown": (
            "MÁY MÀI\n"
            "\n"
            "Mã MMTB . E009\n"
            "\n"
            "Model - GR-200\n"
            "\n"
            "Xưởng : MÀI\n"
            "\n"
            "Vị trí : F3\n"
        ),
    },

    # ── TC-6: Extra bold/asterisk markdown decoration ───────────────────────
    {
        "name": "TC-6  Bold markdown wrappers around keys",
        "raw_blocks": [
            "MÁY KHOAN",
            "**Mã MMTB** : F010",
            "**Model** : DR-13",
            "**Xưởng** : KHOAN",
            "**Vị trí** : G7",
        ],
        "markdown": (
            "MÁY KHOAN\n"
            "\n"
            "**Mã MMTB** : F010\n"
            "\n"
            "**Model** : DR-13\n"
            "\n"
            "**Xưởng** : KHOAN\n"
            "\n"
            "**Vị trí** : G7\n"
        ),
    },

    # ── TC-7: "Xương" instead of "Xưởng" (common OCR diacritic error) ──────
    {
        "name": "TC-7  Diacritic error: 'Xương' instead of 'Xưởng'",
        "raw_blocks": [
            "MÁY CẮT LASER",
            "Mã MMTB : G011",
            "Model : FLC-3015",
            "Xương : LASER",     # OCR mis-reads ưở as ươ
            "Vị trí : H8",
        ],
        "markdown": (
            "MÁY CẮT LASER\n"
            "\n"
            "Mã MMTB : G011\n"
            "\n"
            "Model : FLC-3015\n"
            "\n"
            "Xương : LASER\n"
            "\n"
            "Vị trí : H8\n"
        ),
    },

    # ── TC-8: Key and value on same line but no space after colon ───────────
    {
        "name": "TC-8  No space after colon  (key:value)",
        "raw_blocks": [
            "MÁY UỐN ỐNG",
            "Mã MMTB:H012",
            "Model:PB-50",
            "Xưởng:UỐN",
            "Vị trí:I9",
        ],
        "markdown": (
            "MÁY UỐN ỐNG\n"
            "\n"
            "Mã MMTB:H012\n"
            "\n"
            "Model:PB-50\n"
            "\n"
            "Xưởng:UỐN\n"
            "\n"
            "Vị trí:I9\n"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TRACKED_FIELDS = ["Mã MMTB", "Model", "Xưởng", "Vị trí"]


def raw_contains(blocks: List[str], field: str) -> bool:
    """True if any raw text block contains the field name (fuzzy)."""
    norm_f = _norm(field)
    aliases = [_norm(a) for a in FIELD_NORM_KEYS.get(field, [])] + [norm_f]
    for blk in blocks:
        norm_blk = _norm(blk)
        for alias in aliases:
            if alias in norm_blk:
                return True
    return False


def markdown_contains(md: str, field: str) -> bool:
    """True if the markdown string contains the field name (fuzzy)."""
    norm_f = _norm(field)
    aliases = [_norm(a) for a in FIELD_NORM_KEYS.get(field, [])] + [norm_f]
    norm_md = _norm(md)
    return any(alias in norm_md for alias in aliases)


def analyse(tc: dict):
    raw_blocks   = tc["raw_blocks"]
    markdown     = tc["markdown"]
    parsed_kv    = parse_markdown_to_key_value(markdown)

    results = {}
    for field in TRACKED_FIELDS:
        in_raw  = raw_contains(raw_blocks, field)
        in_md   = markdown_contains(markdown, field)
        found, matched_key = _has_field(parsed_kv, field)
        results[field] = {
            "in_raw":    in_raw,
            "in_md":     in_md,
            "in_parsed": found,
            "value":     parsed_kv.get(matched_key, ""),
            "lost_at":   (
                "OCR→Markdown" if in_raw and not in_md else
                "Markdown→KV"  if in_md and not found  else
                "not_in_raw"   if not in_raw            else
                "OK"
            ),
        }
    return parsed_kv, results


# ─────────────────────────────────────────────────────────────────────────────
# Additional pattern-level tests (unit tests for the regex)
# ─────────────────────────────────────────────────────────────────────────────

REGEX_CASES = [
    # (input_line, expected_key, expected_value_prefix)
    ("Mã MMTB : B22401469",       "Mã MMTB",    "B22401469"),
    ("Model : TTC-500T",           "Model",       "TTC-500T"),
    ("Xương : AH6",                "Xương",       "AH6"),
    ("trí : RHT1",                 "trí",         "RHT1"),   # broken "Vị trí" – key is truncated
    ("Vị trí : C5",                "Vị trí",      "C5"),
    ("**Mã MMTB** : F010",         "Mã MMTB",    "F010"),
    ("**Model** : DR-13",          "Model",       "DR-13"),
    ("Mã MMTB:H012",               "Mã MMTB",    "H012"),
    ("Model - GR-200",             None,          None),      # dash separator – should FAIL
    ("Mã MMTB . E009",             None,          None),      # period separator – should FAIL
    ("| Mã MMTB | C003 |",         None,          None),      # table cell – should FAIL
    (": HAAS ST-10",               None,          None),      # value-only orphan – should FAIL
]


def run_regex_tests(output_lines: List[str]):
    kv_pattern = re.compile(r"^\s*(?:\*\*)?\s*([^*：:]+?)\s*(?:\*\*)?\s*[:：]\s*(.*)$")
    output_lines.append("\n" + "═"*70)
    output_lines.append("REGEX UNIT TESTS")
    output_lines.append("═"*70)
    pass_count = fail_count = 0
    for line, exp_key, exp_val in REGEX_CASES:
        m = kv_pattern.match(line)
        if m:
            k = re.sub(r"^\*+\s*|\s*\*+$", "", m.group(1).strip()).strip()
            v = re.sub(r"^\*+\s*|\s*\*+$", "", m.group(2).strip()).strip()
            parsed = (k, v)
        else:
            parsed = (None, None)

        if exp_key is None:
            ok = parsed == (None, None)
        else:
            ok = parsed[0] == exp_key and (exp_val is None or parsed[1].startswith(exp_val))

        icon = "✔" if ok else "✘"
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        output_lines.append(
            f"  {icon}  Input: {line!r:45s}  →  parsed={parsed}  "
            f"expected=({exp_key!r}, {exp_val!r})"
        )
    output_lines.append(f"\nRegex tests: {pass_count} passed, {fail_count} failed\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main report
# ─────────────────────────────────────────────────────────────────────────────

def main():
    lines: List[str] = []
    W = 70

    lines.append("=" * W)
    lines.append("  PaddleOCR PIPELINE DIAGNOSTIC REPORT")
    lines.append("=" * W)

    summary_rows = []  # for the final table

    for tc in TEST_CASES:
        parsed_kv, results = analyse(tc)

        lines.append(f"\n{'─'*W}")
        lines.append(f"  {tc['name']}")
        lines.append(f"{'─'*W}")

        lines.append("\n[RAW BLOCKS]")
        for i, blk in enumerate(tc["raw_blocks"]):
            lines.append(f"  [{i}] {blk!r}")

        lines.append("\n[MARKDOWN TEXT]")
        for ln in tc["markdown"].splitlines():
            # Truncate HTML/img tags for readability
            if ln.startswith("<div"):
                lines.append(f"  <img block truncated>")
            else:
                lines.append(f"  {ln!r}")

        lines.append("\n[PARSED KEY-VALUE]")
        if parsed_kv:
            for k, v in parsed_kv.items():
                lines.append(f"  {k!r:25s} → {v!r}")
        else:
            lines.append("  (empty)")

        lines.append("\n[FIELD TRACKING]")
        row = {"tc": tc["name"].split("–")[0].strip()}
        for field, r in results.items():
            status_emoji = (
                "✅" if r["lost_at"] == "OK" else
                "⚠️" if r["lost_at"] == "not_in_raw" else
                "❌"
            )
            lines.append(
                f"  {status_emoji} {field:12s} "
                f"raw={'Y' if r['in_raw'] else 'N'}  "
                f"md={'Y' if r['in_md'] else 'N'}  "
                f"parsed={'Y' if r['in_parsed'] else 'N'}  "
                f"value={r['value']!r:20s}  lost_at={r['lost_at']}"
            )
            row[field] = r

        # Detect specific failure patterns
        failures = [f for f, r in results.items() if r["lost_at"] not in ("OK", "not_in_raw")]
        if failures:
            lines.append(f"\n  ⚠ FAILURES: {', '.join(failures)}")
            for f in failures:
                r = results[f]
                lines.append(f"    → {f}: lost at [{r['lost_at']}]")

        summary_rows.append(row)

    # ── Regex tests ───────────────────────────────────────────────────────────
    run_regex_tests(lines)

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append("=" * W)
    lines.append("SUMMARY TABLE")
    lines.append("=" * W)
    hdr = f"{'Test Case':<10} {'Mã MMTB(raw/md/kv)':<22} {'Model(raw/md/kv)':<20} {'Xưởng(raw/md/kv)':<20} {'Vị trí(raw/md/kv)':<20}"
    lines.append(hdr)
    lines.append("-" * W)

    for row in summary_rows:
        tc_short = row["tc"].split()[0] + " " + row["tc"].split()[1]

        def cell(f):
            r = row.get(f, {})
            raw = "Y" if r.get("in_raw") else "N"
            md  = "Y" if r.get("in_md")  else "N"
            kv  = "Y" if r.get("in_parsed") else "N"
            mark = "✅" if kv == "Y" else ("⚠" if md == "N" and raw == "Y" else "❌")
            return f"{mark}{raw}/{md}/{kv}"

        lines.append(
            f"{tc_short:<10} {cell('Mã MMTB'):<22} {cell('Model'):<20} {cell('Xưởng'):<20} {cell('Vị trí'):<20}"
        )

    lines.append("")

    # ── Root cause analysis ───────────────────────────────────────────────────
    lines.append("=" * W)
    lines.append("ROOT CAUSE ANALYSIS")
    lines.append("=" * W)

    rca = """
ISSUE 1 – Broken "Vị trí" (Markdown→KV stage)
   The string "Vị trí" contains a QR/logo image box between "Vị" and
   "trí" in the physical label layout. PaddleOCR emits:
     line A:  "Vị"           (no colon → discarded by regex)
     line B:  "trí : RHT1"  (key extracted as "trí", not "Vị trí")
   Stage lost: OCR→Markdown (PaddleOCR split the text box).
   Also Markdown→KV (the truncated key "trí" is not alias-matched to "Vị trí").

ISSUE 2 – Value on next line (Markdown→KV stage)
   Some labels print the key on one line and the value on the next:
     "Model"        ← no colon → discarded by the kv_pattern regex
     ": HAAS ST-10" ← regex requires a key before the colon → discarded
   The current parser does not do look-ahead / look-behind across lines.
   Stage lost: Markdown→KV

ISSUE 3 – Table layout not parsed (Markdown→KV stage)
   Paddle renders some grid-like labels as markdown tables:
     "| Mã MMTB | C003 |"
   The kv_pattern regex expects "key : value", not "| key | value |".
   Pipe characters are never matched. All table fields are silently dropped.
   Stage lost: Markdown→KV

ISSUE 4 – OCR mis-reads separator character (OCR stage)
   Occasionally PaddleOCR produces "." or "-" instead of ":".
     "Mã MMTB . E009"   → regex requires [:：] → not matched → field lost
     "Model - GR-200"   → same
   Stage lost: OCR (wrong character) then propagated to Markdown→KV.

ISSUE 5 – Diacritic error: "Xương" vs "Xưởng"
   The _get_kv alias lookup in initbot.py covers this, BUT only for
   display/lookup. The raw kv dict stores "Xương" as-is, which does NOT
   match the canonical key "Xưởng" unless the alias check is applied.
   The alias list in helpers.KEY_MAPPING does include "xương" so the
   get_standardized_value function will find it when writing to Sheets.
   For the bot display in FIELDS, _get_kv in initbot.py also handles it.
   Risk: Medium – aliasing is fragile and must stay in sync.

ISSUE 6 – HTML image blocks interrupt parsing
   Paddle wraps extracted images in <div>...</div> blocks inside the
   markdown. These lines are stripped by the "line.strip()" filter since
   they are non-empty, so they count as a line. However the regex never
   matches them (no colon in correct position), so they are silently
   ignored. The real danger is when the image block splits a multi-line
   KV pair (see TC-1 real sample: image sits between "Xương : AH6" and
   "trí : RHT1").
"""
    lines.append(rca)

    lines.append("=" * W)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * W)

    recs = """
R1 – Parse DIRECTLY from the raw OCR JSON blocks (highest priority)
   The raw Paddle JSONL contains a structured list of text blocks with
   bounding boxes. Parsing KV pairs from this list is far more reliable
   than parsing from the re-rendered markdown:
   • No image interruptions
   • Each block has spatial position → can detect split keys
   • Tables appear as adjacent cells, not pipe-separated strings
   Implementation: replace parse_markdown_to_key_value with a function
   that reads from result["layoutParsingResults"][i]["structuredBlocks"]
   or similar raw field in the Paddle JSON.

R2 – Add table-row parser to parse_markdown_to_key_value
   Add a second regex branch:
     r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
   This captures "| key | value |" patterns with no other code change.

R3 – Add multi-line KV merging
   After splitting lines, scan for orphan-value lines (starts with ":")
   and merge them with the previous key-only line:
     if line.startswith(":") and last_key:
         kv[last_key] = line[1:].strip()

R4 – Expand separator matching
   Extend the kv_pattern to accept ".", "–", "-" as separators when
   followed by a known-field name:
     r"^(Mã MMTB|Model|Xưởng|Vị trí)\s*[:.：\-\.]\s*(.+)$"

R5 – Fix "Vị trí" split key
   After collecting all KV pairs, if key "trí" is present but "Vị trí"
   is absent, promote "trí" → "Vị trí":
     if "trí" in kv and "Vị trí" not in kv:
         kv["Vị trí"] = kv.pop("trí")

R6 – Write an integration test that compares raw JSON → final KV
   Capture one Paddle API response per label type and store as a fixture.
   Run the full pipeline on each fixture and assert every tracked field
   is present in the final kv dict.
"""
    lines.append(recs)

    output = "\n".join(lines)
    print(output)

    with open("diagnose_output.txt", "w", encoding="utf-8") as f:
        f.write(output)
    print("\n[Saved to diagnose_output.txt]")


if __name__ == "__main__":
    main()
