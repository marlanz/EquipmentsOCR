"""
diagnose_failed_cases.py
────────────────────────
Runs all 5 failed cases from failed-test/raw.json through BOTH the old
(first-line heuristic) and new (anchor-walk) machine_name extraction algorithms.

Produces a side-by-side comparison table and accuracy percentages.

Usage:
    python diagnose_failed_cases.py
"""

import json
import sys
import re
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.paddle_transformer import (
    _enhanced_markdown_kv,
    extract_machine_name_from_paddle_blocks,
    transform_paddleocr_result,
)

RAW_JSON = Path("failed-test/raw.json")

# ── Ground-truth expected machine_names ──────────────────────────────────────
# Derived from visual inspection of the label images.
EXPECTED = [
    "MÁY HÀN ĐIỆN TỬ/ELECTRICAL WELDING MACHINE",   # Case 1
    "Máy ghép dảm hợp 3000x4000",                    # Case 2
    "MÁY GOUGING JÁSIC",                              # Case 3
    "MÁY HÀN GOUGING MZ-1000 -GOUGING WELDING MACHINE MZ-1000",  # Case 4
    "Máy hàn Ehave, CM500 DC",                        # Case 5
]

CYAN  = "\033[96m"
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).lower().strip()


def _names_match(a: str, b: str) -> bool:
    """Case-insensitive, diacritic-insensitive match."""
    return _normalize(a) == _normalize(b)


def old_machine_name(markdown: str) -> str:
    """Simulates the OLD first-line heuristic."""
    import re as _re
    clean = _re.sub(
        r"<div[^>]*>.*?</div>", "", markdown, flags=_re.IGNORECASE | _re.DOTALL
    )
    lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]
    if not lines:
        return ""
    first = _re.sub(r"^#+\s*", "", lines[0]).strip()
    if ":" not in first and "：" not in first and "|" not in first:
        return first
    return ""


def main():
    cases = json.loads(RAW_JSON.read_text(encoding="utf-8"))

    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  PaddleOCR machine_name Extraction — Before vs After Comparison{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    old_pass = 0
    new_pass = 0

    rows = []

    for i, (case, expected) in enumerate(zip(cases, EXPECTED), start=1):
        md = case.get("markdown", "")

        # --- OLD algorithm ---
        old_name = old_machine_name(md)

        # --- NEW algorithm ---
        res = {"markdown": {"text": md}}
        parsed = transform_paddleocr_result(res)
        new_name = parsed.key_value.get("machine_name", "")

        # --- Direct function output ---
        direct_name = extract_machine_name_from_paddle_blocks([], md)

        old_ok = _names_match(old_name, expected)
        new_ok = _names_match(new_name, expected)

        if old_ok:
            old_pass += 1
        if new_ok:
            new_pass += 1

        rows.append({
            "i": i,
            "expected": expected,
            "old_name": old_name,
            "new_name": new_name,
            "direct":   direct_name or "",
            "old_ok":   old_ok,
            "new_ok":   new_ok,
        })

    # ── Print detailed results ────────────────────────────────────────────────
    for r in rows:
        tick  = f"{GREEN}✔{RESET}"
        cross = f"{RED}✗{RESET}"
        old_mark = tick if r["old_ok"] else cross
        new_mark = tick if r["new_ok"] else cross

        print(f"{BOLD}Case {r['i']}{RESET}")
        print(f"  Expected  : {CYAN}{r['expected']}{RESET}")
        print(f"  OLD (1st) : {old_mark} {r['old_name']!r}")
        print(f"  NEW (anch): {new_mark} {r['new_name']!r}")
        if r["direct"] != r["new_name"]:
            print(f"  direct fn : {r['direct']!r}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(EXPECTED)
    print(f"{BOLD}{'─'*50}{RESET}")
    print(f"  OLD accuracy: {old_pass}/{n}  ({old_pass/n*100:.0f}%)")
    print(f"  NEW accuracy: {new_pass}/{n}  ({new_pass/n*100:.0f}%)")

    improvement = new_pass - old_pass
    label = f"+{improvement}" if improvement >= 0 else str(improvement)
    colour = GREEN if improvement > 0 else (RED if improvement < 0 else "")
    print(f"  Improvement : {colour}{label} case(s) fixed{RESET}")
    print(f"{BOLD}{'='*50}{RESET}\n")

    # ── Per-case JSON output ──────────────────────────────────────────────────
    print(f"{BOLD}JSON validation results:{RESET}")
    results = []
    for r in rows:
        results.append({
            "case": r["i"],
            "expected_machine_name":  r["expected"],
            "old_machine_name":       r["old_name"],
            "new_machine_name":       r["new_name"],
            "old_passed":             r["old_ok"],
            "new_passed":             r["new_ok"],
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
