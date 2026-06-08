"""
Diagnostic script — run from project root:
    .venv\Scripts\python.exe diag_sheet.py
"""
import sys, os
# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from app.config import (
    GOOGLE_SHEETS_ENABLED,
    GOOGLE_SHEETS_NAME,
    GOOGLE_SHEETS_CREDENTIALS_JSON,
    GOOGLE_SHEETS_CREDENTIALS_PATH,
)
from app.helpers import get_gspread_client

print("=" * 60)
print(f"GOOGLE_SHEETS_ENABLED : {GOOGLE_SHEETS_ENABLED}")
print(f"GOOGLE_SHEETS_NAME    : {GOOGLE_SHEETS_NAME!r}")
print(f"Credentials JSON env  : {'SET' if GOOGLE_SHEETS_CREDENTIALS_JSON else 'NOT SET'}")
print(f"Credentials file path : {GOOGLE_SHEETS_CREDENTIALS_PATH}")
print(f"Credentials file exist: {os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH)}")
print("=" * 60)

if not GOOGLE_SHEETS_ENABLED:
    print("ERROR: Google Sheets is DISABLED. Check credentials.")
    sys.exit(1)

print("\nConnecting to gspread...")
client = get_gspread_client()
if not client:
    print("ERROR: Could not get gspread client.")
    sys.exit(1)
print("  [OK] gspread client connected")

print(f"\nOpening sheet: {GOOGLE_SHEETS_NAME!r}...")
try:
    spreadsheet = client.open(GOOGLE_SHEETS_NAME)
    worksheet   = spreadsheet.get_worksheet(0)
    print(f"  [OK] Sheet opened. Worksheet: {worksheet.title!r}")
except Exception as e:
    print(f"  [FAIL] Failed to open sheet: {e}")
    sys.exit(1)

print("\nFetching all values...")
try:
    all_vals  = worksheet.get_all_values()
    total_rows = len(all_vals)
    data_rows  = total_rows - 1  # minus header
    print(f"  Total rows (incl. header): {total_rows}")
    print(f"  Data rows               : {data_rows}")
    if total_rows > 0:
        print(f"  Header row : {all_vals[0]}")
    if total_rows > 1:
        print(f"  Last data  : {all_vals[-1]}")
except Exception as e:
    print(f"  [FAIL] get_all_values() failed: {e}")
    sys.exit(1)

print("\nChecking worksheet dimensions...")
try:
    row_count = worksheet.row_count
    col_count = worksheet.col_count
    print(f"  Worksheet max rows : {row_count}")
    print(f"  Worksheet max cols : {col_count}")
    used = total_rows
    remaining = row_count - used
    if remaining <= 0:
        print(f"  [WARNING] Sheet is FULL! ({used}/{row_count} rows used)")
    else:
        print(f"  [OK] Sheet has {remaining} rows remaining ({used}/{row_count} used)")
except Exception as e:
    print(f"  Could not check dimensions: {e}")

print("\nAttempting TEST WRITE (will be auto-deleted)...")
try:
    test_row = ["[DIAG TEST]", "DIAG-001", "MODEL-TEST", "DIAG_XUONG", "DIAG_VI_TRI", "Chua xac nhan"]
    worksheet.append_row(test_row)
    print("  [OK] append_row() succeeded!")

    # Clean up
    new_total = len(worksheet.get_all_values())
    worksheet.delete_rows(new_total)
    print(f"  [OK] Test row deleted (was row {new_total}). Sheet restored.")
except Exception as e:
    print(f"  [FAIL] append_row() FAILED: {e}")

print("\nDone.")
