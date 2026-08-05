import asyncio
import logging
from typing import Optional, List, Dict
from app.helpers import (
    get_gspread_client,
    GOOGLE_SHEETS_ENABLED,
    GOOGLE_SHEETS_NAME,
    _normalize,
    append_results_to_sheet_sync,
    update_row_in_sheet_sync,
    lookup_row_by_mmtb_sync,
)
from app.schemas import OCRResult

logger = logging.getLogger(__name__)


class SheetService:
    async def append_results(
        self,
        results: List[OCRResult],
        source: str = "UNKNOWN",
        force: bool = False,
    ) -> List[str]:
        """Runs append_results_to_sheet_sync in a thread pool."""
        return await asyncio.to_thread(
            append_results_to_sheet_sync, results, source, force
        )

    async def lookup_row_by_mmtb(self, mmtb_code: str) -> Optional[dict]:
        """Runs lookup_row_by_mmtb_sync in a thread pool."""
        return await asyncio.to_thread(lookup_row_by_mmtb_sync, mmtb_code)

    async def update_row_in_sheet(
        self,
        row_index: int,
        row_data: list,
        source: str = "FIND_UPDATE",
    ) -> bool:
        """Runs update_row_in_sheet_sync in a thread pool."""
        return await asyncio.to_thread(
            update_row_in_sheet_sync, row_index, row_data, source
        )

    async def lookup_row_by_mmtb_raise_errors(self, mmtb_code: str) -> Optional[dict]:
        """
        Looks up a row in the Google Sheet by Mã MMTB value (fuzzy/diacritic-insensitive).
        Unlike lookup_row_by_mmtb_sync, this raises exceptions if Google Sheets cannot be reached
        or is disabled.
        """
        def _lookup():
            if not GOOGLE_SHEETS_ENABLED:
                raise RuntimeError("Google Sheets integration is disabled in configuration.")

            gsclient = get_gspread_client()
            if not gsclient:
                raise RuntimeError("Failed to initialize or authorize Google Sheets client.")

            # This can raise gspread.exceptions.APIError, httpx.HTTPError, etc.
            spreadsheet = gsclient.open(GOOGLE_SHEETS_NAME)
            worksheet   = spreadsheet.get_worksheet(0)
            all_values  = worksheet.get_all_values()

            if not all_values or len(all_values) < 2:
                return None

            norm_search = _normalize(mmtb_code)
            MMTB_COL_IDX = 1

            col_to_key = {
                0: "machine_name",
                1: "Mã MMTB",
                2: "Model",
                3: "Xưởng",
                4: "Vị trí",
                5: "status",
            }

            for row_num, row in enumerate(all_values[1:], start=2):  # skip header row
                if len(row) > MMTB_COL_IDX:
                    cell_val = row[MMTB_COL_IDX].strip()
                    if _normalize(cell_val) == norm_search:
                        kv = {}
                        for col_idx, val in enumerate(row):
                            key = col_to_key.get(col_idx, f"col_{col_idx}")
                            kv[key] = val.strip()
                        logger.info(
                            f"[SheetService] Found '{mmtb_code}' at row {row_num}."
                        )
                        return {"row_index": row_num, "kv": kv}

            logger.info(f"[SheetService] '{mmtb_code}' not found in sheet.")
            return None

        return await asyncio.to_thread(_lookup)
