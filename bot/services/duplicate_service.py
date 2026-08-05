import logging
from bot.services.sheet_service import SheetService

logger = logging.getLogger(__name__)


class DuplicateService:
    def __init__(self, sheet_service: SheetService):
        self.sheet_service = sheet_service

    async def check_duplication_code(self, machine_code: str) -> bool:
        """
        Check if the machine_code exists in the Google Sheet.
        Returns:
            True if the code already exists in Google Sheets.
            False if it does not exist or if the code is empty/exempt.
        """
        if not machine_code:
            logger.info("Machine code is empty; skipping duplication check.")
            return False

        stripped_code = machine_code.strip().lower()
        exempt_codes = {"chưa có mã", "chua co ma", "—", ""}
        if stripped_code in exempt_codes:
            logger.info(f"Machine code '{machine_code}' is exempt; skipping duplication check.")
            return False

        logger.info(f"Checking duplicate equipment code: {machine_code}")

        # lookup_row_by_mmtb_raise_errors will raise an exception on connectivity failure
        result = await self.sheet_service.lookup_row_by_mmtb_raise_errors(machine_code)

        if result is not None:
            logger.info("Duplicate found")
            return True

        logger.info("Duplicate not found")
        return False
