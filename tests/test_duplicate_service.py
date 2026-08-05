import pytest
from unittest.mock import AsyncMock, MagicMock
from bot.services.sheet_service import SheetService
from bot.services.duplicate_service import DuplicateService


@pytest.mark.asyncio
async def test_check_duplication_code_found():
    mock_sheet_service = MagicMock(spec=SheetService)
    mock_sheet_service.lookup_row_by_mmtb_raise_errors = AsyncMock(
        return_value={"row_index": 2, "kv": {"Mã MMTB": "AH-MC-001"}}
    )

    dup_service = DuplicateService(sheet_service=mock_sheet_service)
    result = await dup_service.check_duplication_code("AH-MC-001")

    assert result is True
    mock_sheet_service.lookup_row_by_mmtb_raise_errors.assert_called_once_with("AH-MC-001")


@pytest.mark.asyncio
async def test_check_duplication_code_not_found():
    mock_sheet_service = MagicMock(spec=SheetService)
    mock_sheet_service.lookup_row_by_mmtb_raise_errors = AsyncMock(return_value=None)

    dup_service = DuplicateService(sheet_service=mock_sheet_service)
    result = await dup_service.check_duplication_code("AH-MC-999")

    assert result is False
    mock_sheet_service.lookup_row_by_mmtb_raise_errors.assert_called_once_with("AH-MC-999")


@pytest.mark.asyncio
async def test_check_duplication_code_exempt_and_empty():
    mock_sheet_service = MagicMock(spec=SheetService)
    mock_sheet_service.lookup_row_by_mmtb_raise_errors = AsyncMock()

    dup_service = DuplicateService(sheet_service=mock_sheet_service)

    assert await dup_service.check_duplication_code("") is False
    assert await dup_service.check_duplication_code("chưa có mã") is False
    assert await dup_service.check_duplication_code("CHƯA CÓ MÃ") is False
    assert await dup_service.check_duplication_code("—") is False

    # Should not trigger Google Sheet lookup for exempt or empty codes
    mock_sheet_service.lookup_row_by_mmtb_raise_errors.assert_not_called()


@pytest.mark.asyncio
async def test_check_duplication_code_exception_propagates():
    mock_sheet_service = MagicMock(spec=SheetService)
    mock_sheet_service.lookup_row_by_mmtb_raise_errors = AsyncMock(
        side_effect=RuntimeError("Google Sheets connection error")
    )

    dup_service = DuplicateService(sheet_service=mock_sheet_service)

    with pytest.raises(RuntimeError, match="Google Sheets connection error"):
        await dup_service.check_duplication_code("AH-MC-001")
