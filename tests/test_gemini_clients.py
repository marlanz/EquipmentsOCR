"""
tests/test_gemini_clients.py
─────────────────────────────────────────────────────────────────────────────
Unit tests covering:

  1. Config layer  – GEMINI_API_KEY_1..5 and GEMINI_MODEL_1..5 are exported.
  2. helpers.py    – client1..5 are initialised (or left None) based on keys.
  3. call_gemini_ocr()  – each model_index 1..5 dispatches to the right client/model.
  4. call_gemini_ocr()  – invalid index (e.g. 0, 6) falls back to client1 / raises.
  5. check_gemini_connectivity() – enumerates only the live clients.
  6. validate_upload_gemini()    – extension, MIME, size guards still pass.
  7. validate_upload_paddle()    – PaddleOCR validation still passes.

All network / API calls are mocked – no real keys required.
"""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch, AsyncMock
from PIL import Image
import io


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config layer – all 5 key + model names are exported
# ─────────────────────────────────────────────────────────────────────────────
class TestConfigExports:
    """All GEMINI_API_KEY_N / GEMINI_MODEL_N symbols must be importable."""

    def test_key_1_exported(self):
        from app.config import GEMINI_API_KEY, GEMINI_MODEL
        assert isinstance(GEMINI_API_KEY, str)
        assert isinstance(GEMINI_MODEL, str)
        assert GEMINI_MODEL  # non-empty default

    def test_key_2_exported(self):
        from app.config import GEMINI_API_KEY_2, GEMINI_MODEL_2
        assert isinstance(GEMINI_API_KEY_2, str)
        assert isinstance(GEMINI_MODEL_2, str)
        assert GEMINI_MODEL_2

    def test_key_3_exported(self):
        from app.config import GEMINI_API_KEY_3, GEMINI_MODEL_3
        assert isinstance(GEMINI_API_KEY_3, str)
        assert isinstance(GEMINI_MODEL_3, str)
        assert GEMINI_MODEL_3

    def test_key_4_exported(self):
        from app.config import GEMINI_API_KEY_4, GEMINI_MODEL_4
        assert isinstance(GEMINI_API_KEY_4, str)
        assert isinstance(GEMINI_MODEL_4, str)
        assert GEMINI_MODEL_4

    def test_key_5_exported(self):
        from app.config import GEMINI_API_KEY_5, GEMINI_MODEL_5
        assert isinstance(GEMINI_API_KEY_5, str)
        assert isinstance(GEMINI_MODEL_5, str)
        assert GEMINI_MODEL_5


# ─────────────────────────────────────────────────────────────────────────────
# 2. helpers.py module-level client attributes
# ─────────────────────────────────────────────────────────────────────────────
class TestClientAttributes:
    """client1..5 module attributes must exist in app.helpers."""

    def test_client_attr_exists(self):
        import app.helpers as h
        assert hasattr(h, "client")   # client1

    def test_client2_attr_exists(self):
        import app.helpers as h
        assert hasattr(h, "client2")

    def test_client3_attr_exists(self):
        import app.helpers as h
        assert hasattr(h, "client3")

    def test_client4_attr_exists(self):
        import app.helpers as h
        assert hasattr(h, "client4")

    def test_client5_attr_exists(self):
        import app.helpers as h
        assert hasattr(h, "client5")


# ─────────────────────────────────────────────────────────────────────────────
# 3 & 4.  call_gemini_ocr() dispatch logic
# ─────────────────────────────────────────────────────────────────────────────
def _make_mock_client(name: str) -> MagicMock:
    """Build a minimal genai.Client mock that records which model was called."""
    mc = MagicMock(name=name)
    mc.models.generate_content.return_value = MagicMock(parsed=MagicMock(
        markdown="TEXT",
        machine_name="MACHINE",
        ma_mmtb="B12345678",
        model="ModelX",
        xuong="Xưởng A",
        vi_tri="Khu A",
    ))
    return mc


def _dummy_image() -> Image.Image:
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    return img


class TestCallGeminiOcrDispatch:
    """call_gemini_ocr routes model_index 1-5 to the matching client & model name."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Inject mock clients and model names into app.helpers before each test."""
        self.mc1 = _make_mock_client("client1")
        self.mc2 = _make_mock_client("client2")
        self.mc3 = _make_mock_client("client3")
        self.mc4 = _make_mock_client("client4")
        self.mc5 = _make_mock_client("client5")

        self.model_names = {
            "GEMINI_MODEL":   "gemini-flash-1",
            "GEMINI_MODEL_2": "gemini-flash-2",
            "GEMINI_MODEL_3": "gemini-flash-3",
            "GEMINI_MODEL_4": "gemini-flash-4",
            "GEMINI_MODEL_5": "gemini-flash-5",
        }

        patches = [
            patch("app.helpers.client",  self.mc1),
            patch("app.helpers.client2", self.mc2),
            patch("app.helpers.client3", self.mc3),
            patch("app.helpers.client4", self.mc4),
            patch("app.helpers.client5", self.mc5),
            patch("app.helpers.GEMINI_MODEL",   self.model_names["GEMINI_MODEL"]),
            patch("app.helpers.GEMINI_MODEL_2", self.model_names["GEMINI_MODEL_2"]),
            patch("app.helpers.GEMINI_MODEL_3", self.model_names["GEMINI_MODEL_3"]),
            patch("app.helpers.GEMINI_MODEL_4", self.model_names["GEMINI_MODEL_4"]),
            patch("app.helpers.GEMINI_MODEL_5", self.model_names["GEMINI_MODEL_5"]),
        ]
        self.started = [p.start() for p in patches]
        yield
        for p in patches:
            p.stop()

    # ── index 1 (default) ────────────────────────────────────────────────────

    def test_index_1_uses_client1(self):
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image(), model_index=1)
        self.mc1.models.generate_content.assert_called_once()
        call_args = self.mc1.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-flash-1"
        # Other clients must NOT be touched
        self.mc2.models.generate_content.assert_not_called()
        self.mc3.models.generate_content.assert_not_called()
        self.mc4.models.generate_content.assert_not_called()
        self.mc5.models.generate_content.assert_not_called()

    def test_index_default_uses_client1(self):
        """Calling without model_index should default to client1."""
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image())
        self.mc1.models.generate_content.assert_called_once()

    # ── index 2 ──────────────────────────────────────────────────────────────

    def test_index_2_uses_client2(self):
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image(), model_index=2)
        self.mc2.models.generate_content.assert_called_once()
        call_args = self.mc2.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-flash-2"
        self.mc1.models.generate_content.assert_not_called()
        self.mc3.models.generate_content.assert_not_called()

    # ── index 3 ──────────────────────────────────────────────────────────────

    def test_index_3_uses_client3(self):
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image(), model_index=3)
        self.mc3.models.generate_content.assert_called_once()
        call_args = self.mc3.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-flash-3"
        self.mc1.models.generate_content.assert_not_called()
        self.mc2.models.generate_content.assert_not_called()

    # ── index 4 ──────────────────────────────────────────────────────────────

    def test_index_4_uses_client4(self):
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image(), model_index=4)
        self.mc4.models.generate_content.assert_called_once()
        call_args = self.mc4.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-flash-4"
        self.mc1.models.generate_content.assert_not_called()
        self.mc2.models.generate_content.assert_not_called()
        self.mc3.models.generate_content.assert_not_called()
        self.mc5.models.generate_content.assert_not_called()

    # ── index 5 ──────────────────────────────────────────────────────────────

    def test_index_5_uses_client5(self):
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image(), model_index=5)
        self.mc5.models.generate_content.assert_called_once()
        call_args = self.mc5.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-flash-5"
        self.mc1.models.generate_content.assert_not_called()
        self.mc2.models.generate_content.assert_not_called()
        self.mc3.models.generate_content.assert_not_called()
        self.mc4.models.generate_content.assert_not_called()

    # ── unrecognised index falls back to client1 ──────────────────────────────

    def test_index_0_falls_back_to_client1(self):
        """Index 0 is not valid; the else-branch keeps selected_client=client1."""
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image(), model_index=0)
        self.mc1.models.generate_content.assert_called_once()

    def test_index_6_falls_back_to_client1(self):
        """Index 6 is out of range; falls back to client1."""
        from app.helpers import call_gemini_ocr
        call_gemini_ocr(_dummy_image(), model_index=6)
        self.mc1.models.generate_content.assert_called_once()

    # ── uninitialized client raises RuntimeError ──────────────────────────────

    def test_uninitialised_client_raises(self):
        """If the selected client is None (key missing), a RuntimeError is raised."""
        from app.helpers import call_gemini_ocr
        with patch("app.helpers.client4", None):
            with pytest.raises(RuntimeError, match="Gemini Client 4 is not initialized"):
                call_gemini_ocr(_dummy_image(), model_index=4)

    def test_uninitialised_client5_raises(self):
        from app.helpers import call_gemini_ocr
        with patch("app.helpers.client5", None):
            with pytest.raises(RuntimeError, match="Gemini Client 5 is not initialized"):
                call_gemini_ocr(_dummy_image(), model_index=5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. check_gemini_connectivity() enumerates only live clients
# ─────────────────────────────────────────────────────────────────────────────
class TestCheckGeminiConnectivity:

    @pytest.mark.asyncio
    async def test_returns_true_with_any_live_client(self):
        from app.helpers import check_gemini_connectivity

        mc = _make_mock_client("c1_conn")
        # Simulate models.list returning one item
        mc.models.list.return_value = iter([MagicMock()])

        with (
            patch("app.helpers.client",  mc),
            patch("app.helpers.client2", None),
            patch("app.helpers.client3", None),
            patch("app.helpers.client4", None),
            patch("app.helpers.client5", None),
        ):
            result = await check_gemini_connectivity()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_all_clients_none(self):
        from app.helpers import check_gemini_connectivity

        with (
            patch("app.helpers.client",  None),
            patch("app.helpers.client2", None),
            patch("app.helpers.client3", None),
            patch("app.helpers.client4", None),
            patch("app.helpers.client5", None),
        ):
            result = await check_gemini_connectivity()

        assert result is False

    @pytest.mark.asyncio
    async def test_only_clients_4_and_5_live(self):
        """Connectivity check works even when only clients 4 & 5 are configured."""
        from app.helpers import check_gemini_connectivity

        mc4 = _make_mock_client("c4_conn")
        mc5 = _make_mock_client("c5_conn")
        mc4.models.list.return_value = iter([MagicMock()])
        mc5.models.list.return_value = iter([MagicMock()])

        with (
            patch("app.helpers.client",  None),
            patch("app.helpers.client2", None),
            patch("app.helpers.client3", None),
            patch("app.helpers.client4", mc4),
            patch("app.helpers.client5", mc5),
        ):
            result = await check_gemini_connectivity()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        """If models.list() throws, check returns False gracefully."""
        from app.helpers import check_gemini_connectivity

        mc = _make_mock_client("c1_exc")
        mc.models.list.side_effect = Exception("network error")

        with (
            patch("app.helpers.client",  mc),
            patch("app.helpers.client2", None),
            patch("app.helpers.client3", None),
            patch("app.helpers.client4", None),
            patch("app.helpers.client5", None),
        ):
            result = await check_gemini_connectivity()

        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. validate_upload_gemini() – file validation
# ─────────────────────────────────────────────────────────────────────────────
class TestValidateUploadGemini:

    def test_valid_jpeg_passes(self):
        from app.helpers import validate_upload_gemini
        # Should not raise
        validate_upload_gemini("label.jpg", "image/jpeg", 1024)

    def test_valid_png_passes(self):
        from app.helpers import validate_upload_gemini
        validate_upload_gemini("label.png", "image/png", 2048)

    def test_valid_webp_passes(self):
        from app.helpers import validate_upload_gemini
        validate_upload_gemini("label.webp", "image/webp", 512)

    def test_valid_gif_passes(self):
        from app.helpers import validate_upload_gemini
        validate_upload_gemini("label.gif", "image/gif", 256)

    def test_invalid_extension_raises(self):
        from app.helpers import validate_upload_gemini
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_gemini("label.bmp", "image/bmp", 512)
        assert exc_info.value.status_code == 400

    def test_invalid_mime_raises(self):
        from app.helpers import validate_upload_gemini
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_gemini("label.jpg", "application/octet-stream", 512)
        assert exc_info.value.status_code == 400

    def test_file_too_large_raises(self):
        from app.helpers import validate_upload_gemini
        from fastapi import HTTPException
        over_limit = 5 * 1024 * 1024 + 1  # 5MB + 1 byte
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_gemini("label.jpg", "image/jpeg", over_limit)
        assert exc_info.value.status_code == 400

    def test_empty_filename_raises(self):
        from app.helpers import validate_upload_gemini
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_gemini("", "image/jpeg", 512)
        assert exc_info.value.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 7. validate_upload_paddle() – PaddleOCR validation
# ─────────────────────────────────────────────────────────────────────────────
class TestValidateUploadPaddle:

    def test_valid_jpeg_passes(self):
        from app.helpers import validate_upload_paddle
        validate_upload_paddle("doc.jpg", "image/jpeg", 1024)

    def test_valid_png_passes(self):
        from app.helpers import validate_upload_paddle
        validate_upload_paddle("doc.png", "image/png", 2048)

    def test_valid_pdf_passes(self):
        from app.helpers import validate_upload_paddle
        validate_upload_paddle("doc.pdf", "application/pdf", 4096)

    def test_invalid_extension_raises(self):
        from app.helpers import validate_upload_paddle
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_paddle("doc.docx", "application/msword", 1024)
        assert exc_info.value.status_code == 400

    def test_invalid_mime_raises(self):
        from app.helpers import validate_upload_paddle
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_paddle("doc.jpg", "text/plain", 1024)
        assert exc_info.value.status_code == 400

    def test_file_too_large_raises(self):
        from app.helpers import validate_upload_paddle
        from fastapi import HTTPException
        over_limit = 20 * 1024 * 1024 + 1  # 20MB + 1 byte
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_paddle("doc.pdf", "application/pdf", over_limit)
        assert exc_info.value.status_code == 400

    def test_empty_filename_raises(self):
        from app.helpers import validate_upload_paddle
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_upload_paddle("", "image/png", 1024)
        assert exc_info.value.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 8. Smoke test: all 5 clients can be called sequentially via call_gemini_ocr
# ─────────────────────────────────────────────────────────────────────────────
class TestAllFiveClientsSequential:
    """Smoke test – cycle through all 5 indices and assert each correct client fires."""

    def test_all_five_indices_call_correct_clients(self):
        from app.helpers import call_gemini_ocr

        clients = {i: _make_mock_client(f"client{i}") for i in range(1, 6)}
        model_names = {i: f"gemini-model-{i}" for i in range(1, 6)}

        patches = [
            patch("app.helpers.client",  clients[1]),
            patch("app.helpers.client2", clients[2]),
            patch("app.helpers.client3", clients[3]),
            patch("app.helpers.client4", clients[4]),
            patch("app.helpers.client5", clients[5]),
            patch("app.helpers.GEMINI_MODEL",   model_names[1]),
            patch("app.helpers.GEMINI_MODEL_2", model_names[2]),
            patch("app.helpers.GEMINI_MODEL_3", model_names[3]),
            patch("app.helpers.GEMINI_MODEL_4", model_names[4]),
            patch("app.helpers.GEMINI_MODEL_5", model_names[5]),
        ]

        for p in patches:
            p.start()

        try:
            img = _dummy_image()
            for idx in range(1, 6):
                call_gemini_ocr(img, model_index=idx)

            for idx in range(1, 6):
                mc = clients[idx]
                mc.models.generate_content.assert_called_once()
                call_kwargs = mc.models.generate_content.call_args.kwargs
                assert call_kwargs["model"] == model_names[idx], (
                    f"Client {idx} was called with wrong model name: {call_kwargs['model']}"
                )
        finally:
            for p in patches:
                p.stop()
