"""
tests/test_paddle_transformer.py
─────────────────────────────────
Unit tests for app.paddle_transformer.

Run:  pytest tests/ -v
      pytest tests/ -v -s   (shows debug print output)

No external dependencies required — all fixtures are inline dicts that
simulate PaddleOCR layoutParsingResult entries.
"""

import pytest
from app.paddle_transformer import (
    normalize_key,
    extract_text_blocks,
    parse_kv_from_blocks,
    _enhanced_markdown_kv,
    _recover_missing_fields,
    transform_paddleocr_result,
    extract_machine_name_from_paddle_blocks,
    _is_branding_line,
    _is_metadata_kv_line,
    _machine_name_score,
    _extract_machine_name_from_lines,
    _Block,
    ParsedResult,
    SRC_RAW_BLOCKS,
    SRC_ENHANCED_MARKDOWN,
    TRACKED_FIELDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_res(markdown: str = "", blocks=None, ocr_results=None) -> dict:
    """Build a minimal layoutParsingResult dict."""
    res: dict = {"markdown": {"text": markdown}}
    if blocks is not None:
        res["blocks"] = blocks
    if ocr_results is not None:
        res["ocr_results"] = ocr_results
    return res


def make_block(text: str, x1=0, y1=0, x2=100, y2=20) -> dict:
    return {"content": text, "bbox": [x1, y1, x2, y2]}


def blk(text: str, x1=0.0, y1=0.0, x2=100.0, y2=20.0) -> _Block:
    return _Block(text=text, bbox=[x1, y1, x2, y2])


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — normalize_key
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeKey:

    @pytest.mark.parametrize("raw, expected", [
        # Mã MMTB variants
        ("Mã MMTB",       "Mã MMTB"),
        ("ma mmtb",       "Mã MMTB"),
        ("MA MMTB",       "Mã MMTB"),
        ("ma may",        "Mã MMTB"),
        ("mammtb",        "Mã MMTB"),
        # Model variants
        ("Model",         "Model"),
        ("model",         "Model"),
        ("MODEL",         "Model"),
        ("mo hinh",       "Model"),
        # Xưởng variants
        ("Xưởng",         "Xưởng"),
        ("xuong",         "Xưởng"),
        ("Xuong",         "Xưởng"),
        ("nha may",       "Xưởng"),
        # Vị trí variants
        ("Vị trí",        "Vị trí"),
        ("vi tri",        "Vị trí"),
        ("vitri",         "Vị trí"),
        ("tri",           "Vị trí"),
        ("(i trí",        "Vị trí"),
        ("(i tri",        "Vị trí"),
        ("i tri",         "Vị trí"),
        # Should return None
        ("MÁY HÀN CO2",   None),
        ("TTC-500T",      None),
        ("B22401469",     None),
        ("",              None),
    ])
    def test_normalize_key(self, raw, expected):
        result = normalize_key(raw)
        assert result == expected, f"normalize_key({raw!r}) = {result!r}, expected {expected!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — extract_text_blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTextBlocks:

    def test_blocks_path(self):
        res = make_res(blocks=[
            make_block("Mã MMTB : B001", 0, 0, 200, 20),
            make_block("Model : ABC-100", 0, 25, 200, 45),
        ])
        blocks = extract_text_blocks(res)
        assert len(blocks) == 2
        texts = [b.text for b in blocks]
        assert "Mã MMTB : B001" in texts
        assert "Model : ABC-100" in texts

    def test_sub_blocks_path(self):
        res = make_res(blocks=[{
            "content": "",
            "bbox": [0, 0, 300, 60],
            "sub_blocks": [
                {"content": "Xưởng : AH6",    "bbox": [0, 0, 150, 20]},
                {"content": "Vị trí : C12",    "bbox": [0, 25, 150, 45]},
            ]
        }])
        blocks = extract_text_blocks(res)
        texts = [b.text for b in blocks]
        assert "Xưởng : AH6" in texts
        assert "Vị trí : C12" in texts

    def test_ocr_results_path(self):
        res = make_res(ocr_results=[
            {"text": "Model : XYZ",  "bbox": [0, 0, 150, 20], "confidence": 0.98},
            {"text": "Xưởng : PX01", "bbox": [0, 25, 150, 45]},
        ])
        blocks = extract_text_blocks(res)
        texts = [b.text for b in blocks]
        assert "Model : XYZ" in texts
        assert "Xưởng : PX01" in texts

    def test_rec_texts_path(self):
        res = make_res()
        res["rec_texts"] = ["Mã MMTB : X001", "Vị trí : D5"]
        res["dt_boxes"]  = [[0, 0, 200, 20], [0, 25, 200, 45]]
        blocks = extract_text_blocks(res)
        texts = [b.text for b in blocks]
        assert "Mã MMTB : X001" in texts

    def test_polygon_bbox_normalised(self):
        res = make_res(ocr_results=[{
            "text": "Model : ABC",
            "bbox": [[10, 5], [200, 5], [200, 25], [10, 25]],
        }])
        blocks = extract_text_blocks(res)
        assert len(blocks) == 1
        assert blocks[0].bbox == [10.0, 5.0, 200.0, 25.0]

    def test_empty_res_returns_empty_list(self):
        assert extract_text_blocks({}) == []
        assert extract_text_blocks(make_res()) == []

    def test_deduplication(self):
        res = make_res(blocks=[
            make_block("Model : ABC"),
            make_block("Model : ABC"),   # duplicate
        ])
        blocks = extract_text_blocks(res)
        assert len(blocks) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Spatial block-level KV parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestParseKvFromBlocks:

    def test_single_block_standard_kv(self):
        blocks = [blk("Mã MMTB : B001", 0, 0, 200, 20)]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Mã MMTB") == "B001"

    def test_multi_field_same_column(self):
        """Standard vertical layout — one KV per line."""
        blocks = [
            blk("MÁY HÀN CO2",        0, 0,  250, 20),
            blk("Mã MMTB : B22401469", 0, 30, 250, 50),
            blk("Model : TTC-500T",    0, 60, 250, 80),
            blk("Xưởng : AH6",         0, 90, 250, 110),
            blk("Vị trí : RHT1",       0, 120, 250, 140),
        ]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv["Mã MMTB"] == "B22401469"
        assert kv["Model"]    == "TTC-500T"
        assert kv["Xưởng"]    == "AH6"
        assert kv["Vị trí"]   == "RHT1"

    def test_two_column_layout_no_colon(self):
        """Key and value as separate side-by-side blocks without colon."""
        blocks = [
            blk("Model",   0,  0, 80,  20),   # left  = key
            blk("ABC-123", 90, 0, 200, 20),   # right = value
        ]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Model") == "ABC-123"

    def test_multi_line_kv(self):
        """Key on line N, value on line N+1."""
        blocks = [
            blk("Model",     0, 0,  100, 20),
            blk("HAAS ST-10",0, 30, 100, 50),
        ]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Model") == "HAAS ST-10"

    def test_machine_name_extracted(self):
        blocks = [
            blk("MÁY TIỆN CNC",    0, 0,   250, 20),
            blk("Mã MMTB : A001",  0, 30,  250, 50),
        ]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("machine_name") == "MÁY TIỆN CNC"
        assert kv.get("Mã MMTB") == "A001"

    def test_corrupted_key_xuong(self):
        blocks = [blk("xuong : PX5", 0, 0, 150, 20)]
        kv, _ = parse_kv_from_blocks(blocks)
        assert "Xưởng" in kv
        assert kv["Xưởng"] == "PX5"

    def test_corrupted_key_tri(self):
        blocks = [blk("tri : B5", 0, 0, 100, 20)]
        kv, _ = parse_kv_from_blocks(blocks)
        assert "Vị trí" in kv

    def test_bold_wrapped_keys(self):
        blocks = [
            blk("**Mã MMTB** : F010", 0, 0, 200, 20),
            blk("**Model** : DR-13",  0, 30, 200, 50),
        ]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Mã MMTB") == "F010"
        assert kv.get("Model")   == "DR-13"

    def test_no_space_after_colon(self):
        blocks = [blk("Mã MMTB:H012", 0, 0, 150, 20)]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Mã MMTB") == "H012"

    def test_table_layout_blocks(self):
        blocks = [
            blk("| Mã MMTB | C003 |",    0, 0,  250, 20),
            blk("| Model | VMC-850 |",   0, 30, 250, 50),
            blk("| Xưởng | Phay |",       0, 60, 250, 80),
            blk("| Vị trí | D5 |",        0, 90, 250, 110),
        ]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Mã MMTB") == "C003"
        assert kv.get("Model")   == "VMC-850"
        assert kv.get("Xưởng")   == "Phay"
        assert kv.get("Vị trí")  == "D5"

    def test_alt_separator_dot(self):
        blocks = [blk("Mã MMTB . E009", 0, 0, 200, 20)]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Mã MMTB") == "E009"

    def test_alt_separator_dash(self):
        blocks = [blk("Model - GR-200", 0, 0, 200, 20)]
        kv, _ = parse_kv_from_blocks(blocks)
        assert kv.get("Model") == "GR-200"


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — Enhanced markdown parser
# ─────────────────────────────────────────────────────────────────────────────

class TestEnhancedMarkdownKv:

    def test_standard_kv(self):
        md = "MÁY HÀN CO2\n\nMã MMTB : B001\n\nModel : TTC-500T\n"
        kv, _ = _enhanced_markdown_kv(md)
        assert kv["machine_name"] == "MÁY HÀN CO2"
        assert kv["Mã MMTB"]     == "B001"
        assert kv["Model"]        == "TTC-500T"

    def test_table_rows_parsed(self):
        md = "MÁY PHAY\n\n| Mã MMTB | C003 |\n| Model | VMC-850 |\n| Xưởng | Phay |\n| Vị trí | D5 |\n"
        kv, _ = _enhanced_markdown_kv(md)
        assert kv.get("Mã MMTB") == "C003"
        assert kv.get("Model")   == "VMC-850"
        assert kv.get("Xưởng")   == "Phay"
        assert kv.get("Vị trí")  == "D5"

    def test_orphan_value_line_merged(self):
        md = "MÁY TIỆN\n\nMã MMTB : A001\n\nModel\n\n: HAAS ST-10\n\nXưởng : CNC\n\nVị trí\n\n: C12\n"
        kv, _ = _enhanced_markdown_kv(md)
        assert kv.get("Model")   == "HAAS ST-10"
        assert kv.get("Vị trí")  == "C12"

    def test_html_image_block_stripped(self):
        md = (
            "MÁY HÀN CO2\n"
            "Mã MMTB : B001\n"
            '<div style="text-align:center"><img src="logo.jpg"/></div>\n'
            "Model : TTC-500T\n"
        )
        kv, _ = _enhanced_markdown_kv(md)
        assert kv.get("Mã MMTB") == "B001"
        assert kv.get("Model")   == "TTC-500T"

    def test_bold_wrappers(self):
        md = "**Mã MMTB** : F010\n**Model** : DR-13\n"
        kv, _ = _enhanced_markdown_kv(md)
        assert kv.get("Mã MMTB") == "F010"
        assert kv.get("Model")   == "DR-13"

    def test_alt_separator_dot(self):
        md = "Mã MMTB . E009\nXưởng : MÀI\n"
        kv, _ = _enhanced_markdown_kv(md)
        assert kv.get("Mã MMTB") == "E009"

    def test_no_space_after_colon(self):
        md = "Mã MMTB:H012\nModel:PB-50\n"
        kv, _ = _enhanced_markdown_kv(md)
        assert kv.get("Mã MMTB") == "H012"
        assert kv.get("Model")   == "PB-50"

    def test_diacritic_corrupted_key_xuong(self):
        md = "Xương : AH6\n"
        kv, _ = _enhanced_markdown_kv(md)
        assert "Xưởng" in kv
        assert kv["Xưởng"] == "AH6"

    def test_tri_key_mapped_to_vi_tri(self):
        md = "trí : RHT1\n"
        kv, _ = _enhanced_markdown_kv(md)
        # Either stored as "trí" (recovered later) or directly as "Vị trí"
        assert kv.get("Vị trí") == "RHT1" or kv.get("trí") == "RHT1"


# ─────────────────────────────────────────────────────────────────────────────
# Group 5 — Missing field recovery
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoverMissingFields:

    def test_tri_promoted_to_vi_tri(self):
        kv     = {"trí": "RHT1", "machine_name": "MÁY HÀN"}
        blocks = []
        kv, _  = _recover_missing_fields(kv, blocks, "")
        assert kv.get("Vị trí") == "RHT1"
        assert "trí" not in kv

    def test_i_tri_promoted(self):
        kv     = {"(i trí": "B5"}
        blocks = []
        kv, _  = _recover_missing_fields(kv, blocks, "")
        assert kv.get("Vị trí") == "B5"

    def test_block_recovery_fills_missing_field(self):
        kv = {"machine_name": "MÁY TIỆN", "Model": "ABC"}
        blocks = [
            blk("Mã MMTB : X999", 0, 0, 200, 20),
            blk("Xưởng : PX3",    0, 30, 200, 50),
        ]
        kv, _ = _recover_missing_fields(kv, blocks, "")
        assert kv.get("Mã MMTB") == "X999"
        assert kv.get("Xưởng")   == "PX3"

    def test_markdown_recovery_fills_missing_field(self):
        kv       = {"machine_name": "MÁY"}
        md       = "Mã MMTB : Z100\nModel : ZZZ\n"
        kv, _    = _recover_missing_fields(kv, [], md)
        assert kv.get("Mã MMTB") == "Z100"


# ─────────────────────────────────────────────────────────────────────────────
# Group 6 — transform_paddleocr_result (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformPaddleocrResult:

    # ── 6.1 Normal label via raw blocks ─────────────────────────────────────

    def test_normal_label_raw_blocks(self):
        res = make_res(
            markdown="MÁY HÀN CO2\nMã MMTB : B22401469\nModel : TTC-500T\nXưởng : AH6\nVị trí : RHT1\n",
            blocks=[
                make_block("MÁY HÀN CO2",        0, 0,   250, 20),
                make_block("Mã MMTB : B22401469", 0, 30,  250, 50),
                make_block("Model : TTC-500T",    0, 60,  250, 80),
                make_block("Xưởng : AH6",          0, 90,  250, 110),
                make_block("Vị trí : RHT1",        0, 120, 250, 140),
            ]
        )
        result = transform_paddleocr_result(res)
        assert isinstance(result, ParsedResult)
        assert result.source == SRC_RAW_BLOCKS
        assert result.key_value["Mã MMTB"] == "B22401469"
        assert result.key_value["Model"]    == "TTC-500T"
        assert result.key_value["Xưởng"]    == "AH6"
        assert result.key_value["Vị trí"]   == "RHT1"

    # ── 6.2 Fallback to enhanced markdown when no blocks ────────────────────

    def test_enhanced_markdown_fallback(self):
        res = make_res(
            markdown="MÁY KHOAN\n\nMã MMTB : F010\n\nModel : DR-13\n\nXưởng : KHOAN\n\nVị trí : G7\n"
        )
        result = transform_paddleocr_result(res)
        assert result.source == SRC_ENHANCED_MARKDOWN
        assert result.key_value["Mã MMTB"] == "F010"
        assert result.key_value["Model"]    == "DR-13"

    # ── 6.3 Real sample: doc_0.md (split "Vị trí") ──────────────────────────

    def test_real_sample_doc0(self):
        """Reproduces the output/doc_0.md case verbatim."""
        md = (
            "MÁY HÀN CO2 TÂN THÀNH - TTC-500T\n"
            "\n"
            "Mã MMTB : B22401469\n"
            "\n"
            "Model : TTC-500T\n"
            "\n"
            "Xương : AH6\n"
            "\n"
            '<div style="text-align: center;"><img src="imgs/img_0.jpg" width="7%" /></div>\n'
            "\n"
            "trí : RHT1\n"
        )
        res    = make_res(markdown=md)
        result = transform_paddleocr_result(res)
        kv     = result.key_value
        assert kv["Mã MMTB"] == "B22401469"
        assert kv["Model"]    == "TTC-500T"
        assert kv["Xưởng"]    == "AH6"     # diacritic correction
        assert kv["Vị trí"]   == "RHT1"    # recovered from "trí"

    # ── 6.4 Table layout (all fields in pipe rows) ───────────────────────────

    def test_table_layout(self):
        md = (
            "MÁY PHAY CNC\n"
            "| Mã MMTB | C003 |\n"
            "| Model | VMC-850 |\n"
            "| Xưởng | Phay |\n"
            "| Vị trí | D5 |\n"
        )
        res    = make_res(markdown=md)
        result = transform_paddleocr_result(res)
        kv     = result.key_value
        assert kv.get("Mã MMTB") == "C003"
        assert kv.get("Model")   == "VMC-850"
        assert kv.get("Xưởng")   == "Phay"
        assert kv.get("Vị trí")  == "D5"

    # ── 6.5 OCR-corrupted keys ───────────────────────────────────────────────

    def test_corrupted_key_i_tri(self):
        md = "MÁY\n\nMã MMTB : G011\n\n(i trí : H8\n"
        result = transform_paddleocr_result(make_res(markdown=md))
        assert result.key_value.get("Vị trí") == "H8"

    def test_corrupted_key_xuong_no_diacritics(self):
        md = "MÁY\n\nxuong : MÀI\n\nModel : M1\n"
        result = transform_paddleocr_result(make_res(markdown=md))
        assert result.key_value.get("Xưởng") == "MÀI"

    def test_corrupted_key_ma_may(self):
        md = "MÁY\n\nma may : Z001\n"
        result = transform_paddleocr_result(make_res(markdown=md))
        assert result.key_value.get("Mã MMTB") == "Z001"

    # ── 6.6 Multi-line KV ────────────────────────────────────────────────────

    def test_multiline_kv_orphan_colon(self):
        md = "MÁY TIỆN\n\nModel\n\n: HAAS ST-10\n\nXưởng : CNC\n\nVị trí\n\n: C12\n"
        result = transform_paddleocr_result(make_res(markdown=md))
        assert result.key_value.get("Model")  == "HAAS ST-10"
        assert result.key_value.get("Vị trí") == "C12"

    # ── 6.7 Alt separators ───────────────────────────────────────────────────

    def test_alt_separator_dot(self):
        md = "MÁY MÀI\n\nMã MMTB . E009\n\nModel . GR-200\n\nXưởng : MÀI\n"
        result = transform_paddleocr_result(make_res(markdown=md))
        assert result.key_value.get("Mã MMTB") == "E009"
        assert result.key_value.get("Model")   == "GR-200"

    # ── 6.8 markdown field is always preserved unchanged ─────────────────────

    def test_markdown_field_preserved(self):
        original_md = "MÁY HÀN\n\nMã MMTB : B001\n"
        res         = make_res(markdown=original_md)
        result      = transform_paddleocr_result(res)
        assert result.markdown == original_md

    # ── 6.9 debug mode returns debug_info ────────────────────────────────────

    def test_debug_mode(self):
        res    = make_res(markdown="Mã MMTB : B001\nModel : M1\n")
        result = transform_paddleocr_result(res, debug=True)
        assert result.debug_info is not None
        assert "stage" in result.debug_info

    def test_no_debug_by_default(self):
        res    = make_res(markdown="Mã MMTB : B001\n")
        result = transform_paddleocr_result(res)
        assert result.debug_info is None

    # ── 6.10 Empty / edge cases ───────────────────────────────────────────────

    def test_empty_res(self):
        result = transform_paddleocr_result({})
        assert isinstance(result, ParsedResult)
        assert result.key_value == {} or isinstance(result.key_value, dict)

    def test_empty_markdown(self):
        result = transform_paddleocr_result(make_res(markdown=""))
        assert isinstance(result.key_value, dict)

    # ── 6.11 Vertical label (fields appear out of expected order) ────────────

    def test_vertical_label_with_blocks(self):
        """All text blocks stacked vertically, each on its own line."""
        res = make_res(blocks=[
            make_block("Vị trí : A1",      0, 0,  200, 20),
            make_block("Xưởng : PX2",       0, 30, 200, 50),
            make_block("Mã MMTB : V999",   0, 60, 200, 80),
            make_block("Model : VERT-100",  0, 90, 200, 110),
            make_block("MÁY VERTICAL",      0, 120,200, 140),
        ])
        result = transform_paddleocr_result(res)
        kv = result.key_value
        assert kv.get("Vị trí")  == "A1"
        assert kv.get("Xưởng")   == "PX2"
        assert kv.get("Mã MMTB") == "V999"
        assert kv.get("Model")   == "VERT-100"

    # ── 6.12 Combined: blocks present + missing field recovered via markdown ──

    def test_blocks_plus_markdown_recovery(self):
        """
        Blocks have 3 fields; markdown has the 4th.
        Recovery stage should fill the gap.
        """
        res = make_res(
            markdown="MÁY\nMã MMTB : M001\nModel : MX\nXưởng : PX\nVị trí : V9\n",
            blocks=[
                make_block("Mã MMTB : M001", 0, 0,  200, 20),
                make_block("Model : MX",      0, 30, 200, 50),
                make_block("Xưởng : PX",      0, 60, 200, 80),
                # "Vị trí" intentionally missing from blocks
            ]
        )
        result = transform_paddleocr_result(res)
        # Should be recovered from markdown in Stage 4
        assert result.key_value.get("Vị trí") == "V9"


# ─────────────────────────────────────────────────────────────────────────────
# Group 7 — Branding Detection & Machine Name Anchor Walk (and failed cases)
# ─────────────────────────────────────────────────────────────────────────────

class TestBrandingAndMachineName:

    @pytest.mark.parametrize("line, expected", [
        # Branding / log text should be True
        ("DAIDUNG", True),
        ("DAI DUNG", True),
        ("DAI OUNG", True),
        ("## DAI DUNG", True),
        ("VIET HUNG", True),
        ("VIET NAM", True),
        ("## HÀN:", True),
        ("", True),
        # Normal machine names should be False
        ("MÁY HÀN ĐIỆN TỬ/ELECTRICAL WELDING MACHINE", False),
        ("Máy ghép dảm hợp 3000x4000", False),
        ("MÁY GOUGING JÁSIC", False),
        ("MÁY HÀN GOUGING MZ-1000 -GOUGING WELDING MACHINE MZ-1000", False),
        ("Máy hàn Ehave, CM500 DC", False),
        ("(MMA/MAG/CO2) - Megmeet", False),
    ])
    def test_is_branding_line(self, line, expected):
        assert _is_branding_line(line) == expected

    @pytest.mark.parametrize("line, expected", [
        ("- Mã MMTB/Code MMTB: B22400711", True),
        ("- Model: ZHS(L)-1000", True),
        ("Mã máy : B21600137", True),
        ("Model:", True),
        ("Vị trí : AH2", True),
        ("Mã MMTB: B22300090", True),
        ("Model: MZ1000", True),
        ("Xuất xứ : Trung Quốc", False),
        ("V: Trí : B16 (AH2)", True),  # normalized key starts with 'vi tri'
        ("Vị trí : TỔ HÀN LINE 1 (A20-21)", True),
        ("Xương : AH2", True),
    ])
    def test_is_metadata_kv_line(self, line, expected):
        assert _is_metadata_kv_line(line) == expected

    def test_machine_name_score(self):
        assert _machine_name_score("MÁY HÀN") > _machine_name_score("HÀN")
        assert _machine_name_score("Máy hàn Ehave, CM500 DC") > _machine_name_score("(MMA/MAG/CO2) - Megmeet")

    def test_failed_case_1(self):
        # Case 1 from raw.json
        md = (
            "DAIDUNG\n"
            "MÁY HÀN ĐIỆN TỬ/ELECTRICAL WELDING MACHINE\n"
            "- Mã MMTB/Code MMTB: B22400711\n"
            "- Model: ZHS(L)-1000\n"
            "- Xuất xứ/ Origin: Trung Quốc/China"
        )
        res = make_res(markdown=md)
        result = transform_paddleocr_result(res)
        assert result.key_value.get("machine_name") == "MÁY HÀN ĐIỆN TỬ/ELECTRICAL WELDING MACHINE"
        assert result.key_value.get("Mã MMTB") == "B22400711"
        assert result.key_value.get("Model") == "ZHS(L)-1000"

    def test_failed_case_2(self):
        # Case 2 from raw.json
        md = (
            "## DAI DUNG\n"
            "\n"
            "Máy ghép dảm hợp 3000x4000\n"
            "\n"
            "Mã máy : B21600137\n"
            "\n"
            "Model:\n"
            "\n"
            "Vị trí : AH2"
        )
        res = make_res(markdown=md)
        result = transform_paddleocr_result(res)
        assert result.key_value.get("machine_name") == "Máy ghép dảm hợp 3000x4000"
        assert result.key_value.get("Mã MMTB") == "B21600137"
        assert result.key_value.get("Vị trí") == "AH2"

    def test_failed_case_3(self):
        # Case 3 from raw.json
        md = (
            "## DAI OUNG\n"
            "\n"
            "MÁY GOUGING JÁSIC\n"
            "\n"
            "Mã MMTB: B22300090\n"
            "\n"
            "Model: MZ1000\n"
            "\n"
            "Xuất xứ : Trung Quốc\n"
            "\n"
            "V: Trí : B16 (AH2)\n"
            "\n"
            '<div style="text-align: center;"><img src="imgs/img_in_image_box_783_333_922_460.jpg" alt="Image" width="10%" /></div>\n'
        )
        res = make_res(markdown=md)
        result = transform_paddleocr_result(res)
        assert result.key_value.get("machine_name") == "MÁY GOUGING JÁSIC"
        assert result.key_value.get("Mã MMTB") == "B22300090"
        assert result.key_value.get("Model") == "MZ1000"
        assert result.key_value.get("Vị trí") == "B16 (AH2)"

    def test_failed_case_4(self):
        # Case 4 from raw.json
        md = (
            '<div style="text-align: center;"><img src="imgs/img_in_image_box_435_275_503_359.jpg" alt="Image" width="5%" /></div>\n'
            "\n"
            "\n"
            "## DAIDUNG\n"
            "\n"
            '<div style="text-align: center;"><img src="imgs/img_in_image_box_881_183_1118_426.jpg" alt="Image" width="18%" /></div>\n'
            "\n"
            "\n"
            "MÁY HÀN GOUGING MZ-1000 -GOUGING WELDING MACHINE MZ-1000\n"
            "\n"
            "- Model: MZ-1000\n"
            "\n"
            "-Mä MMTB/Code MMTB: B22300035\n"
            "\n"
            "- Xuất xử/ Origin: Trung Quốc"
        )
        res = make_res(markdown=md)
        result = transform_paddleocr_result(res)
        assert result.key_value.get("machine_name") == "MÁY HÀN GOUGING MZ-1000 -GOUGING WELDING MACHINE MZ-1000"
        assert result.key_value.get("Mã MMTB") == "B22300035"
        assert result.key_value.get("Model") == "MZ-1000"

    def test_failed_case_5(self):
        # Case 5 from raw.json
        md = (
            "## HÀN:\n"
            "\n"
            "19\n"
            "\n"
            "Máy hàn Ehave, CM500 DC\n"
            "\n"
            "(MMA/MAG/CO2) - Megmeet\n"
            "\n"
            "\n"
            "\n"
            "Mã MMTB :B22400848\n"
            "\n"
            "Model : CM-500A\n"
            "\n"
            "Xương : AH2\n"
            "\n"
            "Vị trí : TỔ HÀN LINE 1 (A20-21)\n"
        )
        res = make_res(markdown=md)
        result = transform_paddleocr_result(res)
        assert result.key_value.get("machine_name") == "Máy hàn Ehave, CM500 DC"
        assert result.key_value.get("Mã MMTB") == "B22400848"
        assert result.key_value.get("Model") == "CM-500A"
        assert result.key_value.get("Xưởng") == "AH2"
        assert result.key_value.get("Vị trí") == "TỔ HÀN LINE 1 (A20-21)"
