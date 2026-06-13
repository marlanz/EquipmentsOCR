from app.utils.validators import validate_mmtb

def test_validate_mmtb_valid():
    assert validate_mmtb("B22400711") is True
    assert validate_mmtb("B12345678") is True
    assert validate_mmtb("B00000001") is True
    # Test stripping whitespace
    assert validate_mmtb("  B22400711  ") is True

def test_validate_mmtb_invalid():
    assert validate_mmtb("22400711") is False
    assert validate_mmtb("b22400711") is False
    assert validate_mmtb("B2240071") is False
    assert validate_mmtb("B224007111") is False
    assert validate_mmtb("B22A00711") is False
    assert validate_mmtb("ABC12345") is False
    assert validate_mmtb("CHƯA CÓ MÃ") is False
    assert validate_mmtb("") is False
    assert validate_mmtb(None) is False
