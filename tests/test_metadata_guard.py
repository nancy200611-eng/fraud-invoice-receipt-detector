import struct
import zlib
import pytest
from src.forensics.jumbf_byte_parser import JUMBFByteParser
from src.forensics.approved_exif_registry import (
    ApprovedExifRegistry,
    levenshtein_distance,
    APPROVED
)
from src.forensics.metadata_guard import MetadataGuard
from src.forensics.c2pa_exif_fast_guard import C2PAExifFastGuard, FastPathResult

def create_minimal_jpeg(extra_segments: bytes = b"") -> bytes:
    """Helper to build a valid minimal JPEG byte buffer with optional extra segments."""
    # SOI (2 bytes) + extra segments + DQT + SOF0 + SOS + minimal scan + EOI
    soi = b"\xff\xd8"
    eoi = b"\xff\xd9"
    return soi + extra_segments + eoi

def create_minimal_png(chunks: list = None) -> bytes:
    """Helper to build a valid minimal PNG byte buffer with custom chunks."""
    png_sig = b"\x89PNG\r\n\x1a\n"
    # Minimal IHDR
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    custom_chunks = b""
    if chunks:
        for c_type, c_data, corrupt_crc in chunks:
            c_len = len(c_data)
            crc = zlib.crc32(c_type + c_data) & 0xFFFFFFFF
            if corrupt_crc:
                crc ^= 0xDEADBEEF
            custom_chunks += struct.pack(">I", c_len) + c_type + c_data + struct.pack(">I", crc)

    return png_sig + ihdr_chunk + custom_chunks + iend_chunk

# =========================================================================
# 1. Synthetic Truncated JUMBF Block Test
# =========================================================================
def test_synthetic_truncated_jumbf():
    """
    Synthetic truncated JUMBF: Create a JPEG with LBox=999999 but only 100 bytes payload.
    Should flag TRUNCATED_JUMBF_BLOCK with exact byte offset and length.
    """
    parser = JUMBFByteParser()

    # Build APP11 marker (0xFFEB)
    # LBox = 999999 (4 bytes), TBox = 'jumb' (4 bytes), 92 bytes of mock data = total 100 bytes payload
    declared_lbox = 999999
    jumbf_payload = struct.pack(">I", declared_lbox) + b"jumb" + (b"A" * 92)
    segment_length = len(jumbf_payload) + 2  # includes 2 bytes of length
    app11_segment = b"\xff\xeb" + struct.pack(">H", segment_length) + jumbf_payload

    jpeg_bytes = create_minimal_jpeg(extra_segments=app11_segment)

    res = parser.parse(jpeg_bytes)
    assert res.format == "JPEG"
    assert res.truncated is True

    # Find the truncated diagnostic item
    trunc_items = [d for d in res.details if d.tag_type == "TRUNCATED_JUMBF_BLOCK"]
    assert len(trunc_items) >= 1

    item = trunc_items[0]
    assert item.length == declared_lbox
    assert item.offset == 6  # 2 (SOI) + 2 (0xFFEB) + 2 (length) = 6
    assert item.offset_hex == "0x00000006"
    assert "declared LBox=999999" in item.context
    assert item.severity == "CRITICAL"

# =========================================================================
# 2. Partial AI Tag Scan Test with re.finditer Offsets
# =========================================================================
def test_partial_ai_tag_at_segment_end():
    """
    Partial AI tag: Insert 'traineda...' severed at end of segment.
    Verify regex finditer captures start and end byte offsets.
    """
    parser = JUMBFByteParser()

    partial_sig = b"trainedalgorithmicm"
    com_payload = b"Notice: " + partial_sig
    com_segment = b"\xff\xfe" + struct.pack(">H", len(com_payload) + 2) + com_payload

    jpeg_bytes = create_minimal_jpeg(extra_segments=com_segment)
    res = parser.parse(jpeg_bytes)

    partial_items = [d for d in res.details if d.tag_type == "PARTIAL_AI_TAG"]
    assert len(partial_items) >= 1

    item = partial_items[0]
    assert item.raw_snippet.lower().startswith("trainedalg")
    assert item.offset > 0
    assert item.length == len(item.raw_snippet)
    # Check that bytes at exact offset match the raw snippet
    extracted_slice = jpeg_bytes[item.offset:item.offset + item.length].decode("latin-1")
    assert extracted_slice == item.raw_snippet

# =========================================================================
# 3. Whitelist Spoof: Fake "Canon" Tag + Truncated JUMBF
# =========================================================================
def test_whitelist_spoof_detection(tmp_path):
    """
    Whitelist spoof: Legitimate 'Canon' EXIF tag + truncated JUMBF -> expect SPOOFED_SCANNER_METADATA.
    """
    # Create synthetic JPEG with APP11 truncated JUMBF
    declared_lbox = 50000
    jumbf_payload = struct.pack(">I", declared_lbox) + b"jumb" + (b"X" * 60)
    app11_segment = b"\xff\xeb" + struct.pack(">H", len(jumbf_payload) + 2) + jumbf_payload
    jpeg_bytes = create_minimal_jpeg(extra_segments=app11_segment)

    test_file = tmp_path / "spoofed_canon.jpg"
    test_file.write_bytes(jpeg_bytes)

    # Mock registry returning whitelisted Canon profile
    registry = ApprovedExifRegistry()
    guard = MetadataGuard(registry=registry)

    # Manually seed parser EXIF to simulate Canon tag present
    original_parse = guard.parser.parse
    def mock_parse(inp):
        res = original_parse(inp)
        res.exif = {"Make": "Canon", "Model": "CanoScan LiDE 300", "Software": "Canon IJ Scan Utility"}
        return res
    guard.parser.parse = mock_parse

    eval_res = guard.evaluate(str(test_file))

    assert eval_res["is_tampered_truncated"] is True
    spoofed_items = [d for d in eval_res["byte_offset_diagnostics"] if d["tag_type"] == "SPOOFED_SCANNER_METADATA"]
    assert len(spoofed_items) >= 1
    assert "Anti-forensic Spoofing Detected" in spoofed_items[0]["context"]
    assert eval_res["risk_score"] >= 60.0

# =========================================================================
# 4. Levenshtein Fuzzy Matching for Spoofed Brand Tags
# =========================================================================
def test_levenshtein_distance_calculation():
    assert levenshtein_distance("canon", "canon") == 0
    assert levenshtein_distance("canon", "can0n") == 1
    assert levenshtein_distance("apple", "app1e") == 1
    assert levenshtein_distance("epson", "eps0n") == 1
    assert levenshtein_distance("fujitsu", "fuj1tsu") == 1
    assert levenshtein_distance("brother", "br0ther") == 1

@pytest.mark.parametrize("spoofed_make,approved_vendor", [
    ("Can0n", "Canon"),
    ("App1e", "Apple"),
    ("Eps0n", "Epson"),
    ("Br0ther", "Brother"),
])
def test_fuzzy_spoof_detection(spoofed_make, approved_vendor):
    registry = ApprovedExifRegistry()
    profile = registry.evaluate_exif({
        "Make": spoofed_make,
        "Model": "Test Model",
        "Software": "Scan App"
    })
    assert profile is not None
    assert profile.is_spoofed_attempt is True
    assert profile.vendor == approved_vendor
    assert f"Make '{spoofed_make}' has Levenshtein distance 1" in profile.spoof_reason

# =========================================================================
# 5. Whitelist Registry: Legitimate Scanners & Mobile Capture (Parametrized)
# =========================================================================
@pytest.mark.parametrize("make,model,software,expected_vendor,expected_category", [
    ("Canon", "CanoScan LiDE 300", "Canon IJ Scan Utility", "Canon", "Scanner"),
    ("Canon", "CanoScan LiDE 400", "Canon IJ Scan Utility", "Canon", "Scanner"),
    ("Epson", "Perfection V39", "Epson Scan", "Epson", "Scanner"),
    ("Epson", "WorkForce DS-530", "Document Capture Pro", "Epson", "Scanner"),
    ("HP", "ScanJet Pro 2500 f1", "HP Scan", "HP", "Scanner"),
    ("Fujitsu", "ScanSnap iX1500", "ScanSnap Home", "Fujitsu", "Scanner"),
    ("Brother", "ADS-1700W", "Brother iPrint&Scan", "Brother", "Scanner"),
    ("Xerox", "DocuMate 3125", "Visioneer OneTouch", "Xerox", "Scanner"),
    ("Apple", "iPhone 14 Pro", "Apple Camera", "Apple", "Mobile Camera"),
    ("Samsung", "Galaxy S23", "Samsung Camera", "Samsung", "Mobile Camera"),
    ("Google", "Pixel 7", "Google Camera", "Google", "Mobile Camera"),
    ("Adobe", "Adobe Scan Mobile", "Adobe Scan", "Adobe", "Mobile Scanner App"),
    ("INTSIG", "CamScanner", "CamScanner App", "CamScanner", "Mobile Scanner App"),
])
def test_approved_registry_devices(make, model, software, expected_vendor, expected_category):
    registry = ApprovedExifRegistry()
    profile = registry.evaluate_exif({
        "Make": make,
        "Model": model,
        "Software": software
    })
    assert profile is not None
    assert profile.is_spoofed_attempt is False
    assert profile.vendor == expected_vendor
    assert profile.category == expected_category
    assert profile.confidence >= 0.85

# =========================================================================
# 6. AI Generator Signatures (Parametrized)
# =========================================================================
@pytest.mark.parametrize("ai_payload,expected_keyword", [
    (b"trainedAlgorithmicMedia", "trainedalgorithmicmedia"),
    (b"compositeWithTrainedAlgorithmicMedia", "compositewithtrainedalgorithmicmedia"),
    (b"Generated with DALL-E 3 image engine", "dall-e"),
    (b"midjourney v6 stylize 250", "midjourney"),
    (b"steps: 35, sampler: DPM++ 2M, cfg scale: 7.5, seed: 9948291", "steps:"),
    (b"adobe firefly generative match", "adobe firefly"),
    (b"Flux.1 Schnell diffusion engine by Black Forest Labs", "flux.1"),
])
def test_ai_generator_signatures_offset_capture(ai_payload, expected_keyword):
    parser = JUMBFByteParser()
    jpeg_bytes = create_minimal_jpeg(extra_segments=b"\xff\xfe" + struct.pack(">H", len(ai_payload) + 2) + ai_payload)
    res = parser.parse(jpeg_bytes)

    assert res.has_ai_tags is True
    matching = [d for d in res.details if expected_keyword.lower() in d.context.lower() or (d.raw_snippet and expected_keyword.lower() in d.raw_snippet.lower())]
    assert len(matching) >= 1
    diag = matching[0]
    assert diag.offset > 0
    assert diag.length > 0
    assert diag.offset_hex.startswith("0x")

# =========================================================================
# 7. PNG Chunk Walker & CRC Tampering Detection
# =========================================================================
def test_png_walker_and_crc_tampering():
    parser = JUMBFByteParser()

    # Valid PNG chunk
    valid_png = create_minimal_png(chunks=[
        (b"tEXt", b"Software\x00Adobe Scan", False)
    ])
    res_valid = parser.parse(valid_png)
    assert res_valid.format == "PNG"
    crc_errors = [d for d in res_valid.details if d.tag_type == "CRC_ERROR"]
    assert len(crc_errors) == 0

    # Tampered PNG chunk with corrupted CRC
    corrupt_png = create_minimal_png(chunks=[
        (b"tEXt", b"Software\x00Adobe Scan", True)
    ])
    res_corrupt = parser.parse(corrupt_png)
    assert res_corrupt.format == "PNG"
    assert res_corrupt.truncated is True
    crc_errors = [d for d in res_corrupt.details if d.tag_type == "CRC_ERROR"]
    assert len(crc_errors) >= 1
    assert crc_errors[0].severity == "HIGH"
    assert "failed CRC validation" in crc_errors[0].context

# =========================================================================
# 8. C2PAExifFastGuard Integration & Backward Compatibility
# =========================================================================
def test_c2pa_exif_fast_guard_integration(tmp_path):
    guard = C2PAExifFastGuard()

    # Test genuine image with whitelisted scanner
    clean_jpeg = create_minimal_jpeg()
    clean_file = tmp_path / "clean_canon.jpg"
    clean_file.write_bytes(clean_jpeg)

    res = guard.evaluate_fast_path(str(clean_file))
    # Test backward-compatible tuple unpacking (is_ai, title, explanation)
    is_ai, title, explanation = res
    assert is_ai is False
    assert title is None
    assert explanation is None
    assert hasattr(res, "diagnostics")
    assert "byte_offset_diagnostics" in res.diagnostics
    assert "risk_score" in res.diagnostics

    # Test AI image with DALL-E tag
    ai_jpeg = create_minimal_jpeg(extra_segments=b"\xff\xfe" + struct.pack(">H", 16) + b"dall-e 3 output")
    ai_file = tmp_path / "ai_dalle.jpg"
    ai_file.write_bytes(ai_jpeg)

    res_ai = guard.evaluate_fast_path(str(ai_file))
    is_ai_flag, ai_title, ai_exp = res_ai
    assert is_ai_flag is True
    assert "AI-Generated" in ai_title
    assert "dall-e" in ai_exp.lower()
    assert len(res_ai.diagnostics["byte_offset_diagnostics"]) >= 1
