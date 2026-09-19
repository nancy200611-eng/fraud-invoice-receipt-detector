import os
import re
import struct
import zlib
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any, Iterator

@dataclass
class DiagnosticItem:
    offset: int
    offset_hex: str
    length: int
    tag_type: str
    context: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "INFO"
    raw_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ParsedOffsetsResult:
    format: str
    file_size: int
    markers_or_chunks: List[Dict[str, Any]] = field(default_factory=list)
    has_ai_tags: bool = False
    truncated: bool = False
    details: List[DiagnosticItem] = field(default_factory=list)
    exif: Dict[str, Any] = field(default_factory=dict)
    jumbf_boxes: List[Dict[str, Any]] = field(default_factory=list)

class JUMBFByteParser:
    """
    Low-Level Binary Byte Parser for JPEG & PNG containers.
    Parses segment markers and chunk offsets, validates CRCs, parses C2PA JUMBF boxes,
    and identifies partial/truncated AI metadata tags using regex finditer byte offsets.
    """

    # JUMBF 4-character codes
    JUMB_BOX_TYPE = b"jumb"
    JUMD_BOX_TYPE = b"jumd"
    CBOR_BOX_TYPE = b"cbor"
    C2MA_BOX_TYPE = b"c2ma"

    # Known C2PA UUID for C2PA Manifest Superbox
    C2PA_UUID = b"\x63\x32\x70\x61\x00\x11\x00\x10\x80\x00\x00\xaa\x00\x38\x9b\x71"

    # AI Signatures to locate at byte offsets
    AI_SIGNATURES = [
        re.compile(b"trainedalgorithmicmedia", re.IGNORECASE),
        re.compile(b"compositewithtrainedalgorithmicmedia", re.IGNORECASE),
        re.compile(b"content credentials", re.IGNORECASE),
        re.compile(b"c2pa(\\.manifest|\\.claim|\\.assertions|\\.action)?", re.IGNORECASE),
        re.compile(b"dall[-_]?e(\\s*\\d)?", re.IGNORECASE),
        re.compile(b"chatgpt", re.IGNORECASE),
        re.compile(b"midjourney", re.IGNORECASE),
        re.compile(b"stable\\s*diffusion", re.IGNORECASE),
        re.compile(b"stablediffusion", re.IGNORECASE),
        re.compile(b"comfyui", re.IGNORECASE),
        re.compile(b"automatic1111", re.IGNORECASE),
        re.compile(b"novelai", re.IGNORECASE),
        re.compile(b"adobe\\s+firefly", re.IGNORECASE),
        re.compile(b"flux\\.1", re.IGNORECASE),
        re.compile(b"black\\s+forest\\s+labs", re.IGNORECASE),
        re.compile(b"steps:\\s*\\d+", re.IGNORECASE),
        re.compile(b"cfg\\s+scale:", re.IGNORECASE),
        re.compile(b"seed:\\s*\\d+", re.IGNORECASE),
        re.compile(b"sampler:", re.IGNORECASE),
        re.compile(b"negative\\s+prompt:", re.IGNORECASE)
    ]

    # Partial / Incomplete AI Tag patterns (severed or truncated tags)
    PARTIAL_AI_SIGNATURES = [
        re.compile(b"trainedalg[a-z]{0,15}", re.IGNORECASE),
        re.compile(b"compositewithtrained[a-z]{0,15}", re.IGNORECASE),
        re.compile(b"content\\s+cred[a-z]{0,10}", re.IGNORECASE),
        re.compile(b"midjour[a-z]{0,6}", re.IGNORECASE),
        re.compile(b"stablediff[a-z]{0,8}", re.IGNORECASE),
        re.compile(b"firefl[a-z]{0,4}", re.IGNORECASE)
    ]

    def scan_signature_offsets(self, file_bytes: bytes, regex_patterns: List[re.Pattern]) -> Iterator[Tuple[int, int, bytes]]:
        """
        Yields (start_offset, end_offset, matched_bytes) using re.finditer across raw bytes.
        """
        for pattern in regex_patterns:
            for match in pattern.finditer(file_bytes):
                yield (match.start(), match.end(), match.group(0))

    def parse(self, image_input) -> ParsedOffsetsResult:
        """
        Parses an image file path or raw bytes and extracts structured byte offsets.
        """
        if isinstance(image_input, (str, bytes, os.PathLike)) and os.path.exists(str(image_input)):
            with open(image_input, "rb") as f:
                file_bytes = f.read()
        elif isinstance(image_input, bytes):
            file_bytes = image_input
        else:
            raise ValueError("Input must be a valid file path or raw bytes.")

        file_size = len(file_bytes)
        result = ParsedOffsetsResult(format="UNKNOWN", file_size=file_size)

        if file_bytes.startswith(b"\xff\xd8"):
            result.format = "JPEG"
            self._parse_jpeg(file_bytes, result)
        elif file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            result.format = "PNG"
            self._parse_png(file_bytes, result)
        else:
            result.format = "OTHER"

        # Universal byte offset scan for AI signatures and partial tags
        self._scan_ai_tags(file_bytes, result)

        return result

    def _parse_jpeg(self, file_bytes: bytes, result: ParsedOffsetsResult):
        """
        Walks JPEG markers (0xFFxx) tracking exact offsets, payload boundaries,
        and parses APP11 JUMBF boxes and APP1 EXIF/XMP data.
        """
        pos = 2
        file_size = len(file_bytes)
        result.markers_or_chunks.append({
            "offset": 0,
            "offset_hex": "0x00000000",
            "marker": "0xFFD8",
            "name": "SOI (Start of Image)",
            "length": 0,
            "payload_start": 2
        })

        standalone_markers = {
            0xD8, 0xD9, 0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0x00
        }

        while pos < file_size - 1:
            if file_bytes[pos] != 0xFF:
                # Seek next marker
                next_marker = file_bytes.find(b"\xff", pos)
                if next_marker == -1 or next_marker >= file_size - 1:
                    break
                pos = next_marker

            # Skip padding 0xFF bytes
            while pos < file_size - 1 and file_bytes[pos] == 0xFF and file_bytes[pos + 1] == 0xFF:
                pos += 1

            if pos >= file_size - 1:
                break

            marker_code = file_bytes[pos + 1]
            marker_offset = pos
            marker_hex = f"0xFF{marker_code:02X}"
            pos += 2

            if marker_code in standalone_markers:
                result.markers_or_chunks.append({
                    "offset": marker_offset,
                    "offset_hex": f"0x{marker_offset:08x}",
                    "marker": marker_hex,
                    "name": self._jpeg_marker_name(marker_code),
                    "length": 0,
                    "payload_start": pos
                })
                if marker_code == 0xD9:  # EOI
                    break
                continue

            # Marker with length
            if pos + 2 > file_size:
                result.truncated = True
                result.details.append(DiagnosticItem(
                    offset=marker_offset,
                    offset_hex=f"0x{marker_offset:08x}",
                    length=file_size - marker_offset,
                    tag_type="TRUNCATED_MARKER",
                    context=f"JPEG marker {marker_hex} truncated before 2-byte length field",
                    severity="CRITICAL"
                ))
                break

            length = struct.unpack(">H", file_bytes[pos:pos + 2])[0]
            payload_start = pos + 2
            payload_length = length - 2
            payload_end = payload_start + payload_length

            result.markers_or_chunks.append({
                "offset": marker_offset,
                "offset_hex": f"0x{marker_offset:08x}",
                "marker": marker_hex,
                "name": self._jpeg_marker_name(marker_code),
                "length": length,
                "payload_start": payload_start
            })

            if payload_end > file_size:
                result.truncated = True
                result.details.append(DiagnosticItem(
                    offset=marker_offset,
                    offset_hex=f"0x{marker_offset:08x}",
                    length=length,
                    tag_type="TRUNCATED_SEGMENT",
                    context=f"JPEG marker {marker_hex} specifies length {length} exceeding file size (deficit: {payload_end - file_size} bytes)",
                    severity="CRITICAL"
                ))
                # Stop parsing further segments if truncated beyond EOF
                break

            payload = file_bytes[payload_start:min(payload_end, file_size)]

            # Check APP11 (0xFFEB) for JUMBF
            if marker_code == 0xEB:
                self._parse_jumbf_in_payload(payload, payload_start, result)

            # Check APP1 (0xFFE1) for EXIF or XMP
            elif marker_code == 0xE1:
                self._parse_app1_payload(payload, payload_start, result)

            # Check SOS (0xFFDA) - Scan Data follows
            if marker_code == 0xDA:
                # Find EOI or next marker
                eoi_pos = file_bytes.find(b"\xff\xd9", payload_end)
                if eoi_pos != -1:
                    pos = eoi_pos
                else:
                    break
            else:
                pos = payload_end

    def _parse_png(self, file_bytes: bytes, result: ParsedOffsetsResult):
        """
        Iterates PNG chunks tracking (offset, type, length, CRC).
        Validates CRC to catch tampering and inspects chunks for JUMBF and metadata.
        """
        pos = 8  # Skip 8-byte PNG header
        file_size = len(file_bytes)

        result.markers_or_chunks.append({
            "offset": 0,
            "offset_hex": "0x00000000",
            "type": "PNG_HEADER",
            "length": 8,
            "payload_start": 8
        })

        while pos + 8 <= file_size:
            chunk_offset = pos
            chunk_length = struct.unpack(">I", file_bytes[pos:pos + 4])[0]
            chunk_type_bytes = file_bytes[pos + 4:pos + 8]
            chunk_type = chunk_type_bytes.decode("ascii", errors="replace")
            payload_start = pos + 8
            payload_end = payload_start + chunk_length
            crc_offset = payload_end

            if payload_end + 4 > file_size:
                result.truncated = True
                result.details.append(DiagnosticItem(
                    offset=chunk_offset,
                    offset_hex=f"0x{chunk_offset:08x}",
                    length=chunk_length,
                    tag_type="TRUNCATED_PNG_CHUNK",
                    context=f"PNG chunk '{chunk_type}' claims length {chunk_length} extending past EOF by {payload_end + 4 - file_size} bytes",
                    severity="CRITICAL"
                ))
                break

            expected_crc = struct.unpack(">I", file_bytes[crc_offset:crc_offset + 4])[0]
            # CRC is calculated on chunk type and chunk data, but NOT length
            crc_data = file_bytes[pos + 4:crc_offset]
            actual_crc = zlib.crc32(crc_data) & 0xFFFFFFFF

            crc_valid = (expected_crc == actual_crc)
            if not crc_valid:
                result.truncated = True
                result.details.append(DiagnosticItem(
                    offset=crc_offset,
                    offset_hex=f"0x{crc_offset:08x}",
                    length=4,
                    tag_type="CRC_ERROR",
                    context=f"PNG chunk '{chunk_type}' at offset 0x{chunk_offset:08x} failed CRC validation (expected 0x{expected_crc:08X}, computed 0x{actual_crc:08X}) - Tampering detected",
                    severity="HIGH"
                ))

            result.markers_or_chunks.append({
                "offset": chunk_offset,
                "offset_hex": f"0x{chunk_offset:08x}",
                "type": chunk_type,
                "length": chunk_length,
                "payload_start": payload_start,
                "crc_valid": crc_valid
            })

            payload = file_bytes[payload_start:payload_end]

            # Inspect JUMBF / C2PA chunks in PNG
            if chunk_type.lower() in ["jumb", "c2pa"]:
                self._parse_jumbf_in_payload(payload, payload_start, result)

            # Inspect EXIF chunk
            elif chunk_type == "eXIf":
                self._parse_raw_exif_tiff(payload, payload_start, result)

            # Inspect text chunks for EXIF/AI parameters
            elif chunk_type in ["tEXt", "iTXt", "zTXt"]:
                self._parse_png_text_chunk(chunk_type, payload, payload_start, result)

            pos = crc_offset + 4
            if chunk_type == "IEND":
                break

    def _parse_jumbf_in_payload(self, payload: bytes, base_offset: int, result: ParsedOffsetsResult):
        """
        Parses JUMBF superbox structure (ISO/IEC 19566-5).
        Validates LBox, TBox, description box (jumd), and catches truncations.
        """
        box_pos = 0
        payload_len = len(payload)

        # Check for JP / JUMBF sub-headers if APP11 wrapper is present
        if payload.startswith(b"JP\x00\x00") or payload.startswith(b"JUMBF"):
            box_pos = 4 if payload.startswith(b"JP\x00\x00") else 5

        while box_pos + 8 <= payload_len:
            curr_box_offset = base_offset + box_pos
            lbox = struct.unpack(">I", payload[box_pos:box_pos + 4])[0]
            tbox = payload[box_pos + 4:box_pos + 8]

            # If LBox == 1, 64-bit XLBox follows
            header_size = 8
            actual_box_len = lbox
            if lbox == 1:
                if box_pos + 16 > payload_len:
                    result.truncated = True
                    result.details.append(DiagnosticItem(
                        offset=curr_box_offset,
                        offset_hex=f"0x{curr_box_offset:08x}",
                        length=8,
                        tag_type="TRUNCATED_JUMBF_BLOCK",
                        context="JUMBF XLBox length header truncated before 64-bit size field",
                        severity="CRITICAL"
                    ))
                    break
                actual_box_len = struct.unpack(">Q", payload[box_pos + 8:box_pos + 16])[0]
                header_size = 16

            box_type_str = tbox.decode("ascii", errors="replace")
            box_info = {
                "offset": curr_box_offset,
                "offset_hex": f"0x{curr_box_offset:08x}",
                "type": box_type_str,
                "declared_length": actual_box_len,
                "header_size": header_size
            }
            result.jumbf_boxes.append(box_info)

            # Truncation check: LBox > remaining bytes
            remaining_bytes = payload_len - box_pos
            if actual_box_len > remaining_bytes:
                result.truncated = True
                result.details.append(DiagnosticItem(
                    offset=curr_box_offset,
                    offset_hex=f"0x{curr_box_offset:08x}",
                    length=actual_box_len,
                    tag_type="TRUNCATED_JUMBF_BLOCK",
                    context=f"JUMBF box '{box_type_str}' declared LBox={actual_box_len} bytes but only {remaining_bytes} bytes remain in payload",
                    severity="CRITICAL"
                ))
                break

            if actual_box_len < header_size:
                result.truncated = True
                result.details.append(DiagnosticItem(
                    offset=curr_box_offset,
                    offset_hex=f"0x{curr_box_offset:08x}",
                    length=actual_box_len,
                    tag_type="CORRUPTED_JUMBF_BLOCK",
                    context=f"JUMBF box '{box_type_str}' has invalid LBox={actual_box_len} less than minimum header size {header_size}",
                    severity="CRITICAL"
                ))
                break

            # If this is a 'jumb' superbox, inspect child description box 'jumd'
            if tbox == self.JUMB_BOX_TYPE:
                sub_pos = box_pos + header_size
                box_end = box_pos + actual_box_len
                if sub_pos + 8 <= box_end:
                    sub_lbox = struct.unpack(">I", payload[sub_pos:sub_pos + 4])[0]
                    sub_tbox = payload[sub_pos + 4:sub_pos + 8]
                    if sub_tbox == self.JUMD_BOX_TYPE:
                        # Check UUID
                        if sub_pos + 24 <= box_end:
                            uuid = payload[sub_pos + 8:sub_pos + 24]
                            if uuid == self.C2PA_UUID or b"c2pa" in uuid:
                                result.has_ai_tags = True
                                result.details.append(DiagnosticItem(
                                    offset=base_offset + sub_pos,
                                    offset_hex=f"0x{base_offset + sub_pos:08x}",
                                    length=sub_lbox,
                                    tag_type="C2PA_MANIFEST",
                                    context="Verified C2PA Content Credentials JUMBF Manifest superbox detected",
                                    severity="HIGH"
                                ))
                else:
                    # Superbox has no room for jumd description box
                    result.truncated = True
                    result.details.append(DiagnosticItem(
                        offset=curr_box_offset,
                        offset_hex=f"0x{curr_box_offset:08x}",
                        length=actual_box_len,
                        tag_type="TRUNCATED_JUMBF_BLOCK",
                        context="JUMBF superbox truncated before child description box (jumd)",
                        severity="CRITICAL"
                    ))

            box_pos += actual_box_len

    def _parse_app1_payload(self, payload: bytes, base_offset: int, result: ParsedOffsetsResult):
        """
        Parses APP1 EXIF and XMP payloads.
        """
        if payload.startswith(b"Exif\x00\x00"):
            tiff_payload = payload[6:]
            self._parse_raw_exif_tiff(tiff_payload, base_offset + 6, result)

        elif payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00") or b"<x:xmpmeta" in payload or b"<?xpacket" in payload:
            self._parse_xmp_packet(payload, base_offset, result)

    def _parse_xmp_packet(self, payload: bytes, base_offset: int, result: ParsedOffsetsResult):
        """
        Checks XMP packet boundaries and unclosed/truncated tags.
        """
        start_idx = payload.find(b"<?xpacket begin")
        end_idx = payload.find(b"<?xpacket end")

        if start_idx != -1 and end_idx == -1:
            result.truncated = True
            result.details.append(DiagnosticItem(
                offset=base_offset + start_idx,
                offset_hex=f"0x{base_offset + start_idx:08x}",
                length=len(payload) - start_idx,
                tag_type="TRUNCATED_XMP_PACKET",
                context="XMP packet opened with <?xpacket begin but lacks closing <?xpacket end packet trailer",
                severity="HIGH"
            ))

        # Check for C2PA or AI IPTC in XMP
        xmp_text = payload.lower()
        if b"trainedalgorithmicmedia" in xmp_text:
            result.has_ai_tags = True
            off = base_offset + xmp_text.find(b"trainedalgorithmicmedia")
            result.details.append(DiagnosticItem(
                offset=off,
                offset_hex=f"0x{off:08x}",
                length=23,
                tag_type="IPTC_DIGITAL_SOURCE_TYPE",
                context="XMP specifies IPTC DigitalSourceType: trainedAlgorithmicMedia",
                severity="CRITICAL"
            ))

    def _parse_raw_exif_tiff(self, tiff_bytes: bytes, base_offset: int, result: ParsedOffsetsResult):
        """
        Parses TIFF header and extracts Make, Model, Software tags from IFD0.
        """
        if len(tiff_bytes) < 8:
            return

        byte_order = tiff_bytes[:2]
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            return

        try:
            magic = struct.unpack(f"{endian}H", tiff_bytes[2:4])[0]
            if magic != 42:
                return

            ifd0_offset = struct.unpack(f"{endian}I", tiff_bytes[4:8])[0]
            if ifd0_offset + 2 > len(tiff_bytes):
                return

            num_entries = struct.unpack(f"{endian}H", tiff_bytes[ifd0_offset:ifd0_offset + 2])[0]
            pos = ifd0_offset + 2

            tag_names = {
                271: "Make",
                272: "Model",
                305: "Software",
                315: "Artist",
                33432: "Copyright",
                37510: "UserComment"
            }

            for _ in range(num_entries):
                if pos + 12 > len(tiff_bytes):
                    break
                tag_id, tag_type, tag_count, tag_val_or_offset = struct.unpack(f"{endian}HHI I", tiff_bytes[pos:pos + 12])
                pos += 12

                if tag_id in tag_names and tag_type == 2:  # ASCII string
                    str_len = tag_count
                    if str_len <= 4:
                        raw_str = struct.pack(f"{endian}I", tag_val_or_offset)[:str_len]
                    else:
                        val_offset = tag_val_or_offset
                        raw_str = tiff_bytes[val_offset:val_offset + str_len]
                    val_clean = raw_str.decode("utf-8", errors="ignore").rstrip("\x00").strip()
                    result.exif[tag_names[tag_id]] = val_clean

        except Exception:
            pass

    def _parse_png_text_chunk(self, chunk_type: str, payload: bytes, base_offset: int, result: ParsedOffsetsResult):
        """
        Parses PNG text chunks (tEXt, iTXt) and extracts software or generation parameters.
        """
        try:
            if chunk_type == "tEXt":
                parts = payload.split(b"\x00", 1)
                if len(parts) == 2:
                    keyword = parts[0].decode("latin-1", errors="ignore")
                    text = parts[1].decode("latin-1", errors="ignore")
                    if keyword.lower() == "software":
                        result.exif["Software"] = text
                    elif keyword.lower() in ["prompt", "parameters", "workflow"]:
                        result.has_ai_tags = True
                        result.details.append(DiagnosticItem(
                            offset=base_offset,
                            offset_hex=f"0x{base_offset:08x}",
                            length=len(payload),
                            tag_type="AI_PROMPT_PARAMETER",
                            context=f"PNG tEXt chunk contains AI generation keyword '{keyword}'",
                            severity="CRITICAL",
                            raw_snippet=text[:100]
                        ))
        except Exception:
            pass

    def _scan_ai_tags(self, file_bytes: bytes, result: ParsedOffsetsResult):
        """
        Universal scan using regex finditer byte offsets.
        Identifies full AI signatures and flags partial/truncated AI tag remnants.
        """
        # 1. Full signatures
        for start, end, match_bytes in self.scan_signature_offsets(file_bytes, self.AI_SIGNATURES):
            matched_str = match_bytes.decode("latin-1", errors="ignore")
            if any(d.offset == start for d in result.details):
                continue
            result.has_ai_tags = True
            result.details.append(DiagnosticItem(
                offset=start,
                offset_hex=f"0x{start:08x}",
                length=end - start,
                tag_type="AI_PROVENANCE_SIGNATURE",
                context=f"AI provenance signature '{matched_str}' detected at byte offset 0x{start:08x}",
                severity="CRITICAL" if any(k in matched_str.lower() for k in ["trained", "dall", "c2pa"]) else "HIGH",
                raw_snippet=matched_str
            ))

        # 2. Partial / Truncated AI signatures
        for start, end, match_bytes in self.scan_signature_offsets(file_bytes, self.PARTIAL_AI_SIGNATURES):
            # Check if this match is part of an already detected full signature
            if any(d.offset <= start and (d.offset + d.length) >= end for d in result.details):
                continue
            matched_str = match_bytes.decode("latin-1", errors="ignore")
            result.truncated = True
            result.has_ai_tags = True
            result.details.append(DiagnosticItem(
                offset=start,
                offset_hex=f"0x{start:08x}",
                length=end - start,
                tag_type="PARTIAL_AI_TAG",
                context=f"Partial/truncated AI provenance tag '{matched_str}' detected at byte offset 0x{start:08x} (incomplete or severed block)",
                severity="CRITICAL",
                raw_snippet=matched_str
            ))

    def _jpeg_marker_name(self, code: int) -> str:
        names = {
            0xD8: "SOI (Start of Image)",
            0xD9: "EOI (End of Image)",
            0xDA: "SOS (Start of Scan)",
            0xDB: "DQT (Quantization Table)",
            0xC0: "SOF0 (Baseline DCT)",
            0xC2: "SOF2 (Progressive DCT)",
            0xC4: "DHT (Huffman Table)",
            0xEB: "APP11 (JPEG Universal Metadata / JUMBF)",
            0xE1: "APP1 (EXIF / XMP)",
            0xE0: "APP0 (JFIF)",
            0xE2: "APP2 (ICC Profile)",
            0xED: "APP13 (Photoshop IPTC)",
            0xFE: "COM (Comment)",
        }
        return names.get(code, f"Marker 0xFF{code:02X}")
