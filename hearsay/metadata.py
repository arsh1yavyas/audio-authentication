"""Check container metadata before decoding audio.

These checks can find malformed or mislabeled files. A coherent container is
weak evidence about provenance, so it receives only a small score adjustment.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


METADATA_REAL_WEIGHT = 0.05
ASF_MAGIC = bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")


@dataclass(frozen=True)
class MetadataAnalysis:
    status: str
    checks: tuple[str, ...] = ()
    inconsistencies: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"status": self.status, "checks": list(self.checks),
                "inconsistencies": list(self.inconsistencies)}


def _inconsistent(reason: str) -> MetadataAnalysis:
    return MetadataAnalysis("inconsistent", inconsistencies=(reason,))


def _wav_metadata(stream, file_size: int) -> MetadataAnalysis:
    header = stream.read(12)
    if len(header) == 12 and header[:4] in (b"RF64", b"RIFX") and header[8:] == b"WAVE":
        return MetadataAnalysis("unknown", checks=("WAVE variant not checked for consistency",))
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
        return _inconsistent("The .wav extension does not match a RIFF/WAVE header")
    declared_end = struct.unpack_from("<I", header, 4)[0] + 8
    if declared_end > file_size:
        return _inconsistent("RIFF size extends past the end of the file")
    if declared_end < 44:
        return _inconsistent("RIFF size is too small for WAVE audio")

    position = 12
    audio_format = channels = sample_rate = byte_rate = block_align = bit_depth = None
    data_size = None
    while position + 8 <= declared_end:
        stream.seek(position)
        chunk = stream.read(8)
        chunk_size = struct.unpack_from("<I", chunk, 4)[0]
        # RIFF chunks occupy an even number of bytes; odd payloads have padding.
        next_position = position + 8 + chunk_size + (chunk_size % 2)
        if next_position > declared_end:
            return _inconsistent("A WAVE chunk extends past the declared RIFF size")
        if chunk[:4] == b"fmt ":
            if chunk_size < 16:
                return _inconsistent("WAVE format chunk is too short")
            fields = stream.read(16)
            audio_format, channels, sample_rate, byte_rate, block_align, bit_depth = struct.unpack(
                "<HHIIHH", fields)
        elif chunk[:4] == b"data":
            data_size = chunk_size if data_size is None else data_size + chunk_size
        position = next_position
    if position != declared_end:
        return _inconsistent("WAVE chunk boundaries do not match the RIFF size")
    if audio_format is None or data_size is None or data_size == 0:
        return _inconsistent("WAVE format or audio data chunk is missing")
    if not 1 <= channels <= 32 or not 1 <= sample_rate <= 768_000 or block_align == 0:
        return _inconsistent("WAVE channel count, sample rate, or block alignment is invalid")
    if audio_format in (1, 3):
        if bit_depth == 0 or bit_depth % 8 or block_align != channels * (bit_depth // 8):
            return _inconsistent("PCM sample width and block alignment disagree")
        if byte_rate != sample_rate * block_align:
            return _inconsistent("PCM byte rate disagrees with sample rate and block alignment")
        if data_size % block_align:
            return _inconsistent("PCM data size is not a whole number of audio frames")
    return MetadataAnalysis("consistent", checks=("RIFF/WAVE signature and chunk sizes",
                                                  "audio format and frame alignment"))


def _flac_metadata(header: bytes, file_size: int) -> MetadataAnalysis:
    if not header.startswith(b"fLaC"):
        return _inconsistent("The .flac extension does not match a FLAC header")
    if len(header) < 42 or header[4] & 0x7F or int.from_bytes(header[5:8], "big") != 34:
        return _inconsistent("FLAC STREAMINFO block is missing or malformed")
    if file_size < 42:
        return _inconsistent("FLAC file is shorter than its STREAMINFO block")
    # STREAMINFO packs sample rate, channels, and bit depth into these 8 bytes.
    audio_info = int.from_bytes(header[18:26], "big")
    sample_rate = (audio_info >> 44) & 0xFFFFF
    channels = ((audio_info >> 41) & 0x7) + 1
    bit_depth = ((audio_info >> 36) & 0x1F) + 1
    if sample_rate == 0 or not 1 <= channels <= 8 or bit_depth < 4:
        return _inconsistent("FLAC STREAMINFO audio parameters are invalid")
    return MetadataAnalysis("consistent", checks=("FLAC signature and STREAMINFO",))


def _mp3_metadata(header: bytes, file_size: int) -> MetadataAnalysis:
    if header.startswith(b"ID3"):
        if len(header) < 10 or header[3] not in (2, 3, 4) or any(byte & 0x80 for byte in header[6:10]):
            return _inconsistent("MP3 ID3 header is malformed")
        tag_size = sum(byte << shift for byte, shift in zip(header[6:10], (21, 14, 7, 0)))
        if tag_size + 10 >= file_size:
            return _inconsistent("MP3 ID3 tag extends past the audio file")
        return MetadataAnalysis("consistent", checks=("MP3 ID3 header and declared size",))
    if len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0:
        return MetadataAnalysis("consistent", checks=("MPEG audio frame signature",))
    return _inconsistent("The .mp3 extension does not match an ID3 or MPEG audio header")


def analyze_metadata(path: Path) -> MetadataAnalysis:
    """Inspect the container header without decoding or trusting editable tags."""
    suffix = path.suffix.lower()
    with path.open("rb") as stream:
        file_size = path.stat().st_size
        if suffix in (".wav", ".wave"):
            return _wav_metadata(stream, file_size)
        header = stream.read(64)
    if suffix == ".flac":
        return _flac_metadata(header, file_size)
    if suffix == ".mp3":
        return _mp3_metadata(header, file_size)
    if suffix in (".m4a", ".mp4"):
        box_size = int.from_bytes(header[:4], "big")
        if header[4:8] != b"ftyp" or not 8 <= box_size <= file_size:
            return _inconsistent("MP4 file type box is missing or extends past the file")
        return MetadataAnalysis("consistent", checks=("MP4 file type box and declared size",))
    signatures = {
        ".ogg": (header.startswith(b"OggS\x00"), "Ogg page signature"),
        ".opus": (header.startswith(b"OggS\x00"), "Ogg page signature"),
        ".aif": (header.startswith(b"FORM") and header[8:12] in (b"AIFF", b"AIFC"), "AIFF signature"),
        ".aiff": (header.startswith(b"FORM") and header[8:12] in (b"AIFF", b"AIFC"), "AIFF signature"),
        ".webm": (header.startswith(bytes.fromhex("1a45dfa3")), "EBML signature"),
        ".wma": (header.startswith(ASF_MAGIC), "ASF signature"),
        ".aac": (len(header) >= 2 and header[0] == 0xFF and header[1] & 0xF0 == 0xF0,
                 "AAC ADTS frame signature"),
    }
    if suffix not in signatures:
        return MetadataAnalysis("unknown")
    matches, check = signatures[suffix]
    return (MetadataAnalysis("consistent", checks=(check,)) if matches else
            _inconsistent(f"The {suffix} extension does not match its container signature"))


def adjust_score(synthetic_score: float, analysis: MetadataAnalysis) -> float:
    """Give a small real-side weight only when checked metadata is coherent."""
    # This is a transparent heuristic after audio classification, not a
    # calibrated probability update or evidence that the speaker is real.
    if analysis.status == "consistent":
        return float(synthetic_score * (1 - METADATA_REAL_WEIGHT))
    return float(synthetic_score)
