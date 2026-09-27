"""Safely unpack the Hearsay test ZIP, which contains a nested TAR archive."""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

SAFE_WAV_NAME = re.compile(r"HGT[A-Za-z0-9_.-]*\.wav\Z")
MAX_AUDIO_SIZE = 30_000_000


@contextmanager
def archive_tar(archive_path: Path):
    """Stream the raw challenge TAR or its ZIP-wrapped form."""
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as outer:
            names = [name for name in outer.namelist() if not name.endswith("/")]
            if len(names) != 1 or not names[0].endswith(".tar"):
                raise ValueError("Expected exactly one TAR file in the ZIP")
            with outer.open(names[0]) as tar_stream:
                with tarfile.open(fileobj=tar_stream, mode="r|*") as tar:
                    yield tar
    else:
        with tarfile.open(archive_path, mode="r:*") as tar:
            yield tar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wav_count = 0
    template_count = 0
    seen_wavs: set[str] = set()
    with archive_tar(args.archive) as inner:
        for member in inner:
            if not member.isfile():
                continue
            source = PurePosixPath(member.name)
            if len(source.parts) != 2 or source.parts[0] != "HackGTHearsayTesting":
                raise ValueError(f"Unexpected TAR path: {member.name}")
            name = source.name
            if SAFE_WAV_NAME.fullmatch(name):
                if name in seen_wavs:
                    raise ValueError(f"Duplicate test WAV in archive: {name}")
                if member.size > MAX_AUDIO_SIZE:
                    raise ValueError(f"Unusually large test WAV: {name}")
                seen_wavs.add(name)
                destination = args.output / "test" / name
                wav_count += 1
            elif name == "HGT_Hearsay_score_template.csv":
                destination = args.output / name
                template_count += 1
            else:
                raise ValueError(f"Unexpected TAR file: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = inner.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read TAR member: {member.name}")
            if destination.exists():
                if destination.stat().st_size != member.size:
                    raise ValueError(f"Existing file has different size: {destination}")
                with destination.open("rb") as existing:
                    while chunk := existing.read(1024 * 1024):
                        if chunk != extracted.read(len(chunk)):
                            raise ValueError(f"Existing file differs from archive: {destination}")
                continue
            with destination.open("xb") as target:
                shutil.copyfileobj(extracted, target)
    if wav_count == 0 or template_count != 1:
        raise ValueError("Archive did not contain test WAV files and one score template")
    print(f"Prepared {wav_count} WAV files in {args.output / 'test'}")
    print(f"Template: {args.output / 'HGT_Hearsay_score_template.csv'}")


if __name__ == "__main__":
    main()
