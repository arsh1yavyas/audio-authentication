"""Safely unpack the Hearsay test ZIP, which contains a nested TAR archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wav_count = 0
    template_count = 0
    with zipfile.ZipFile(args.archive) as outer:
        names = outer.namelist()
        if len(names) != 1 or not names[0].endswith(".tar"):
            raise ValueError("Expected exactly one TAR file in the ZIP")
        with outer.open(names[0]) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode="r|*") as inner:
                for member in inner:
                    if not member.isfile():
                        continue
                    source = PurePosixPath(member.name)
                    if len(source.parts) != 2 or source.parts[0] != "HackGTHearsayTesting":
                        raise ValueError(f"Unexpected TAR path: {member.name}")
                    name = source.name
                    if name.endswith(".wav") and name.startswith("HGT"):
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
                        continue
                    with destination.open("xb") as target:
                        shutil.copyfileobj(extracted, target)
    if wav_count == 0 or template_count != 1:
        raise ValueError("Archive did not contain test WAV files and one score template")
    print(f"Prepared {wav_count} WAV files in {args.output / 'test'}")
    print(f"Template: {args.output / 'HGT_Hearsay_score_template.csv'}")


if __name__ == "__main__":
    main()
