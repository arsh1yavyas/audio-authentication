"""Command line interface for training and submitting Hearsay predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .features import extract_features
from .metadata import adjust_score, analyze_metadata
from .model import explain, load_model, save_json, train

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac",
                    ".aac", ".wma", ".aif", ".aiff", ".webm", ".opus"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hearsay", description="Random forest audio authenticity baseline")
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train from a labeled CSV or TSV manifest")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--test-manifest", type=Path, help="Labeled holdout manifest; excluded from model fitting")
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--report", type=Path, help="Write validation metrics and feature importance as JSON")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--feature-cache", type=Path, help="Reuse feature vectors if audio files have not changed")

    scoring = commands.add_parser("score", help="Score one audio file with a path explanation")
    scoring.add_argument("--model", type=Path, required=True)
    scoring.add_argument("--audio", type=Path, required=True)

    predicting = commands.add_parser("predict", help="Write assignment-format filename/cm-score TSV")
    predicting.add_argument("--model", type=Path, required=True)
    predicting.add_argument("--input", type=Path, required=True, help="Unlabeled audio file or directory for output")
    predicting.add_argument("--output", type=Path, required=True)
    predicting.add_argument("--template", type=Path, help="Use the provided score template's filenames and row order")
    predicting.add_argument("--explanations", type=Path, help="Optional JSONL path explanations")
    return parser


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Audio input does not exist: {path}")
    files = sorted((item for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS),
                   key=lambda item: str(item).lower())
    if not files:
        raise ValueError(f"No supported audio files found in {path}")
    names = [item.name for item in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate audio basenames in input folders would make filename values ambiguous")
    return files


def generate_predictions(model_path: Path, input_path: Path, output_path: Path,
                         template_path: Path | None = None,
                         explanations_path: Path | None = None) -> int:
    """Score unlabeled audio and write a prediction TSV; return the clip count."""
    model = load_model(model_path)
    files = _input_files(input_path)
    if template_path:
        with template_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames or "filename" not in reader.fieldnames:
                raise ValueError("Template needs a tab-delimited filename column")
            ordered_names = [(row.get("filename") or "").strip() for row in reader]
        by_name = {path.name: path for path in files}
        if (len(ordered_names) != len(set(ordered_names)) or
                set(ordered_names) != set(by_name)):
            raise ValueError("Template filenames must match input audio files exactly once")
        files = [by_name[name] for name in ordered_names]
    vectors = []
    metadata = []
    for path in files:
        metadata.append(analyze_metadata(path))
        vectors.append(extract_features(path))
        if len(vectors) % 100 == 0 or len(vectors) == len(files):
            print(f"Processed {len(vectors)}/{len(files)} clips", file=sys.stderr, flush=True)
    if explanations_path:
        explanations = [
            {"filename": path.name, **explain(model, vector, analysis)}
            for path, vector, analysis in zip(files, vectors, metadata)
        ]
        scores = [result["cm-score"] for result in explanations]
    else:
        audio_scores = model.predict_proba(np.vstack(vectors))[:, 1]
        scores = [adjust_score(score, analysis)
                  for score, analysis in zip(audio_scores, metadata)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("filename", "cm-score"))
        writer.writerows((path.name, f"{score:.6f}") for path, score in zip(files, scores))
    if explanations_path:
        explanations_path.parent.mkdir(parents=True, exist_ok=True)
        with explanations_path.open("w", encoding="utf-8") as stream:
            for result in explanations:
                stream.write(json.dumps(result) + "\n")
    return len(files)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            report = train(args.manifest, args.model, args.seed, args.feature_cache, args.test_manifest)
            if args.report:
                save_json(report, args.report)
            print(json.dumps(report, indent=2))
        elif args.command == "score":
            model = load_model(args.model)
            analysis = analyze_metadata(args.audio)
            result = {"filename": args.audio.name,
                      **explain(model, extract_features(args.audio), analysis)}
            print(json.dumps(result, indent=2))
        else:
            count = generate_predictions(args.model, args.input, args.output,
                                         args.template, args.explanations)
            print(f"Wrote {count} predictions to {args.output}", file=sys.stderr)
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
