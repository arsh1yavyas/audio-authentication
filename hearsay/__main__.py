"""Command line interface for training and submitting Hearsay predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_NAMES, decode_audio, extract_features, lowpass_audio
from .model import explain, load_model, save_json, train
from .forensics import extract_quality_features
from .metadata import adjust_score, analyze_metadata
from .routing import ConditionalRouter
from .temporal import TEMPORAL_FEATURE_NAMES, extract_temporal_features
from .lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features

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

    routed = commands.add_parser("predict-routed", help="Score using a saved condition-aware blend")
    routed.add_argument("--model", type=Path, required=True)
    routed.add_argument("--input", type=Path, required=True)
    routed.add_argument("--output", type=Path, required=True)
    routed.add_argument("--template", type=Path)
    routed.add_argument("--router", type=Path, help="Router artifact from scripts/evaluate_routing.py")
    routed.add_argument("--julia-scores", type=Path,
                        help="TSV/CSV with filename,julia-score columns from Julia's predictor")

    fused = commands.add_parser(
        "predict-fused", help="Opt-in score blend using Arshiya temporal and Julia-style metadata evidence")
    fused.add_argument("--model", type=Path, required=True, help="Fitted fused baseline model")
    fused.add_argument("--temporal-model", type=Path, required=True,
                       help="Fitted Arshiya temporal candidate")
    fused.add_argument("--policy", type=Path, required=True,
                       help="Nested group-validation fusion policy JSON")
    fused.add_argument("--input", type=Path, required=True)
    fused.add_argument("--output", type=Path, required=True)
    fused.add_argument("--template", type=Path)

    lfcc = commands.add_parser(
        "predict-lfcc", help="Score with one forest trained on baseline plus LFCC features")
    lfcc.add_argument("--model", type=Path, required=True)
    lfcc.add_argument("--input", type=Path, required=True)
    lfcc.add_argument("--output", type=Path, required=True)
    lfcc.add_argument("--template", type=Path)
    return parser


def _read_external_scores(path: Path | None, column: str) -> dict[str, float]:
    if path is None:
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        score_column = next((name for name in (column, "cm-score", "score")
                             if reader.fieldnames and name in reader.fieldnames), None)
        if not reader.fieldnames or "filename" not in reader.fieldnames or score_column is None:
            raise ValueError(f"External score file needs filename and one of {column}, cm-score, or score columns")
        result = {}
        for row in reader:
            name = (row.get("filename") or "").strip()
            if not name or name in result:
                raise ValueError("External score file contains an empty or duplicate filename")
            result[name] = float(row[score_column])
        return result


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
        elif args.command == "predict":
            count = generate_predictions(args.model, args.input, args.output,
                                         args.template, args.explanations)
            print(f"Wrote {count} predictions to {args.output}", file=sys.stderr)
        elif args.command == "predict-fused":
            model = load_model(args.model)
            temporal_bundle = joblib.load(args.temporal_model)
            if (temporal_bundle.get("version") != 1 or
                    tuple(temporal_bundle.get("feature_names", ())) != TEMPORAL_FEATURE_NAMES or
                    temporal_bundle.get("kind") != "experimental-temporal"):
                raise ValueError("Temporal model was trained with an incompatible feature extractor")
            policy = json.loads(args.policy.read_text(encoding="utf-8"))
            temporal_weight = float(policy["temporal_weight"])
            metadata_weight = float(policy["metadata_real_weight"])
            if not 0.0 <= temporal_weight <= 1.0 or not 0.0 <= metadata_weight <= 1.0:
                raise ValueError("Fusion policy weights must be between zero and one")
            files = _input_files(args.input)
            if args.template:
                with args.template.open("r", newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if not reader.fieldnames or "filename" not in reader.fieldnames:
                        raise ValueError("Template needs a tab-delimited filename column")
                    ordered_names = [(row.get("filename") or "").strip() for row in reader]
                by_name = {path.name: path for path in files}
                if len(ordered_names) != len(set(ordered_names)) or set(ordered_names) != set(by_name):
                    raise ValueError("Template filenames must match input audio files exactly once")
                files = [by_name[name] for name in ordered_names]
            baseline_vectors, temporal_vectors, metadata = [], [], []
            for index, path in enumerate(files, start=1):
                audio = decode_audio(path)
                baseline_vectors.append(extract_features(path, audio))
                temporal_vectors.append(extract_temporal_features(path, audio))
                metadata.append(analyze_metadata(path))
                if index % 100 == 0 or index == len(files):
                    print(f"Processed {index}/{len(files)} clips", file=sys.stderr, flush=True)
            baseline_scores = model.predict_proba(np.vstack(baseline_vectors))[:, 1]
            temporal_scores = temporal_bundle["model"].predict_proba(np.vstack(temporal_vectors))[:, 1]
            adjusted_baseline = np.asarray([
                adjust_score(score, analysis, metadata_weight)
                for score, analysis in zip(baseline_scores, metadata)
            ])
            final_scores = np.clip((1.0 - temporal_weight) * adjusted_baseline
                                   + temporal_weight * temporal_scores, 0.0, 1.0)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(("filename", "cm-score"))
                writer.writerows((path.name, f"{score:.6f}")
                                 for path, score in zip(files, final_scores))
            print(f"Wrote {len(files)} predictions to {args.output}", file=sys.stderr)
        elif args.command == "predict-lfcc":
            bundle = joblib.load(args.model)
            expected_names = (*FEATURE_NAMES, *LFCC_FEATURE_NAMES)
            if (bundle.get("version") != 1 or
                    bundle.get("kind") != "experimental-baseline-plus-lfcc" or
                    tuple(bundle.get("feature_names", ())) != expected_names):
                raise ValueError("LFCC model was trained with an incompatible feature extractor")
            metadata_weight = float(bundle.get("metadata_real_weight", 0.0))
            if not 0.0 <= metadata_weight <= 1.0:
                raise ValueError("LFCC metadata weight must be between zero and one")
            files = _input_files(args.input)
            if args.template:
                with args.template.open("r", newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if not reader.fieldnames or "filename" not in reader.fieldnames:
                        raise ValueError("Template needs a tab-delimited filename column")
                    ordered_names = [(row.get("filename") or "").strip() for row in reader]
                by_name = {path.name: path for path in files}
                if len(ordered_names) != len(set(ordered_names)) or set(ordered_names) != set(by_name):
                    raise ValueError("Template filenames must match input audio files exactly once")
                files = [by_name[name] for name in ordered_names]
            vectors = []
            for index, path in enumerate(files, start=1):
                audio = decode_audio(path)
                cutoff = bundle.get("lowpass_hz")
                if cutoff is not None:
                    audio = lowpass_audio(audio, float(cutoff))
                vectors.append(np.concatenate((extract_features(path, audio),
                                               extract_lfcc_features(path, audio))))
                if index % 100 == 0 or index == len(files):
                    print(f"Processed {index}/{len(files)} clips", file=sys.stderr, flush=True)
            scores = bundle["model"].predict_proba(np.vstack(vectors))[:, 1]
            if metadata_weight:
                scores = np.asarray([
                    adjust_score(score, analyze_metadata(path), metadata_weight)
                    for score, path in zip(scores, files)
                ])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(("filename", "cm-score"))
                writer.writerows((path.name, f"{score:.6f}")
                                 for path, score in zip(files, scores))
            print(f"Wrote {len(files)} predictions to {args.output}", file=sys.stderr)
        elif args.command == "predict-routed":
            model = load_model(args.model)
            files = _input_files(args.input)
            if args.template:
                with args.template.open("r", newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if not reader.fieldnames or "filename" not in reader.fieldnames:
                        raise ValueError("Template needs a tab-delimited filename column")
                    ordered_names = [(row.get("filename") or "").strip() for row in reader]
                by_name = {path.name: path for path in files}
                if len(ordered_names) != len(set(ordered_names)) or set(ordered_names) != set(by_name):
                    raise ValueError("Template filenames must match input audio files exactly once")
                files = [by_name[name] for name in ordered_names]
            julia_scores = _read_external_scores(args.julia_scores, "julia-score")
            if julia_scores and set(julia_scores) != {path.name for path in files}:
                raise ValueError("Julia score filenames must match the input audio files exactly")
            router = None
            if args.router:
                bundle = joblib.load(args.router)
                if bundle.get("version") != 1 or not isinstance(bundle.get("router"), ConditionalRouter):
                    raise ValueError("Incompatible routing artifact")
                router = bundle["router"]
            vectors, quality = [], []
            for index, path in enumerate(files, start=1):
                audio = decode_audio(path)
                vectors.append(extract_features(path, audio))
                quality.append(extract_quality_features(path, audio))
                if index % 100 == 0 or index == len(files):
                    print(f"Processed {index}/{len(files)} clips", file=sys.stderr, flush=True)
            arshiya_scores = model.predict_proba(np.vstack(vectors))[:, 1]
            if not julia_scores:
                # Missing second scorer cleanly falls back to Arshiya's score.
                final_scores = arshiya_scores
            elif router is None:
                final_scores = 0.5 * arshiya_scores + 0.5 * np.asarray([julia_scores[p.name] for p in files])
            else:
                final_scores = router.predict(
                    arshiya_scores,
                    np.asarray([julia_scores[p.name] for p in files]),
                    np.vstack(quality),
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(("filename", "cm-score"))
                writer.writerows((path.name, f"{score:.6f}") for path, score in zip(files, final_scores))
            print(f"Wrote {len(files)} predictions to {args.output}", file=sys.stderr)
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
