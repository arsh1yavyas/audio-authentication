"""Evaluate LFCC evidence as a standalone expert and as added model inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.features import (FEATURE_NAMES, decode_audio, extract_features,
                              lowpass_audio)
from hearsay.lfcc import LFCC_FEATURE_NAMES, extract_lfcc_features
from hearsay.model import _fit_ensemble, feature_matrix, read_manifest
from hearsay.routing import min_dcf


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"normalized_minDCF": float(min_dcf(labels, scores)[0]),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def _expert(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                max_features="sqrt", class_weight="balanced",
                                n_jobs=-1, random_state=seed)


def _transcode_probe(files: list[Path], labels: np.ndarray, models: dict,
                     limit: int = 40) -> dict:
    """Paired stability measurement; it is not an accuracy estimate."""
    chosen = []
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        chosen.extend(positions[np.linspace(0, len(positions) - 1,
                                           min(limit // 2, len(positions)), dtype=int)])
    original = {name: [] for name in models}
    converted = {name: [] for name in models}
    with tempfile.TemporaryDirectory(prefix="hearsay-lfcc-transcode-") as temporary:
        folder = Path(temporary)
        for position in chosen:
            path = files[int(position)]
            source = decode_audio(path)
            mp3 = folder / f"probe_{position}.mp3"
            result = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
                 "-i", str(path), "-b:a", "64k", str(mp3)], capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
            recoded = decode_audio(mp3)
            for name, bundle in models.items():
                cutoff = bundle.get("lowpass_hz")
                before = source if cutoff is None else lowpass_audio(source, float(cutoff))
                after = recoded if cutoff is None else lowpass_audio(recoded, float(cutoff))
                before_x = np.concatenate((extract_features(path, before),
                                           extract_lfcc_features(path, before)))
                after_x = np.concatenate((extract_features(mp3, after),
                                          extract_lfcc_features(mp3, after)))
                original[name].append(float(bundle["model"].predict_proba(before_x[None, :])[0, 1]))
                converted[name].append(float(bundle["model"].predict_proba(after_x[None, :])[0, 1]))
    output = {}
    for name in models:
        before, after = np.asarray(original[name]), np.asarray(converted[name])
        correlation = spearmanr(before, after).statistic
        output[name] = {"clips": len(before),
                        "mean_absolute_score_change": float(np.mean(np.abs(before - after))),
                        "max_absolute_score_change": float(np.max(np.abs(before - after))),
                        "rank_correlation": float(correlation) if np.isfinite(correlation) else None}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/train_features.npz"))
    parser.add_argument("--report", type=Path, default=Path("models/arshiya_lfcc_validation.json"))
    parser.add_argument("--baseline-model", type=Path, default=Path("models/arshiya_lfcc_baseline.joblib"))
    parser.add_argument("--lowpass-model", type=Path, default=Path("models/arshiya_lfcc_lowpass.joblib"))
    parser.add_argument("--lowpass-hz", type=float, default=7_000.0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    files, labels, groups = read_manifest(args.manifest)
    if groups is None:
        raise ValueError("Grouped LFCC evaluation requires a group column")
    groups = np.asarray(groups)
    baseline_x = feature_matrix(files, args.feature_cache)
    lfcc_x = np.vstack([extract_lfcc_features(path) for path in files])
    combined_x = np.hstack((baseline_x, lfcc_x))
    lowpass_baseline_x, lowpass_lfcc_x = [], []
    for index, path in enumerate(files, start=1):
        filtered = lowpass_audio(decode_audio(path), args.lowpass_hz)
        lowpass_baseline_x.append(extract_features(path, filtered))
        lowpass_lfcc_x.append(extract_lfcc_features(path, filtered))
        if index % 100 == 0 or index == len(files):
            print(f"Prepared common-band features {index}/{len(files)}", file=sys.stderr, flush=True)
    lowpass_baseline_x = np.vstack(lowpass_baseline_x)
    lowpass_lfcc_x = np.vstack(lowpass_lfcc_x)
    lowpass_combined_x = np.hstack((lowpass_baseline_x, lowpass_lfcc_x))
    oof = {name: np.full(len(labels), np.nan)
           for name in ("baseline", "lfcc_only", "baseline_plus_lfcc",
                        "lowpass_baseline", "lowpass_baseline_plus_lfcc")}
    fold_details = []
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    for fold, (train, valid) in enumerate(splitter.split(baseline_x, labels, groups), start=1):
        base = _fit_ensemble(baseline_x[train], labels[train], seed=40_000 + fold)
        oof["baseline"][valid] = base.predict_proba(baseline_x[valid])[:, 1]
        lfcc = _expert(40_000 + fold).fit(lfcc_x[train], labels[train])
        oof["lfcc_only"][valid] = lfcc.predict_proba(lfcc_x[valid])[:, 1]
        combined = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=40_000 + fold,
        ).fit(combined_x[train], labels[train])
        oof["baseline_plus_lfcc"][valid] = combined.predict_proba(combined_x[valid])[:, 1]
        lowpass_base = _fit_ensemble(lowpass_baseline_x[train], labels[train], seed=50_000 + fold)
        oof["lowpass_baseline"][valid] = lowpass_base.predict_proba(lowpass_baseline_x[valid])[:, 1]
        lowpass_combined = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=50_000 + fold,
        ).fit(lowpass_combined_x[train], labels[train])
        oof["lowpass_baseline_plus_lfcc"][valid] = lowpass_combined.predict_proba(
            lowpass_combined_x[valid])[:, 1]
        fold_details.append({
            "fold": fold,
            "held_groups": sorted(set(groups[valid])),
            "scores": {name: _metrics(labels[valid], values[valid])
                       for name, values in oof.items()},
        })
        print(f"Completed grouped LFCC fold {fold}/{args.folds}; "
              f"held groups={sorted(set(groups[valid]))}", file=sys.stderr, flush=True)
    if any(not np.all(np.isfinite(value)) for value in oof.values()):
        raise ValueError("LFCC validation did not produce a complete OOF score vector")

    final_model = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    ).fit(combined_x, labels)
    final_lowpass_model = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=43,
    ).fit(lowpass_combined_x, labels)
    raw_bundle = {"model": final_model, "lowpass_hz": None}
    lowpass_bundle = {"model": final_lowpass_model, "lowpass_hz": args.lowpass_hz}
    args.baseline_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**raw_bundle, "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
                 "version": 1, "kind": "experimental-baseline-plus-lfcc",
                 }, args.baseline_model)
    args.lowpass_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**lowpass_bundle,
                 "feature_names": (*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
                 "version": 1, "kind": "experimental-baseline-plus-lfcc"}, args.lowpass_model)
    report = {
        "samples": len(labels), "real": int(sum(labels == 0)),
        "synthetic": int(sum(labels == 1)), "groups": len(set(groups)),
        "validation": f"{args.folds}-fold StratifiedGroupKFold; random_state=2026",
        "metric": "normalized minDCF; Pspoof=.3, Cmiss=1, Cfa=4",
        "feature_dimensions": {"baseline": int(baseline_x.shape[1]),
                               "lfcc": int(lfcc_x.shape[1]), "combined": int(combined_x.shape[1]),
                               "lowpass_combined": int(lowpass_combined_x.shape[1])},
        "out_of_fold_results": {name: _metrics(labels, scores) for name, scores in oof.items()},
        "fold_details": fold_details,
        "artifacts": {"raw": str(args.baseline_model), "common_band": str(args.lowpass_model)},
        "common_band_cutoff_hz": args.lowpass_hz,
        "top_combined_feature_importance": [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(zip((*FEATURE_NAMES, *LFCC_FEATURE_NAMES),
                                           final_model.feature_importances_),
                                      key=lambda pair: -pair[1])[:20]
        ],
        "paired_64kbps_mp3_stability": _transcode_probe(
            files, labels, {"raw_lfcc": raw_bundle, "lowpass_lfcc": lowpass_bundle}),
        "paired_stability_note": "Training clips are used in this paired, in-sample stability diagnostic; it does not estimate accuracy.",
        "note": "LFCC is concatenated with the baseline inputs and trained as one forest; common-band results test a fixed low-pass against source-bandwidth shortcuts.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
