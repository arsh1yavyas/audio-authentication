"""Grouped evaluation of baseline, temporal, and encoding feature views.

The primary metric follows Hearsay's announced parameters (Pspoof=.3,
Cmiss=1, Cfa=4). The challenge answer key is never read by this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.encoding import (ENCODING_FEATURE_NAMES, ENCODING_HEADER_FEATURE_NAMES,
                              ENCODING_SIGNAL_FEATURE_NAMES, extract_encoding_features,
                              extract_signal_encoding_features)
from hearsay.features import decode_audio, extract_features
from hearsay.model import _fit_ensemble, feature_matrix, read_manifest
from hearsay.routing import min_dcf
from hearsay.temporal import TEMPORAL_FEATURE_NAMES, extract_temporal_features


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "normalized_min_dcf": min_dcf(labels, scores)[0],
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def new_classifier(seed: int) -> ExtraTreesClassifier:
    # Small sample count and heterogeneous source groups favor a compact,
    # class-balanced tree model over a large neural net trained from scratch.
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                max_features="sqrt", class_weight="balanced",
                                n_jobs=-1, random_state=seed)


def transcode_probe(paths: list[Path], labels: np.ndarray, models: dict,
                    extractors: dict, output: Path, limit: int = 40) -> dict:
    """Measure prediction movement after lossy MP3 conversion on train examples."""
    import imageio_ffmpeg

    chosen = []
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        chosen.extend(positions[np.linspace(0, len(positions) - 1,
                                           min(limit // 2, len(positions)), dtype=int)])
    original: dict[str, list[float]] = {name: [] for name in models}
    converted: dict[str, list[float]] = {name: [] for name in models}
    with tempfile.TemporaryDirectory(prefix="hearsay-transcode-") as temp:
        folder = Path(temp)
        for position in chosen:
            path = paths[int(position)]
            decoded = decode_audio(path)
            mp3 = folder / f"probe_{position}.mp3"
            result = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
                 "-i", str(path), "-b:a", "64k", str(mp3)],
                capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
            for name, model in models.items():
                before_x = extractors[name](path, decoded)
                after_audio = decode_audio(mp3)
                after_x = extractors[name](mp3, after_audio)
                original[name].append(float(model.predict_proba(before_x.reshape(1, -1))[0, 1]))
                converted[name].append(float(model.predict_proba(after_x.reshape(1, -1))[0, 1]))
    report = {}
    for name in models:
        before = np.asarray(original[name])
        after = np.asarray(converted[name])
        rho = spearmanr(before, after).statistic
        report[name] = {
            "clips": len(before), "mean_absolute_score_change": float(np.mean(np.abs(after - before))),
            "max_absolute_score_change": float(np.max(np.abs(after - before))),
            "score_rank_correlation": float(rho) if np.isfinite(rho) else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/train_features.npz"))
    parser.add_argument("--report", type=Path, default=Path("models/arshiya_feature_views_validation.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-transcode", action="store_true")
    args = parser.parse_args()

    paths, labels, groups = read_manifest(args.manifest)
    if groups is None:
        raise ValueError("Grouped validation requires a group column in the manifest")
    groups_array = np.asarray(groups)
    base_x = feature_matrix(paths, args.cache)
    temporal_x = np.vstack([extract_temporal_features(path) for path in paths])
    encoding_x = np.vstack([extract_encoding_features(path) for path in paths])
    metadata_x = encoding_x[:, :len(ENCODING_HEADER_FEATURE_NAMES)]
    signal_encoding_x = encoding_x[:, len(ENCODING_HEADER_FEATURE_NAMES):]
    print(f"Feature matrices: baseline={base_x.shape}, temporal={temporal_x.shape}, "
          f"encoding_header={metadata_x.shape}, encoding_signal={signal_encoding_x.shape}, "
          f"encoding_combined={encoding_x.shape}", file=sys.stderr, flush=True)

    matrices = {"baseline": base_x, "temporal": temporal_x,
                "encoding_header": metadata_x, "encoding_signal": signal_encoding_x,
                "encoding_combined": encoding_x}
    oof = {name: np.full(len(labels), np.nan) for name in matrices}
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    for fold, (train, valid) in enumerate(splitter.split(base_x, labels, groups_array), start=1):
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[valid])) != 2:
            raise ValueError(f"Fold {fold} lacks one of the classes")
        base = _fit_ensemble(base_x[train], labels[train], seed=700 + fold)
        oof["baseline"][valid] = base.predict_proba(base_x[valid])[:, 1]
        for name in ("temporal", "encoding_header", "encoding_signal", "encoding_combined"):
            estimator = new_classifier(700 + fold).fit(matrices[name][train], labels[train])
            oof[name][valid] = estimator.predict_proba(matrices[name][valid])[:, 1]
        held_groups = sorted(set(groups_array[valid]))
        print(f"Completed fold {fold}/{args.folds}; held-out groups={held_groups}",
              file=sys.stderr, flush=True)

    if any(not np.all(np.isfinite(scores)) for scores in oof.values()):
        raise ValueError("Grouped validation did not generate exactly one OOF score per clip")
    blends = {
        "baseline_plus_temporal_equal": (oof["baseline"] + oof["temporal"]) / 2,
        "baseline_plus_encoding_signal_equal": (oof["baseline"] + oof["encoding_signal"]) / 2,
        "all_three_signal_equal": (oof["baseline"] + oof["temporal"] + oof["encoding_signal"]) / 3,
    }
    report = {
        "validation": "5-fold StratifiedGroupKFold; real speakers and synthetic generators kept within folds",
        "samples": len(labels), "real": int((labels == 0).sum()),
        "synthetic": int((labels == 1).sum()), "group_count": len(set(groups_array)),
        "metric": "normalized minDCF (Pspoof=.3, Cmiss=1, Cfa=4); lower is better",
        "feature_dimensions": {key: int(value.shape[1]) for key, value in matrices.items()},
        "out_of_fold_results": {name: metrics(labels, scores) for name, scores in oof.items()},
        "fixed_equal_weight_blends": {name: metrics(labels, scores) for name, scores in blends.items()},
        "blend_note": "Fixed equal weights avoid selecting weights on these same OOF labels; not a nested tuning result.",
        "encoding_header_by_class": {
            "real_median_source_sample_rate": float(np.median(metadata_x[labels == 0, 3])),
            "synthetic_median_source_sample_rate": float(np.median(metadata_x[labels == 1, 3])),
            "real_median_source_bytes_per_second": float(np.median(metadata_x[labels == 0, 5])),
            "synthetic_median_source_bytes_per_second": float(np.median(metadata_x[labels == 1, 5])),
        },
        "fold_group_assignments": [],
    }
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    for fold, (_, valid) in enumerate(splitter.split(base_x, labels, groups_array), start=1):
        report["fold_group_assignments"].append({"fold": fold,
                                                  "groups": sorted(set(groups_array[valid]))})

    # Save separate candidates; the production baseline artifact and CLI remain untouched.
    temporal_model = new_classifier(42).fit(temporal_x, labels)
    # Header and file-size clues identify dataset preparation; retain them in
    # the diagnostics above, but exclude them from the candidate authenticity score.
    encoding_model = new_classifier(43).fit(signal_encoding_x, labels)
    joblib.dump({"model": temporal_model, "feature_names": TEMPORAL_FEATURE_NAMES,
                 "version": 1, "kind": "experimental-temporal"},
                Path("models/arshiya_temporal_candidate.joblib"))
    joblib.dump({"model": encoding_model, "feature_names": ENCODING_SIGNAL_FEATURE_NAMES,
                 "version": 1, "kind": "experimental-signal-encoding-forensics"},
                Path("models/arshiya_encoding_candidate.joblib"))
    if not args.skip_transcode:
        report["lossy_reencoding_probe"] = transcode_probe(
            paths, labels,
            {"baseline": _fit_ensemble(base_x, labels, 42),
             "temporal": temporal_model, "encoding_signal": encoding_model},
            {"baseline": extract_features,
             "temporal": extract_temporal_features,
             "encoding_signal": extract_signal_encoding_features}, args.report.parent)
        report["lossy_reencoding_note"] = (
            "Exploratory paired stability probe on training clips after 64 kbps MP3 transcoding; "
            "the scores are in-sample and do not estimate accuracy."
        )
    report["encoding_provenance_diagnostic"] = {
        "header_feature_names": list(ENCODING_HEADER_FEATURE_NAMES),
        "signal_feature_names": list(ENCODING_SIGNAL_FEATURE_NAMES),
        "combined_feature_names": list(ENCODING_FEATURE_NAMES),
        "warning": "High header-only performance is corpus provenance leakage, not evidence of synthesis.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
