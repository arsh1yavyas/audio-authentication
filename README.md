# Hearsay audio authentication

This repository contains a trained audio classifier for the HackGT 13 Hearsay challenge. It returns a score from 0 (real) to 1 (synthetic). The recommended model is a **75% RBF support-vector machine and 25% logistic-regression blend**, selected using generator-held-out validation. The earlier two-random-forest model remains in `models/hearsay-split.joblib` for comparison. Retrain the model if your data or feature code changes.

## Approach

FFmpeg decodes the first 120 seconds of the first audio stream to mono 16 kHz audio. A fixed-size feature vector contains all signal feature families identified in `Research.md`:

| Research feature | Representation in this model |
| --- | --- |
| STFT | Short-time spectrum, spectral flux, relative low/mid/high frequency energy; the spectral statistics below also derive from STFT |
| MFCC | Mean and standard deviation of 20 coefficients |
| Chroma | Mean and standard deviation of 12 pitch classes |
| Zero-crossing rate | Mean and standard deviation |
| RMS energy | Mean and standard deviation after peak normalization |
| Spectral centroid | Mean and standard deviation |
| Spectral bandwidth | Mean and standard deviation |
| Spectral rolloff | Mean and standard deviation, at 85% cumulative energy |
| Spectral flatness | Mean and standard deviation |

The voice-cloning, speaker-embedding, CNN, RNN, and LSTM sections of `Research.md` describe background and alternative models. They are not implemented in either saved classifier. Compression forensics, acoustic environment, splice, and neural anti-spoofing approaches are further extensions, not detectors claimed by this version.

Before decoding, the metadata check inspects the file's container signature and, for WAV, declared chunk sizes and audio parameters. FLAC and MP3 headers are also checked. When the inspected metadata is coherent, the final synthetic score is multiplied by **0.95**, giving a small additional weight to “real.” Inconsistent or unrecognized metadata leaves the audio score unchanged. The score explanation shows both the audio score and this adjustment. A clean header can be forged and is common in synthetic audio, so this is a weak heuristic rather than proof of authenticity.

The optimized model standardizes all 84 features before fitting an RBF SVM and logistic regression. Their synthetic scores are averaged with weights 0.75 and 0.25. Its score explanation shows each component score and the metadata adjustment; it does not claim an additive feature explanation. The original model averages two 300-tree forests, one using all 84 features and the other using 20 spectral and statistical features. Only the forest model provides tree-path feature contributions. These describe model behavior, not proof that a clip was generated.

## Install

Use Python 3.12, matching the model and Docker image:

```sh
python -m venv .venv
# Activate the virtual environment using your shell's usual command.
python -m pip install -r requirements.txt
```

`imageio-ffmpeg` supplies an FFmpeg executable. WAV, MP3, M4A, MP4 audio, OGG, FLAC and other formats supported by that executable can be decoded. Encrypted, corrupt, silent, or unsupported files produce a clear error.

The provided `HackGTHearsayTesting.zip` contains a TAR archive with 1,671 test WAV files and a tab-delimited score template. Prepare it with:

```sh
python scripts/prepare_test_data.py --archive "/path/to/HackGTHearsayTesting.zip" --output data
```

This creates `data/test` and `data/HGT_Hearsay_score_template.csv`. The template's repeated `0.006` scores are placeholders, not labels. The test archive cannot be used to train or evaluate the classifier.

## Train

The supplied training archives are `LJRealResampled.zip` (real) and `DiffSSD.zip` (synthetic). Both contain nested TAR files. Prepare a reproducible sample and manifest with:

```sh
python scripts/prepare_training_data.py --real-archive "/path/to/LJRealResampled.zip" --synthetic-archive "/path/to/DiffSSD.zip" --output data
```

The script uses all 242 LJ real clips and up to twice as many synthetic clips, sampled deterministically across ten DiffSSD generator families. The ZIPs remain untouched. It retains additional synthetic candidates on disk for the later LibriSpeech rebalancing step; only selected clips appear in `data/train.csv`. The `group` column keeps LJ chapters and synthetic generator families whole in validation. LJ chapters still share one speaker, so this grouping alone does not test an unseen real voice. The first model marked a spot check of challenge files almost uniformly synthetic.

Both saved models use a deterministic selection of real clips from 26 speakers in [OpenSLR Mini LibriSpeech SLR31](https://openslr.org/31/) (`dev-clean-2.tar.gz`, CC BY 4.0). Download that archive from the official page, verify its MD5 is `6d7ab67ac6a1d2c993d050e16d61080d`, then run:

```sh
python scripts/add_librispeech_real.py --archive data/external/dev-clean-2.tar.gz --data data
```

This updates `data/train.csv` with 242 LJ real clips, 488 LibriSpeech real clips, and 730 DiffSSD synthetic clips balanced across the ten generators. The `group` column identifies LibriSpeech speakers, LJ chapters, and synthetic generators so related clips stay together in a split. The included training summary is in `data/train_summary.json` after preparation.

Prepare a UTF-8 CSV (or `.tsv`) with `filename,label` columns. Audio paths are relative to the manifest. Accepted labels are `real`, `bonafide`, or `0` and `synthetic`, `spoof`, or `1`. An optional `group` column should identify a speaker, source recording, or generator family shared by clips. The validation split keeps entire groups together when provided. Example:

```csv
filename,label,group
audio/real_001.wav,real,speaker_01
audio/fake_001.mp3,synthetic,generator_a
```

Use `hearsay/utils.py` to create reproducible labeled training and test manifests. The CSVs point to the original audio files, so the clips are not copied:

```sh
python -m hearsay.utils --manifest data/train.csv --output data/split --seed 42
python -m hearsay train --manifest data/split/train.csv --test-manifest data/split/test.csv --model models/hearsay-split.joblib --report models/split-validation.json --feature-cache data/split/split_features.npz
python -m scripts.train_optimized
```

The `hearsay train` command reproduces the forest baseline. `scripts.train_optimized` uses only the 1,167 training clips for model selection, holding out each synthetic generator in turn alongside disjoint real groups. It then fits the selected blend on all training clips and evaluates it once on the 293 test clips. The script saves `models/hearsay-optimized.joblib` and `models/optimized-comparison.json`. `utils.py` keeps entire groups together when the manifest has a `group` column; without it, the split is stratified by label at the clip level and can overestimate performance when speakers or generators overlap. You can also pass the original `data/train.csv` directly to `hearsay train`; it uses the same split internally. On datasets too small for a two-class holdout, it fits all rows and reports that validation was unavailable. The optional feature cache avoids decoding unchanged audio on repeat fits. Model files are Python joblib pickles; load only ones you trust. The 0–1 values are classifier scores, not independently calibrated probabilities.

`data/split/test.csv` is a labeled holdout for checking the model. The separate `data/test` directory from `HackGTHearsayTesting.zip` has no labels and is used only for predictions.

### What validation found

The supplied training set is difficult to validate across unseen generators. The original `models/hearsay.joblib` and its saved reports predate the split utility: its one group holdout (Grad-TTS plus unseen real groups) yielded ROC AUC **0.567** and balanced accuracy **0.489** at 0.5. A separate leave-one-generator-family-out check across all ten families, using one fixed real-speaker split, averaged ROC AUC **0.914** with a worst family of **0.715**. On five different real-speaker splits for Grad-TTS alone, its mean AUC was **0.695** and its lowest was **0.571**. Results depend strongly on which real speakers are held out. See `models/validation.json`, `models/ensemble_holdout.json`, and `models/grad_holdout.json` for the exact splits and scores. Retraining with the commands above writes a new model and report.

On the current group-disjoint split (1,167 training and 293 test clips), the recommended model improves both accuracy at the fixed 0.5 cutoff and minDCF:

| Evaluation | Forest baseline | Optimized blend |
| --- | ---: | ---: |
| Training-only generator-held-out validation accuracy | 0.7464 | 0.8509 |
| Training-only validation minDCF | 0.6007 | 0.3487 |
| Held-out test accuracy | 0.8874 | 0.9454 |
| Held-out test ROC AUC | 0.9482 | 0.9930 |
| Held-out test minDCF | 0.2929 | 0.1088 |

minDCF is the normalized [ASVspoof 5 Track 1 detection cost](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf), minimized over thresholds, with a 5% spoof prior, cost 1 for rejecting real speech, and cost 10 for accepting spoofed speech. The original and optimized models have 0.8703 and 0.9454 audio-only held-out accuracy, respectively; the optimized model's accuracy gain does not depend on metadata adjustment. The held-out synthetic groups are only PlayHT and Pro Diff. PlayHT uses `.wav` names with MP3 headers, a source-specific mismatch that may not recur with other generators or transcoded files. Some synthetic speaker IDs recur across generator families, and LJ chapters share a narrator. See `models/optimized-comparison.json` for all metrics and confusion matrices.

No independent labeled Hearsay test results are available.

## Score one clip

```sh
python -m hearsay score --model models/hearsay-optimized.joblib --audio data/sample.m4a
```

The JSON includes the final `cm-score`, audio-only `audio_score`, both component scores, and the metadata adjustment. The original forest additionally reports tree-path feature contributions.

## Create the required prediction TSV

```sh
python -m hearsay predict --model models/hearsay-optimized.joblib --input data/test --template data/HGT_Hearsay_score_template.csv --output predictions/teamName_predictions.tsv --explanations predictions/explanations.jsonl
```

The `predict` command calls a separate `generate_predictions` function for unlabeled audio when an output is needed. These challenge clips are never part of the labeled train/test split or accuracy calculation. The TSV has the required `filename` and `cm-score` columns. Files are discovered recursively in the test directory; basenames must be unique. `--template` verifies that every expected file is present exactly once and preserves the template row order. The optional JSONL explains each prediction. If any file cannot be decoded, the command fails rather than silently omitting it.

## Docker

In PowerShell, with Docker Desktop running, run:

```powershell
docker build -t hearsay .
docker run --rm -v "${PWD}/data:/data:ro" -v "${PWD}/predictions:/predictions" hearsay predict --model /app/models/hearsay-optimized.joblib --input /data/test --template /data/HGT_Hearsay_score_template.csv --output /predictions/hearsay_predictions.tsv
```

The image includes the optimized model. The command scores `/data/test`, checks filenames against the provided template, and writes `predictions/hearsay_predictions.tsv` in template order.

Run the software smoke test with `python -m unittest discover -s tests -v`. Its generated tones only verify the pipeline; they do not estimate deepfake detection performance.

## Limits

The detector needs labeled, varied training data. The added LibriSpeech voices reduce the original one-speaker bias, but the real data still consists of audiobook recordings while the synthetic data comes from DiffSSD. Studio, phone, outdoor, noisy, replayed, and partially edited clips remain underrepresented. Evaluate on speakers, generators, and recording conditions that training did not include. A low score is not authentication; replay, partial edits, and unseen voice generators may evade either classifier. The first 120 seconds are analyzed, so later edits in longer recordings are not covered.
