# Hearsay audio authentication

This repository contains a trained baseline for the HackGT 13 Hearsay challenge. It returns a score from 0 (real) to 1 (synthetic). The classifier is an **equal blend of two random forests of decision trees** trained on labeled real and generated speech. The included model can be run directly; retrain it if your data or feature code changes.

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

The voice-cloning, speaker-embedding, CNN, RNN, and LSTM sections of `Research.md` describe background and alternative models. They are not implemented in this decision-tree baseline. The assignment's metadata, compression, acoustic environment, splice, and neural anti-spoofing approaches are further extensions, not detectors claimed by this version.

One forest uses all 84 features. A second uses the 20 STFT and statistical features, giving less weight to speaker-specific MFCC and chroma patterns. Both have 300 trees and their scores are averaged equally. Each prediction can include the features that moved the forests' probabilities most along their tree paths. These explain the model's decision, not proof that a clip was generated. A validation report gives ROC AUC and accuracy at 0.5.

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

The script uses all 242 LJ real clips and up to twice as many synthetic clips, sampled deterministically across ten DiffSSD generator families. The ZIPs remain untouched. The `group` column keeps LJ chapters and synthetic generator families whole in validation. The LJ archive contains only one real speaker, which caused the first model to mark a spot check of test files almost uniformly synthetic.

The included model also uses a deterministic selection of real clips from 26 speakers in [OpenSLR Mini LibriSpeech SLR31](https://openslr.org/31/) (`dev-clean-2.tar.gz`, CC BY 4.0). Download that archive from the official page, verify its MD5 is `6d7ab67ac6a1d2c993d050e16d61080d`, then run:

```sh
python scripts/add_librispeech_real.py --archive data/external/dev-clean-2.tar.gz --data data
```

This updates `data/train.csv` with 242 LJ real clips, 488 LibriSpeech real clips, and 730 DiffSSD synthetic clips balanced across the ten generators. The split groups LibriSpeech by speaker, LJ by chapter, and synthetic clips by generator. The included training summary is in `data/train_summary.json` after preparation.

Prepare a UTF-8 CSV (or `.tsv`) with `filename,label` columns. Audio paths are relative to the manifest. Accepted labels are `real`, `bonafide`, or `0` and `synthetic`, `spoof`, or `1`. An optional `group` column should identify a speaker, source recording, or generator family shared by clips. The validation split keeps entire groups together when provided. Example:

```csv
filename,label,group
audio/real_001.wav,real,speaker_01
audio/fake_001.mp3,synthetic,generator_a
```

```sh
python -m hearsay train --manifest data/train.csv --model models/hearsay.joblib --report models/validation.json --feature-cache data/train_features.npz
```

Training fits the two 300-tree forests. If there are enough examples, it first evaluates a holdout split, then refits on all labeled examples to create the final model. The optional feature cache avoids decoding unchanged audio on repeat fits. Without the `group` column, the clip-level split may leak speaker or generator cues and overestimate performance. The model file is a Python joblib pickle; load only one you trust. The model is not calibrated on an independent calibration set, so its 0–1 values should be treated as classifier scores until calibrated and checked on held-out data.

### What validation found

The supplied training set is difficult to validate across unseen generators. The saved model's one group holdout (Grad-TTS plus unseen real groups) yielded ROC AUC **0.567** and balanced accuracy **0.489** at 0.5. A separate leave-one-generator-family-out check across all ten families, using one fixed real-speaker split, averaged ROC AUC **0.914** with a worst family of **0.715**. On five different real-speaker splits for Grad-TTS alone, its mean AUC was **0.695** and its lowest was **0.571**. Results depend strongly on which real speakers are held out. See `models/validation.json`, `models/ensemble_holdout.json`, and `models/grad_holdout.json` for the exact splits and scores.

No independent labeled Hearsay test results are available.

## Score one clip

```sh
python -m hearsay score --model models/hearsay.joblib --audio data/sample.m4a
```

The JSON includes `cm-score`, the forest baseline probability, and the strongest positive or negative feature contributions.

## Create the required prediction TSV

```sh
python -m hearsay predict --model models/hearsay.joblib --input data/test --template data/HGT_Hearsay_score_template.csv --output predictions/teamName_predictions.tsv --explanations predictions/explanations.jsonl
```

The TSV has the required `filename` and `cm-score` columns. Files are discovered recursively in the test directory; basenames must be unique. `--template` verifies that every expected file is present exactly once and preserves the template row order. The optional JSONL explains each prediction. If any file cannot be decoded, the command fails rather than silently omitting it.

## Docker

In PowerShell, with Docker Desktop running, run:

```powershell
docker build -t hearsay .
docker run --rm -v "${PWD}/data:/data:ro" -v "${PWD}/predictions:/predictions" hearsay predict --model /app/models/hearsay.joblib --input /data/test --template /data/HGT_Hearsay_score_template.csv --output /predictions/hearsay_predictions.tsv
```

The image includes the trained model. The command scores `/data/test`, checks filenames against the provided template, and writes `predictions/hearsay_predictions.tsv` in template order. The image build and one-file inference were verified with Docker Desktop.

Run the software smoke test with `python -m unittest discover -s tests -v`. Its generated tones only verify the pipeline; they do not estimate deepfake detection performance.

## Limits

The detector needs labeled, varied training data. The added LibriSpeech voices reduce the original one-speaker bias, but the real data still consists of audiobook recordings while the synthetic data comes from DiffSSD. Studio, phone, outdoor, noisy, replayed, and partially edited clips remain underrepresented. Evaluate on speakers, generators, and recording conditions that training did not include. A low score is not authentication; replay, partial edits, and unseen voice generators may evade this baseline. The first 120 seconds are analyzed, so later edits in longer recordings are not covered.
