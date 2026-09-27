# Initial Research on Voice Cloning

- From Less Data + Faster + Lower Quality to More Data + Slower + Higher Quality:
  - Zero-shot (no training, just inference and reference audio provided at generation time)
  - Few-shot (minimal adaptation of general speech model and minutes of target speaker audio)
  - Fine-tuning (adapt pre-trained, general speech model to specific voice and hours of target speaker audio)
  - Training from scratch (no pre-trained knowledge and weeks/months of audio)
  - Training from scratch is rarely used in modern voice cloning since pre-trained models have learned general speech and transfer learning is much more efficient
- VC models learn to separate linguistic content from speaker identity
  - They can then use the same model for outputs (same words) in several different speakers' voices by just using different speaker embeddings each time
  - Speaker embedding: vector representation of a speaker's voice that represents that voice identity with numbers ("voice fingerprint")
  - Speaker embeddings can be extracted from audio via self-supervised speech models like WavLM
  - Zero-shot and few-shot learning (both instant) train model on many different speakers and also learn to extract speaker embeddings
  - Then, once given a target text and voice, they can extract the speaker embedding and generate speech using it
  - For few-shot, having more data leads to a better embedding, which leads to higher quality cloning
- With fine-tuning, you take a pre-trained Text-to-Speech model and use hours of desired speaker audio to find specialized model weights
  - Modern approach is to use adapter layers (LoRA or Low-Rank Adaptation of LLMs) on top of the model
  - Instead of changing model weights, can change widths of the adapter matrix
- ElevenLabs is currently dominating the voice cloning field
- Some open-source modern voice cloning systems are XTTS, YourTTS, VALL-E, Bark for zero and few-shot and Coqui TTS, and Tortoise TTS for fine-tuning
- There are many labeling legislations that require AI companies that generate synthetic audio to find a way to utilize watermarks/labels/metadata to indicate or disclose the fact that it is AI-generated (_note that we shouldn't rely on this since real voices that are altered or changed in some way with AI should still be identified as real voices in this project_)
- High frequencies are really hard to generate correctly; in AI music, sometimes you can see in the spectrograph that they cut you off at a certain frequency (like 16,000 Hz)
  - Very muddy high-detail sound, sounds like a highly compressed MP3

# Research from "Ensemble learning model for deepfake audio detection using multi-feature extraction approach":

Feature extraction is the first important step for detecting deepfake audio, popular techniques are short-time Fourier transform (STFT) and Mel-frequency cepstral coefficients (MFCC), chroma features, and zero-crossing rate (ZCR).
    1. STFT - decomposes audio signals into time-divided frequency information, so in short produces a time vs frequency 2D representation called a spectrogram that is used for analysis on audio
    2. MFCC - used for speech recognition, provides a low-dimensional representation of the short-term spectral envelope of audio, it does this by imitating human auditory perception and the way that human hearing has finer resolution at low frequencies over high frequencies
    3. Chroma features and zero-crossing rate - used to identify unnatural frequency changes
        > chroma features captures the musical pitch class content in frequency domain, relative energy of each pitch class (C, C#, D, ...) regardless of octave, computed from STFT!! or CQT
        > ZCR measures temporal rate of waveform sign changes in the time domain, counts the number of times the audio waveform crosses the zero amplitude level per unit time, indicates temporal complexity of signal, (Ex: higher ZCR means more percussive or high-frequency content, lower ZCR suggests sustained tones)
    

After extracting these characteristics AI algorithms can be trained to discriminate between AI voices and real ones, for example CNNs identify spatial patterns within spectrograms, RNNs and Long Short-Term Memory (LSTM) models record temporal correlations in speech patterns, Multi-layer perceptrons (MLP) also help

Modern detection systems usually integrate several AI models, combining a lot of feature extraction methods. Standard ML models can be effective in conditions with effective preprocessing and cross-validation. Ensemble learning has become crucial to preventing overfitting. Models tend to be highly accurate but needing excessive computational power. 

Preprocessing of audio deepfakes means getting useful features in speech recording that allow our deep learning architectures to identify AI generated versus an original audio. Feature extraction focuses on utilising time-frequency representation and statistics to note audio attributes.

Their data preprocessing pipeline:
    1. STFT - converts time-domain signal into frequency-domain 2D rep, by preserving time and frequency info through the Fourier Transform to short, overlapping pieces of a signal. This helps CNNS which use structured 2D reps for pattern recognition a lot.
    2. MFCCs - spectral envelope of audio stream, related to how people perceive speech, RNNs and LSTM networks are tasked with detecting time-series data's sequential relationships. RNN can learn temporal patterns and anomalies in simulated speech since MFCCs retain phonetic and speech structure
    3. 1D Statistical Feature Vectors - they extracted 6 features: ZCR (provides details on speech texture), Root Mean Square (RMS) Energy captures global loudness and amplitude variations of signal, measured pitch of sound is related to the center of mass of the spectrum, or spectral centroid. Measurement of frequencies cluster about the spectral centroid, indicates signal's richness in timbre. Spectral Rolloff shows high-frequency emphasis by locating frequency point below where 85% of the energy in the spectrum is. Spectral Flatness quantifies how noise-like or tonal a signal is by comparing geometric and arithmetic means

# Research from Paper 2 and 3:

- Random Forest Classifier Model is very good especially with using MFCC
- For imitation deepfakes: spectral analysis
- Replay deepfakes: speech flow disruptions, sudden background noise change and emotion change
- Librosa is a good Python library that does MFCC, generates feature extraction functions
  
- Uncompressed and Lossless files like .wav, .flac, high resolution spectral domain analysis
- Compressed containers, looking at compression forensics, MAC timestamps and container metadata

## Forest baseline review and proposed experiments (September 2026)

These proposals remain future work. The figures in this section describe the original two-forest baseline; the optimized SVM/logistic blend and its comparison are documented in `README.md`. The labeled holdout has 147 real and 146 synthetic clips. At the baseline's 0.5 cutoff it has 132 true negatives, 15 false positives, 18 false negatives, and 128 true positives: 88.74% accuracy, 10.20% false positive rate, and 12.33% false negative rate. ROC AUC is 0.9482. The separate 1,671 challenge clips have no labels, so they cannot establish accuracy.

For a population with 70% real and 30% synthetic audio, and a false positive (calling real audio AI) costing four times a false negative, compare candidate decisions using `2.8 * false_positive_rate + 0.3 * false_negative_rate`. If the holdout's class-conditional error rates transferred unchanged, the present cutoff would cost 32.27 units per 100 clips; calling everything real would cost 30. This is an illustrative projection, not an estimate measured on that population. Report cost, false positive rate, fake recall, and precision at the intended prevalence alongside AUC. Choose any cutoff using group-disjoint development data, then evaluate it once on an untouched test set; changing a cutoff changes decisions without improving the underlying score ranking. [scikit-learn threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html)

1. **Make validation more independent and representative.** LJ chapters in different splits still share the same speaker and recording corpus, while the present synthetic holdout contains only PlayHT and Pro Diff. Collect more independent real speakers and recording conditions, hold out whole speakers and source corpora, and repeat with different unseen generator families. Aim for labeled evaluation data close to the expected 70/30 mix, or use class-conditional rates to project that mix with uncertainty. More independent real clips are particularly valuable when the goal is a low false positive rate. ASVspoof 2021 found weak cross-dataset generalization in its deepfake task; ASVspoof 5 used diverse acoustic conditions, many speakers, attacks, and speaker-disjoint partitions. [ASVspoof 2021](https://arxiv.org/abs/2210.02437), [ASVspoof 5](https://arxiv.org/abs/2502.08857)
2. **Calibrate scores and select the operating point for the stated cost.** Neither saved classifier's final score has been calibrated to the deployment population; its 0–1 values should not be read as verified probabilities. Fit a calibrator using predictions from groups disjoint from model fitting, compare reliability and cost, and choose the decision cutoff on development data. Sigmoid calibration is a reasonable first comparison with a small calibration set; isotonic calibration can overfit with few examples. A theoretical 0.8 cutoff applies only to an accurately calibrated deployment posterior under the stated 4:1 loss, not to these raw scores. [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html), [calibrator API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
3. **Check for source and metadata shortcuts.** All 73 held-out PlayHT files have `.wav` names but MP3 content; the real holdout files have consistent metadata. This can reward source identification instead of voice authenticity. Compare audio-only and metadata-adjusted scores by generator and corpus, then test both classes after matched container and codec conversion. Also match duration, silence, loudness, and recording channel where practical. Treat coherent metadata as weak provenance evidence, because it is easy to reproduce in synthetic files. [ASVspoof 2021 cross-source findings](https://arxiv.org/abs/2210.02437), [ASVspoof 5 data design](https://arxiv.org/abs/2502.08857)
4. **Only then compare richer audio models.** The present 84-feature vector summarizes whole-clip means and standard deviations, which can hide short manipulated segments. As separately measured experiments, compare short-window or spectrogram features and a temporal anti-spoofing model on the same independent groups and cost metric. Such models may help with local artifacts but may also learn new source shortcuts; an external test is needed before claiming an improvement. [AASIST original paper](https://arxiv.org/abs/2110.01200), [partially fake audio study](https://www.isca-archive.org/interspeech_2021/yi21_interspeech.pdf)
