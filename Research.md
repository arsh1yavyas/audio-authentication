
=======
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

# Research from "Ensemble learning model for deepfake audio detection using multi-feature extraction approach":

Feature extraction is the first important step for detecting deepfake audio, popular techniques are short-time Fourier transform (STFT) and Mel-frequency cepstral coefficients (MFCC), chroma features, and zero-crossing rate (ZCR).
    1. STFT - decomposes audio signals into time-divided frequency information, so in short produces a time vs frequency 2D representation called a spectrogram that is used for analysis on audio
    2. MFCC - used for speech recognition, provides a low-dimensional representation of the short-term spectral envelope of audio, it does this by imitating human auditory perception and the way that human hearing has finer resolution at low frequencies over high frequencies
    3. Chroma features and zero-crossing rate - used to identify unnatural frequency changes
        > chroma features captures the musical pitch class content in frequency domain, relative energy of each pitch class (C, C#, D, ...) regardless of octave, computed from STFT!! or CQT
        > ZCR measures temporal rate of waveform sign changes in the time domain, counts the number of times the audio waveform crosses the zero amplitude level per unit time, indicates temporal complexity of signal, (Ex: higher ZCR means more percussive or high-frequency content, lower ZCR suggests sustained tones)
    

After extracting these characteristics AI algorithms can be trained to discriminate between AI voices and real ones, for example CNNs identify spatial patterns within spectrograms, RNNs and Long Short-Term Memory (LSTM) models record temporal correlations in speech patterns, Multi-layer perceptrons (MLP) also help

Modern detection systems usually integrate several AI models, combining a lot of feature extraction methods. Standard ML models can be effective in conditions with effective preprocessing and cross-validation. Ensemble learning has become crucial to preventing overfitting. Models tend to be highly accurate but needing excessive computational power. 

Preprocessing of audio deeppfakes means getting useful features in speech recording that allow our deep learning architectures to identify AI generated versus an original audio. Feature extraction focuses on utilising time-frequency representation and statistics to note audio attributes.

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
- Im so sleepy right now, going to bed ಥ_ಥ
