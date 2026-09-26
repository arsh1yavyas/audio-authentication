Research from "Ensemble learning model for deepfake audio detection using multi-feature extraction approach":

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
    3. 1D Statistical Feature Vectors - they extracted 6 features: ZCR (provides details on speech texture), Root Mean Square (RMS) Energy captures global loudness and amplitude variations of signal, measured pitch of sound is related to the center of mass of the spectrum, or spectral centroid. Measurement of frequencies cluster about the spectral centroid, indicates signal's richness in timbre. Spectral Rolloff shows high-frequency emphasis by locating frequency point below where 85% of the energy in the spectrum is. Spectral Flatness quantifies how noise-like or tonal a signal is by comparing geometric and arighmetic means 

