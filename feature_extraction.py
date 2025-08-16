import librosa
import numpy as np

def extract_mfcc(audio, sr=16000, n_mfcc=13):
    """Extract MFCC features."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def extract_chroma(audio, sr=16000):
    """Extract chroma features."""
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return np.mean(chroma.T, axis=0)

def extract_spectral_contrast(audio, sr=16000):
    """Extract spectral contrast."""
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    return np.mean(contrast.T, axis=0)

def extract_zero_crossing_rate(audio):
    """Extract zero-crossing rate."""
    zcr = librosa.feature.zero_crossing_rate(audio)
    return np.mean(zcr)

def extract_features(audio, sr=16000):
    """Combine all features into a single vector."""
    mfcc = extract_mfcc(audio, sr)
    chroma = extract_chroma(audio, sr)
    contrast = extract_spectral_contrast(audio, sr)
    zcr = extract_zero_crossing_rate(audio)
    return np.hstack([mfcc, chroma, contrast, zcr])