import librosa
import numpy as np

def load_audio(file_path, sr=16000):
    """
    Load audio file and resample to 16kHz.
    Uses librosa for loading and ensures compatibility with various formats.
    """
    try:
        # Commented out debug logs to suppress verbose output
        # print(f"Attempting to load audio file: {file_path}")
        audio, _ = librosa.load(file_path, sr=sr)
        if len(audio) == 0:
            # print(f"Audio file is empty or corrupted: {file_path}")
            return None
        # print("Audio loaded successfully.")  # Debug log
        return audio
    except Exception as e:
        # print(f"Error loading audio file {file_path}: {e}")
        return None

def normalize_audio(audio):
    """
    Normalize audio amplitude to prevent distortion.
    """
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        # print("Audio is silent or corrupted. Skipping normalization.")
        return None  # Skip processing if audio is silent
    return audio / max_amplitude

def remove_silence(audio, top_db=20):
    """
    Remove silent parts from audio to improve feature extraction.
    Adjust `top_db` for quieter segments if needed.
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        # print("No non-silent intervals detected. Returning original audio.")
        return audio  # Return the original audio if no intervals are found
    return np.concatenate([audio[start:end] for start, end in intervals])

def preprocess_audio(file_path):
    """
    Full preprocessing pipeline for audio files.
    Includes loading, normalizing, and removing silence.
    """
    # Step 1: Load the audio file
    audio = load_audio(file_path)
    if audio is None:
        return None  # Skip processing if loading fails

    # Step 2: Normalize the audio
    audio = normalize_audio(audio)
    if audio is None:
        return None  # Skip processing if normalization fails

    # Step 3: Remove silence
    audio = remove_silence(audio)

    return audio