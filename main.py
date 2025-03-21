import os
import re
import shutil

import librosa
import numpy as np
import pandas as pd

dataset_path = "Music Data"
data_csv_path = "Music Data/data.csv"

# List of emotion annotation columns
emotion_columns = [
    'amazement',
    'solemnity',
    'tenderness',
    'nostalgia',
    'calmness',
    'power',
    'joyful_activation',
    'tension',
    'sadness'
]


def natural_key(string):
    """
    Splits a string into a list of strings and numbers for natural sorting.

    Parameters:
        string (str): The input string that may contain digits.

    Returns:
        list: A list containing substrings and integer conversions of digits,
              allowing for natural (human-friendly) sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]


def reformat_music_files(path):
    """
    Copies and renames music files from genre subdirectories into a unified directory.

    Parameters:
        path (str): Base directory path containing the genre folders.

    The function iterates over predefined genre folders, sorts the files naturally,
    and copies them into a "Musics" folder with sequential naming (e.g., "1.mp3", "2.mp3", etc.).
    """
    track_id = 1
    copy_path = os.path.join(path, 'Musics')
    os.makedirs(copy_path, exist_ok=True)

    genre_list = ["classical", "rock", "electronic", "pop"]

    for genre in genre_list:
        genre_path = os.path.join(path, genre)

        # Iterate through the files in the genre directory and copy with new names
        files = sorted(os.listdir(genre_path), key=natural_key)
        for file in files:
            source_file = os.path.join(genre_path, file)
            if os.path.isfile(source_file):
                new_name = f"{track_id}.mp3"
                new_file_path = os.path.join(copy_path, new_name)
                shutil.copyfile(source_file, new_file_path)
                track_id += 1


def reformat_data():
    """
    Reads and reformats the CSV data file for the Emotify dataset.

    Returns:
        pd.DataFrame or None: DataFrame containing only the desired columns if successful,
                              otherwise returns None.

    The function reads the CSV file at 'data_csv_path', strips extra spaces from column names,
    and selects a subset of columns that are relevant for analysis.
    """
    try:
        data_file = pd.read_csv(data_csv_path)

        # List the columns you want to keep:
        columns_to_use = [
            'track id',  # Music file identifier
            'genre',  # Genre of the music file
            'amazement',  # Emotion annotation 1
            'solemnity',  # Emotion annotation 2
            'tenderness',  # Emotion annotation 3
            'nostalgia',  # Emotion annotation 4
            'calmness',  # Emotion annotation 5
            'power',  # Emotion annotation 6
            'joyful_activation',  # Emotion annotation 7
            'tension',  # Emotion annotation 8
            'sadness',  # Emotion annotation 9
        ]

        # Strip any leading/trailing spaces in column names
        data_file.columns = data_file.columns.str.strip()

        # Create a new DataFrame with only the desired columns
        data_subset = data_file[columns_to_use]

        return data_subset

    except Exception as e:
        print(f"Error: {e}")
        return None


def aggregate_data(data_subset, threshold=0.4, apply_threshold=True):
    """
    Aggregates participant annotations into a single label per song.

    Parameters:
        data_subset (pd.DataFrame): DataFrame containing the original annotations.
        threshold (float): Value above which an emotion is considered present (Default: 0.4).
        apply_threshold (bool): If True, threshold averaged emotions to produce binary labels.
                                If False, returns soft labels (average values).

    Returns:
        pd.DataFrame: Aggregated DataFrame with one row per song.

    The function groups the data by 'track id' and 'genre', computes the mean for each emotion,
    and then (optionally) thresholds the averages to create a binary multi-label vector per song.
    """

    # Group by 'track id' and 'genre', and compute mean of each emotion column
    aggregated = data_subset.groupby(['track id', 'genre'], as_index=False).mean()

    if apply_threshold:
        # Apply thresholding: mark emotion as 1 if its mean exceeds the threshold, else 0.
        for col in emotion_columns:
            aggregated[col] = (aggregated[col] > threshold).astype(int)

    return aggregated


def extract_audio_segments_and_labels(audio_path, aggregated_data, sr=44100, segment_length=5):
    """
    Load audio files from a directory, segment each into fixed-length chunks,
    and associate each segment with its corresponding labels.

    Parameters:
        audio_path (str): Directory containing audio files.
        aggregated_data (pd.DataFrame): DataFrame containing emotion labels for each track ID.
        sr (int): Sampling rate (default: 44100).
        segment_length (int): Duration in seconds for each segment (default: 5).

    Returns:
        tuple: (X, y, track_ids) where:
            X (np.ndarray): Array of audio segments with shape (n_segments, segment_length * sr).
            y (np.ndarray): Array of corresponding emotion labels with shape (n_segments, n_emotions).
            track_ids (np.ndarray): Array of track IDs for each segment.
    """
    X_segments = []
    y_labels = []
    track_ids = []

    try:
        track_id = 1
        audios = sorted(os.listdir(audio_path), key=natural_key)

        for audio in audios:
            y_audio, sr = librosa.load(os.path.join(audio_path, audio), sr=sr, mono=True)
            segment_samples = segment_length * sr

            # Split audio into segments; only include segments of exact desired length
            segments = [y_audio[i:i + segment_samples] for i in range(0, len(y_audio), segment_samples)
                        if len(y_audio[i:i + segment_samples]) == segment_samples]

            if segments:  # Only add to results if we have valid segments
                if track_id in aggregated_data['track id'].values:
                    # Extract emotion labels for this track
                    track_labels = aggregated_data[aggregated_data['track id'] == track_id][emotion_columns].values[0]

                    for segment in segments:
                        X_segments.append(segment)
                        y_labels.append(track_labels)
                        track_ids.append(track_id)
                else:
                    print(f"Warning: No labels found for track ID {track_id}")

            track_id += 1

        # Convert lists to numpy arrays
        X_segments = np.array(X_segments)
        y_labels = np.array(y_labels)
        track_ids = np.array(track_ids)

        return X_segments, y_labels, track_ids

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


def extract_features(audio_segments, sr=44100, n_mels=128, augment=False):
    """Extract combined audio features (mel spectrogram, MFCC, spectral contrast) with optional augmentation.

    Args:
        audio_segments: Input audio segments as numpy arrays
        sr: Sample rate (default: 44100)
        n_mels: Number of mel bands (default: 128)
        augment: Enable pitch-shift augmentation (default: False)

    Returns:
        np.ndarray: Combined features array. Shape: (samples, features, time_steps)
        When augmented, returns original and pitch-shifted versions concatenated
    """
    try:
        features = []

        for segment in audio_segments:
            # Basic feature - mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_mels=n_mels,
                n_fft=2048,
                hop_length=512,
                fmin=20,
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # MFCC
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=20)

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)

            # Combine features
            combined = np.vstack([mel_spec_db, mfcc, contrast])

            # Optional augmentation
            if augment:
                # Pitch shift (mild)
                segment_shifted = librosa.effects.pitch_shift(segment, sr=sr, n_steps=1)
                mel_spec_shifted = librosa.feature.melspectrogram(
                    y=segment_shifted, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
                )
                mel_spec_db_shifted = librosa.power_to_db(mel_spec_shifted, ref=np.max)
                mfcc_shifted = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec_shifted), n_mfcc=20)
                contrast_shifted = librosa.feature.spectral_contrast(y=segment_shifted, sr=sr)
                combined_shifted = np.vstack([mel_spec_db_shifted, mfcc_shifted, contrast_shifted])

                features.append(combined)
                features.append(combined_shifted)
            else:
                features.append(combined)

        return np.array(features)
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None


def split_data_by_track(X_segments, y_labels, track_ids, train_size=0.7, val_size=0.15, test_size=0.15,
                        random_state=42):
    """
    Split data into training, validation, and test sets while ensuring all segments
    from the same track stay together in the same set.

    Parameters:
        X_segments (np.ndarray): Array of audio segments or extracted features.
        y_labels (np.ndarray): Array of corresponding emotion labels.
        track_ids (np.ndarray): Array of track IDs for each segment.
        train_size (float): Proportion of data to use for training (default: 0.7).
        val_size (float): Proportion of data to use for validation (default: 0.15).
        test_size (float): Proportion of data to use for testing (default: 0.15).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    try:
        # Ensure proportions sum to 1
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split proportions must sum to 1"

        # Get unique track IDs
        unique_tracks = np.unique(track_ids)

        # Shuffle tracks
        np.random.seed(random_state)
        np.random.shuffle(unique_tracks)

        # Split tracks according to proportions
        n_tracks = len(unique_tracks)
        n_train = int(n_tracks * train_size)
        n_val = int(n_tracks * val_size)

        train_tracks = unique_tracks[:n_train]
        val_tracks = unique_tracks[n_train:n_train + n_val]
        test_tracks = unique_tracks[n_train + n_val:]

        # Create masks for each split
        train_mask = np.isin(track_ids, train_tracks)
        val_mask = np.isin(track_ids, val_tracks)
        test_mask = np.isin(track_ids, test_tracks)

        # Apply masks to get split datasets
        X_train = X_segments[train_mask]
        y_train = y_labels[train_mask]

        X_val = X_segments[val_mask]
        y_val = y_labels[val_mask]

        X_test = X_segments[test_mask]
        y_test = y_labels[test_mask]

        # Print split information
        print(f"Data split complete by track ID:")
        print(
            f"  Training set:   {X_train.shape[0]} samples from {len(train_tracks)} tracks ({X_train.shape[0] / X_segments.shape[0] * 100:.1f}%)")
        print(
            f"  Validation set: {X_val.shape[0]} samples from {len(val_tracks)} tracks ({X_val.shape[0] / X_segments.shape[0] * 100:.1f}%)")
        print(
            f"  Test set:       {X_test.shape[0]} samples from {len(test_tracks)} tracks ({X_test.shape[0] / X_segments.shape[0] * 100:.1f}%)")

        # Print label distribution in each split
        print("\nLabel distribution across splits:")
        for i, emotion in enumerate(emotion_columns):
            train_dist = np.mean(y_train[:, i])
            val_dist = np.mean(y_val[:, i])
            test_dist = np.mean(y_test[:, i])
            total_dist = np.mean(y_labels[:, i])
            print(
                f"  {emotion:<20}: Total: {total_dist:.3f}, Train: {train_dist:.3f}, Val: {val_dist:.3f}, Test: {test_dist:.3f}")

        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        print(f"Error in split_data_by_track: {e}")
        return None, None, None, None, None, None
