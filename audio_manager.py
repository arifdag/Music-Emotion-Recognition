import os
import shutil
import librosa
import numpy as np
from utils import natural_key


class AudioManager:
    """
    Class for managing audio file operations such as reformatting, segmentation,
    and feature extraction.
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def reformat_music_files(self):
        """
        Copies and renames music files from genre subdirectories into a unified directory.
        """
        track_id = 1
        copy_path = os.path.join(self.dataset_path, "Musics")
        os.makedirs(copy_path, exist_ok=True)

        genre_list = ["classical", "rock", "electronic", "pop"]
        for genre in genre_list:
            genre_path = os.path.join(self.dataset_path, genre)

            # Iterate through the files in the genre directory and copy with new names
            files = sorted(os.listdir(genre_path), key=natural_key)
            for file in files:
                source_file = os.path.join(genre_path, file)
                if os.path.isfile(source_file):
                    new_name = f"{track_id}.mp3"
                    new_file_path = os.path.join(copy_path, new_name)
                    shutil.copyfile(source_file, new_file_path)
                    track_id += 1

    def extract_audio_segments_and_labels(self, audio_path, aggregated_data, emotion_columns, sr=44100,
                                          segment_length=5):
        """
        Load audio files from a directory, segment each into fixed-length chunks,
        and associate each segment with its corresponding labels.

        Parameters:
            audio_path (str): Directory containing audio files.
            aggregated_data (pd.DataFrame): DataFrame with emotion labels.
            emotion_columns (list): List of emotion column names.
            sr (int): Sampling rate.
            segment_length (int): Duration in seconds for each segment.

        Returns:
            tuple: (X_segments, y_labels, track_ids) as numpy arrays.
        """
        X_segments = []
        y_labels = []
        track_ids = []

        try:
            track_id = 1
            audios = sorted(os.listdir(audio_path), key=natural_key)
            for audio in audios:
                y_audio, _ = librosa.load(os.path.join(audio_path, audio), sr=sr, mono=True)
                segment_samples = segment_length * sr

                # Split audio into segments; only include segments of exact desired length
                segments = [
                    y_audio[i: i + segment_samples]
                    for i in range(0, len(y_audio), segment_samples)
                    if len(y_audio[i: i + segment_samples]) == segment_samples
                ]
                if segments:  # Only add to results if we have valid segments
                    if track_id in aggregated_data["track id"].values:
                        # Extract emotion labels for this track
                        track_labels = aggregated_data[aggregated_data["track id"] == track_id][emotion_columns].values[
                            0]
                        for segment in segments:
                            X_segments.append(segment)
                            y_labels.append(track_labels)
                            track_ids.append(track_id)
                    else:
                        print(f"Warning: No labels found for track ID {track_id}")

                track_id += 1

            X_segments = np.array(X_segments)
            y_labels = np.array(y_labels)
            track_ids = np.array(track_ids)

            return X_segments, y_labels, track_ids
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def extract_features(self, audio_segments, sr=44100, n_mels=128, augment=False):
        """
        Extract combined audio features from segments with optional augmentation.

        Parameters:
            audio_segments (list or np.ndarray): Audio segments.
            sr (int): Sampling rate.
            n_mels (int): Number of mel bands.
            augment (bool): Whether to perform pitch-shift augmentation.

        Returns:
            np.ndarray: Array of combined features.
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
