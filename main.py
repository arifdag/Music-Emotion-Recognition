import os
import re
import shutil

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

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


def aggregate_data(data_subset, emotion_columns):
    """
    Aggregate emotion annotations with dynamic thresholds.

    Parameters:
        data_subset (pd.DataFrame): DataFrame with emotion annotations.
        emotion_columns (list): List of emotion column names.

    Returns:
        pd.DataFrame: Aggregated DataFrame with binary emotion labels.
    """
    aggregated = data_subset.groupby(['track id', 'genre'], as_index=False).mean()

    # Find optimal threshold for each emotion based on distribution
    thresholds = {}
    for emotion in emotion_columns:
        values = aggregated[emotion].values
        # Use the median as threshold if the distribution is skewed
        if np.std(values) > 0.2:
            threshold = np.median(values)
        else:
            # Otherwise use a fixed threshold
            threshold = 0.4
        thresholds[emotion] = threshold
        print(f"Emotion: {emotion}, Threshold: {threshold:.3f}")

    # Apply thresholds
    for emotion in emotion_columns:
        aggregated[emotion] = (aggregated[emotion] > thresholds[emotion]).astype(int)

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


def preprocess_data(X_train, X_val, X_test):
    """
    Standardize input data using training statistics and reshape 3D arrays for 2D CNN.

    Parameters:
        X_train (np.ndarray): Training data.
        X_val (np.ndarray): Validation data.
        X_test (np.ndarray): Test data.

    Returns:
        tuple: Preprocessed training, validation, and test data.
    """
    # Calculate mean and std from training data
    mean = np.mean(X_train)
    std = np.std(X_train)

    # Standardize all sets using training statistics
    X_train_norm = (X_train - mean) / (std + 1e-6)
    X_val_norm = (X_val - mean) / (std + 1e-6)
    X_test_norm = (X_test - mean) / (std + 1e-6)

    # Reshape for 2D CNN if necessary
    if len(X_train_norm.shape) == 3:
        # For spectrograms (assuming shape is [samples, features, time])
        # Reshape to [samples, features, time, 1] for 2D CNN
        X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], X_train_norm.shape[1], X_train_norm.shape[2], 1)
        X_val_norm = X_val_norm.reshape(X_val_norm.shape[0], X_val_norm.shape[1], X_val_norm.shape[2], 1)
        X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], X_test_norm.shape[2], 1)

    return X_train_norm, X_val_norm, X_test_norm

def build_model(input_shape, num_emotions, dropout_rate=0.5):
    """
    Build a CNN model for music emotion recognition.

    Parameters:
        input_shape (tuple): Shape of input features.
        num_emotions (int): Number of emotion classes.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),

        # First Conv2D block
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate / 2),

        # Second Conv2D block
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate / 2),

        # Third Conv2D block
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        # Output layer
        tf.keras.layers.Dense(num_emotions, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # Binary crossentropy for multi-label classification
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model


def train_model(X_train, y_train, X_val, y_val, input_shape, num_emotions, epochs=50, batch_size=32,
                patience=10):
    """
    Train the CNN model with data augmentation, learning rate scheduling, and early stopping.

    Parameters:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        input_shape (tuple): Shape of input features.
        num_emotions (int): Number of emotion labels.
        epochs (int): Maximum training epochs.
        batch_size (int): Batch size.
        patience (int): Early stopping patience.

    Returns:
        tuple: Trained model and training history.
    """
    # Data augmentation if train set is small
    if X_train.shape[0] < 1000:
        print("Data augmentation in progress...")
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    model = build_model(input_shape, num_emotions)

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # Model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train model
    if X_train.shape[0] < 1000:
        # Use data augmentation
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[lr_scheduler, early_stopping, model_checkpoint],
            verbose=1
        )
    else:
        # Train without augmentation for larger datasets
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_scheduler, early_stopping, model_checkpoint],
            verbose=1
        )

    return model, history
