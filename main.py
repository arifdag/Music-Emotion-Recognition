import json
import os
import re
import shutil

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score

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
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10

        # Get unique tracks and their associated emotion labels
        unique_tracks = np.unique(track_ids)
        track_to_emotions = {}

        for track in unique_tracks:
            # Find indices where this track appears
            track_indices = np.where(track_ids == track)[0]
            # Get the emotion labels for this track (should all be the same)
            track_emotions = y_labels[track_indices[0]]
            track_to_emotions[track] = track_emotions

        # Initialize track lists for each split
        train_tracks = []
        val_tracks = []
        test_tracks = []

        np.random.seed(random_state)

        # For each emotion, ensure proportional representation
        for emotion_idx, emotion in enumerate(emotion_columns):
            # Find tracks that have this emotion
            positive_tracks = [track for track, emotions in track_to_emotions.items()
                               if emotions[emotion_idx] == 1]

            # Skip if no tracks have this emotion
            if not positive_tracks:
                print(f"Warning: No tracks found with emotion '{emotion}'")
                continue

            # Shuffle tracks with this emotion
            np.random.shuffle(positive_tracks)

            # Calculate split sizes
            n_tracks = len(positive_tracks)
            n_train = max(1, int(n_tracks * train_size))  # Ensure at least 1 track in train
            n_val = max(1, int(n_tracks * val_size))  # Ensure at least 1 track in val

            # Ensure at least 1 track in test if there are enough tracks
            if n_tracks > 2:  # We need at least 3 tracks to distribute across 3 splits
                # Adjust to ensure we have at least one track in each split
                if n_train + n_val >= n_tracks:
                    n_train = max(1, n_tracks - 2)
                    n_val = 1

                train_tracks.extend(positive_tracks[:n_train])
                val_tracks.extend(positive_tracks[n_train:n_train + n_val])
                test_tracks.extend(positive_tracks[n_train + n_val:])
            else:
                # If we have only 1-2 tracks, prioritize training data
                train_tracks.extend(positive_tracks)
                print(f"Warning: Only {n_tracks} tracks with emotion '{emotion}', all added to training")

        # Remove duplicates
        train_tracks = list(set(train_tracks))
        val_tracks = list(set(val_tracks))
        test_tracks = list(set(test_tracks))

        # Handle tracks without any of the target emotions
        remaining_tracks = [track for track in unique_tracks
                            if track not in train_tracks and
                            track not in val_tracks and
                            track not in test_tracks]

        # Distribute remaining tracks proportionally
        np.random.shuffle(remaining_tracks)
        n_remaining = len(remaining_tracks)
        n_train_remaining = int(n_remaining * train_size)
        n_val_remaining = int(n_remaining * val_size)

        train_tracks.extend(remaining_tracks[:n_train_remaining])
        val_tracks.extend(remaining_tracks[n_train_remaining:n_train_remaining + n_val_remaining])
        test_tracks.extend(remaining_tracks[n_train_remaining + n_val_remaining:])

        # Create masks for each split
        train_mask = np.isin(track_ids, train_tracks)
        val_mask = np.isin(track_ids, val_tracks)
        test_mask = np.isin(track_ids, test_tracks)

        X_train = X_segments[train_mask]
        y_train = y_labels[train_mask]
        X_val = X_segments[val_mask]
        y_val = y_labels[val_mask]
        X_test = X_segments[test_mask]
        y_test = y_labels[test_mask]

        # Print statistics
        print(f"Training set:   {X_train.shape[0]} samples from {len(train_tracks)} tracks")
        print(f"Validation set: {X_val.shape[0]} samples from {len(val_tracks)} tracks")
        print(f"Test set:       {X_test.shape[0]} samples from {len(test_tracks)} tracks")

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
        # 'binary_crossentropy' is used as the loss function, but focal loss can be an alternative.
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


def evaluate_model(model, X_test, y_test, emotion_columns):
    """
    Evaluate model with multiple metrics for multi-label classification.

    Parameters:
        model: Trained model
        X_test: Processed test data
        y_test: Test labels
        emotion_columns: List of emotion column names

    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)

    # Use 0.5 as threshold for predictions
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'hamming_loss': hamming_loss(y_test, y_pred),
        'sample_f1': f1_score(y_test, y_pred, average='samples'),
        'micro_f1': f1_score(y_test, y_pred, average='micro'),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'precision_micro': precision_score(y_test, y_pred, average='micro'),
        'recall_micro': recall_score(y_test, y_pred, average='micro')
    }

    # Per-emotion metrics
    for i, emotion in enumerate(emotion_columns):
        metrics[f'{emotion}_f1'] = f1_score(y_test[:, i], y_pred[:, i])
        metrics[f'{emotion}_precision'] = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
        metrics[f'{emotion}_recall'] = recall_score(y_test[:, i], y_pred[:, i])

    return metrics


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance.

    Parameters:
        gamma: Focusing parameter
        alpha: Balancing parameter

    Returns:
        loss_function: Focal loss function
    """

    def focal_loss_fixed(y_true, y_pred):
        # Clip prediction values to avoid log(0)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Calculate standard binary cross-entropy loss
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # Calculate focal loss components
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        # Calculate focal loss
        loss = alpha_factor * modulating_factor * bce

        return tf.reduce_mean(loss)

    return focal_loss_fixed


def main():
    """
        Executes the music emotion recognition pipeline by loading and preprocessing data,
        extracting features, training and evaluating the model, and saving the results.

        Returns:
            tuple: A tuple containing the trained model, training history, and evaluation metrics.
        """
    print("Starting music emotion recognition pipeline...")

    # Load and reformat the data
    print("Loading and reformatting data...")
    data_subset = reformat_data()
    if data_subset is None:
        print("Failed to load data. Exiting.")
        return
    print(f"Data loaded successfully. Shape: {data_subset.shape}")

    # Aggregate data
    print("Aggregating annotations...")
    aggregated_data = aggregate_data(data_subset, emotion_columns)
    print(f"Data aggregated successfully. Shape: {aggregated_data.shape}")

    # Extract audio segments and their labels
    print("Extracting audio segments...")
    audio_path = os.path.join(dataset_path, 'Musics')
    X_segments, y_labels, track_ids = extract_audio_segments_and_labels(audio_path, aggregated_data, segment_length=5)
    if X_segments is None or y_labels is None:
        print("Failed to extract audio segments. Exiting.")
        return
    print(f"Audio segments extracted. Shape: {X_segments.shape}, Labels shape: {y_labels.shape}")

    # Extract features
    print("Extracting features...")
    features = extract_features(X_segments, augment=False)
    print(f"Features extracted. Shape: {features.shape}")

    # Split the data
    print("Splitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_track(
        features, y_labels, track_ids, train_size=0.7, val_size=0.15, test_size=0.15
    )

    # Preprocess data
    print("Preprocessing data...")
    X_train_proc, X_val_proc, X_test_proc = preprocess_data(X_train, X_val, X_test)
    print(f"Preprocessed data shapes: {X_train_proc.shape}, {X_val_proc.shape}, {X_test_proc.shape}")

    # Check the final input shape for the model
    input_shape = X_train_proc.shape[1:]
    num_emotions = y_train.shape[1]
    print(f"Model input shape: {input_shape}, Number of emotions: {num_emotions}")

    # Train the model
    print("Training the model...")
    model, history = train_model(
        X_train_proc,
        y_train,
        X_val_proc,
        y_val,
        input_shape,
        num_emotions,
        epochs=50,
        batch_size=32,
        patience=10
    )

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test_proc, y_test, emotion_columns)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.3f}")

    # Save the model
    model.save("music_emotion_model.keras")
    print("Model saved successfully.")

    # Save the training history
    with open('Music_emotion_history.json', 'w') as f:
        json.dump(history.history, f)
        print("Training history saved successfully.")

    # Save the metrics
    with open('Music_emotion_metrics.json', 'w') as f:
        json.dump(metrics, f)
        print("Metrics saved successfully.")

    return model, history, metrics


if __name__ == "__main__":
    main()
