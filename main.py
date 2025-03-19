import os
import re
import shutil

import librosa
import numpy as np
import pandas as pd

dataset_path = "Music Data"
data_csv_path = "Music Data/data.csv"


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
            'mood'  # Participant's mood prior to playing
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
        threshold (float): Value above which an emotion is considered present.
        apply_threshold (bool): If True, threshold averaged emotions to produce binary labels.
                                If False, returns soft labels (average values).

    Returns:
        pd.DataFrame: Aggregated DataFrame with one row per song.

    The function groups the data by 'track id' and 'genre', computes the mean for each emotion,
    and then (optionally) thresholds the averages to create a binary multi-label vector per song.
    """
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
    # Group by 'track id' and 'genre', and compute mean of each emotion column
    aggregated = data_subset.groupby(['track id', 'genre'], as_index=False).mean()

    if apply_threshold:
        # Apply thresholding: mark emotion as 1 if its mean exceeds the threshold, else 0.
        for col in emotion_columns:
            aggregated[col] = (aggregated[col] > threshold).astype(int)

    return aggregated
