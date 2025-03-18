import os
import re
import shutil

import librosa
import numpy as np
import pandas as pd

dataset_path = "Music Data"
data_csv_path = "Music Data/data.csv"


# Split the string into a list of strings and numbers for natural sorting
def natural_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]


def reformat_music_files(path):
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
