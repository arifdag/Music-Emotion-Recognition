# Music Emotion Recognition System

## Overview
This repository contains a comprehensive system for music emotion recognition (MER) using deep learning. The pipeline processes audio files, extracts acoustic features, and trains a convolutional neural network to predict multiple emotion labels for music tracks.

## Repository Structure

- `audio_manager.py`: Audio file operations including reformatting, segmentation, and feature extraction
- `augmentor.py`: Class balancing using SMOTE to address underrepresented emotions
- `data_manager.py`: CSV data operations including reformatting and aggregating emotion annotations
- `dataset_manager.py`: Dataset splitting and preprocessing while maintaining track integrity
- `model_manager.py`: CNN model building, training, and evaluation
- `utils.py`: Utility functions
- `main.py`: Main pipeline that orchestrates the entire workflow


## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/music-emotion-recognition.git
cd music-emotion-recognition

# Install dependencies
pip install tensorflow numpy pandas librosa scikit-learn imbalanced-learn
```

## Usage
1. Prepare your dataset with audio files organized in genre subdirectories and corresponding emotion annotations in a CSV file
2. Update the paths in `main.py` to match your dataset structure
3. Run the pipeline:
```bash
python main.py
```

## Dataset Requirements
- Audio files organized in genre subdirectories
- CSV file with emotion annotations for each track
- The CSV should contain columns for track ID, genre, and emotion annotations
