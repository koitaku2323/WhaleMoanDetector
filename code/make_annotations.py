# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:25:24 2024

@author: Michaela ALksne

Make spectrograms and bounding box annotations for each spectrogram in a given wav file
- starts with "modififed annotations" which contain the start and end times of the annotations in seconds since the start of the wav file
- makes "chunks" for each 60 second window in the wav file (with 30 seconds of overlap)
- finds the annotation time stamps for each chunk and makes a spectrogram around them with bounding box annotations in the format:
    [xmin, ymin, xmax, ymax]
- saves the spectrograms and an annotation csv pointing to each spectrogram and its corresponding bounding box
- if multiple bounding boxes exist per spectrogram, the spectrogram annotation gets repeated row-wise
- loops through whatever files you point it to
    
"""

import glob
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import ImageOps
from PIL import Image, ImageDraw


directory_path = "C:\\Users\\aryye\\OneDrive\\Documents\\GitHub\\WhaleMoanDetector\\labeled_data\\logs\\CalCOFI_split_by_deployment" # point to modified annotation files
all_files = glob.glob(os.path.join(directory_path,'*.csv')) # path for all files



def generate_spectrogram_and_annotations(unique_name_part,annotations_df, output_dir, window_size=60, overlap_size=30, n_fft=48000, hop_length=4800):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_list = []  # To collect annotations for each deployment

    # Group annotations by audio file
    grouped = annotations_df.groupby('audio_file')

    for audio_file_path, group in grouped:
        audio_path = Path(audio_file_path)  # Convert to Path object for easier handling
        audio_basename = audio_path.stem  # Get the basename without the extension

        # Load the audio file
        y, sr = librosa.load(audio_file_path, sr=None)
        samples_per_window = window_size * sr
        samples_overlap = overlap_size * sr

        # Process each chunk of the audio file
        for start_idx in range(0, len(y), samples_per_window - samples_overlap):
            end_idx = start_idx + samples_per_window
            chunk = y[start_idx:end_idx] if end_idx <= len(y) else np.pad(y[start_idx:], (0, end_idx - len(y)), 'constant') #Pad with zeros when we reach the end of the file
            S = librosa.stft(chunk, n_fft=sr, hop_length=int(sr/10))
            S_dB_all = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            S_dB = S_dB_all[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz

            chunk_start_time = start_idx / sr
            chunk_end_time = chunk_start_time + window_size
            
            spectrogram_filename = f"{audio_basename}_second_{int(chunk_start_time)}_to_{int(chunk_end_time)}.png"
        
            # Filter and adjust annotations for this chunk
            relevant_annotations = group[(group['start_time'] >= chunk_start_time) & (group['end_time'] <= chunk_end_time)]

            # Adjust annotation times relative to the start of the chunk
            for _, row in relevant_annotations.iterrows():
                adjusted_start_time = row['start_time'] - chunk_start_time
                adjusted_end_time = row['end_time'] - chunk_start_time

                normalized_S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
                S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L') # apply grayscale colormap
                # Flip the image vertically
                S_dB_img = ImageOps.flip(S_dB_img)
                S_dB_img.save(output_dir / spectrogram_filename)
                # Map annotation times and frequencies to spectrogram pixels
                xmin, xmax = time_to_pixels(adjusted_start_time, adjusted_end_time, S_dB.shape[1], window_size)
                ymin, ymax = freq_to_pixels(row['low_f'], row['high_f'], S_dB.shape[0], sr, sr)

                annotations_list.append({
                    "spectrogram_path": f"{output_dir}\{spectrogram_filename}",
                    "label": row['annotation'],
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
                
                
    # Convert annotations list to DataFrame and save as CSV
    df_annotations = pd.DataFrame(annotations_list)
    df_annotations.to_csv(f"{output_dir}/{unique_name_part}_annotations.csv", index=False)

def time_to_pixels(start_time, end_time, spec_width, window_size):
    """Map start and end times in seconds to pixel positions in the spectrogram."""
    pixels_per_second = spec_width / window_size
    xmin = int(np.floor(start_time * pixels_per_second))
    xmax = int(np.ceil(end_time * pixels_per_second))
    return xmin, xmax

def freq_to_pixels(low_f, high_f, spec_height, sr, n_fft):
    """Map frequencies to pixel positions in the spectrogram, accounting for the inversion."""
    # The frequency range we are mapping into (10 Hz to 150 Hz)
    low_freq_bound = 10
    high_freq_bound = 150

    # Calculate the frequency resolution of the spectrogram
    freq_resolution = (sr / 2) / (n_fft // 2)

    # How many Hz each frequency bin in the spectrogram represents
    freq_per_bin = freq_resolution

    # Calculate the total number of frequency bins
    total_freq_bins = spec_height

    # Calculate pixel position for the low and high frequencies, taking into account the inversion
    # The pixel position is calculated from the bottom (low frequencies) because of the inversion
    ymin = total_freq_bins - int(np.round((high_f - low_freq_bound) / freq_per_bin))
    ymax = total_freq_bins - int(np.round((low_f - low_freq_bound) / freq_per_bin))

    # Ensure ymin and ymax are within the bounds of the spectrogram's height
    ymin = max(0, ymin)
    ymax = min(spec_height - 1, ymax)  # Ensure ymax does not exceed the spectrogram's height
    
    return ymin, ymax

for file in all_files:
    # Parse the unique part of the filename you want to use for naming
    unique_name_part = Path(file).stem.split('_')[0]  # Adjust index as needed
    annotations_df = pd.read_csv(file)
    output_directory = 'C:\\Users\\aryye\\OneDrive\\Documents\\GitHub\\WhaleMoanDetector\\labeled_data\\spectrograms\\CalCOFI_2008_08'

    # Call your function to process the annotations and generate spectrograms
    generate_spectrogram_and_annotations(unique_name_part,annotations_df, output_directory, window_size=60, overlap_size=30)



