# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:35:17 2024

@author: Michaela Alksne

plot bounding box annotations generated in Python for faster-rCNN 
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display  # Import the display function
import os
# Assuming annotations.csv is located in the proper path
# Load annotations
# annotations = pd.read_csv('L:\\Sonobuoy_faster-rCNN\\labeled_data\\train_val_test_annotations\\train.csv')
annotations = pd.read_csv('C:\\Users\\aryye\\OneDrive\\Documents\\GitHub\\WhaleMoanDetector\\labeled_data\\train_val_test_annotations\\train.csv')


# Function to plot bounding boxes and labels on the spectrograms
def plot_annotated_spectrograms(annotations):
    grouped_annotations = annotations.groupby('spectrogram_path')
    
    for spectrogram_path, group in grouped_annotations:
        # Load the spectrogram image
        image = Image.open(spectrogram_path)
        draw = ImageDraw.Draw(image)  # Create a drawing context
        font = ImageFont.truetype("arial.ttf", 16)  # Adjust the font and size as needed

        # Plot each bounding box and label for this spectrogram
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            label = row['label']
            # Draw rectangle on the image
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='black', width=3)
            # Place a label near the top-left corner of the bounding box, adjust positioning as needed
            draw.text((xmin, ymin-17), label, fill='black', font = font)
        
        file_name = os.path.basename(spectrogram_path)

        # Display the image with bounding boxes and labels
        draw.text((0, 10), "File Path: " + file_name, fill='black', font = font)

        display(image)

# Call the function with the annotations dataframe
plot_annotated_spectrograms(annotations)

annotations.label.value_counts()

