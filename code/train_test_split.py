# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:01:09 2024

@author: Michaela Alksne

make train and test splits for faster rCNN model for sonobuoys

"""
import glob
import os
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


HARP_directory_path = "L:\\Sonobuoy_faster-rCNN\\labeled_data\\spectrograms\\HARP" # point to spectrogram annotation files
CalCOFI_directory_path = "L:\\Sonobuoy_faster-rCNN\\labeled_data\\spectrograms\\CalCOFI"
all_HARP = glob.glob(os.path.join(HARP_directory_path,'*.csv')) # path for all files
all_CalCOFI = glob.glob(os.path.join(CalCOFI_directory_path,'*.csv'))

all_files = all_HARP.copy()
all_files.extend(all_CalCOFI)


# Group the data by image name and get unique image names
image_groups = all_files.groupby('spectrogram_path').groups.keys()

# Split the unique image names into train and validation sets
train_images, val_images = train_test_split(list(image_groups), test_size=0.1, random_state=0)

# Filter the data based on the train and validation image names
train_data = all_files[all_files['ImageName'].isin(train_images)]
val_data = all_files[all_files['ImageName'].isin(val_images)]

train_data.CallType.value_counts()
val_data.CallType.value_counts()

train_data.to_csv('L:\\Sonobuoy_faster-rCNN\\labeled_data\\train_val_test_annotations\\train.csv')
val_data.to_csv('L:\\Sonobuoy_faster-rCNN\\labeled_data\\train_val_test_annotations\\validation.csv')


