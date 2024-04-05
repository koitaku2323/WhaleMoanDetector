# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:41:34 2024

@author: Michaela Alksne
"""

import librosa
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.colors import Normalize
import torch
from PIL import Image, ImageDraw
import torchvision.ops as ops
from PIL import ImageOps
from IPython.display import display


# Function to load audio file and chunk it into overlapping windows
def chunk_audio(file_path, window_size=60, overlap_size=30):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    # Calculate the number of samples per window
    samples_per_window = window_size * sr
    samples_overlap = overlap_size * sr
    # Calculate the number of chunks
    chunks = []
    for start in range(0, len(y), samples_per_window - samples_overlap):
        end = start + samples_per_window
        # If the last chunk is smaller than the window size, pad it with zeros
        if end > len(y):
            y_pad = np.pad(y[start:], (0, end - len(y)), mode='constant')
            chunks.append(y_pad)
        else:
            chunks.append(y[start:end])
    return chunks

# Function to convert audio chunks to spectrograms
def audio_to_spectrogram(chunks, sr, n_fft=48000, hop_length=4800):
    spectrograms = []
    for chunk in chunks:
        # Use librosa to compute the spectrogram
        S = librosa.stft(chunk, n_fft=sr, hop_length=int(sr/10))
        # Convert to dB
        S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        S_dB_restricted = S_dB[10:151, :]  # 151 is exclusive, so it includes up to 150 Hz
        spectrograms.append(S_dB_restricted)
    return spectrograms

# Function to preprocess and predict on spectrograms
def predict_on_spectrograms(spectrograms, model):
    predictions = []
    for S_dB in spectrograms:
        # Preprocess the spectrogram
        
        normalized_S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
        # Apply colormap (using matplotlib's viridis)
        S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L') # apply grayscale colormap
        # Flip the image vertically
        S_dB_img = ImageOps.flip(S_dB_img)        
        
        # Example: Convert to PyTorch tensor, resize, normalize
        # This needs to be adjusted based on your model's requirements
        S_dB_tensor = F.to_tensor(S_dB_img).unsqueeze(0)  # Add batch dimension
        #S_dB_tensor = torch.tensor(S_dB_img).permute(2, 0, 1).unsqueeze(0).float()  # Rearrange dimensions to CxHxW and add batch dimension

        
        # Run prediction
        model.eval()
        with torch.no_grad():
            prediction = model(S_dB_tensor)
        predictions.append(prediction)
    return predictions

# Load your trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 6 # 5 classes plus background
in_features = model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)

model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
model.load_state_dict(torch.load('L:\\Sonobuoy_faster-rCNN\\trained_model\\Sonobuoy_model_epoch_14.pth'))
model.eval()

# File path
file_path = 'L:/Sonobuoy_faster-rCNN/labeled_data/wav/CalCOFI-Sonobuoy/CalCOFI-2019/CC1907BH_DF_SB03_190712-190000.wav'


def visualize_predictions_on_spectrograms(spectrograms, predictions):
    for S_dB, prediction in zip(spectrograms, predictions):
        # Normalize spectrogram
        normalized_S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
        # Apply colormap (using matplotlib's viridis)
        S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L') # apply grayscale colormap
        # Flip the image vertically
        S_dB_img = ImageOps.flip(S_dB_img) 
        # Flip the image vertically
        
        draw = ImageDraw.Draw(S_dB_img)
        
        # Assuming `prediction` contains `boxes`, `labels`, and `scores`
        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']
        
        # Apply Non-Maximum Suppression (NMS) for cleaner visualization
        keep_indices = ops.nms(boxes, scores, 0.2)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # Draw each bounding box and label on the spectrogram image
        for box, score, label in zip(boxes, scores, labels):
            score_formatted = round(score.item(), 2)
            # Convert box coordinates (considering the flip if necessary)
            box = box.tolist()
            draw.rectangle(box, outline="black")
            draw.text((box[0], box[1]-10), f"Label: {label}, Score: {score_formatted}", fill="black")
        
        # Display the spectrogram with drawn predictions
       #S_dB_img.show()
        display(S_dB_img)




chunks = chunk_audio(file_path)
spectrograms = audio_to_spectrogram(chunks, sr=librosa.get_samplerate(file_path))
predictions = predict_on_spectrograms(spectrograms, model)
# Visualize the predictions on the spectrograms
visualize_predictions_on_spectrograms(spectrograms, predictions)





