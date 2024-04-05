# WhaleMoanDetector
 A fine-tuned faster-rCNN for detecting blue and fin whale moans in audio data

Michaela Alksne WhaleMoanDetector readme 04/04/24

This folder contains all of the necessary code to train and test WhaleMoanDetector, a fine-tuned faster-rCNN model for blue and fin whale calls, on wav files. The wav files are not uploaded. Neither is the original model. 

WhaleMoanDetector is trained to identify blue whale A, B, and D calls, and fin whale 20 Hz and 40 Hz calls in 60 second windows. See the attached spectrograms in the "figures" folder for examples of each category. Best performance so far is on A, B, and D calls. See the attached PR curve for a look at test performance (so far).  

If you would like to retrain the model, you need access to the required wav files. The annotations are uploaded here. Steps to retrain WhaleMoanDetector:

1. Start by running the "modify_annotations.py" script to generate start and end times in seconds. This will save the modified annotations into a new subfolder:

	"L:\WhaleMoanDetector\labeled_data\logs\CalCOFI_split_by_deployment\modified_annotations" &
	"L:\WhaleMoanDetector\labeled_data\logs\HARP_split_by_deployment\modified_annotations"

	You'll need to run this script twice pointing to each of the logger.xls files and to your wav files. 

2. Next, run the "make_annotations.py" script to generate spectrograms and bounding box annotations for each call in a given spectrogram.

3. Run "plot_groundtruth" to visualize the annotations and check how they look. 

4. Run "train_test_split.py" to make train and val splits. As of now, I have not uploaded the test data. So just work on train and val. 

5. Once you have made your train and val split, you are ready to train the model! Run "train.py" which will retrain WhaleMoanDetector and save a model every epoch and print the loss value. 

6. Run "test_compute_metrics.py" to generate Precision-Recall curve and calculate mean average precision, and run test_plot_predictions.py" to visualize model predictions in real time. 

7. Run "inference.py" when you are ready to deploy the model on new data and generate predictions. 
