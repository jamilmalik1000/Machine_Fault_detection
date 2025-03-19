# Machine_Fault_detection
Machine Fault Detection is an AI-powered system designed to automatically identify and classify faults in machinery, enhancing maintenance efficiency and minimizing downtime. This project leverages advanced machine learning and deep learning techniques to detect anomalies and predict potential failures from sensor data or images.
## Contributors:
Group Members - Jamil Malik, Wasia Abid & Arooj Fatima.

Supervisor Name - Engineeer Ahmed Khwaja.

# Automatic Machine Fault Detection from Acoustic Data:

This project detects faults in machines using deep learning techniques applied to spectrogram images derived from acoustic data. The model classifies machine faults into four categories: Arcing, Corona, Looseness, and Tracking.

## Datasets:
Source: Provided by the instructor as a ZIP folder containing audio samples.

Processing: Audio files are converted into grayscale spectrogram images of size 128x128 pixels.

## Classes:
Arcing

Corona

Looseness

Tracking
## Model Architecture:
The CNN model consists of:

3 Convolutional layers with ReLU activation

MaxPooling layers to reduce spatial dimensions

Fully connected Dense layers

Dropout for regularization

Softmax activation for classification

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Regularization
    Dense(4, activation='softmax')  # 4 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


## Model Performance:
Metric                :                   Score

Accuracy              :           80.67%

Precision             :           93%

Recall                :          94%

F1-Score              :          93%
## Clone the Repository:
git clone https://github.com/your-repo-link.git
cd your-project-folder

## Install Dependencies:
pip install -r requirements.txt

## Run the Model Training:
python train.py
## Predict a Fault from an Image:
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("fault_detection_model.h5")

# Load an image
img_path = "test_image.png"
img = image.load_img(img_path, target_size=(128, 128), color_mode="grayscale")
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
fault_classes = ["Arcing", "Corona", "Looseness", "Tracking"]
predicted_label = fault_classes[np.argmax(predictions)]

print(f"Predicted Fault: {predicted_label}")

## Tools & Technologies Used:
Google Colab – Model training and testing

TensorFlow/Keras – CNN implementation

Matplotlib – Data visualization

NumPy – Data preprocessing

## How to Use:
Run the model in Google Colab: Ensure TensorFlow and required libraries are installed.

Upload a spectrogram image: The model accepts 128x128 grayscale spectrograms.

Make a prediction:
img = preprocess_image("path_to_image.png")
prediction = model.predict(img)
print("Predicted Fault:", class_labels[np.argmax(prediction)])

## Future Improvements:
Increase dataset size for better generalization.
Experiment with advanced architectures like ResNet.
Implement real-time fault detection from live audio.

## License:
This project is licensed under the University of Azad Jammu and Kashmir.
