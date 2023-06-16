# ASL Alphabet Recognition using Transfer Learning

This project focuses on developing a machine learning model that can recognize the American Sign Language (ASL) alphabet using transfer learning. The model is trained on a dataset of ASL alphabet images and utilizes the VGG16 pre-trained model for feature extraction.

## Installation

To run this project, follow these steps:

1. Clone the repository:
git clone https://github.com/TranslatorBahasaIsyarat/TASYA-ML.git


2. Install the required dependencies:
pip install tensorflow opencv-python scikit-image matplotlib


## Data Preparation

The dataset used for training the model is the ASL Alphabet dataset, which contains images of hand signs representing letters of the alphabet. The dataset is organized into folders, with each folder representing a different letter or symbol.

The dataset directory structure is as follows:
dataset/
└── asl_alphabet_train/
├── A/
├── B/
├── C/
├── ...
├── space/
├── del/
└── nothing/


Each letter folder contains multiple images of that particular letter sign.

To prepare the data, the `get_data` function is used to load and preprocess the images. The images are resized to a consistent size of 64x64 pixels and converted to a numpy array. The function returns the processed image data (`X`) and corresponding labels (`y`).

## Model Architecture

The model architecture consists of two parts: an initial convolutional neural network (CNN) model and a pre-trained VGG16 model.

The initial CNN model is designed to capture low-level features from the input images. It consists of three sets of convolutional, activation, and max-pooling layers.

The pre-trained VGG16 model is used for feature extraction. The model is loaded with pre-trained weights from the ImageNet dataset and its fully connected layers are excluded.

The outputs of both models are then combined and passed through fully connected layers to make predictions for the 29 different classes (letters and symbols) of the ASL alphabet.

## Training Process

The model is trained using the Adam optimizer and categorical cross-entropy loss. Early stopping is applied with a patience of 2 epochs to prevent overfitting. The training is performed on a training set, and the model's performance is evaluated on a separate test set.

## Evaluation

The trained model is evaluated on the test set using the categorical cross-entropy loss and accuracy metrics. The evaluation results provide an understanding of the model's performance in classifying the ASL alphabet.

## Model Export

The trained model is saved in the HDF5 format as `model-with-transfer-learning.h5`. Additionally, a TensorFlow Lite version of the model is created and saved as `model-with-transfer-learning.tflite`. These exported models can be used for inference on different platforms.

## Usage

To use the trained model for inference, follow these steps:

1. Load the model:
   ```python
   import tensorflow as tf

   model = tf.keras.models.load_model('model-with-transfer-learning.h5')

2. Preprocess the input image:
```python
#Assuming `image` is a NumPy array containing the image
#Resize the image to the required input shape
processed_image = skimage.transform.resize(image, (64, 64, 3))
```

4. Make predictions:
```python
import numpy as np

# Expand dimensions to match the model's input shape
input_image = np.expand_dims(processed_image, axis=0)

# Perform inference
predictions = model.predict(input_image)

# Get the predicted class label
predicted_class = np.argmax(predictions)
```
