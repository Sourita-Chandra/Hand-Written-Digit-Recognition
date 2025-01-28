# Hand-Written-Digit-Recognition

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for recognizing hand-written digits using the MNIST dataset. The model is built using Keras and TensorFlow, employing the LeNet-5 architecture, a classic model for image classification tasks.

## Key Features:
Data Preprocessing: The MNIST dataset is preprocessed by reshaping the images to 28x28 pixels, normalizing the pixel values, and converting the labels to one-hot encoding.
LeNet-5 Architecture: The model consists of convolutional layers for feature extraction, followed by fully connected layers for classification.
Conv2D layers are used for extracting features from the images.
AveragePooling2D layers are used for downsampling.
The final fully connected layers classify the images into one of 10 classes (digits 0-9).
Model Training: The model is trained on a training dataset, and validation is performed on a validation dataset to prevent overfitting.
Evaluation: The model is evaluated on a test dataset, with performance metrics such as accuracy being displayed.
## Steps:
Data Loading: The MNIST dataset is loaded using Keras' datasets.mnist.load_data() function.
Preprocessing: Images are reshaped and normalized, while the labels are converted to one-hot encoding.
Model Construction: The LeNet-5 architecture is implemented with convolutional and pooling layers.
Training: The model is trained for 20 epochs, with a batch size of 32, using the Adam optimizer and categorical crossentropy loss.
Evaluation: The model's accuracy on the test dataset is printed, and a few sample predictions are visualized using matplotlib.
## Requirements:
-Python 3.x
-TensorFlow 2.x
-Keras
-Matplotlib
-Numpy
