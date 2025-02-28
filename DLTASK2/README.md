# Handwritten Digit Classification using TensorFlow

## Project Description  
This project implements a deep learning-based **handwritten digit classifier** using the **MNIST dataset**. The goal is to recognize digits (0-9) from grayscale images of handwritten numbers. Two different neural network architectures are explored:  

1. **Artificial Neural Network (ANN)**  
2. **Convolutional Neural Network (CNN)**  

A comparison is made between the two models to evaluate their accuracy and performance.

## Dataset  
- **MNIST Dataset**: A collection of 60,000 training images and 10,000 test images of handwritten digits (0-9).  
- Each image is **28x28 pixels** in grayscale.  

## Implementation Details  
### **1. Artificial Neural Network (ANN) Model**  
- Built using TensorFlow and Keras.  
- Fully connected dense layers with activation functions.  
- Trained using categorical cross-entropy loss and Adam optimizer.  

### **2. Convolutional Neural Network (CNN) Model**  
- Uses convolutional layers to extract spatial features.  
- Includes max pooling layers for dimensionality reduction.  
- Trained with the same dataset for comparison with ANN.  

## Model Evaluation  
- The accuracy of both models is compared to determine which performs better in classifying handwritten digits.  
- Metrics such as **accuracy, loss, and confusion matrix** are analyzed.  

## Results
- CNN typically performs better than ANN for image-based tasks due to its feature extraction capabilities.
- Accuracy comparison and performance metrics are presented in graphs.

