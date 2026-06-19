# oral-cancer-image-classification
Deep learning based oral cancer image classification using Inception V4 and Inception ResNet V2 during IISc Bangalore research internship.
# Image Classification Model for Oral Cancer Detection

## Overview

This project was developed during my Research Internship at the Indian Institute of Science (IISc), Bangalore under the guidance of Prof. Dr. Rajesh Sundaresan.

The objective of the project was to classify oral cancer images into 11 categories using deep convolutional neural networks. The work involved implementing and evaluating advanced image classification architectures including Inception V4 and Inception ResNet V2.

---

## Research Internship Details

* Institution: Indian Institute of Science (IISc), Bangalore
* Mentor: Prof. Dr. Rajesh Sundaresan
* Duration: June 2024 – July 2024

---

## Problem Statement

Oral cancer is a significant healthcare challenge where early identification can improve treatment outcomes. This project focuses on automated classification of oral cancer images using deep learning techniques to assist in accurate categorization of oral lesions.

---

## Dataset

### Dataset Statistics

| Split      | Images |
| ---------- | ------ |
| Training   | 1344   |
| Validation | 412    |
| Testing    | 422    |

Total Images: 2178

Number of Classes: 11

Note: The dataset used for this research is not publicly available and therefore cannot be shared.

---

## Model Architectures

### Inception V4

Implemented Inception V4 architecture for multi-class image classification.

### Inception ResNet V2

Implemented Inception ResNet V2 to evaluate the impact of residual connections on classification performance.

---

## Results

| Metric              | Score |
| ------------------- | ----- |
| Validation Accuracy | 76%   |
| Test Accuracy       | 75%   |

---

## Technology Stack

* Python
* TensorFlow
* NumPy
* OpenCV
* Matplotlib

---

## Repository Structure

src/

* training.py
* testing.py
* regularize.py
* inceptionv4.py
* dividescript.py
* implementationcode.py

models/

* best_model.h5

---

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run training:

python training.py

Run evaluation:

python testing.py

---

## Disclaimer

The dataset used during this research internship is proprietary and cannot be distributed. Only the implementation code has been made publicly available for educational and research purposes.
