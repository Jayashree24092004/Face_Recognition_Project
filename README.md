# Face Recognition Project using MediaPipe and SVM

## Overview
This project implements a **face recognition system** using **MediaPipe Face Mesh** for feature extraction and **Support Vector Machine (SVM)** for classification. The system can detect faces in images, extract facial landmarks, and identify individuals based on a trained dataset.

## Key Features
- **Face Detection & Landmark Extraction**: Uses MediaPipe to extract 468 3D facial landmarks.
- **Face Recognition**: SVM classifier trained on landmark features for identifying individuals.
- **Supports Image Dataset**: Can process multiple images organized per person.
- **Prediction**: Can predict the person in new images with probabilities for all known classes.
- **Model Persistence**: Trained SVM model is saved for reuse (`.pkl` file).

## Technologies Used
- Python 3
- MediaPipe – for face landmark detection
- OpenCV – for image processing
- Scikit-learn – for SVM classifier and evaluation
- NumPy – for data handling
- Joblib – for saving/loading the trained model

## Installation
1. Clone the repository:
   
   git clone https://github.com/YourUsername/Face_Recognition_Project.git
   cd Face_Recognition_Project
2.Install dependencies:
pip install opencv-python mediapipe scikit-learn numpy joblib
## ML Techniques Used

| Stage              | Technique / Algorithm           | Purpose                                   |
| ------------------ | ------------------------------- | ----------------------------------------- |
| Feature Extraction | MediaPipe Face Mesh             | Convert images to numeric feature vectors |
| Classification     | SVM (Support Vector Machine)    | Identify person from features             |
| Data Splitting     | Train-Test Split                | Validate performance on unseen data       |
| Model Evaluation   | Accuracy, Precision, Recall, F1 | Measure performance of classifier         |
## OUTPUT:##
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b333330b-66a4-4e4d-a08f-d4dfa351bf1f" />


