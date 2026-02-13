🧠 Brain MRI Tumor Segmentation
Comparison of Otsu and Sauvola Thresholding Methods

📌 Project Description

This project performs brain tumor segmentation on MRI images using two classical thresholding techniques:

Otsu's Global Thresholding
Sauvola's Adaptive Thresholding

The segmented tumor regions are compared with ground truth masks using evaluation metrics.

📂 Dataset

Dataset used:
Brain MRI Tumor Segmentation Dataset

Kaggle Link:
https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation

📁 Dataset Structure:
dataset/
   ├── images/
   ├── masks/

Methods Used

1️. Otsu Thresholding

Global thresholding method
Computes a single threshold for the entire image
Based on maximizing inter-class variance

2. Sauvola Thresholding

Adaptive/local thresholding method
Uses local mean and standard deviation

Parameters used:
Window size = 25
k = 0.02
r = 128

Evaluation Metrics

Segmentation performance is evaluated using:

Dice Similarity Coefficient
Jaccard Index

These metrics measure the overlap between predicted tumor region and ground truth mask.

