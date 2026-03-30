Phishing URL Detection System

A machine learning–based phishing detection system that analyzes URLs using feature engineering and a LightGBM model to classify them as benign or phishing.

 Overview

This project builds a real-time phishing detection pipeline using:

Custom URL feature extraction
A trained LightGBM classifier
Smart threshold selection to minimize false positives
Clean, modular training pipeline

The system is designed for integration into a browser extension or backend API for real-world use.

 Key Features
 Custom Feature Extraction
URL structure analysis (length, dots, symbols)
Domain-based signals (IP usage, suspicious TLDs)
Behavioral indicators (auth flows, suspicious words)
Ratio-based features (path/query complexity)
 Fast Inference
No external API calls required
All features computed locally → near-instant predictions
 Smart Thresholding
Optimized using validation data
Minimizes false positives (important for user experience)
 Model Performance
ROC-AUC ≈ 0.99
0 false positives in test set
~96% overall accuracy
