---
title: ASL Sign Language Detection
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🤟 ASL Hand Gesture Recognition: End-to-End Machine Learning Pipeline & Web Application

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow.svg)](https://huggingface.co/spaces/darryll08/asl-alphabet-detection)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg)](https://fastapi.tiangolo.com/)

## 🎯 Project Overview

This repository hosts an end-to-end machine learning project designed to recognize American Sign Language (ASL) alphabets. The project demonstrates a complete data science workflow: from exploratory data analysis (EDA) and data pipeline engineering to model training, evaluation, and final deployment as an interactive web application.

The system is capable of classifying **29 distinct classes**, including the letters `A-Z` and special functional labels (`del`, `nothing`, `space`).

To ensure a robust real-world application, this project explores and compares two distinct architectural approaches:
1. **Baseline CNN Model:** An image-based approach utilizing transfer learning.
2. **Final Landmark Model:** A feature-engineered approach utilizing spatial coordinates for real-time inference.

---

## 🧠 Architectural Methodologies

### 1. Baseline Image Model (MobileNetV2)
The initial approach frames the problem as a standard image classification task using RGB data.
* **Pipeline:** Images are resized to 224x224, normalized to `[0,1]`, and fed through a lightweight data augmentation pipeline.
* **Architecture:** Utilizes a pre-trained **MobileNetV2** base (frozen) with custom top layers (`GlobalAveragePooling`, `Dropout`, and a `Softmax` classifier).
* **Performance:** Achieves strong results on structured, static datasets but exhibits vulnerability to varying backgrounds and lighting conditions in live webcam tests.

### 2. Final Landmark Model (MediaPipe + MLP) - *Selected for Deployment*
To address the limitations of the baseline model in real-world environments, the final architecture relies on spatial hand landmarks.
* **Pipeline:** Utilizes **MediaPipe Hands** to extract 21 3D coordinates per hand. 
* **Feature Engineering:** Extracted coordinates are normalized and enriched with additional geometric features to account for varying distances and angles. Handedness (left/right) is also handled.
* **Architecture:** A Multi-Layer Perceptron (`MLPClassifier`) trained on the engineered landmark features, scaled via `StandardScaler`. Synthetic data was generated for the `nothing` class to improve robustness.

### ⚖️ Technical Trade-off Analysis

| Feature | Baseline CNN (Image-Based) | Final MLP (Landmark-Based) |
| :--- | :--- | :--- |
| **Input Data** | Raw RGB Pixels | 21 Spatial Coordinates (x, y, z) |
| **Robustness** | Susceptible to background noise & lighting | Highly resilient to varied backgrounds |
| **Computational Cost** | High (Heavy inference logic) | Low (Lightweight, fast execution) |
| **Deployment Use Case** | Static image batch processing | Real-time interactive webcam feeds |

*Result: The Landmark MLP was selected for production deployment due to its superior latency and stability in dynamic, real-world environments.*

---

## 🌐 Web Application Features

The project is deployed via **FastAPI** and **Uvicorn**, providing a responsive web interface with two primary operational modes:

* **Single Capture Inference:** Captures a single webcam frame, isolates the gesture Region of Interest (ROI), and returns the top-K predictions alongside confidence scores.
* **Real-Time Spell Builder:** Continuously streams predictions from the webcam, employing prediction stabilization heuristics. It strings recognized letters into words and dynamically handles functional gestures (e.g., using `del` as a backspace and `space` for word separation).

---

## 📂 Repository Structure

The codebase is modularized to separate experimental notebooks from production-ready source code.

```bash
.
├── notebooks/
│   └── asl_recognition_notebook.ipynb  # Comprehensive EDA, training, and evaluation workflows
├── src/                                # Modularized source code for the ML pipeline
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hand_roi.py
│   ├── landmark_features.py
│   ├── train_landmark_model.py
│   ├── inference.py
│   ├── landmark_inference.py
│   └── utils.py
├── app/                                # FastAPI web application backend and frontend
│   ├── main.py
│   ├── templates/
│   └── static/
├── models/                             # Serialized model artifacts and label mappings
│   ├── best_model.keras
│   ├── label_map.json
│   ├── landmark_mlp.joblib
│   ├── landmark_label_encoder.joblib
│   └── landmark_label_map.json
├── reports/                            # Performance metrics and visualizations
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── confusion_matrix.png
├── dataset/                            # Raw and processed datasets (ignored in git)
└── README.md
