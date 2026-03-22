# Electrical Steel Flow Stress Predictor

A deep learning-powered Streamlit web application for predicting high-temperature flow stress of electrical steels based on deformation conditions.

## Overview

This project was developed as part of PhD research at the University of New Brunswick (UNB) in collaboration with CanmetMATERIALS, Natural Resources Canada. It applies deep learning to predict the deformation stress behavior of electrical steels — a critical parameter in steel processing and forming operations.

Two separate neural network models are deployed, trained on experimental data for two silicon content variants:
- **1.3 wt% Si** electrical steel
- **3.2 wt% Si** electrical steel

## Features

- Interactive Streamlit interface for real-time stress prediction
- Neural network models trained using TensorFlow/Keras
- Supports input parameters: strain, strain rate, and deformation temperature
- Separate pre-trained models and scalers for each steel grade
- Achieves ~25% improvement in prediction accuracy over conventional empirical models

## Input Parameters

| Parameter | Range | Unit |
|---|---|---|
| Strain | 0.0 – 0.7 | - |
| Strain Rate | 0.01, 0.1, 1.0 | 1/s |
| Temperature | 850 – 1050 | °C |

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- NumPy
- Pickle

## Project Structure

```
steel-stress-predictor/
├── app.py               # Streamlit application
├── models/              # Trained Keras model files (.keras)
├── scalers/             # Pre-fitted scaler objects (.pkl)
├── requirements.txt     # Python dependencies
```

## How to Run Locally

```bash
git clone https://github.com/gyanaranjanmishra/steel-stress-predictor.git
cd steel-stress-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Research Context

This work is part of a broader PhD study on data-driven modeling of electrical steel deformation behavior using multi-source experimental datasets exceeding 200,000 samples. The models demonstrate the applicability of deep learning to materials science problems where physics-based models have limitations.

## Author

**Gyanaranjan Mishra, PhD**
Applied Data Scientist | Materials Engineer
[LinkedIn](https://www.linkedin.com/in/gyanaranjanmishra/) | gyanaranjanmishra06@gmail.com
