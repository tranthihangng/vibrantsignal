# Equipment Condition Monitoring System

A machine learning-based system for real-time monitoring of equipment conditions using ESP32 sensors. The system can detect normal operations and fault conditions, store results in a database, and send notifications when equipment state changes.

## Overview

This project implements a complete pipeline for equipment condition monitoring:
- Data collection from sensors via ESP32 microcontrollers
- Data transmission through HTTP/TCP
- Machine learning classification of equipment state
- Database storage of results
- Notification system via Telegram

## Components

### Hardware
- ESP32 microcontrollers with sensors
- Equipment/machinery for monitoring

### Software
- Python-based machine learning models
- Flask web server for data reception
- Database for storing results
- Telegram notification system

## File Descriptions

- **trainmodel_practical.py**: Trains the machine learning model with cross-validation, regularization, and hyperparameter tuning
- **visualize_classification.py**: Visualizes the classification results
- **pred_test.py**: Flask server that receives sensor data, makes predictions, and handles notifications
- **merged_data_final4c.csv**: Merged dataset containing normal and fault condition data

## Setup Instructions

1. **Install Dependencies**
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn flask joblib
   ```

2. **Train the Model**
   ```
   python trainmodel_practical.py
   ```
   This generates:
   - `practical_mlp_best_model.joblib`: Trained neural network model
   - `practical_scaler.joblib`: Data scaler for normalization
   - Various visualization files showing model performance

3. **Configure ESP32**
   - Program ESP32 devices to collect sensor data and send to the server
   - Ensure they are configured to send data to the correct endpoint

4. **Run the Prediction Server**
   ```
   python pred_test.py
   ```
   This starts the Flask server to receive data and make predictions

## Usage

1. The ESP32 devices collect sensor data from the equipment
2. Data is sent to the Flask server via HTTP
3. The server processes the data, makes predictions about equipment state
4. Results are stored in the database
5. If an abnormal condition is detected, notifications are sent via Telegram

## Model Information

The system classifies equipment states into categories:
- normal0: Normal operating condition
- rung5_18: Fault condition type 1
- rung10_1: Fault condition type 2

The model is a neural network trained with:
- Cross-validation for reliable performance estimation
- L2 regularization to prevent overfitting
- Early stopping to optimize training time
- Hyperparameter tuning for optimal configuration 