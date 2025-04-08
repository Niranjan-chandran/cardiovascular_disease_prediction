# Cardiovascular Disease Prediction Model

This project implements a machine learning model to predict cardiovascular disease using multiple algorithms including SVM, KNN, Decision Tree, Logistic Regression, and Random Forest.

## Requirements

- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`)

## Dataset

The model uses the Cardiovascular Disease dataset. You can download it from:
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Place the dataset file (`cardio_train.csv`) in the same directory as the script.

## Features Used

The model uses the following features:
- Age (converted from days to years)
- Gender
- Height
- Weight
- Systolic blood pressure
- Diastolic blood pressure
- Cholesterol
- Glucose
- Smoking status
- Alcohol intake
- Physical activity

## How to Run

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Place the dataset file (`cardio_train.csv`) in the project directory

3. Run the script:
```bash
python cardiovascular_prediction.py
```

## Output

The script will:
1. Train and evaluate multiple machine learning models
2. Generate confusion matrices for each model
3. Create a comparison plot of model performances
4. Print detailed performance metrics for each model

## Performance Metrics

The following metrics are calculated for each model:
- Accuracy
- Precision
- Recall
- F1 Score

## Visualization

The script generates:
- Confusion matrices for each model (saved as PNG files)
- A comparison plot of all models' performances (saved as 'model_comparison.png') 