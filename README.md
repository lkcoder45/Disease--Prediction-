# 🩺 Disease Prediction System

An Advanced Machine Learning project that predicts a disease from selected symptoms.

## Run the project

### 1. Open terminal in this folder

### 2. Install libraries

```bash
pip install -r requirements.txt
```

### 3. Train and compare ML models

```bash
python train_model.py
```

The script compares:
- Logistic Regression
- SVM
- Random Forest
- Gradient Boosting

It saves the best model inside `models/`.

### 4. Start the web app

```bash
streamlit run app.py
```

Your browser will open the Streamlit application.

## Project workflow

Symptoms → Data preprocessing → Train/Test Split → Multiple ML Models → Cross Validation → Model Selection → Prediction → Streamlit UI

## Folder structure

```text
Disease_Prediction_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── symptom_disease.csv
├── models/
│   └── (generated after training)
└── notebooks/
```


