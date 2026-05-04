# 🧠 Stroke Prediction using Machine Learning

## 📌 Overview
This project builds an end-to-end machine learning pipeline to predict the likelihood of stroke using patient health data.

Stroke prediction is a highly imbalanced classification problem, where stroke cases are rare compared to non-stroke cases. This project focuses on improving detection of stroke cases using imbalance handling techniques and threshold tuning.

---

## 🎯 Objectives
- Perform data cleaning and preprocessing  
- Handle missing values and categorical variables  
- Address class imbalance using SMOTE  
- Train machine learning models  
- Optimize prediction threshold for better recall  
- Evaluate model performance  

---

## 📂 Dataset
The dataset contains patient information such as:
- Age  
- Gender  
- BMI  
- Average Glucose Level  
- Hypertension  
- Heart Disease  
- Smoking Status  
- Work Type  
- Residence Type  

### Target Variable
- `Stroke`  
  - 0 → No Stroke  
  - 1 → Stroke  

---

## ⚙️ Workflow

### 1. Data Cleaning
- Removed duplicates  
- Filled missing values:
  - BMI → mean  
  - Smoking Status → mode  
- Dropped unnecessary column (`ID`)  

---

### 2. Feature Engineering
Created additional features:
- `Age_Risk` → Age > 60  
- `Glucose_Risk` → Glucose > 140  
- `BMI_Risk` → BMI > 30  
- `Cardio_Risk` → Hypertension OR Heart Disease  

---

### 3. Preprocessing
- Numerical features → StandardScaler  
- Categorical features → OneHotEncoder  
- Combined using ColumnTransformer  

---

### 4. Handling Imbalance
- Applied SMOTE to training data  
- Used Balanced Random Forest Classifier  

---

### 5. Models
- Random Forest Classifier  
- Balanced Random Forest Classifier  

---

### 6. Threshold Tuning
```python
threshold = 0.15
y_pred = (y_prob >= threshold).astype(int)
