# =====================================
# IMPORT LIBRARIES
# =====================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("C:/Users/admin/Documents/python/stroke.csv")

print("\nDataset Description:")
print(df.describe())

# =====================================
# DATA CLEANING
# =====================================

# Drop duplicates
df = df.drop_duplicates()

# Check missing values
print("\nMissing values before filling:")
print(df.isnull().sum())

# Fill missing values
df["BMI"] = df["BMI"].fillna(df["BMI"].mean())
df["Smoking_Status"] = df["Smoking_Status"].fillna(df["Smoking_Status"].mode()[0])

print("\nMissing values after filling:")
print(df.isnull().sum())

# Drop unwanted column
df = df.drop("ID", axis=1)

# =====================================
# TARGET DISTRIBUTION (IMBALANCE CHECK)
# =====================================
print("\nTarget distribution (%):")
print(df["Stroke"].value_counts(normalize=True) * 100)

df["Stroke"].value_counts().plot(kind="bar")
plt.title("Target Variable Distribution")
plt.xlabel("Stroke")
plt.ylabel("Count")
plt.show()

# =====================================
# FEATURE / TARGET SPLIT
# =====================================
X = df.drop("Stroke", axis=1)
y = df["Stroke"]

# Separate numerical & categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

print("\nNumerical Columns:", list(num_cols))
print("Categorical Columns:", list(cat_cols))

# =====================================
# PREPROCESSING PIPELINE
# =====================================
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# =====================================
# TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocess data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# =====================================
# BALANCE DATA USING SMOTE
# =====================================
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed, y_train
)

print("\nBalanced target distribution:")
print(pd.Series(y_train_balanced).value_counts())

from sklearn.ensemble import RandomForestClassifier
df["Age_Risk"] = (df["Age"] > 60).astype(int)
df["Glucose_Risk"] = (df["Avg_Glucose_Level"] > 140).astype(int)
df["BMI_Risk"] = (df["BMI"] > 30).astype(int)
df["Cardio_Risk"] = df["Hypertension"] | df["Heart_Disease"]


# ================================
# TRAIN RANDOM FOREST
# ================================
rf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    min_samples_leaf=10,
    random_state=42
)

rf.fit(X_train_processed, y_train)

# ================================
# PROBABILITIES + THRESHOLD
# ================================
y_prob = rf.predict_proba(X_test_processed)[:, 1]

threshold = 0.15
y_pred = (y_prob >= threshold).astype(int)

# ================================
# EVALUATION
# ================================
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ================================
# CONFUSION MATRIX (USE y_pred!)
# ================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Random Forest Confusion Matrix (threshold={threshold})")
plt.show()

from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

brf.fit(X_train_processed, y_train)
y_pred = brf.predict(X_test_processed)
print("\nBalancedRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"BalancedRandom Forest Confusion Matrix (threshold={threshold})")
plt.show()