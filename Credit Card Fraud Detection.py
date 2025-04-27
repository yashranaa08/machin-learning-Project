# Credit Card Fraud Detection Project in Python

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import zipfile
import os

# Step 2: Unzip the dataset if zipped
zip_path = 'creditcard.csv.zip'
csv_filename = 'creditcard.csv'

if not os.path.exists(csv_filename):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print(f"Extracted {csv_filename} from {zip_path}")

# Step 3: Load the dataset
data = pd.read_csv(csv_filename)

# Step 4: Exploratory Data Analysis (EDA)
print("Dataset shape:", data.shape)
print(data['Class'].value_counts())

# Plot class distribution
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Genuine, 1: Fraud)')
plt.show()

# Step 5: Preprocess data
# Features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Scale the 'Amount' and 'Time' features
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Step 6: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("After resampling, counts of label '1': {}".format(sum(y_resampled==1)))
print("After resampling, counts of label '0': {} \n".format(sum(y_resampled==0)))

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 8: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-AUC score and curve
y_pred_proba = model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Step 10: Save the trained model
model_filename = 'credit_card_fraud_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Step 11: Instructions to run
print("Run this script with: python credit_card_fraud_detection.py")



