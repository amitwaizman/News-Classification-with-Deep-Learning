import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data import loadData

X_train, y_train, X_test,y_test, X_valid , y_valid = loadData()


y_train = torch.tensor(y_train.values, dtype=torch.long)
y_valid = torch.tensor(y_valid.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

majority_class = y_train.mode().values.item()
print(majority_class)
y_pred_class_train = np.full_like(y_train, majority_class)
y_pred_class_test = np.full_like(y_test, majority_class)
y_pred_class_validation = np.full_like(y_valid, majority_class)

accuracy_test = accuracy_score(y_test, y_pred_class_test)
precision_test = precision_score(y_test, y_pred_class_test, zero_division=0, average="macro")
recall_test = recall_score(y_test, y_pred_class_test, zero_division=0, average="macro")

accuracy_validation = accuracy_score(y_valid, y_pred_class_validation)
precision_validation = precision_score(y_valid, y_pred_class_validation, zero_division=0, average="macro")
recall_validation = recall_score(y_valid, y_pred_class_validation, zero_division=0, average="macro")


print("\nClassification Baseline - Test Set:")
print(f"Accuracy: {accuracy_test*100:.2f}%, Precision: {precision_test*100:.2f}%, Recall: {recall_test*100:.2f}%")

print("\nClassification Baseline - Validation Set:")
print(f"Accuracy: {accuracy_validation*100:.2f}%, Precision: {precision_validation*100:.2f}%, Recall: {recall_validation*100:.2f}%")
