import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import load_and_process_data
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score

X_train, y_train, X_test, y_test, X_valid, y_valid = load_and_process_data()

output_file = 'softmax.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Test Loss", "Training Accuracy", "Test Accuracy"])

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train.numpy()))
model = SoftmaxClassifier(input_dim, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  # L2 regularization via weight_decay
# optimizer = optim.SGD(model.parameters(), lr=0.01)  

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    
    loss.backward()
    optimizer.step()

    _, train_predictions = torch.max(outputs, 1)
    train_accuracy = accuracy_score(y_train.numpy(), train_predictions.numpy())
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid)
        val_loss = loss_fn(val_outputs, y_valid)
        
        _, val_predictions = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_valid.numpy(), val_predictions.numpy())

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, loss.item(), val_loss.item(), train_accuracy, val_accuracy])

    if (epoch + 1) % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, '
            f'Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%'
        )

model.eval()
with torch.no_grad():
    # Test Set
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    accuracy_test = accuracy_score(y_test.numpy(), predicted.numpy())
    precision_test = precision_score(y_test.numpy(), predicted.numpy(), zero_division=0, average="weighted")
    recall_test = recall_score(y_test.numpy(), predicted.numpy(), zero_division=0, average="weighted")

    # Validation Set
    y_valid_pred = model(X_valid)
    _, valid_predicted = torch.max(y_valid_pred, 1)
    accuracy_valid = accuracy_score(y_valid.numpy(), valid_predicted.numpy())
    precision_valid = precision_score(y_valid.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")
    recall_valid = recall_score(y_valid.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")

    # Training Set
    y_train_pred = model(X_train)
    _, train_predicted = torch.max(y_train_pred, 1)
    accuracy_train = accuracy_score(y_train.numpy(), train_predicted.numpy())
    precision_train = precision_score(y_train.numpy(), train_predicted.numpy(), zero_division=0, average="weighted")
    recall_train = recall_score(y_train.numpy(), train_predicted.numpy(), zero_division=0, average="weighted")

print("\nClassification SoftMax - Test Set:")
print(f"Accuracy: {accuracy_test * 100:.2f}%, Precision: {precision_test * 100:.2f}%, Recall: {recall_test * 100:.2f}%")

print("\nClassification SoftMax - Validation Set:")
print(f"Accuracy: {accuracy_valid * 100:.2f}%, Precision: {precision_valid * 100:.2f}%, Recall: {recall_valid * 100:.2f}%")

print("\nClassification SoftMax - Training Set:")
print(f"Accuracy: {accuracy_train * 100:.2f}%, Precision: {precision_train * 100:.2f}%, Recall: {recall_train * 100:.2f}%")
