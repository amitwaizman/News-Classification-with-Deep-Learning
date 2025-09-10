import torch
import torch.nn as nn
import torch.optim as optim
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score

from data import load_and_process_data

X_train, y_train, X_test, y_test, X_valid, y_valid = load_and_process_data()
X_train = X_train.unsqueeze(1)  
X_test = X_test.unsqueeze(1)  
X_valid = X_valid.unsqueeze(1)  


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out


input_dim = X_train.shape[2]
hidden_dim = 64
output_dim = 5
model_lstm = LSTMClassifier(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.01)

output_file = 'LSTM.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Test Loss", "Training Accuracy", "Test Accuracy"])

num_epochs = 100
for epoch in range(num_epochs):
    model_lstm.train()
    optimizer.zero_grad()
    outputs = model_lstm(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate on test set
    with torch.no_grad():
        model_lstm.eval()
        test_outputs = model_lstm(X_test)
        test_loss = criterion(test_outputs, y_test)
        _, train_predicted = torch.max(outputs, 1)
        train_accuracy = accuracy_score(y_train.numpy(), train_predicted.numpy())
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = accuracy_score(y_test.numpy(), test_predicted.numpy())

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, loss.item(), test_loss.item(), train_accuracy, test_accuracy])

        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, "
                  f"Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")

model_lstm.eval()
y_test_pred = model_lstm(X_test)
_, test_predicted = torch.max(y_test_pred, 1)
valid_accuracy = accuracy_score(y_test.numpy(), test_predicted.numpy())
precision_validation = precision_score(y_test.numpy(), test_predicted.numpy(), zero_division=0, average="weighted")
recall_validation = recall_score(y_test.numpy(), test_predicted.numpy(), zero_division=0, average="weighted")

print("\nClassification model_lstm - Test Set:")
print(
    f"Accuracy: {valid_accuracy * 100:.2f}%, Precision: {precision_validation * 100:.2f}%, Recall: {recall_validation * 100:.2f}%")

y_valid_pred = model_lstm(X_valid)
_, valid_predicted = torch.max(y_valid_pred, 1)
valid_accuracy = accuracy_score(y_valid.numpy(), valid_predicted.numpy())
precision_validation = precision_score(y_valid.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")
recall_validation = recall_score(y_valid.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")

print("\nClassification model_lstm - Validation Set:")
print(
    f"Accuracy: {valid_accuracy * 100:.2f}%, Precision: {precision_validation * 100:.2f}%, Recall: {recall_validation * 100:.2f}%")
