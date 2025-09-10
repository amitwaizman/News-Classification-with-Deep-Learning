import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data import load_and_process_data

# Load data
X_train, y_train, X_test, y_test, X_valid, y_valid = load_and_process_data()

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes, n_num):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, n_num)
        self.layer2 = nn.Linear(n_num, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # ReLU activation function
        x = self.layer2(x)  
        return x

model = MLPModel(input_dim=X_train.shape[1], num_classes=5, n_num=12)
Loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.01)


output_file = 'neural_network.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Test Loss", "Training Accuracy", "Test Accuracy"])

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)

    loss = Loss_fun(outputs, y_train)

    loss.backward()
    optimizer.step()

    _, train_predicted = torch.max(outputs, 1)
    train_accuracy = accuracy_score(y_train.numpy(), train_predicted.numpy())

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = Loss_fun(test_outputs, y_test).item()
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = accuracy_score(y_test.numpy(), test_predicted.numpy())

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, loss.item(), test_loss, train_accuracy * 100, test_accuracy * 100])

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    precision = precision_score(y_test.numpy(), predicted.numpy(), zero_division=0, average="weighted")
    recall = recall_score(y_test.numpy(), predicted.numpy(), zero_division=0, average="weighted")
    f1 = f1_score(y_test.numpy(), predicted.numpy(), zero_division=0, average="weighted")

print("***Test***")
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

# Validation metrics
y_valid_pred = model(X_valid)
_, valid_predicted = torch.max(y_valid_pred, 1)
valid_accuracy = accuracy_score(y_valid.numpy(), valid_predicted.numpy())
precision = precision_score(y_valid.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")
recall = recall_score(y_valid.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")
f1 = f1_score(y_valid.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")

print("-------------------------------------------------------")
print("***Validation***")
print(f'Accuracy: {valid_accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
