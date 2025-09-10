import csv
import pandas as pd
import numpy as np
import spacy
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    valid_df = pd.read_csv("validation.csv")

    X_train, y_train = train_df["Text"], train_df["Label"]
    X_test, y_test = test_df["Text"], test_df["Label"]
    X_valid, y_valid = valid_df["Text"], valid_df["Label"]

    return X_train, y_train, X_test, y_test, X_valid, y_valid


nlp = spacy.load("en_core_web_sm")


def preprocess_texts(texts):
    processed = []
    for doc in nlp.pipe(texts):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        processed.append(tokens)
    return processed


def text_to_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)


X_train, y_train, X_test, y_test, X_valid, y_valid = load_data()

train_tokens = preprocess_texts(X_train)
test_tokens = preprocess_texts(X_test)
valid_tokens = preprocess_texts(X_valid)

w2v_model = Word2Vec(sentences=train_tokens)
w2v_model.save("word2vec.model")

X_train_vectors = np.array([text_to_vector(tokens, w2v_model) for tokens in train_tokens])
X_test_vectors = np.array([text_to_vector(tokens, w2v_model) for tokens in test_tokens])
X_valid_vectors = np.array([text_to_vector(tokens, w2v_model) for tokens in valid_tokens])

X_train_tensor = torch.tensor(X_train_vectors, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32).unsqueeze(1)
X_valid_tensor = torch.tensor(X_valid_vectors, dtype=torch.float32).unsqueeze(1)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)


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


input_dim = X_train_tensor.shape[2]
hidden_dim = 64
output_dim = 5

model = LSTMClassifier(input_dim, hidden_dim, output_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

output_file = 'W2V.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Test Loss", "Training Accuracy", "Test Accuracy"])

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    train_loss = loss_fn(outputs, y_train_tensor)
    train_loss.backward()
    optimizer.step()

    train_predictions = torch.argmax(outputs, dim=1)
    train_accuracy = accuracy_score(y_train_tensor.numpy(), train_predictions.numpy())

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = loss_fn(test_outputs, y_test_tensor)
        test_predictions = torch.argmax(test_outputs, dim=1)
        test_accuracy = accuracy_score(y_test_tensor.numpy(), test_predictions.numpy())

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss.item(), test_loss.item(), train_accuracy, test_accuracy])

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Training Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, "
        f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

model.eval()
y_test_pred = model(X_test_tensor)
_, test_predicted = torch.max(y_test_pred, 1)
valid_accuracy = accuracy_score(y_test_tensor.numpy(), test_predicted.numpy())
precision_validation = precision_score(y_test_tensor.numpy(), test_predicted.numpy(), zero_division=0, average="weighted")
recall_validation = recall_score(y_test_tensor.numpy(), test_predicted.numpy(), zero_division=0, average="weighted")

print("\nClassification model_lstm - Test Set:")
print(
    f"Accuracy: {valid_accuracy * 100:.2f}%, Precision: {precision_validation * 100:.2f}%, Recall: {recall_validation * 100:.2f}%")

y_valid_pred = model(X_valid_tensor)
_, valid_predicted = torch.max(y_valid_pred, 1)
valid_accuracy = accuracy_score(y_valid_tensor.numpy(), valid_predicted.numpy())
precision_validation = precision_score(y_valid_tensor.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")
recall_validation = recall_score(y_valid_tensor.numpy(), valid_predicted.numpy(), zero_division=0, average="weighted")

print("\nClassification model_lstm - Validation Set:")
print(
    f"Accuracy: {valid_accuracy * 100:.2f}%, Precision: {precision_validation * 100:.2f}%, Recall: {recall_validation * 100:.2f}%")

