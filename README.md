# 📰 News Classification with Deep Learning

## 📌 Project Overview
The goal of this project is to **classify news articles into different categories** based on their textual content using **Deep Learning models**.  

**Categories:**
- `0` : Politics  
- `1` : Sports  
- `2` : Technology  
- `3` : Entertainment  
- `4` : Business  

Dataset size: **2,225 examples**  
- Politics: 417  
- Sports: 511  
- Technology: 401  
- Entertainment: 386  
- Business: 510  

---

## ⚙️ Preprocessing & Feature Extraction
We used **Bag of Words (BoW)** representation to convert text into numerical vectors:  
1. Build a vocabulary from all documents (**24,331 words**).  
2. Count word frequencies for each document.  
3. Preprocessing with **NLTK**:  
   - Removing stop words  
   - Removing punctuation  
   - Lemmatization  

---

## 🧪 Models & Results

### 🔹 Baseline
- Simple classifier: predicts the most frequent class.  
- **Accuracy:** ~21–24%  

---

### 🔹 SoftMax Classifier
- Implemented with different learning rates and optimizers (SGD, Adam).  
- Best results with Ridge Regularization + Adam:  
  - **Accuracy:** ~97%  

---

### 🔹 Neural Network (Fully Connected)
- Input layer: 24,331 (vocabulary size)  
- Hidden layers: tested with 1–3 layers (sizes: 12–20 neurons)  
- Output layer: 5 categories  
- Best model (4 layers + Adam):  
  - **Accuracy:** 97.75–97.98%  

---

### 🔹 LSTM
- Tested with **Word2Vec** and **Bag of Words** embeddings.  
- Best results with BoW + 2 LSTM layers (64 neurons):  
  - **Accuracy:** 98.43%  

---

## 📊 Comparison of Models

| Model                | Test Accuracy | Validation Accuracy |
|-----------------------|---------------|----------------------|
| Baseline             | 20.67%        | 24.49%              |
| SoftMax (Adam)       | 97.08%        | 96.18%              |
| Neural Network (Adam)| 97.75%        | 97.98%              |
| LSTM (BoW, 2 layers) | **98.43%**    | **97.98%**          |

---

## 🚀 Technologies Used

- **Python** – main programming language  
- **PyTorch** – deep learning models  
- **scikit-learn** – data splitting & metrics  
- **pandas & numpy** – data handling & numerical computing  
- **NLTK & spaCy** – natural language processing  
- **Gensim (Word2Vec)** – word embeddings


