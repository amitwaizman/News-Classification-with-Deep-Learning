import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pandas as pd
from sklearn.model_selection import train_test_split


def splitData():
    df = pd.read_csv('df_file.csv')
    X = df['Text']
    y = df['Label']
    label_counts = y.value_counts()
    print("Label Distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} examples")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_df = pd.DataFrame({'Text': X_train, 'Label': y_train})
    test_df = pd.DataFrame({'Text': X_test, 'Label': y_test})
    valid_df = pd.DataFrame({'Text': X_valid, 'Label': y_valid})

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    valid_df.to_csv('validation.csv', index=False)

    return X_train, y_train, X_test, y_test, X_valid, y_valid


def process_bag_of_words(df):
    text_data = df.dropna().astype(str)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    word2location = {}
    idx = 0
    for sentence in text_data:
        for word in word_tokenize(sentence):
            word = word.lower().translate(str.maketrans('', '', string.punctuation))
            if word and word not in stop_words:
                word = stemmer.stem(word)
                if word not in word2location:
                    word2location[word] = idx
                    idx += 1
    vocabulary_size = len(word2location)
    print(f"Vocabulary Size: {vocabulary_size}")

    def convert2vec(sentence):
        res_vec = np.zeros(vocabulary_size)
        for word in word_tokenize(sentence):
            word = word.lower().translate(str.maketrans('', '', string.punctuation))
            if word and word not in stop_words:
                word = stemmer.stem(word)
                if word in word2location:
                    res_vec[word2location[word]] += 1
        return res_vec

    vectors = np.array([convert2vec(sentence) for sentence in text_data])
    output_df = pd.DataFrame(vectors, columns=[f"{word}" for word in word2location.keys()])
    return output_df, df


def load_and_process_data():
    text_column = "Text"
    label_column = "Label"

    train_df = pd.read_csv('train.csv').reset_index(drop=True)
    test_df = pd.read_csv('test.csv').reset_index(drop=True)
    valid_df = pd.read_csv('validation.csv').reset_index(drop=True)

    all_texts = pd.concat([train_df[text_column], test_df[text_column], valid_df[text_column]]).reset_index(drop=True)

    features_df, _ = process_bag_of_words(all_texts)

    X_train = features_df.iloc[:len(train_df)].values
    X_test = features_df.iloc[len(train_df):len(train_df) + len(test_df)].values
    X_valid = features_df.iloc[len(train_df) + len(test_df):].values

    y_train = train_df[label_column].values
    y_test = test_df[label_column].values
    y_valid = valid_df[label_column].values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test, X_valid, y_valid

