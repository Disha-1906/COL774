import argparse
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from copy import deepcopy
import time

# Command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Naive Bayes Classifier for Truthfulness Detection")
    parser.add_argument('--train', type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument('--test', type=str, required=True, help="Path to the validation data CSV file")
    parser.add_argument('--out', type=str, required=True, help="Path to output the predictions")
    parser.add_argument('--stop', type=str, required=True, help="Path to stopwords file")
    return parser.parse_args()

# Stopword removal and stemming
def preprocess_text(text, stop_words, ps):
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return words

# Bernoulli Naive Bayes function (based on code 1 logic)
def train_naive_bayes(X_train, y_train, vocab, label_map):
    n_classes = len(label_map)
    class_probabilities = np.zeros(n_classes)
    feature_probabilities = np.zeros((n_classes, len(vocab)))

    y_train_mapped = np.array([label_map[label] for label in y_train])  # Convert labels to integers
    class_counts = np.bincount(y_train_mapped)

    # Calculate class probabilities: P(y)
    class_probabilities = np.log(class_counts / len(y_train))
    word_counts = np.zeros((n_classes, len(vocab)))

    # Calculate feature counts for each class
    for i, label in enumerate(y_train_mapped):
        word_counts[label] += X_train[i]
    
    for c in range(n_classes):
        total_words = sum(word_counts[c])
        feature_probabilities[c] = np.log((word_counts[c] + 1) / (total_words + len(vocab)))

    return class_probabilities, feature_probabilities, word_counts

def predict_naive_bayes(X_test, test_texts, vocab, class_probabilities, feature_probabilities, labels, word_counts):
    predictions = []
    for x, xx in zip(X_test, test_texts):
        doc = list(xx)
        class_scores = deepcopy(class_probabilities)
        for c in range(len(class_probabilities)):
            total_words = sum(word_counts[c])
            class_scores[c] += np.sum(x * feature_probabilities[c])
            unseen = sum(1 for word in doc if word not in vocab)
            if unseen > 0:
                class_scores[c] += unseen * np.log(1 / (total_words + len(vocab)))
        predictions.append(labels[np.argmax(class_scores)])
    return predictions

def main():
    start_time = time.time()
    args = parse_args()

    # Load stopwords from the file, skipping blank lines
    with open(args.stop, 'r') as f:
        stop_words = set([line.strip() for line in f if line.strip()])  # Skip blank lines

    # Initialize the stemmer
    ps = PorterStemmer()

    # Load train and validation data
    train_data = pd.read_csv(args.train, sep="\t", header=None, quoting=3)
    test_data = pd.read_csv(args.test, sep="\t", header=None, quoting=3)

    # Preprocess the training and validation data
    train_texts = train_data[2].apply(lambda x: preprocess_text(x, stop_words, ps)).values
    train_labels = train_data[1].values

    test_texts = test_data[2].apply(lambda x: preprocess_text(x, stop_words, ps)).values

    # Build vocabulary from the training data (unigram features)
    vocab = list(set(word for text in train_texts for word in text))

    # Convert texts to Bernoulli event model (word presence as 0/1)
    def text_to_features(text, vocab):
        return np.array([text.count(word) for word in vocab])

    # Convert train and test sets to feature matrices
    X_train = np.array([text_to_features(text, vocab) for text in train_texts])
    X_test = np.array([text_to_features(text, vocab) for text in test_texts])
    y_train = train_labels

    # Create a mapping from labels to integers
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Train the model
    class_probabilities, feature_probabilities, word_counts = train_naive_bayes(X_train, y_train, vocab, label_map)

    # Predict on the test set
    y_pred = predict_naive_bayes(X_test, test_texts, vocab, class_probabilities, feature_probabilities, unique_labels, word_counts)

    # Output the predictions to the specified output file, ensuring no extra newlines
    with open(args.out, 'w') as f_out:
        f_out.write('\n'.join(y_pred))  # Ensure no extra newline at the end

    end_time = time.time()
    # print(f'Execution Time: {(end_time - start_time) / 60:.2f} minutes')

if __name__ == "__main__":
    main()
