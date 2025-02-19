import argparse
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.util import ngrams  # To handle bigrams
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
    bigrams = ['_'.join(bigram) for bigram in ngrams(words, 2)]  # Generate bigrams
    return words + bigrams  # Combine unigrams and bigrams

# Bernoulli Naive Bayes function
def train_naive_bayes(X_train, y_train, vocab_size, label_map):
    n_classes = len(label_map)
    class_probabilities = np.zeros(n_classes)
    feature_probabilities = np.zeros((n_classes, vocab_size))
    feature_log_complements = np.zeros((n_classes, vocab_size))

    y_train_mapped = np.array([label_map[label] for label in y_train])  # Convert labels to integers
    class_counts = np.bincount(y_train_mapped)

    # Calculate class probabilities: P(y)
    class_probabilities = np.log(class_counts / len(y_train))
    word_counts = np.zeros((n_classes, vocab_size))

    # Calculate feature counts for each class
    for i, label in enumerate(y_train_mapped):
        word_counts[label] += X_train[i]

    # Calculate P(xj=1|y) with Laplace smoothing and P(xj=0|y)
    for c in range(n_classes):
        feature_probabilities[c] = np.log((word_counts[c] + 1) / (class_counts[c] + 2))
        feature_log_complements[c] = np.log(1 - np.exp(feature_probabilities[c]))

    return class_probabilities, feature_probabilities, feature_log_complements

def predict_naive_bayes(X_test, test_texts, vocab, class_probabilities, feature_probabilities, feature_log_complements, labels, y_train, label_map):
    predictions = []
    y_train_mapped = np.array([label_map[label] for label in y_train])  # Convert labels to integers
    class_counts = np.bincount(y_train_mapped)  # Class counts for Laplace smoothing

    for x, doc in zip(X_test, test_texts):
        class_scores = deepcopy(class_probabilities)

        for c in range(len(class_probabilities)):
            # Compute the class score using feature probabilities
            class_scores[c] += x.dot(feature_probabilities[c]) + (1 - x).dot(feature_log_complements[c])

            # Check for missing words (words in the doc not in the vocab)
            missing_words = set(doc) - set(vocab)
            if missing_words:
                # Apply penalty for unseen words using Laplace smoothing
                class_scores[c] += np.log(1 / (class_counts[c] + 2))

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

    # Build vocabulary from the training data (unigram and bigram features)
    vocab = list(set(word for text in train_texts for word in text))
    vocab_index_map = {word: idx for idx, word in enumerate(vocab)}
    
    # Convert texts to a binary event model (word presence as 0/1) using numpy
    def text_to_features(text, vocab_index_map):
        features = np.zeros(len(vocab_index_map))
        for word in text:
            if word in vocab_index_map:
                features[vocab_index_map[word]] = 1
        return features

    # Convert train and test sets to binary feature matrices
    X_train = np.array([text_to_features(text, vocab_index_map) for text in train_texts])
    X_test = np.array([text_to_features(text, vocab_index_map) for text in test_texts])

    y_train = train_labels

    # Create a mapping from labels to integers
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Train the model
    class_probabilities, feature_probabilities, feature_log_complements = train_naive_bayes(X_train, y_train, len(vocab), label_map)

    # Predict on the test set
    y_pred = predict_naive_bayes(X_test, test_texts, vocab, class_probabilities, feature_probabilities, feature_log_complements, unique_labels, y_train, label_map)

    # Output the predictions to the specified output file, ensuring no extra newlines
    with open(args.out, 'w') as f_out:
        f_out.write('\n'.join(y_pred))  # Write predictions without extra newlines

    end_time = time.time()
    # print(f'Execution Time: {(end_time - start_time) / 60:.2f} minutes')

if __name__ == "__main__":
    main()
