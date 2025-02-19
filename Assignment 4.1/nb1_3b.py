import argparse
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk import ngrams
from copy import deepcopy
from collections import Counter
import time

# Command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Naive Bayes Classifier for Truthfulness Detection")
    parser.add_argument('--train', type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument('--test', type=str, required=True, help="Path to the validation data CSV file")
    parser.add_argument('--out', type=str, required=True, help="Path to output the predictions")
    parser.add_argument('--stop', type=str, required=True, help="Path to stopwords file")
    return parser.parse_args()

# Stopword removal and stemming, including bigram extraction
def preprocess_text(text, stop_words, ps):
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    # Generate bigrams
    bigrams = list(ngrams(words, 2))
    bigrams = [' '.join(bigram) for bigram in bigrams]
    
    return words + bigrams  # Return both unigrams and bigrams

# Convert text into features using dense numpy arrays
def text_to_features(text, vocab_index_map, vocab_size):
    word_count = Counter(text)
    feature_vector = np.zeros(vocab_size, dtype=np.float32)
    for word in word_count:
        if word in vocab_index_map:
            feature_vector[vocab_index_map[word]] = word_count[word]
    return feature_vector

# Bernoulli Naive Bayes function
def train_naive_bayes(X_train, y_train, vocab_size, label_map):
    n_classes = len(label_map)
    class_probabilities = np.zeros(n_classes)
    feature_probabilities = np.zeros((n_classes, vocab_size))

    y_train_mapped = np.array([label_map[label] for label in y_train])  # Convert labels to integers
    class_counts = np.bincount(y_train_mapped)

    # Calculate class probabilities: P(y)
    class_probabilities = np.log(class_counts / len(y_train))

    # Sum word counts per class
    word_counts = np.zeros((n_classes, vocab_size))
    for i in range(X_train.shape[0]):
        word_counts[y_train_mapped[i]] += X_train[i]

    # Calculate feature probabilities: P(x|y)
    for c in range(n_classes):
        total_words = np.sum(word_counts[c])
        feature_probabilities[c] = np.log((word_counts[c] + 1) / (total_words + vocab_size))

    return class_probabilities, feature_probabilities, word_counts

# Predict using Naive Bayes
def predict_naive_bayes(X_test, test_texts, vocab, class_probabilities, feature_probabilities, labels, word_counts):
    predictions = []
    vocab_size = len(vocab)
    
    for i in range(X_test.shape[0]):
        class_scores = deepcopy(class_probabilities)
        doc = test_texts[i]
        unseen = sum(1 for word in doc if word not in vocab)

        for c in range(len(class_probabilities)):
            class_scores[c] += X_test[i].dot(feature_probabilities[c].T)
            if unseen > 0:
                total_words = np.sum(word_counts[c])
                class_scores[c] += unseen * np.log(1 / (total_words + vocab_size))
        
        predictions.append(labels[np.argmax(class_scores)])
    return predictions

def main():
    start_time = time.time()
    args = parse_args()

    # Load stopwords from the file, skipping any blank lines
    with open(args.stop, 'r') as f:
        stop_words = set(line.strip() for line in f if line.strip())  # Skip blank lines

    # Initialize the stemmer
    ps = PorterStemmer()

    # Load train and validation data
    train_data = pd.read_csv(args.train, sep="\t", header=None, quoting=3)
    test_data = pd.read_csv(args.test, sep="\t", header=None, quoting=3)

    # Preprocess text
    train_texts = [preprocess_text(text, stop_words, ps) for text in train_data[2].values]
    test_texts = [preprocess_text(text, stop_words, ps) for text in test_data[2].values]
    train_labels = train_data[1].values

    # Build vocabulary from the training data
    vocab = list(set(word for text in train_texts for word in text))
    vocab_index_map = {word: idx for idx, word in enumerate(vocab)}

    # Create feature vectors for training and test sets using dense numpy arrays
    vocab_size = len(vocab)
    X_train = np.array([text_to_features(text, vocab_index_map, vocab_size) for text in train_texts])
    X_test = np.array([text_to_features(text, vocab_index_map, vocab_size) for text in test_texts])

    y_train = train_labels

    # Create a mapping from labels to integers
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Train the model
    class_probabilities, feature_probabilities, word_counts = train_naive_bayes(X_train, y_train, vocab_size, label_map)

    # Predict on the test set
    y_pred = predict_naive_bayes(X_test, test_texts, vocab, class_probabilities, feature_probabilities, unique_labels, word_counts)

    # Output the predictions to the specified output file, ensuring no extra blank lines
    with open(args.out, 'w') as f_out:
        f_out.write('\n'.join(y_pred) + '\n')  # Ensure no extra blank line, but a single newline at the end
    
    end_time = time.time()
    # print(f'Execution Time: {(end_time - start_time) / 60:.2f} minutes')

if __name__ == "__main__":
    main()
