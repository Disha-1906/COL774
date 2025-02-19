import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
import argparse
from itertools import zip_longest
from sklearn.preprocessing import OneHotEncoder
import time

# Argument parser to take command-line inputs
def parse_args():
    parser = argparse.ArgumentParser(description='Naive Bayes Multinomial for Fake News Detection')
    parser.add_argument('--train', required=True, type=str, help='Path to the train.csv file')
    parser.add_argument('--test', required=True, type=str, help='Path to the val.csv file')
    parser.add_argument('--out', required=True, type=str, help='Path to save the output predictions')
    parser.add_argument('--stop', required=True, type=str, help='Path to the stopwords.txt file')
    return parser.parse_args()

# Initialize stemmer
stemmer = PorterStemmer()

# Load custom stopwords from a file
def load_stopwords(file_path):
    with open(file_path, 'r') as f:
        stop_words = set(word.strip() for word in f.readlines())
    return stop_words

# Text preprocessing: stopword removal and stemming
def preprocess_text(text, stop_words):
    words = text.lower().split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return filtered_words

# Build vocabulary based on training data, including unigrams and bigrams
def build_vocabulary(texts):
    vocab = set()
    for text in texts:
        vocab.update(text)
        bigrams = zip_longest(text[:-1], text[1:], fillvalue='')
        for bigram in bigrams:
            vocab.add('_'.join(bigram))
    return list(vocab)

# Convert text to term frequency matrix (Multinomial model)
def text_to_term_frequency_matrix(texts, vocab):
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    term_freq_matrix = np.zeros((len(texts), len(vocab)), dtype=int)    
    for i, text in enumerate(texts):
        # Count unigrams
        for word in text:
            if word in vocab_index:
                term_freq_matrix[i, vocab_index[word]] += 1
        # Count bigrams
        bigrams = zip_longest(text[:-1], text[1:], fillvalue='')
        for bigram in bigrams:
            bigram_str = '_'.join(bigram)
            if bigram_str in vocab_index:
                term_freq_matrix[i, vocab_index[bigram_str]] += 1
    return term_freq_matrix

# Helper function to merge the text-based and feature-based matrices
def merge_features(text_matrix, feature_matrix):
    return np.hstack((text_matrix, feature_matrix))

# Map string labels to integer values
def map_labels_to_integers(labels):
    label_mapping = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true': 5}
    return [label_mapping[label] for label in labels]

# Map integer predictions back to string labels
def map_integers_to_labels(predictions):
    reverse_label_mapping = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}
    return [reverse_label_mapping[pred] for pred in predictions]

# Main Naive Bayes Multinomial Class
class NaiveBayesMultinomial:
    def __init__(self, smoothing=1):
        self.smoothing = smoothing
        self.class_probs = {}
        self.word_probs = defaultdict(dict)
        self.vocab_size = 0

    def fit(self, X, y):
        self.vocab_size = X.shape[1]
        class_counts = np.bincount(y)
        total_samples = len(y)
        total_word_count_per_class = defaultdict(int)

        self.class_probs = {c: math.log(count / total_samples) for c, count in enumerate(class_counts)}
        for c in np.unique(y):
            X_c = X[y == c]
            word_counts = X_c.sum(axis=0)
            total_word_count_per_class[c] = word_counts.sum()
            for i in range(self.vocab_size):
                self.word_probs[c][i] = math.log((word_counts[i] + self.smoothing) / 
                                                  (total_word_count_per_class[c] + self.vocab_size * self.smoothing))

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            class_scores = {}
            for c in self.class_probs:
                class_scores[c] = self.class_probs[c]
                for j in range(self.vocab_size):
                    if X[i, j] > 0:
                        class_scores[c] += X[i, j] * self.word_probs[c].get(j, math.log(self.smoothing / (self.vocab_size * self.smoothing)))
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions

# Main function
def main():
    start_time = time.time()
    args = parse_args()

    # Load stopwords
    stop_words = load_stopwords(args.stop)

    # Load the training and test datasets
    column_names = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state', 'party', 
                    'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 
                    'pants_on_fire_counts', 'context']

    train_df = pd.read_csv(args.train, sep="\t", header=None, quoting=3, names=column_names)
    test_df = pd.read_csv(args.test, sep="\t", header=None, quoting=3, names=column_names)

    # Text preprocessing
    train_df['processed_text'] = train_df['statement'].apply(lambda text: preprocess_text(text, stop_words))
    test_df['processed_text'] = test_df['statement'].apply(lambda text: preprocess_text(text, stop_words))

    # Build vocabulary using only the training data
    vocab = build_vocabulary(train_df['processed_text'].values)

    # Convert text to term frequency matrix using the training set vocabulary
    X_train_text = text_to_term_frequency_matrix(train_df['processed_text'].values, vocab)
    X_test_text = text_to_term_frequency_matrix(test_df['processed_text'].values, vocab)

    # Feature engineering
    feature_columns = ['subject', 'speaker', 'job_title', 'state', 'party', 
                       'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']

    # Fit the encoder on training data only and transform both train and test
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_features = encoder.fit_transform(train_df[feature_columns])
    X_test_features = encoder.transform(test_df[feature_columns])

    # Merge text and feature matrices
    X_train = merge_features(X_train_text, X_train_features)
    X_test = merge_features(X_test_text, X_test_features)

    # Map labels to integers
    y_train = map_labels_to_integers(train_df['label'].values)
    y_test = map_labels_to_integers(test_df['label'].values)

    # Train Naive Bayes model
    nb_model = NaiveBayesMultinomial(smoothing=1)
    nb_model.fit(X_train, np.array(y_train))

    # Predict on test data
    y_test_pred = nb_model.predict(X_test)

    # Output results
    y_test_pred_labels = map_integers_to_labels(y_test_pred)
    with open(args.out, 'w') as f:
        for label in y_test_pred_labels:
            f.write(f'{label}\n')

    test_accuracy = np.mean(np.array(y_test) == np.array(y_test_pred))
    # print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    end_time = time.time() 
    # print(f'Execution Time: {(end_time - start_time) / 60:.2f} minutes')

if __name__ == '__main__':
    main()
