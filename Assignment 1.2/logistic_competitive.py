import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
import sys
import json

def feature_engineering(data):
    data['Total_Costs_Log'] = np.log1p(data['Total Costs'])
    data['Length_of_Stay_Log'] = np.log1p(data['Length of Stay'])
    data['Age_Log'] = np.log1p(data['Age Group'])

    data['Age_TotalCost_Interaction'] = data['Age Group'] * data['Total Costs']

    data['Zip_Code_First_Digit'] = data['Zip Code - 3 digits'] // 10

    data = data.drop(columns=['Operating Certificate Number', 'Permanent Facility Id', 'Facility Name'])

    return data

def drop_zero_variance_columns(df):
    return df.loc[:, df.var() != 0]

def apply_one_hot_encoding_fc(df, mapping_file):
    with open(mapping_file, 'r') as json_file:
        data_dict = json.load(json_file)

    for key in data_dict.keys():
        if key in df.columns:
            possible_values = data_dict[key]
            one_hot = pd.get_dummies(df[key])
            one_hot = one_hot.reindex(columns=sorted(possible_values), fill_value=False)
            one_hot = one_hot.iloc[:, 1:]  # Drop first column to avoid multicollinearity
            one_hot = one_hot.astype(int)
            one_hot.columns = [f"{key}_{col}" for col in one_hot.columns]

            # Replace the original column with the new one-hot encoded columns
            column_index = df.columns.get_loc(key)
            df = df.drop(columns=[key])
            df = pd.concat([df.iloc[:, :column_index], one_hot, df.iloc[:, column_index:]], axis=1)

    return df

def load_feature_selection_results():
    # Load created features
    with open("created.txt", 'r') as f:
        created_features = [line.strip() for line in f]
    
    # Load selected features
    with open("selected.txt", 'r') as f:
        selected_features = [int(line.strip()) for line in f]
    
    return created_features, selected_features


def apply_one_hot_encoding(df, mapping_file):
    with open(mapping_file, 'r') as json_file, open(mapping_file, 'r') as json_file:
        data_dict = json.load(json_file)

    for key in data_dict.keys():
        if key in df.columns:
            # print(f"Applying one-hot encoding to {key}")
            possible_values = data_dict[key]
            one_hot = pd.get_dummies(df[key])
            one_hot = one_hot.reindex(columns=sorted(possible_values), fill_value=0)
            one_hot = one_hot.astype(int)
            one_hot.columns = [f"{key}_{col}" for col in one_hot.columns]

            # Replace the original column with the new one-hot encoded columns
            df = df.drop(columns=[key])
            df = pd.concat([df, one_hot], axis=1)

    return df

def preprocess(filepath, istransform):
    data = pd.read_csv(filepath)
    data = apply_one_hot_encoding(data, "mapping.json")
    
    # Extracting the newly created Gender columns (e.g., Gender_0 and Gender_1)
    gender_columns = [col for col in data.columns if col.startswith('Gender_')]

    # Assuming binary gender: Gender_0 and Gender_1, we'll use Gender_1 as the target
    X_train = data.drop(columns=gender_columns).values.astype(np.float64)
    # print(gender_columns)
    y = data[gender_columns].iloc[:, 1].values  # Using the second column Gender_1 as the label

    scaler = None
    y_encoded = pd.get_dummies(y).values.astype(np.float64)  # One-hot encoding

    if istransform:
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis=1)
    else:
        X_train = np.hstack([np.ones((X_train.shape[0], 1), dtype=np.float64), X_train])
    
    return X_train, y, y_encoded, scaler


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_gradient(X, Y, y_encoded, w, freq):
    n, m = X.shape  # n = number of samples, m = number of features
    k = w.shape[1] 
    z = np.dot(X, w)
    prob = softmax(z)
    Y_one_hot = np.zeros((n, k), dtype=np.float64)
    Y_one_hot[np.arange(n), Y] = 1
    freq_values = freq[Y].reshape(-1, 1)  
    gradient = np.dot(X.T, (prob - Y_one_hot) / (2 * n * freq_values))
    return gradient

# def compute_loss(X, Y, y_encoded, w, freq):
#     n = X.shape[0]  # Number of samples
#     pred = np.dot(X, w)  # Predicted logits
#     prob = softmax(pred)  # Apply softmax to get probabilities
    
#     # Vectorized loss computation
#     weighted_loss = y_encoded * np.log(prob) / (2 * freq)
#     loss = -np.sum(weighted_loss) / n    
#     return loss

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_loss(X, Y, y_encoded, w, freq, epsilon=1e-8):
    n = X.shape[0]  # Number of samples
    # print(X.shape)
    # print(w.shape)
    # print(y_encoded.shape)
    pred = np.dot(X, w)  # Predicted logits
    sigmoid_pred = sigmoid(pred)
    modified_term = 2 * sigmoid_pred - 1
    # print(modified_term.shape)
    # print("-----------------------")
    # print(Y.shape)
    loss = np.sum(y_encoded * modified_term) / n
    
    return loss

def compute_loss_and_gradient(X,Y, y_encoded, weights, freq, batch_size):
    loss = compute_loss(X, Y, y_encoded, weights, freq)
    gradient = compute_gradient(X,Y, y_encoded, weights, freq)
    return loss, gradient

def mini_batch_gradient_descent_adam(X, Y, y_encoded, batch_size=128, epochs=50, learning_rate=10, beta1=0.9, beta2=0.99999, epsilon=1e-8, regularization_term=0.01):
    n_samples, n_features = X.shape
    n_classes = y_encoded.shape[1]
    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    freq = np.sum(y_encoded, axis=0)
    m_t = np.zeros_like(weights)
    v_t = np.zeros_like(weights)
    t = 0
    for epoch in range(epochs):
        # start_time = time.time()
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            y_batch = y_encoded[i:i+batch_size]
            actual_batch_size = X_batch.shape[0]
            loss = compute_loss(X_batch, Y_batch, y_batch, weights, freq)
            gradient = compute_gradient(X_batch, Y_batch, y_batch, weights, freq)
            t += 1
            m_t = beta1 * m_t + (1 - beta1) * gradient
            v_t = beta2 * v_t + (1 - beta2) * (gradient ** 2)
            m_hat = m_t / (1 - beta1 ** t)
            v_hat = v_t / (1 - beta2 ** t)
            weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            # print(f'Epoch {epoch+1}/{epochs}, Batch {i/batch_size} Loss: {loss}')
    
    return weights


train_file = sys.argv[1]
created_file = "created.txt"
selected_file = "selected.txt"
mapping_file = 'mapping.json'  # Specify the path to your mapping file

train_data = pd.read_csv(train_file)

# Apply one-hot encoding using the provided mapping file

# train_data = feature_engineering(train_data)
train_data = apply_one_hot_encoding_fc(train_data, mapping_file)

# train_data = feature_engineering(train_data)

# Drop zero variance columns
train_data = drop_zero_variance_columns(train_data)

# categorical_features = list(pd.read_json(mapping_file).keys())  # Extract categorical features from mapping file
# print([col for col in train_data.columns])
X_train = train_data.drop(columns=['Gender_1']).values
y_train = train_data['Gender_1'].values

# Feature Selection
selector = SelectKBest(f_classif, k=1000)
X_train_selected = selector.fit_transform(X_train, y_train)

# Save the created features to created.txt
created_features = train_data.drop(columns=['Gender_1']).columns.tolist()
with open(created_file, 'w') as f:
    for feature in created_features:
        f.write(f"{feature}\n")

# Save the selected features to selected.txt
selected_mask = selector.get_support().astype(int)
with open(selected_file, 'w') as f:
    for select in selected_mask:
        f.write(f"{select}\n")

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

created_features, selected_features = load_feature_selection_results()

batch_size = 8152
epochs = 50
X_train, Y_train, y_encoded, scaler = preprocess(train_file, True)

# Ensure selected_features is not empty
if len(selected_features) > 0:
    X_train = X_train[:, selected_features]
    # print(X_train.shape)

weights = mini_batch_gradient_descent_adam(X_train, Y_train, y_encoded, batch_size, epochs)

test_data = pd.read_csv(test_file)
test_data = apply_one_hot_encoding(test_data, "mapping.json")
X_test = test_data.values.astype(np.float64)
X_test = scaler.transform(X_test)
X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1)

if len(selected_features) > 0:
    X_test = X_test[:, selected_features]

# print(X_test.shape)
# print(weights.shape)
Y_pred = X_test @ weights
Y_pred_probs = softmax(Y_pred)
Y_pred_labels = np.argmax(Y_pred_probs, axis=1)  # Assign the label based on the highest probability

with open(output_file, 'w') as f:
    for pred in Y_pred_labels:
        if(pred==0):
            f.write("-1\n")
        else:
            f.write(f"{pred}\n")

