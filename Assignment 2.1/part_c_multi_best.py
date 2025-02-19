import numpy as np
import os
import pickle
import argparse
import time
import matplotlib.pyplot as plt
from preprocessor import CustomImageDataset, DataLoader, numpy_transform
input_size = 625
layer1_size = 512
layer2_size = 256
layer3_size = 128
layer4_size = 32
output_size = 8

# Adam Optimiser Parameters
beta1 = 0.9
beta2 = 0.9999
epsilon = 1e-8

def xavier_initialization(output, input):
    W = np.random.randn(input,output)*np.sqrt(2/(input))
    return W.astype(np.float64)

def get_params():
    W1 = xavier_initialization(512,625).T
    W2 = xavier_initialization(256,512).T
    W3 = xavier_initialization(128,256).T
    W4 = xavier_initialization(32,128).T
    W5 = xavier_initialization(8,32).T
    weights = (W1,W2,W3,W4,W5)

    b1 = np.zeros((512, 1), dtype=np.float64)
    b2 = np.zeros((256, 1), dtype=np.float64)
    b3 = np.zeros((128, 1), dtype=np.float64)
    b4 = np.zeros((32, 1), dtype=np.float64)
    b5 = np.zeros((8,1), dtype=np.float64)
    biases = (b1,b2,b3,b4,b5)

    return weights,biases

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z)  # Subtract max for numerical stability
    return e_z / np.sum(e_z, axis=0, keepdims=True)

def forward_propagation(X, weights, biases):
    W1,W2,W3,W4,W5 = weights
    b1,b2,b3,b4,b5 = biases

    Z1 = np.dot(W1,X.T) + b1  #x = 256*625 w1 = 512*625
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2   
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    Z4 = np.dot(W4,A3) + b4
    A4 = sigmoid(Z4)
    Z5 = np.dot(W5,A4) + b5
    A5 = softmax(Z5)
    cache = (Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5)
    return A5,cache

def back_propogation(X,Y,cache,weights):
    n = X.shape[0]
    W1,W2,W3,W4,W5 = weights
    Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5 = cache

    num_classes = A5.shape[0]
    Y_one_hot = np.zeros((num_classes, n))
    Y_one_hot[Y, np.arange(n)] = 1

    dZ5 = 1/n*(A5 - Y_one_hot)
    dW5 = (dZ5 @ A4.T)
    db5 = np.sum(dZ5,axis=1)

    dZ4 = (A4 * (1 - A4)) * (W5.T @ dZ5)
    dW4 =  (dZ4 @ A3.T)
    db4 = np.sum(dZ4,axis=1)

    dZ3 = (A3 * (1 - A3)) * (W4.T @ dZ4)
    dW3 =  (dZ3 @ A2.T)
    db3 = np.sum(dZ3,axis=1)

    dZ2 = (A2 * (1 - A2)) * (W3.T @ dZ3)
    dW2 = (dZ2 @ A1.T)
    db2 =  np.sum(dZ2,axis=1)

    dZ1 = (A1 * (1 - A1)) * (W2.T @ dZ2)
    dW1 = (dZ1 @ X)
    db1 = np.sum(dZ1,axis=1)

    gradients = {
        'dW5': dW5, 'db5': db5,
        'dW4': dW4, 'db4': db4,
        'dW3': dW3, 'db3': db3,
        'dW2': dW2, 'db2': db2,
        'dW1': dW1, 'db1': db1
    }

    return gradients    

def initialize_adam_parameters(weights,biases):
    m = {}
    v = {}
    # print("-----------------------")
    for i in range(1, 6):
        m[f"dW{i}"] = np.zeros_like(weights[i-1])
        m[f"db{i}"] = np.zeros_like(biases[i-1])
        v[f"dW{i}"] = np.zeros_like(weights[i-1])
        v[f"db{i}"] = np.zeros_like(biases[i-1])
        
        # print(m[f"db{i}"].shape)
    # print("-----------------------")
    return m, v

def update_parameters_adam(weights, biases, gradients, learning_rate, m, v, t, beta1, beta2, epsilon):
    for i in range(1,6):
        # print(m[f"db{i}"].shape)
        # print("****************")
        m[f"dW{i}"] = beta1 * m[f"dW{i}"] + (1 - beta1) * gradients[f"dW{i}"]
        m[f"db{i}"] = beta1 * m[f"db{i}"] + (1 - beta1) * gradients[f"db{i}"].reshape(-1, 1)
        v[f"dW{i}"] = beta2 * v[f"dW{i}"] + (1 - beta2) * np.square(gradients[f"dW{i}"])
        v[f"db{i}"] = beta2 * v[f"db{i}"] + (1 - beta2) * np.square(gradients[f"db{i}"]).reshape(-1, 1)

        m_hat_dW = m[f"dW{i}"] / (1 - beta1 ** t)
        m_hat_db = m[f"db{i}"] / (1 - beta1 ** t)
        v_hat_dW = v[f"dW{i}"] / (1 - beta2 ** t)
        v_hat_db = v[f"db{i}"] / (1 - beta2 ** t)

        # print(biases[i-1].shape)
        # print(m_hat_db.shape)
        # print(v_hat_db.shape)
        # print(gradients[f"db{i}"].shape)
        # print(m[f"db{i}"].shape)

        weights[i-1] -= learning_rate * m_hat_dW / (np.sqrt(v_hat_dW) + epsilon)
        biases[i-1] -= learning_rate * m_hat_db.reshape(biases[i - 1].shape) / (np.sqrt(v_hat_db.reshape(biases[i - 1].shape)) + epsilon)

    return weights, biases,m,v


def compute_loss(Y, A4):
    m = Y.shape[0]
    logprobs = -np.log(A4[Y,range(m)])
    loss = np.sum(logprobs) / m
    return loss

def get_mini_batches(X, Y, batch_size):
    m = X.shape[0]  # Number of examples
    mini_batches = []

    # Partition the data into mini-batches without shuffling
    num_complete_batches = m // batch_size
    for i in range(num_complete_batches):
        X_mini = X[i * batch_size:(i + 1) * batch_size,:]
        Y_mini = Y[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_mini, Y_mini))

    # If the number of examples is not a multiple of batch_size, take the remaining examples
    if m % batch_size != 0:
        X_mini = X[num_complete_batches * batch_size:,:]
        Y_mini = Y[num_complete_batches * batch_size:]
        mini_batches.append((X_mini, Y_mini))

    return mini_batches

def train(X, Y, weights, biases, learning_rate, num_epochs, batch_size,dataloader):
    # Generate mini-batches once
    weights = list(weights)
    biases = list(biases)
    m,v = initialize_adam_parameters(weights,biases)
    t = 0
    loss_history = []
    mini_batches = get_mini_batches(X, Y, batch_size)
    start_time = time.time()
    max_time_seconds = 13*60
    epoch = 0
    while True:
        epoch_loss = 0
        for X_mini, Y_mini in mini_batches:
            # print(X_mini.shape)
            t+=1
            A4, cache = forward_propagation(X_mini, weights, biases)
            loss = compute_loss(Y_mini, A4)
            gradients = back_propogation(X_mini, Y_mini, cache, weights)
            weights, biases, m, v = update_parameters_adam(weights, biases, gradients, learning_rate, m, v, t, beta1, beta2, epsilon)
            epoch_loss += loss
        # if epoch % 100 == 0:
        avg_epoch_loss = epoch_loss / len(mini_batches)
        # loss_history.append(avg_epoch_loss)
        # print(f"Epoch {epoch} - Loss: {avg_epoch_loss}")
        elapsed_time = time.time()-start_time
        if elapsed_time >max_time_seconds:
            # print("Training stopped due to time limit")
            # print(elapsed_time)
            break
        epoch+=1

    return weights, biases, loss_history

def predict(X, weights, biases):
    A4, _ = forward_propagation(X, weights, biases)
    predictions = np.argmax(A4, axis=0)
    return predictions


def save_weights(weights, biases, filename="weights.pkl"):
    # Creating the dictionary structure as per the given guidelines
    weights_dict = {
        "weights": {
            "fc1": weights[0].T,
            "fc2": weights[1].T,
            "fc3": weights[2].T,
            "fc4": weights[3].T,
            "fc5": weights[4].T
        },
        "bias": {
            "fc1": biases[0].reshape(-1),
            "fc2": biases[1].reshape(-1),
            "fc3": biases[2].reshape(-1),
            "fc4": biases[3].reshape(-1),
            "fc5": biases[4].reshape(-1)
        }
    }

    # Saving the dictionary as a pickle file
    with open(filename, "wb") as file:
        pickle.dump(weights_dict, file)

def plot_loss_vs_epochs(loss_history, save_path='loss_vs_epochs.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    # plt.show()

def create_xy(dataloader):
    x = []
    y = []
    for images, labels in dataloader:
        x.append(images)
        y.append(labels)
    # Concatenate the lists to create numpy arrays
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y

def main(dataset_root, save_weights_path):
    np.random.seed(0)
    dataset = CustomImageDataset(root_dir=dataset_root, csv=os.path.join(dataset_root, "train.csv"), transform=numpy_transform)
    dataloader = DataLoader(dataset, batch_size=256)
    X, Y = create_xy(dataloader)
    weights,biases = get_params()
    batch_size = 256
    num_epochs = 1000
    learning_rate = 0.001
    w,b,loss_history = train(X, Y, weights, biases, learning_rate, num_epochs, batch_size,dataloader)
    save_weights(w,b,save_weights_path)
    # dataset_val = CustomImageDataset(root_dir=dataset_root, csv=os.path.join(dataset_root, "val.csv"), transform=numpy_transform)
    # dataloader_val = DataLoader(dataset_val, batch_size=256)
    # X_val,_ = create_xy(dataloader_val)
    # predictions = predict(X,w,b)
    # predictions = np.argmax(predictions, axis=1)
    # Save predictions as a pickle file
    # with open('predictions.pkl', 'wb') as f:
    #     pickle.dump(predictions, f)

    # plot_loss_vs_epochs(loss_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights as weights.pkl')

    args = parser.parse_args()
    main(args.dataset_root, args.save_weights_path)
