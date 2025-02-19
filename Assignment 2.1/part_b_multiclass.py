import numpy as np
import os
import pickle
import argparse
from preprocessor import CustomImageDataset, DataLoader, numpy_transform
input_size = 625
layer1_size = 512
layer2_size = 256
layer3_size = 128
output_size = 8



def xavier_initialization(output, input):
    W = np.random.randn(input,output)*np.sqrt(2/(input))
    return W.astype(np.float64)

def get_params():
    W1 = xavier_initialization(512,625).T
    W2 = xavier_initialization(256,512).T
    W3 = xavier_initialization(128,256).T
    W4 = xavier_initialization(8,128).T
    weights = (W1,W2,W3,W4)

    b1 = np.zeros((512, 1), dtype=np.float64)
    b2 = np.zeros((256, 1), dtype=np.float64)
    b3 = np.zeros((128, 1), dtype=np.float64)
    b4 = np.zeros((8, 1), dtype=np.float64)
    biases = (b1,b2,b3,b4)

    return weights,biases

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z)  # Subtract max for numerical stability
    return e_z / np.sum(e_z, axis=0, keepdims=True)

def forward_propagation(X, weights, biases):
    W1,W2,W3,W4 = weights
    b1,b2,b3,b4 = biases
    # print("for")
    # print(X.shape)
    Z1 = np.dot(W1,X.T) + b1  #x = 256*625 w1 = 512*625
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2   
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    Z4 = np.dot(W4,A3) + b4
    A4 = softmax(Z4)
    cache = (Z1,A1,Z2,A2,Z3,A3,Z4,A4)
    return A4,cache

def back_propogation(X,Y,cache,weights):
    n = X.shape[0]
    W1,W2,W3,W4 = weights
    Z1,A1,Z2,A2,Z3,A3,Z4,A4 = cache

    num_classes = A4.shape[0]
    Y_one_hot = np.zeros((num_classes, n))
    Y_one_hot[Y, np.arange(n)] = 1

    # print("in bq")
    # print(A4.shape)
    # print(Y.shape)
    # dZ4 = A4.copy()
    # dZ4[Y, np.arange(n)] -= 1
    # dZ4 = dZ4 / n
    dZ4 = 1/n*(A4 - Y_one_hot)
    dW4 = (dZ4 @ A3.T)
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
    # print(dZ1.shape)
    # print(db1.shape)
    gradients = {
        'dW4': dW4, 'db4': db4,
        'dW3': dW3, 'db3': db3,
        'dW2': dW2, 'db2': db2,
        'dW1': dW1, 'db1': db1
    }

    return gradients    

def update_parameters(weights, biases, gradients, learning_rate):
    W1, W2, W3, W4 = weights
    b1, b2, b3, b4 = biases
    # print("in")
    # print(b1.shape)
    # print(gradients['db1'].shape)
    W1 -= learning_rate * gradients['dW1']
    b1 -= learning_rate * gradients['db1'].reshape(b1.shape)
    
    W2 -= learning_rate * gradients['dW2']
    b2 -= learning_rate * gradients['db2'].reshape(b2.shape)
    
    W3 -= learning_rate * gradients['dW3']
    b3 -= learning_rate * gradients['db3'].reshape(b3.shape)
    
    W4 -= learning_rate * gradients['dW4']
    b4 -= learning_rate * gradients['db4'].reshape(b4.shape)
    
    (W1,W2,W3,W4) = weights
    (b1,b2,b3,b4) = biases

    return weights, biases


def compute_loss(Y, A4):
    m = Y.shape[0]
    # print(A4.sAhape)
    # print(Y.shape)
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

def train(X, Y, weights, biases, learning_rate, num_epochs, batch_size):
    # Generate mini-batches once
    
    mini_batches = get_mini_batches(X, Y, batch_size)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_mini, Y_mini in mini_batches:
            # print(X_mini.shape)
            A4, cache = forward_propagation(X_mini, weights, biases)
            loss = compute_loss(Y_mini, A4)
            gradients = back_propogation(X_mini, Y_mini, cache, weights)
            weights, biases = update_parameters(weights, biases, gradients, learning_rate)
            epoch_loss += loss
        # if epoch % 100 == 0:
        avg_epoch_loss = epoch_loss / len(mini_batches)
        # print(f"Epoch {epoch} - Loss: {avg_epoch_loss}")

    return weights, biases

def predict(X, weights, biases):
    A4, _ = forward_propagation(X, weights, biases)
    predictions = A4 > 0.5
    return predictions

def save_weights(weights, biases, filename="weights.pkl"):
    # Creating the dictionary structure as per the given guidelines
    weights_dict = {
        "weights": {
            "fc1": weights[0].T,
            "fc2": weights[1].T,
            "fc3": weights[2].T,
            "fc4": weights[3].T
        },
        "bias": {
            "fc1": biases[0].reshape(-1),
            "fc2": biases[1].reshape(-1),
            "fc3": biases[2].reshape(-1),
            "fc4": biases[3].reshape(-1)
        }
    }

    # Saving the dictionary as a pickle file
    with open(filename, "wb") as file:
        pickle.dump(weights_dict, file)

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
    # dataset_root = sys.argv[1]
    # train_data = pd.read_csv(f"{dataset_root}/train.csv")
    # X = train_data.iloc[:, :-1].values
    # Y = train_data.iloc[:, -1].values.reshape(-1,1)
    np.random.seed(0)
    dataset = CustomImageDataset(root_dir=dataset_root, csv=os.path.join(dataset_root, "train.csv"), transform=numpy_transform)
    dataloader = DataLoader(dataset, batch_size=256)
    X, Y = create_xy(dataloader)
    # print(X.shape)
    # print(Y.shape)
    # print(Y)
    weights,biases = get_params()
    batch_size = 256
    num_epochs = 15
    learning_rate = 0.001
    w,b = train(X, Y, weights, biases, learning_rate, num_epochs, batch_size)
    # w,b = weights,biases
    save_weights(w,b,save_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights as weights.pkl')

    args = parser.parse_args()
    main(args.dataset_root, args.save_weights_path)
