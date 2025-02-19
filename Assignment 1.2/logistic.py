import numpy as np
import pandas as pd
import sys
import time
import math
from sklearn import preprocessing

def preprocess(filepath, istransform):
    data = pd.read_csv(filepath)
    X_train = data.drop(columns=['Race']).values.astype(np.float64)
    y = data['Race'].values
    scaler = None
    y_encoded = pd.get_dummies(y).values.astype(np.float64)
    if istransform:
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1)
    else:
        X_train = np.hstack([np.ones((X_train.shape[0],1), dtype=np.float64), X_train])
    return X_train, y, y_encoded, scaler
# def preprocess(filepath,istransform):
#     data = pd.read_csv(filepath)
#     X_train = data.drop(columns=['Race']).values.astype(np.float64)
#     scaler = preprocessing.StandardScaler().fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1) #(Inserting the columns of 1 for bias).
    
#     y = data['Race'].values
#     y_encoded = pd.get_dummies(y).values.astype(np.float64)
#     # X_train = np.hstack([np.ones((X_train.shape[0],1), dtype=np.float64), X_train])
#     return X_train, y, y_encoded, scaler

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_gradient(X, Y, y_encoded, w, freq):
    n, m = X.shape  # n = number of samples, m = number of features
    k = w.shape[1] 
    z = np.dot(X, w)
    # prob = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    prob = softmax(z)
    Y_one_hot = np.zeros((n, k), dtype=np.float64)
    Y_one_hot[np.arange(n), Y - 1] = 1
    freq_values = freq[Y - 1].reshape(-1, 1)  
    gradient = np.dot(X.T, (prob - Y_one_hot) / (2 * n * freq_values))
    return gradient

def compute_loss(X, Y, y_encoded, w, freq):
    n = X.shape[0]  # Number of samples
    pred = np.dot(X, w)  # Predicted logits
    prob = softmax(pred)  # Apply softmax to get probabilities
    
    # Vectorized loss computation
    weighted_loss = y_encoded * np.log(prob) / (2 * freq)
    loss = -np.sum(weighted_loss) / n    
    return loss

def compute_loss_and_gradient(X,Y, y_encoded, weights, freq, batch_size):
    loss = compute_loss(X, Y, y_encoded, weights, freq)
    gradient = compute_gradient(X,Y, y_encoded, weights, freq)
    return loss, gradient

def ternary_search_learning_rate(X, Y,y_encoded, w, freq, loss, gradient, eta0, max_iterations=20):
    eta_l = 0
    eta_h = eta0
    g = gradient 
    while compute_loss(X,Y,y_encoded,w - eta_h * g,freq) < loss:
        eta_h *= 2
    for _ in range(max_iterations):
        eta_1 = (2 * eta_l + eta_h) / 3
        eta_2 = (eta_l + 2 * eta_h) / 3
        loss_eta1 = compute_loss(X,Y,y_encoded,w - eta_1 * g,freq)
        loss_eta2 = compute_loss(X,Y,y_encoded,w - eta_2 * g,freq)
        
        if loss_eta1 > loss_eta2:
            eta_l = eta_1
        elif loss_eta1 < loss_eta2:
            eta_h = eta_2
        else:
            eta_l = eta_1
            eta_h = eta_2
    eta = (eta_l + eta_h) / 2
    return eta

def mini_batch_gradient_descent(X,Y, y_encoded, batch_size, epochs, learning_rate):
    n_samples, n_features = X.shape
    n_classes = y_encoded.shape[1]
    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    freq = np.sum(y_encoded, axis=0)
    
    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            y_batch = y_encoded[i:i+batch_size]
            actual_batch_size = X_batch.shape[0]
            loss, gradient = compute_loss_and_gradient(X_batch,Y_batch, y_batch, weights, freq, actual_batch_size)
            weights -= learning_rate * gradient
            # print(f'Epoch {epoch+1}/{epochs}, Batch {i/batch_size} Loss: {loss}')
    return weights

def mini_batch_gradient_descent_2(X,Y, y_encoded, batch_size, epochs, eta, k):
    n_samples, n_features = X.shape
    n_classes = y_encoded.shape[1]
    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    freq = np.sum(y_encoded, axis=0)
    
    for epoch in range(epochs):
        learning_rate = eta / (1 + k * (epoch+1))
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            y_batch = y_encoded[i:i+batch_size]
            actual_batch_size = X_batch.shape[0]
            loss, gradient = compute_loss_and_gradient(X_batch,Y_batch, y_batch, weights, freq, actual_batch_size)
            weights -= learning_rate * gradient
            # print(f'Epoch {epoch+1}/{epochs}, Batch {i/batch_size} Loss: {loss}')
    return weights

def mini_batch_gradient_descent_3(X,Y, y_encoded, batch_size, epochs, eta0):
    n_samples, n_features = X.shape
    n_classes = y_encoded.shape[1]
    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    freq = np.sum(y_encoded, axis=0)
    for epoch in range(epochs):
        start_time = time.time()
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            y_batch = y_encoded[i:i+batch_size]
            actual_batch_size = X_batch.shape[0]
            loss, gradient = compute_loss_and_gradient(X_batch,Y_batch, y_batch, weights, freq, actual_batch_size)
            learning_rate = ternary_search_learning_rate(X_batch, Y_batch,y_batch, weights, freq, loss, gradient, eta0)
            weights -= learning_rate * gradient
            # print(f'Epoch {epoch+1}/{epochs}, Batch {i/batch_size} Loss: {loss}')
        # print(time.time()-start_time)
    return weights

def mini_batch_gradient_descent_adam(X, Y, y_encoded, batch_size=128, epochs=50, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, regularization_term=0.01):
    n_samples, n_features = X.shape
    n_classes = y_encoded.shape[1]
    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    freq = np.sum(y_encoded, axis=0)
    m_t = np.zeros_like(weights)
    v_t = np.zeros_like(weights)
    t = 0
    for epoch in range(epochs):
        start_time = time.time()
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
            # if((i//batch_size + 1 )== 1):
                # print(f'Epoch {epoch+1}/{epochs}, Batch {i//batch_size + 1} Loss: {loss}')
        # print(f'Time elapsed for epoch {epoch+1}: {time.time() - start_time} seconds')
    
    return weights

if __name__ == '__main__':
    if len(sys.argv) < 5 or (sys.argv[1] != 'a' and sys.argv[1] != 'b'):
        print("Usage: python3 logistic.py a train1.csv params.txt modelweights.txt")
        sys.exit(1)
    if sys.argv[1] == 'a':
        train_file = sys.argv[2]
        params_file = sys.argv[3]
        output_file = sys.argv[4]
        with open(params_file, 'r') as f:
            params = f.readlines()
        strategy = int(params[0].strip())
        batch_size = int(params[3].strip())
        epochs = int(params[2].strip())
        X,Y, y_encoded,_ = preprocess(train_file,False)
        if(strategy == 1):
            learning_rate = float(params[1].strip())
            final_weights = mini_batch_gradient_descent(X,Y, y_encoded, batch_size,  epochs,learning_rate)
        elif(strategy == 2):
            eta_k_string = params[1]
            eta_string, k_string = eta_k_string.split(',')
            eta = float(eta_string.strip())
            k = float(k_string.strip())
            final_weights = mini_batch_gradient_descent_2(X,Y, y_encoded, batch_size,  epochs,eta,k)
        elif(strategy==3):
            eta0 = float(params[1].strip())
            # print(eta0)
            final_weights = mini_batch_gradient_descent_3(X,Y, y_encoded, batch_size,  epochs,eta0)

        with open(output_file, 'w') as f:
            for row in final_weights:
                for weight in row:
                    f.write(f"{weight}\n")
    elif sys.argv[1] == 'b':
        train_file = sys.argv[2]
        test_file = sys.argv[3]
        model_weights = sys.argv[4]
        model_pred = sys.argv[5]
        batch_size = 16384
        epochs = 50
        X,Y, y_encoded, scaler = preprocess(train_file,True)
        weights = mini_batch_gradient_descent_adam(X,Y,y_encoded,batch_size,epochs)

        with open(model_weights, 'w') as f:
            for row in weights:
                for value in row:
                    f.write(f"{value}\n")

        data = pd.read_csv(test_file)
        X_test= data.values.astype(np.float64)
        X_test = scaler.transform(X_test)
        X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)
        Y_pred = X_test @ weights
        # Y_pred = softmax(Y_pred.T).T
        Y_pred = np.exp(Y_pred) / np.sum(np.exp(Y_pred), axis=1, keepdims=True)
        np.savetxt(model_pred, Y_pred, delimiter=',',fmt="%0.18e")

    
