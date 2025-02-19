import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Hyperparameters for the CNN architecture
conv1_input_channels = 1
conv1_output_channels = 32
conv1_kernel_size = 3
pool1_kernel_size = 2
conv2_input_channels = 32
conv2_output_channels = 64
conv2_kernel_size = 3
pool2_kernel_size = 2
conv3_input_channels = 64
conv3_output_channels = 128
conv3_kernel_size = 3
pool3_kernel_size = 2

# Model class definition for multi-class classification
class CNNMultiClassClassifier(nn.Module):
    def __init__(self):
        super(CNNMultiClassClassifier, self).__init__()
        self.conv1 = nn.Conv2d(conv1_input_channels, conv1_output_channels, conv1_kernel_size, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(pool1_kernel_size, stride=2, padding=0)
        self.conv2 = nn.Conv2d(conv2_input_channels, conv2_output_channels, conv2_kernel_size, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(pool2_kernel_size, stride=2, padding=0)
        self.conv3 = nn.Conv2d(conv3_input_channels, conv3_output_channels, conv3_kernel_size, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(pool3_kernel_size, stride=1, padding=0)
        self.fc1 = nn.Linear(128 * 11 * 24, 512)  # Adjust input size based on image dimensions after pooling
        self.fc2 = nn.Linear(512, 8)  # Output size is 8 for the 8 classes

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = self.pool2(x)
        x = f.relu(self.conv3(x))
        x = self.pool3(x)
        # print(f"Shape before flattening: {x.shape}") 
        x = x.view(-1, 128 * 11 * 24)  # Flatten the output for the fully connected layer
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, train_loader,save_weights_path, epochs=8):
    
    model = model.float()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    accuracy_list = []
    loss_list = []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.float(), labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # accuracy = test_model(model,test_loader)
        # accuracy_list.append(accuracy)
        # avg_loss = running_loss/len(train_loader)
        # loss_list.append(avg_loss)
        # print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        torch.save(model.state_dict(),save_weights_path)
    return accuracy_list,loss_list

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            inputs = data.float()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds).reshape(-1)
    all_labels = np.concatenate(all_labels).reshape(-1)

    if all_preds.ndim != 1 or all_labels.ndim != 1:
        print("Predictions or labels are not 1D arrays.")
        return
    
    if len(all_preds) != len(all_labels):
        print("Mismatch between number of predictions and labels.")
        return
    accuracy = np.mean(all_preds == all_labels) * 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def plot_loss_vs_epochs(history, save_path='loss_vs_epochs_b.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    # plt.show()

def plot_accuracy_vs_epochs(history, save_path='accuracy_vs_epochs_b.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    # plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)
    from trainloader import CustomImageDataset, transform  # Ensure this script has the correct imports and definitions
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train a CNN for multi-class classification.')
    parser.add_argument('--train_dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights file')

    args = parser.parse_args()
    csv_path_train = os.path.join(args.train_dataset_root, "public_train.csv")
    csv_path_test = os.path.join(args.train_dataset_root, "public_test.csv")

    # Create datasets and loaders
    train_dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=csv_path_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)  # Do not shuffle as per instructions

    # Initialize and train the model
    model = CNNMultiClassClassifier()
    # test_dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=csv_path_test, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    accuracy_history , loss_history = train_model(model, train_loader,args.save_weights_path)
    # plot_loss_vs_epochs(loss_history)
    # plot_accuracy_vs_epochs(accuracy_history)
