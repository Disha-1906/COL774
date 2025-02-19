import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pickle


# Define a basic block used in ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def train_model(model, train_loader, save_weights_path, epochs=1000):
    
    model = model.float()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    # accuracy_list = []
    # loss_list = []
    start_time = time.time()
    max_time_seconds = 27*60
    epoch = 0
    while True:
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.long()

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
        elapsed_time = time.time()-start_time
        if elapsed_time>max_time_seconds:
            print("Training stopped due to time limit")
            print(elapsed_time)
            break
        epoch+=1
    torch.save(model.state_dict(), save_weights_path)  # Save model weights after each epoch
    # return accuracy_list,loss_list

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in test_loader:
            inputs = data
            inputs = inputs.float()
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
    
    print(f'Accuracy is {accuracy}%')
    return accuracy

def plot_loss_vs_epochs(history, save_path='loss_vs_epochs_c.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    # plt.show()

def plot_accuracy_vs_epochs(history, save_path='accuracy_vs_epochs_c.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    # plt.show()

def save_predictions(predictions):
    pred_array = np.concatenate(predictions)
    with open("predictions.pkl", "wb") as f:
        pickle.dump(pred_array, f)

if __name__ == "__main__":
    torch.manual_seed(0)
    from trainloader import CustomImageDataset, transform  # Update this import based on your dataloader script
    # from testloader import CustomImageDataset as TestDataset  # Update this import based on your test dataloader script

    parser = argparse.ArgumentParser(description='Train a neural network for multi-class classification.')
    parser.add_argument('--train_dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights')

    args = parser.parse_args()
    csv_path_train = os.path.join(args.train_dataset_root, "public_train.csv")
    csv_path_test = os.path.join(args.train_dataset_root, "public_test.csv")

    train_dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=csv_path_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    # test_dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=csv_path_test, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=8)  # ResNet-18
    train_model(model, train_loader, args.save_weights_path, epochs=10)
    # plot_loss_vs_epochs(loss_history)
    # plot_accuracy_vs_epochs(accuracy_history)

    # model.load_state_dict(torch.load("resnet_model_epoch_b.pth"))
    # predictions = test_model(model, test_loader)
    # save_predictions(predictions)
