import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
conv1_input_channels = 1
conv1_output_channels = 32
conv1_kernel_size = 3
pool1_kernel_size = 2
conv2_input_channels = 32
conv2_output_channels = 64
conv2_kernel_size = 3
pool2_kernel_size = 2


def create_xy(dataloader):
    all_images = []
    all_labels = []
    
    for data in dataloader:
        # This line should match the __getitem__ output in CustomImageDataset
        print(data.shape)
        images,labels = data  # This assumes each batch contains exactly 2 elements: images and labels
        all_images.append(images)
        all_labels.append(labels)

    # Concatenate all the images and labels into a single tensor
    x = torch.cat(all_images, dim=0)
    y = torch.cat(all_labels, dim=0)

    return x, y

class CNNBinaryClassifier(nn.Module):
  def __init__(self):
    super(CNNBinaryClassifier,self).__init__()
    self.conv1 = nn.Conv2d(conv1_input_channels, conv1_output_channels, conv1_kernel_size, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(pool1_kernel_size, stride=2, padding=0)
    self.conv2 = nn.Conv2d(conv2_input_channels, conv2_output_channels, conv2_kernel_size, stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(pool2_kernel_size, stride=2, padding=0)
    self.fc1 = nn.Linear(64*12*25,1)

  def forward(self,x):
    x = f.relu(self.conv1(x))
    x = self.pool1(x)
    x = f.relu(self.conv2(x))
    x = self.pool2(x)
    x = x.view(-1,64*12*25)
    x = self.fc1(x)
    return x

def train_model(model, train_loader, save_weights_path, epochs=8):
    
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    accuracy_list = []
    loss_list = []
    # Create x and y using create_xy
    # x, y = create_xy(train_loader)
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        # Ensure to iterate through the dataset in batches
        # print(train_loader.shape())
        for i, data in enumerate(train_loader):
            # print(f"Batch {i}: {len(data)} elements")
            # print(f"Data type: {type(data)}, Data shape: {data[0].shape}, Labels: {data[1].shape}")
            inputs,labels = data
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)
            
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
        # print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        torch.save(model.state_dict(), save_weights_path)
    return accuracy_list,loss_list

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            inputs = data.float()
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
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


def save_predictions(predictions):
    import numpy as np
    import pickle
    pred_array = np.concatenate(predictions)
    with open("predictions.pkl", "wb") as f:
        pickle.dump(pred_array, f)


def plot_loss_vs_epochs(history, save_path='loss_vs_epochs.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    # plt.show()

def plot_accuracy_vs_epochs(history, save_path='accuracy_vs_epochs.png'):
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
    from trainloader import CustomImageDataset, transform
    # from testloader import CustomImageDataset
    from torch.utils.data import DataLoader
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--train_dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights as weights.pkl')

    args = parser.parse_args()
    csv_path_train = os.path.join(args.train_dataset_root, "public_train.csv")
    csv_path_test = os.path.join(args.train_dataset_root, "public_test.csv")
    train_dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=csv_path_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    model = CNNBinaryClassifier()
    # test_dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=csv_path_test, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # model = CNNBinaryClassifier()
    accuracy_history , loss_history = train_model(model, train_loader,args.save_weights_path)
    # plot_loss_vs_epochs(loss_history)
    # plot_accuracy_vs_epochs(accuracy_history)
    # model.load_state_dict(torch.load("part_a_binary_model_epoch.pth"))
    # predictions = test_model(model, test_loader)
    # save_predictions(predictions)