import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
from torch.utils.data import DataLoader
from part_b_train import CNNMultiClassClassifier

# Function to test the model and get predictions
def test_model(model, test_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data
            inputs = inputs.float().to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)  # Get the predicted class
            all_preds.append(preds.view(-1).cpu().numpy())

    return all_preds

# Function to save predictions to a file
def save_predictions(predictions,save_pred_path):
    import numpy as np
    import pickle
    pred_array = np.concatenate(predictions)
    with open(save_pred_path, "wb") as f:
        pickle.dump(pred_array, f)

if __name__ == "__main__":
    from testloader import CustomImageDataset, transform  # Ensure this script has the correct imports and definitions
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train a CNN for multi-class classification.')
    parser.add_argument('--test_dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--load_weights_path', type=str, required=True, help='Path to save the weights file')
    parser.add_argument('--save_predictions_path',type=str,required=True,help='Path to save the predictions')
    

    args = parser.parse_args()
    csv_path_test = os.path.join(args.test_dataset_root, "public_test.csv")

    test_dataset = CustomImageDataset(root_dir=args.test_dataset_root, csv=csv_path_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)  # Do not shuffle as per instructions

    # Initialize and train the model
    model = CNNMultiClassClassifier()

    # Load the trained weights and test the model
    model.load_state_dict(torch.load(args.load_weights_path))
    predictions = test_model(model, test_loader)
    save_predictions(predictions,args.save_predictions_path)