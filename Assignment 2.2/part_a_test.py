import torch
import torch.nn as nn
import numpy as np
from part_a_train import CNNBinaryClassifier
from torch.utils.data import DataLoader

def test_model(model, test_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data
            inputs = inputs.float()
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.append(preds.view(-1).cpu().numpy())

    return all_preds

def save_predictions(predictions,save_pred_path):
    import numpy as np
    import pickle
    pred_array = np.concatenate(predictions)
    with open(save_pred_path, "wb") as f:
        pickle.dump(pred_array, f)

if __name__ == "__main__":
    from testloader import CustomImageDataset, transform
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Predict using a trained model.')
    parser.add_argument('--test_dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--load_weights_path', type=str, required=True, help='Path to load the saved weights file')
    parser.add_argument('--save_predictions_path',type=str,required=True,help='Path to save the predictions')
    
    args = parser.parse_args()
    csv_path_test = os.path.join(args.test_dataset_root, "public_test.csv")
    
    # Create test dataset and dataloader
    test_dataset = CustomImageDataset(root_dir=args.test_dataset_root, csv=csv_path_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize the model (replace with your model class)
    model = CNNBinaryClassifier()  # Make sure this class is defined elsewhere in your code
    
    # Load the saved model weights
    model.load_state_dict(torch.load(args.load_weights_path))
    # model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

    # Get predictions on the test set
    predictions = test_model(model, test_loader)
    
    # Save predictions
    save_predictions(predictions,args.save_predictions_path)
