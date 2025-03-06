import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

def main():
    # ------------------------------
    # Setup Device
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Data Transformations for Test Set
    # ------------------------------
    # Use the same transformations as in validation.
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ------------------------------
    # Test Data Loading
    # ------------------------------
    test_dir = os.path.join('state_dataset', 'valid')
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ------------------------------
    # Model Setup
    # ------------------------------
    # Initialize the ResNet18 model with the same architecture used during training.
    model = models.resnet18(pretrained=False)  # No need for pretrained weights as we load saved ones.
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),  # Keep dropout as used during training.
        nn.Linear(num_features, 11)
    )
    model = model.to(device)

    # Load the saved model state.
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()  # Set model to evaluation mode.

    # ------------------------------
    # Evaluation on Test Set
    # ------------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and weighted F1 score.
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print("Test F1 Score: {:.4f}".format(test_f1))


if __name__ == '__main__':
    main()
