# **Fine-grained-Object-State-Recognition-in-Robotic-Cooking**

## **EEL4810 Project Group #7**
- Corey Barbera
- Jack Wilson
- Sean Kenney

## **Items Needed**
- CNN.ipynb
- state_dataset folder (for train and valid)

## **Items Created**
- best_model.pth

## **Features**
- CUDA acceleration for faster training.
- Early stopping to prevent overfitting.
- Tracks training progress with loss, accuracy, and F1-score.
- Saves as best_model.pth

## **How to Set Up**
Step-by-Step Guide to set up PyTorch within Jupyter Notebook.
https://medium.com/@kajaani1705/a-step-by-step-guide-to-using-pytorch-with-vscode-in-jupyter-notebook-f09c427f84e4

## **Usage**
Run CNN.ipynb inside Jupyter Notebook.
This will:
- Load images from state_dataset/train/ and state_dataset/valid/
- Train the model for 50 epochs (or until early stopping)
- Save the best model as best_model.pth

The notebook will:
- Load best_model.pth
- Show validation accuracy and F1-score
- Create training loss and accuracy plots

## **Troubleshooting**
- Make sure you installed PyTorch with CUDA

## **Issues with Code**
- Current warning displayed at beginning of run (Looking for workaround to fix that doesnt cause flucuation)
- Adjusting rates to stabilize train/valid
