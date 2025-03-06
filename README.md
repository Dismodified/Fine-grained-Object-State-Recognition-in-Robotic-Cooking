# **Fine-grained-Object-State-Recognition-in-Robotic-Cooking**

## **EEL4810 Project Group #7**
- Corey Barbera
- Jack Wilson
- Sean Kenney

## **Features**
- **Uses ResNet-18** with a modified classifier layer.
- **Supports CUDA acceleration** for faster training.
- **Applies data augmentation** (random rotations, flips, and color jittering).
- **Implements early stopping** to prevent overfitting.
- **Tracks training progress** with loss, accuracy, and F1-score.
- **Saves the best model** as `best_model.pth`.

## **Usage**
### **1. Train the Model**
Run `CNN.ipynb` inside Jupyter Notebook.
This will:
- Load images from `state_dataset/train/` and `state_dataset/valid/`
- Train the model for `50` epochs (or until early stopping)
- Save the best model as `best_model.pth`

### **2. Evaluate the Model**
The notebook will:
- Load `best_model.pth`
- Compute validation accuracy and F1-score
- Generate training loss/accuracy plots

### **3. Test on New Images**
To classify new images, modify the notebook's inference section and pass a test image.

## **Troubleshooting**
**CUDA Not Available?**
- Run `!nvidia-smi` to check your GPU status.
- Ensure you installed PyTorch with CUDA (`cu121`).
- Restart Jupyter Kernel if using Jupyter Notebook.

**Validation Performance is Worse than Training?**
- Increase **dropout rate**.
- Use **stronger data augmentation**.
- Reduce **learning rate** if validation loss fluctuates.

## **Submission**
This project will be submitted as a **ZIP file** containing:
- `CNN.ipynb` (Jupyter Notebook with training code)
- `state_dataset/` (Dataset folder)



