# FreshTrack AI — Trained Model Weights Directory

This folder is the target destination for your trained PyTorch weights file.

## Expected Weights File
- File name: `freshtrack_model.pth` (or `freshtrack_model.pt`)
- Path: `backend/models/freshtrack_model.pth`

## Current Status
- When **no file** is present here, FreshTrack AI operates in **DEMO MODE**.
  In Demo Mode, it analyzes image RGB, hue, saturation, and texture features to provide realistic simulated predictions, with clear labels and disclaimers indicating demo operation.
- Once you train your transfer learning model (using `notebooks/FreshTrack_AI_Training_TransferLearning.ipynb` in Google Colab), download `freshtrack_model.pth` and place it in this directory.
- Restart the backend server. FreshTrack AI will automatically detect the file, load the weights, and transition from `demo_heuristic` mode to `pytorch_trained` mode!

## Model Architecture Specification
The provided Colab notebook uses:
- **Base Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Heads**:
  1. Produce Classification Head (Classes: Apple, Banana, Orange, Tomato, Potato, Carrot, Bell Pepper, Broccoli, Strawberry, Cucumber, Onion, Lettuce, Avocado, Lemon)
  2. Freshness Head (Classes: Fresh, Fair / Consume Soon, Overripe / Spoiled)
- **Input Size**: 224 x 224 RGB, Normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
