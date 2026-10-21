# FreshTrack AI — Trained & Pretrained Model Weights Directory

This folder contains the active computer vision model weights for FreshTrack AI.

## Active Model Files

1. **`mobilenetv2.onnx`** (~14.2 MB)
   - **Type**: Pretrained MobileNetV2 Neural Network in Open Neural Network Exchange (ONNX) format.
   - **Execution Engine**: `onnxruntime` (CPU execution provider).
   - **Memory Footprint**: Extremely lightweight (<30 MB RAM), ideal for free-tier cloud platforms (e.g., Render 512MB RAM limit, Hugging Face Spaces).
   - **Capabilities**: Genuine deep convolutional feature extraction and produce recognition across supported fruit and vegetable classes (including Lemon, Banana, Apple, Orange, Bell Pepper, Broccoli, Cucumber, Strawberry, Lettuce, Potato, etc.).
   - **Solves**: Accurately classifies green lemons and limes as **Lemon** without falling back to Potato.

2. **`imagenet_classes.txt`**
   - Synset mapping for ImageNet-1000 categories used by MobileNetV2.

3. **`freshtrack_model.pth`** (Optional Custom Transfer Learning Weights)
   - Destination for custom dual-head PyTorch weights trained via `notebooks/FreshTrack_AI_Training_TransferLearning.ipynb` in Google Colab.
   - If placed here and PyTorch is installed, the service seamlessly prioritizes custom trained PyTorch weights.

## Loading Priority

1. **Pretrained MobileNetV2 ONNX (`mobilenetv2.onnx`)**: Loaded automatically via `onnxruntime`. Zero heavy dependencies, instant cold starts.
2. **Custom PyTorch Model (`freshtrack_model.pth`)**: Loaded if PyTorch is installed and custom weights exist.
3. **Demo Heuristics Fallback**: Only activated if neither ONNX nor PyTorch models can be loaded. Includes explicit warnings and disclaimers.
