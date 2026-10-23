# FreshTrack AI — Trained & Pretrained Model Weights Directory

This folder contains the active computer vision model weights for FreshTrack AI.

## Active Model Files

1. **`freshtrack_produce_v2.onnx`** (8.54 MB)
   - **Type**: Dedicated Produce Classifier v2 based on fine-tuned MobileNetV2 architecture in ONNX format.
   - **Execution Engine**: `onnxruntime` (CPU execution provider).
   - **Memory Footprint**: Ultra-lightweight (<20 MB RAM, 11-15ms latency), ideal for free-tier cloud platforms (e.g., Render 512MB RAM limit).
   - **Capabilities**: Genuine produce recognition across 16 categories with zero cross-confusion between Persimmon and Orange.
   - **Supported Classes**: Defined in `produce_classes.json` (Apple, Avocado, Banana, Bell Pepper, Broccoli, Eggplant, Lemon, Lettuce, Mango, Onion, Orange, Persimmon, Pomegranate, Potato, Tomato, Watermelon).

2. **`produce_classes.json`**
   - Class manifest and normalization metadata for `freshtrack_produce_v2.onnx`.

3. **`mobilenetv2.onnx`** (~14.2 MB)
   - **Type**: Pretrained MobileNetV2 Neural Network on ImageNet-1000 (secondary fallback).
   - **Execution Engine**: `onnxruntime` (CPU execution provider).

4. **`imagenet_classes.txt`**
   - Synset mapping for ImageNet-1000 categories used by fallback MobileNetV2.

5. **`freshtrack_model.pth`** (Optional Custom Transfer Learning Weights)
   - PyTorch checkpoint fallback.

## Loading Priority

1. **Produce Classifier v2 ONNX (`freshtrack_produce_v2.onnx`)**: Primary production model. Instant cold start, broad produce recognition including Persimmon.
2. **Fallback MobileNetV2 ONNX (`mobilenetv2.onnx`)**: Secondary fallback for ImageNet produce categories.
3. **Custom PyTorch Model (`freshtrack_model.pth`)**: Loaded if PyTorch is installed and custom weights exist.
4. **Model Unavailable Status**: If no model weights are found, the app reports `model_status='unavailable'` rather than using fake demo predictions.
