# DeepLense_SSL

This repository contains code and results for multi-class classification and self-supervised learning (SSL) tasks using deep learning models. The project is focused on classifying images into lenses using PyTorch or Keras.

## Common Test I: Multi-Class Classification

### Task
Build a model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy.

### Approach
ResNet-18 was chosen due to its ability to capture intricate patterns and features in the data, leading to better generalization and higher accuracy. It is not excessively deep, which prevents overfitting for grayscale images. Research supports that ResNet-18 is highly effective for lens classification.

### Results
- **Model**: ResNet-18
- **Training**: 5-fold cross-validation, 5 epochs each
- **Dataset**: 3X10k training images, 3X2.5k test images
- **Performance**: ROC-AUC score of 0.99 on test data

For more details, refer to the [notebook](https://github.com/ShubhamChauhan22222/DeepLense_SSL/blob/main/Common%20Test.ipynb).

## Specific Test VI: SSL on Real Dataset

### Task
Build a Self-Supervised Learning model for classifying the images into lenses using PyTorch or Keras. Pick the most appropriate approach and discuss your strategy.

### Approaches
1. **ResNet-18 with Rotation Pretext Training**
   - **Model**: ResNet-18
   - **Training**: Rotation pretext learning, projection head after feature extraction
   - [Notebook](https://github.com/ShubhamChauhan22222/DeepLense_SSL/blob/main/Rot_SSL_resnet18.ipynb)

2. **CNN + Self-Attention with Gaussian Noise Pretext Training**
   - **Model**: Small CNN with self-attention, projection head after feature extraction
   - **Pretext Training**: Gaussian Noise
   - [Notebook](https://github.com/ShubhamChauhan22222/DeepLense_SSL/blob/main/CNN-self-attention.ipynb)

### Fine-Tuning
Both models were fine-tuned on the same dataset of 215 images with 5-fold cross-validation. The backbone architecture weights were frozen, and the projection head was replaced with a classification head for lens and not-lens classification.

### Results
- **Dataset**: 215 images
- **Evaluation**: ROC-AUC metrics
- **Observations**: The first approach (ResNet-18 with rotation) outperformed the second approach due to the more robust backbone architecture. The small CNN with self-attention yielded lower scores.

### Future Work
The second approach suggests further exploration into combining different CNNs and self-attention mechanisms. Potential improvements include using Vision Transformers (VITs) and hybrid architectures, as well as diverse pretext training on larger datasets.

## Repository Structure
- `Common Test.ipynb`: Multi-Class Classification using ResNet-18
- `Rot_SSL_resnet18.ipynb`: SSL with ResNet-18 and rotation pretext training
- `CNN-self-attention.ipynb`: SSL with CNN, self-attention, and Gaussian noise pretext training

## References
- [Common Test Notebook](https://github.com/ShubhamChauhan22222/DeepLense_SSL/blob/main/Common%20Test.ipynb)
- [Rotation SSL ResNet-18 Notebook](https://github.com/ShubhamChauhan22222/DeepLense_SSL/blob/main/Rot_SSL_resnet18.ipynb)
- [CNN Self-Attention Notebook](https://github.com/ShubhamChauhan22222/DeepLense_SSL/blob/main/CNN-self-attention.ipynb)
