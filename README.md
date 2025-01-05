# Affective Computing with DarkFace, AffectNet, and FER Datasets

## Overview
This project focuses on training and fine-tuning models for affective computing and facial recognition tasks. The primary datasets used are:

- **DarkFace** – for low-light face detection
- **AffectNet** – for facial expression recognition
- **FER (Facial Expression Recognition)** – for classifying emotions in facial images

## Project Structure
```
Affective-Computing/
│
├── darknet_model/               # Main model training and evaluation code
│   ├── checkpoints/            # Saved models and weights
│   ├── data/                   # Datasets and annotations
│   ├── layers/                 # Custom neural network layers
│   ├── models/                 # Model architectures
│   ├── splits/                 # Data split configurations (train/val/test)
│   ├── utils/                  # Utility scripts for preprocessing and augmentation
│   ├── main.py                 # Training and evaluation script
│   ├── prepare_data.py         # Data preparation script
│   ├── requirements.txt        # Project dependencies
│   └── check_intermediate.ipynb  # Jupyter notebook for testing
│
└── README.md                   # Project documentation
```

## Datasets
- **DarkFace**: Contains annotated low-light images for face detection.
- **AffectNet**: Large dataset for facial expressions and affect analysis.
- **FER2013**: Widely-used facial emotion recognition dataset.

## Model
The primary model used is based on **Faster R-CNN** and **ResNet** backbones, fine-tuned on facial datasets for detection and recognition tasks.
FER2013 Model

## Training
### Training Configuration
- **Batch Size**: 8
- **Learning Rate**: 0.001 (with decay)
- **Optimizer**: SGD with momentum (0.9)
- **Epochs**: 100
- **Augmentations**: Brightness/Contrast enhancement, Gaussian blur, Histogram Equalization

### Training Progress
- **Accuracy**:
  - Training: Increases from 40% to 92%
  - Validation: Increases from 42% to 90%
- **Loss**:
  - Training: Decreases from 0.9 to 0.04
  - Validation: Decreases from 0.8 to 0.05
  - 
## FER2013 Model

For the FER2013 dataset, a VGG-inspired convolutional neural network (CNN) is used. This model consists of multiple convolutional layers with increasing filter sizes, batch normalization, and ReLU activation functions. MaxPooling layers are applied to progressively reduce the spatial dimensions. Fully connected layers at the end are followed by dropout for regularization and a final softmax layer for emotion classification.

### Model Configuration (FER2013)
- **Input Size**: 48x48 grayscale images (single channel)  
- **Architecture**: VGG-style CNN with 4 convolutional blocks  
- **Batch Size**: 64  
- **Learning Rate**: 0.001 (Adam optimizer)  
- **Dropout**: 50% on fully connected layers  
- **Epochs**: 120  
- **Augmentations**: Rotation, width/height shift, shear, zoom, horizontal flip  

### Training Progress (FER2013)
- **Accuracy**:  
  - **Training**: Increases from 50% to 73%  
  - **Validation**: Increases from 48% to 68%  
- **Loss**:  
  - **Training**: Decreases from 1.9 to 0.72  
  - **Validation**: Decreases from 1.8 to 0.75
      
# Emotion Recognition - MobileNetV2 with CNN Blocks

This project implements an emotion recognition model based on MobileNetV2 as the base architecture. On top of MobileNetV2, 4 additional convolutional blocks were added to enhance feature extraction and classification performance. The model classifies facial images into various emotion categories.

## Model Configuration
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Additional Layers:** 4 convolutional blocks (Conv2D, BatchNormalization, ReLU, MaxPooling)
- **Fully Connected Layers:** Dense (512, 256) with Dropout (0.5)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Image Size:** 48x48 (grayscale)
- **Epochs:** 100
- **Batch size:** 64

### Training Progress (FER2013)
- **Accuracy**:  
  - **Training**: 97.5%  
  - **Validation**: 88.3% 
- **Loss**:  
  - **Training**: 0.05  
  - **Validation**: 0.18
## Confusion Matrix
The model performance was evaluated using a confusion matrix to visualize the classification results for each emotion category.

## Usage
To train the model:
```bash
python train_model.py

## Visualization
Learning curves for accuracy and loss during training and validation.
```python
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title("Training Progress")
plt.show()
```

## Requirements
Install dependencies with:
```
pip install -r requirements.txt
```

## Running the Project
1. Prepare the datasets by placing them in the `data/` folder.
2. Adjust configurations in `prepare_data.py`.
3. Run training:
```bash
python main.py
```
4. Evaluate the model on test data:
```bash
python main.py --evaluate
```

## Model Checkpoints
- The trained models are saved in the `checkpoints/` directory.
- Ensure Git LFS is used for large files (`*.pth` files > 100MB).

## Git Large File Storage (LFS)
If you encounter file size limits, install and configure Git LFS:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

## Contributing
Feel free to fork and contribute by submitting pull requests.

## License
This project is licensed under the MIT License.

