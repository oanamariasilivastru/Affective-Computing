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

