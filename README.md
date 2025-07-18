# Facemask Detection Project

A real-time face mask detection system using PyTorch and OpenCV. This project uses a CNN to classify whether people are wearing masks or not through live camera feed.

## Features

- **Real-time detection**: Live camera feed analysis
- **High accuracy**: CNN model trained on balanced dataset
- **Visual feedback**: Bounding boxes and confidence scores
- **Robust face detection**: Works with and without masks

## Project Structure

```
facemask-detection/
├── data/
│   ├── with_mask/       # Training images with masks
│   └── without_mask/    # Training images without masks
├── models/
│   ├── cnn.py          # CNN architecture
│   └── facemask_cnn.pth # Trained model weights
├── notebooks/
│   ├── LiveDetectionDocumentation.ipynb # Live detection walkthrough
│   └── ModelDocumentation.ipynb         # Model training & evaluation guide
├── src/
│   ├── download_dataset.py # Dataset download script
│   ├── evaluate.py     # Model evaluation
│   ├── live_detect.py  # Real-time detection
│   ├── load_data.py    # Data loading utilities
│   ├── organize_images.py # Image organization script
│   ├── test_model.py   # Model testing
│   └── train.py        # Training script
└── README.md           # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd facemask-detection
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision opencv-python datasets matplotlib
   ```

3. **Download the dataset**
   ```bash
   python src/download_dataset.py
   ```

4. **Organize images** (if needed)
   ```bash
   python src/organize_images.py
   ```

## Usage

### Training the Model

1. **Organize your data** (if using custom dataset)
   - Place masked face images in `data/with_mask/`
   - Place non-masked face images in `data/without_mask/`

2. **Train the model**
   ```bash
   python src/train.py
   ```
   - Trains for 5 epochs by default
   - Saves model weights to `models/facemask_cnn.pth`

### Evaluating the Model

```bash
python src/evaluate.py
```
This will output the validation accuracy of your trained model.

### Live Detection

```bash
python src/live_detect.py
```

**Controls:**
- Press `q` to quit the application
- The system will show bounding boxes around detected faces
- Green box = Mask detected
- Red box = No mask detected
- Confidence percentages displayed above each detection

### Testing the Model

```bash
python src/test_model.py
```
Quick test to verify the model architecture is working correctly.

## Documentation

For detailed step-by-step guides and explanations:

- **Model Training & Evaluation**: `notebooks/ModelDocumentation.ipynb`
  - Dataset preparation and analysis
  - Model architecture breakdown
  - Training process walkthrough
  - Performance evaluation and metrics

- **Live Detection System**: `notebooks/LiveDetectionDocumentation.ipynb`
  - Real-time detection implementation
  - Face detection optimization
  - Camera integration guide
  - Troubleshooting common issues

## Technical Details

### Model Architecture
- **CNN with 2 convolutional layers**
- **Input**: 224x224 RGB images
- **Output**: 2 classes (with_mask, without_mask)
- **Activation**: ReLU + MaxPooling
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss

### Face Detection
- Uses OpenCV's Haar Cascade classifier
- Optimized parameters for mask detection
- Fallback detection for challenging cases

## Troubleshooting

### Camera Issues
- **No camera found**: Change camera index in `src/live_detect.py` line 29
  ```python
  cap = cv2.VideoCapture(1)  # Try 0, 1, or 2
  ```

### Poor Detection
- **Faces not detected with masks**: The system uses multiple detection passes with different sensitivity levels
- **Wrong predictions**: Ensure your training data folders are named correctly (`with_mask`, `without_mask`)

### Performance Issues
- **Slow detection**: Reduce image resolution or use GPU if available
- **Memory errors**: Reduce batch size in training scripts

## Dataset

This project uses the [Face-Mask-Detection dataset](https://huggingface.co/datasets/DamarJati/Face-Mask-Detection) from Hugging Face, which contains balanced examples of faces with and without masks.

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- torchvision
- datasets
- matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues and pull requests to improve the project!

## Acknowledgments

- Dataset: [DamarJati/Face-Mask-Detection](https://huggingface.co/datasets/DamarJati/Face-Mask-Detection) on Hugging Face
- OpenCV for computer vision utilities
- PyTorch for deep learning framework
