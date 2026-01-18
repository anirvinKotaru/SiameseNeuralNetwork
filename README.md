# Face Verification System

A deep learning project that implements facial verification using Siamese Neural Networks. The system learns to determine whether two face images belong to the same person by calculating similarity between embeddings.

## Overview

This project uses a Siamese network architecture to perform one-shot face verification. The model learns to create embeddings for face images and compares them using L1 distance to determine if two faces belong to the same person.

## Features

- **Real-time Data Collection**: Capture anchor and positive images directly from webcam
- **Siamese Architecture**: Twin neural networks with shared weights for embedding generation
- **L1 Distance Layer**: Custom similarity calculation between embeddings
- **Binary Classification**: Determines same person (1) or different person (0)
- **LFW Dataset Integration**: Uses Labeled Faces in the Wild for negative samples

## Tech Stack

- **TensorFlow 2.12.1**: Deep learning framework
- **Keras**: High-level neural network API
- **OpenCV**: Image processing and webcam capture
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-verification.git
cd face-verification
```

2. Install required packages:
```bash
pip install tensorflow==2.12.1 opencv-python matplotlib numpy
```

## Usage

### 1. Data Collection

Run the webcam capture cell to collect images:
- Press **'a'** to capture anchor images (your face)
- Press **'p'** to capture positive images (your face from different angles)
- Press **'q'** to quit

Collect at least 300-400 images of each type for best results.

### 2. Training

The notebook includes cells to:
- Load and preprocess the LFW dataset as negative samples
- Create training and test data pipelines
- Build the Siamese model architecture
- Train the model with binary cross-entropy loss

### 3. Model Architecture

**Embedding Network:**
- Input: 100x100x3 RGB images
- 4 Convolutional blocks with ReLU activation
- Max pooling layers for dimensionality reduction
- Flattening and dense layer
- Output: 4096-dimensional embedding vector

**Siamese Network:**
- Takes two images as input (anchor and validation)
- Passes both through the same embedding network
- Calculates L1 distance between embeddings
- Dense layer with sigmoid activation for final prediction

## Project Structure

```
face-verification/
├── data/
│   ├── anchor/         # Your face images
│   ├── positive/       # Your face (different angles)
│   └── negative/       # LFW dataset images
├── training_checkpoints/  # Model checkpoints
├── FacialVerification.ipynb
├── archive.zip         # LFW dataset
└── README.md
```

## Model Details

- **Input Shape**: (100, 100, 3)
- **Embedding Dimension**: 4096
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam (learning rate: 0.0001)
- **Total Parameters**: ~38.9M trainable parameters

## Dataset

- **Anchor Images**: User-captured via webcam
- **Positive Images**: Same person from different angles
- **Negative Images**: LFW (Labeled Faces in the Wild) dataset with 13,000+ images

## Configuration

GPU memory growth is enabled to avoid OOM errors:
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## Training Pipeline

1. Load anchor, positive, and negative image paths
2. Create labeled dataset (positive pairs = 1, negative pairs = 0)
3. Preprocess images (resize to 100x100, normalize to 0-1)
4. Split into 70% training, 30% testing
5. Batch and prefetch for optimal performance
6. Train with binary cross-entropy loss

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- LFW (Labeled Faces in the Wild) dataset
- Original Siamese Networks paper by Koch et al.
- Tutorial inspiration from Nicholas Renotte

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project requires Python 3.10 and TensorFlow 2.12.1 for compatibility. Using Python 3.14 may cause issues with TensorFlow installation.
