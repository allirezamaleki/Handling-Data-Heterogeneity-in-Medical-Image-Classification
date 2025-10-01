# Handling Data Heterogeneity in Medical Image Classification

A federated learning implementation for medical image classification using the SCAFFOLD algorithm on skin lesion datasets (ISIC 2019).

##  Overview

This project implements a federated learning approach to handle data heterogeneity in medical image classification. It uses the SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) algorithm to train Vision Transformer (ViT) models on distributed medical datasets while preserving data privacy.

### Key Features

- **Federated Learning**: Trains models across multiple medical centers without centralizing data
- **SCAFFOLD Algorithm**: Advanced federated optimization that handles data heterogeneity
- **Vision Transformers**: Uses pre-trained ViT models for superior image classification performance
- **Medical Image Focus**: Specialized data augmentations for medical imaging (ISIC skin lesion dataset)
- **Comprehensive Metrics**: Tracks accuracy, F1-score, precision, recall, and AUC-ROC

##  Dataset

The project uses the **ISIC 2019** (International Skin Imaging Collaboration) dataset, specifically the federated version:
- **Dataset**: `flwrlabs/fed-isic2019`
- **Task**: Multi-class skin lesion classification
- **Distribution**: Naturally partitioned by medical centers to simulate real-world federated scenarios
- **Classes**: Multiple skin lesion types (melanoma, nevus, basal cell carcinoma, etc.)

##  Architecture

### Model Architecture
- **Base Model**: Vision Transformer (ViT-Base/16-224)
- **Alternative Models**: CaiT and Swin Transformer (available in `model.py`)
- **Input Size**: 224×224 pixels
- **Pre-training**: ImageNet pre-trained weights

### Federated Learning Setup
- **Algorithm**: SCAFFOLD (Stochastic Controlled Averaging)
- **Aggregation**: FedAvg with control variates
- **Client Selection**: All available medical centers participate
- **Communication**: Model parameters only (no raw data sharing)

##  Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `timm>=0.9.0` - Pre-trained model library
- `tqdm>=4.65.0` - Progress bars
- `numpy>=1.24.0` - Numerical computing
- `torchmetrics>=1.0.0` - Evaluation metrics
- `flwr-datasets>=0.0.2` - Federated datasets
- `datasets>=2.14.0` - Hugging Face datasets

##  Usage

### Basic Usage
```bash
python main.py
```

### Configuration
Modify hyperparameters in `main.py`:

```python
LOCAL_EPOCHS = 5    # Local training epochs per round
ROUNDS = 15         # Total federated rounds
LR = 0.01          # Learning rate
SEED = 42          # Random seed for reproducibility
```

### Data Augmentation
The project includes medical image-specific augmentations in `data_loader.py`:
- Conservative rotations (15°)
- Horizontal flips
- Color jittering (brightness, contrast, saturation, hue)
- Small translations and scaling

##  Project Structure

```
.
├── main.py           # Entry point and configuration
├── federated.py      # SCAFFOLD federated learning implementation
├── model.py          # Neural network architectures (ViT, CaiT, Swin)
├── train.py          # Local training and evaluation functions
├── data_loader.py    # Data loading and preprocessing
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

### File Descriptions

#### `main.py`
- Configures federated learning parameters
- Initializes dataset and determines number of clients
- Orchestrates the federated training process
- Sets deterministic seed for reproducible results

#### `federated.py`
- Implements SCAFFOLD algorithm with control variates
- Manages federated aggregation (FedAvg)
- Coordinates training across multiple clients
- Tracks global model performance

#### `model.py`
- Defines Vision Transformer architectures
- Supports ViT-Base, CaiT, and Swin Transformer
- Pre-trained model loading and adaptation
- GPU/CPU device management

#### `train.py`
- Local client training with SCAFFOLD control variates
- Comprehensive evaluation with multiple metrics
- SGD optimization with weight decay
- Medical image-specific metric tracking

#### `data_loader.py`
- Federated dataset partitioning by medical center
- Medical image augmentation pipeline
- DataLoader creation for training and testing
- Natural heterogeneity preservation

##  Algorithms

### SCAFFOLD (Stochastic Controlled Averaging)

SCAFFOLD addresses client drift in federated learning by maintaining control variates:

1. **Global Control Variate (c)**: Tracks global gradient direction
2. **Client Control Variates (c_i)**: Track individual client gradient directions
3. **Gradient Correction**: Adjusts local gradients during training
4. **Controlled Aggregation**: Updates global model with drift correction

### Training Process

1. **Initialization**: 
   - Load pre-trained ViT model
   - Initialize global and client control variates

2. **Federated Rounds**:
   - For each client:
     - Download global model
     - Train locally with SCAFFOLD corrections
     - Compute control variate updates
   - Aggregate model updates
   - Update global control variate

3. **Evaluation**:
   - Test on local client data
   - Evaluate on global test set
   - Track comprehensive metrics

##  Metrics

The system tracks comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity/true positive rate
- **AUC-ROC**: Area under receiver operating characteristic curve

##  Customization

### Adding New Models
Extend `model.py` with new architectures:

```python
def custom_model(num_classes: int) -> nn.Module:
    model = timm.create_model('your_model_name', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model.to(DEVICE)
```

### Modifying Data Augmentation
Adjust augmentation pipeline in `data_loader.py`:

```python
def get_custom_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        # Add your augmentations here
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
```

### Hyperparameter Tuning
Key parameters to experiment with:
- `LOCAL_EPOCHS`: Balance between local overfitting and communication
- `LR`: Learning rate affects convergence speed
- `ROUNDS`: Total federated training duration
- Batch size in `data_loader.py`

##  Research Context

### Medical Image Federated Learning
- **Privacy Preservation**: No patient data leaves medical institutions
- **Data Heterogeneity**: Different patient populations across centers
- **Regulatory Compliance**: Meets healthcare data protection requirements
- **Collaborative Learning**: Leverages collective knowledge without data sharing

### SCAFFOLD Advantages
- **Handles Non-IID Data**: Effective with heterogeneous client data
- **Reduces Communication**: Fewer rounds needed for convergence
- **Theoretical Guarantees**: Proven convergence properties
- **Client Drift Correction**: Maintains global optimization direction

##  Results

The system provides detailed logging:

```
Round 1/15
>> Client 0
  Epoch  1 → Train Loss: 1.2345, Train Accuracy: 45.67%  |  Test Loss: 1.1234, Test Accuracy: 48.90%
  Test Metrics → F1: 0.4567, Precision: 0.4321, Recall: 0.4890, AUC: 0.7123

→ Global Results →
  Loss: 1.0987, Accuracy: 52.34%
  F1: 0.5234, Precision: 0.5123, Recall: 0.5345, AUC: 0.7456
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For questions, issues, or collaborations, please open an issue in the repository.

---

**Note**: This implementation is designed for research purposes. For production medical applications, ensure compliance with relevant healthcare regulations and data protection laws.
