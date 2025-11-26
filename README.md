# Lane Detection with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

**Binary lane segmentation using ENet architecture with custom dataset - UIUC ECE 484 Safe Autonomy**

## 🎯 Key Results

- ✅ **High Accuracy** - Validation loss: 0.0014 after 10 epochs
- ✅ **Real-time Performance** - Inference at 30+ FPS on GPU
- ✅ **Robust Segmentation** - Binary lane mask generation
- ✅ **ROS2 Integration** - Live processing in Gazebo simulation
- ✅ **Custom Dataset** - Self-collected training data from simulator

---

## 📋 Table of Contents

- [Overview](#overview)
- [Network Architecture](#network-architecture)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Course Context](#course-context)
- [Team](#team)

---

## Overview

This project implements **binary lane segmentation** using a simplified ENet (Efficient Neural Network) architecture. The system detects lane markings from front-facing camera images and outputs a binary mask suitable for downstream path planning and control.

### Problem Statement

**Given:**
- Grayscale camera images (640×384 pixels)
- Lane-marked roads in Gazebo simulation
- Various lighting and track conditions

**Find:**
- Binary segmentation mask indicating lane pixels
- Real-time inference suitable for autonomous driving

### System Pipeline
```
Camera Image (640×384)
        ↓
  ENet Encoder
  (Downsampling)
        ↓
   Feature Maps
        ↓
  ENet Decoder
  (Upsampling)
        ↓
 Binary Mask (640×384)
        ↓
Bird's Eye View Transform
        ↓
Lane Following Controller
```

---

## 🏗️ Network Architecture

### Simplified ENet

ENet is designed for efficient semantic segmentation with fewer parameters than traditional architectures like FCN or SegNet.
```
Input: [B, 1, 384, 640]  # Grayscale images

Encoder:
├── InitialBlock: Conv(1→13) + MaxPool(1→3)
│   Output: [B, 16, 192, 320]
│
├── BottleneckBlock (×4): Downsampling + Residual
│   Output: [B, 64, 48, 80]
│
└── BottleneckBlock (×2): Feature refinement
    Output: [B, 128, 48, 80]

Decoder:
├── UpsamplingBottleneck (×2): Transpose Conv
│   Output: [B, 64, 192, 320]
│
└── UpsamplingBottleneck: Final upsampling
    Output: [B, 16, 384, 640]

Output Head:
└── Conv(16→1) + Sigmoid
    Output: [B, 1, 384, 640]  # Binary mask
```

### Key Components

**1. Bottleneck Block**
```python
class Bottleneck(nn.Module):
    """
    Efficient residual block with:
    - 1×1 Conv (dimension reduction)
    - 3×3 Conv (spatial processing)
    - 1×1 Conv (dimension expansion)
    - Skip connection
    """
```

**2. Downsampling**
```python
# Combines max pooling with convolution
MaxPool2d(kernel_size=2, stride=2)
# Reduces spatial dimensions by 2×
```

**3. Upsampling**
```python
# Transpose convolution for learned upsampling
ConvTranspose2d(kernel_size=3, stride=2, padding=1)
```

---

## 📊 Dataset

### Data Collection

- **Source**: Custom Gazebo simulation environment
- **Collection Method**: Manual driving + automated script
- **Total Images**: ~5000-10000 frames
- **Train/Val Split**: 80/20

### Dataset Structure
```
data/
├── train/
│   ├── images/          # Raw camera frames
│   │   ├── frame_0000.png
│   │   ├── frame_0001.png
│   │   └── ...
│   └── masks/           # Ground truth binary masks
│       ├── mask_0000.png
│       ├── mask_0001.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

### Preprocessing
```python
# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(),           # Convert to grayscale
    transforms.Resize((384, 640)),    # Standardize size
    transforms.ToTensor(),            # Convert to tensor [0,1]
])

# Mask preprocessing
mask_transform = transforms.Compose([
    transforms.Resize((384, 640)),
    transforms.ToTensor(),
    lambda x: (x > 0.5).float()      # Binarize
])
```

### Data Augmentation

Applied during training to improve generalization:
- Random brightness adjustment (±20%)
- Random horizontal flips
- Gaussian noise injection (σ=0.01)
- Random rotations (±5°)

---

## 🎓 Training Pipeline

### Loss Function

**Binary Cross-Entropy with Logits**
```python
criterion = nn.BCEWithLogitsLoss()

# Advantages:
# - Numerically stable (combines sigmoid + BCE)
# - Handles class imbalance (lane pixels << background)
# - Smooth gradients for optimization
```

### Optimizer

**Adam with Weight Decay**
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,              # Learning rate
    weight_decay=1e-4      # L2 regularization
)
```

### Training Hyperparameters (Final)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 8 | Max GPU memory allows; stable gradients |
| **Learning Rate** | 0.001 | Adam default; smooth convergence |
| **Epochs** | 10 | Validation plateaus after epoch 10 |
| **Optimizer** | Adam | Adaptive learning rate, good for CNNs |
| **Weight Decay** | 1e-4 | Prevents overfitting |

### Training Loop
```python
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, masks in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = validate(model, val_loader)
    
    # Log to WandB
    wandb.log({
        'epoch': epoch,
        'train_loss': loss.item(),
        'val_loss': val_loss
    })
```

---

## 🚀 Installation

### Prerequisites
```bash
# System requirements
Ubuntu 22.04
ROS2 Humble
Gazebo 11
Python 3.8+
CUDA 11.8+ (for GPU training)
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
pillow>=9.0.0
wandb>=0.15.0
tqdm>=4.65.0
```

### ROS2 Package Dependencies
```bash
sudo apt install ros-humble-cv-bridge \
                 ros-humble-image-transport \
                 ros-humble-sensor-msgs
```

### Build Instructions
```bash
# Clone repository into ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/ansh1113/ece484-lane-detection.git mp1

# Build workspace
cd ~/ros2_ws
colcon build --symlink-install

# Source workspace
source install/setup.bash
```

---

## 💻 Usage

### 1. Data Collection (Optional)
```bash
# Launch Gazebo simulation
ros2 launch mp1 gem_vehicle.launch.py

# Run data collection script
cd ~/ros2_ws/src/mp1/scripts
python3 generate_map.py --output_dir ../data/train
```

### 2. Train Model
```bash
cd ~/ros2_ws/src/mp1/scripts

# Train with default hyperparameters
python3 simple_train.py \
    --data_dir ../data \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.001

# Train with WandB logging
python3 simple_train.py \
    --wandb_project "lane-detection" \
    --wandb_entity "your-username"
```

### 3. Evaluate Model
```bash
# Evaluate on validation set
python3 eval.py \
    --checkpoint checkpoints/simple_enet_checkpoint_epoch_10.pth \
    --data_dir ../data/val

# Visualize predictions
python3 eval.py \
    --checkpoint checkpoints/simple_enet_checkpoint_epoch_10.pth \
    --visualize \
    --num_samples 10
```

### 4. Run Lane Detection in Simulation

**Terminal 1: Launch Gazebo**
```bash
ros2 launch mp1 gem_vehicle.launch.py
```

**Terminal 2: Run Lane Detection Node**
```bash
cd ~/ros2_ws/src/mp1/src
python3 run_lane_detection.py \
    --checkpoint ../checkpoints/simple_enet_checkpoint_epoch_10.pth
```

**Terminal 3: View Visualizations**
```bash
ros2 run rqt_image_view rqt_image_view
# Select topic: /mp1/lane_mask
```

### Command Line Arguments

**Training (`simple_train.py`):**
```
--data_dir          Path to dataset directory
--epochs            Number of training epochs (default: 10)
--batch_size        Batch size (default: 8)
--lr                Learning rate (default: 0.001)
--checkpoint_dir    Where to save checkpoints
--wandb_project     Weights & Biases project name
```

**Evaluation (`eval.py`):**
```
--checkpoint        Path to model checkpoint
--data_dir          Path to validation data
--visualize         Show prediction visualizations
--num_samples       Number of samples to visualize
```

---

## 📊 Experimental Results

### Training Curves

**Validation Loss over Epochs**
```
Epoch 1:  Val Loss = 0.0283
Epoch 2:  Val Loss = 0.0146
Epoch 3:  Val Loss = 0.0089
Epoch 4:  Val Loss = 0.0051
Epoch 5:  Val Loss = 0.0033
Epoch 6:  Val Loss = 0.0024
Epoch 7:  Val Loss = 0.0019
Epoch 8:  Val Loss = 0.0016
Epoch 9:  Val Loss = 0.0015
Epoch 10: Val Loss = 0.0014 ✓
```

**Observations:**
- Rapid convergence in first 5 epochs
- Loss plateaus after epoch 10
- No signs of overfitting (train/val losses similar)

### Qualitative Results

**Sample Predictions:**

| Input Image | Ground Truth | Prediction | Notes |
|-------------|--------------|------------|-------|
| Track straight section | Binary lane mask | High accuracy | Clean detection |
| Curved section | Binary lane mask | Good accuracy | Slight edge blur |
| Intersection | Binary lane mask | Moderate accuracy | Some confusion |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Val Loss** | 0.0014 |
| **Inference Time** | ~30ms (GPU) / ~200ms (CPU) |
| **Model Size** | ~15 MB |
| **Parameters** | ~3.7M |
| **FPS (GPU)** | 33 FPS |

---

## 🔧 Hyperparameter Tuning

### Batch Size Selection

**Tested**: 4, 8, 16

| Batch Size | GPU Memory | Training Speed | Gradient Quality | Selected |
|------------|------------|----------------|------------------|----------|
| 4 | 3.2 GB | Slow | Noisy | ❌ |
| **8** | **6.8 GB** | **Good** | **Stable** | **✅** |
| 16 | OOM (Out of Memory) | - | - | ❌ |

**Conclusion**: Batch size 8 is the largest that fits in GPU memory while providing stable gradient estimates.

### Learning Rate Selection

**Tested**: 0.0001, 0.001, 0.01

| Learning Rate | Convergence | Final Loss | Stability | Selected |
|---------------|-------------|------------|-----------|----------|
| 0.0001 | Very Slow | 0.0025 (20 epochs) | Stable | ❌ |
| **0.001** | **Fast** | **0.0014 (10 epochs)** | **Stable** | **✅** |
| 0.01 | Unstable | Diverged | Oscillating | ❌ |

**Conclusion**: 0.001 (Adam default) provides smooth, fast convergence with stable training.

### Number of Epochs

**Observation**: Validation loss plateaus after epoch 10
```
Epoch 10: Val Loss = 0.0014
Epoch 15: Val Loss = 0.0013 (minimal improvement)
Epoch 20: Val Loss = 0.0013 (no improvement)
```

**Conclusion**: 10 epochs is optimal - training longer risks overfitting without significant accuracy gains.

### Preventing Overfitting

**Techniques Applied:**
1. **L1/L2 Regularization**: Weight decay = 1e-4
2. **Dropout**: Applied in bottleneck blocks (p=0.1)
3. **Data Augmentation**: Brightness, rotation, flipping
4. **Early Stopping**: Monitor validation loss

**Result**: Training and validation losses remain close, indicating good generalization.

### Preventing Underfitting

**If Model Underperforms:**
1. **Increase Model Complexity**
   - Add more bottleneck blocks
   - Increase feature channels

2. **Train Longer with Better Features**
   - Increase epochs if loss still decreasing
   - Use pretrained weights (transfer learning)
   - Add skip connections (U-Net style)

3. **Improve Data Quality**
   - Collect more diverse training samples
   - Better data augmentation
   - Fix labeling errors

---

## 🌦️ Domain Adaptation Challenge

### Problem: Sunny → Snowy Conditions

**Challenge**: Model trained on sunny-day images fails on snowy-day images due to:
- Low contrast (snow-covered lanes)
- Different texture (snow vs. asphalt)
- Reduced visibility
- Changed color distribution

### Solutions

**1. Data Augmentation with Snow Simulation**
```python
def simulate_snow(image):
    # Reduce contrast
    image = image * 0.7 + 0.3
    
    # Add gaussian noise (snowflakes)
    noise = np.random.normal(0, 0.1, image.shape)
    image = image + noise
    
    # Adjust brightness (overcast sky)
    image = image * 0.8
    
    return np.clip(image, 0, 1)
```

**Benefits:**
- Cheap (no real snow data needed)
- Controllable (vary snow intensity)
- Improves generalization

**2. Transfer Learning + Fine-tuning**
```python
# Step 1: Pre-train on large sunny dataset
model.train_on_sunny_data(epochs=10)

# Step 2: Fine-tune on small snowy dataset
model.freeze_encoder()  # Keep general features
model.train_on_snowy_data(epochs=5, lr=0.0001)
```

**Benefits:**
- Reuses general features (edges, shapes)
- Requires fewer snowy training samples
- Faster convergence on new domain

---

## 📁 Project Structure
```
ece484-lane-detection/
├── src/
│   ├── run_lane_detection.py   # ROS2 inference node
│   ├── line_fit.py             # Polynomial lane fitting
│   └── util.py                 # Utility functions
├── models/
│   ├── __init__.py
│   ├── simple_enet.py          # ENet architecture
│   └── losses.py               # Loss functions
├── datasets/
│   ├── __init__.py
│   └── simple_lane_dataset.py  # PyTorch Dataset class
├── scripts/
│   ├── simple_train.py         # Training script
│   ├── eval.py                 # Evaluation script
│   ├── generate_map.py         # Data collection
│   ├── preprocess_data.py      # Data preprocessing
│   └── run_bev_conversion.py   # Bird's eye view
├── utils/
│   ├── __init__.py
│   ├── Line.py                 # Lane line class
│   └── ground_truth_generator.py
├── launch/
│   └── gem_vehicle.launch.py   # ROS2 launch file
├── config/
│   └── mp1.rviz               # RViz configuration
├── checkpoints/                # Saved model weights
├── data/                       # Training data (gitignored)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📚 Course Context

**Course**: ECE 484 - Principles of Safe Autonomy  
**Institution**: University of Illinois Urbana-Champaign  
**Semester**: Fall 2025  
**Project Type**: Machine Problem 1 (Group Project)  
**Team**: Autoshield

---

## 👥 Team

**Team Autoshield**
- Ansh Bhansali - anshb3@illinois.edu
- Het Patel - hcp4@illinois.edu
- Sunny Deshpande - sunnynd2@illinois.edu
- Keisuke Ogawa - ogawa3@illinois.edu

---

## 🎥 Demo Videos

**Video Links**: [Google Drive - MP1 Demos](https://drive.google.com/drive/folders/1tji1z8HDMwo6BdeU8uxJp2NpQ15aoZ15)

### Visualization

- **RViz**: Camera feed + binary lane mask overlay
- **WandB Dashboard**: Training curves and sample predictions
- **Real-time Inference**: Lane detection at 30+ FPS

---

## 📖 Academic Integrity Statement

This repository contains coursework from ECE 484 - Principles of Safe Autonomy at UIUC.  
Shared for portfolio and educational purposes after course completion.

**If you are currently enrolled in this course:**
- ❌ Do NOT copy this code for your assignments
- ✅ Use only as a learning reference
- ✅ Follow your course's academic integrity policy

Violations of academic integrity policies will be reported.

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- ECE 484 course staff for simulation environment and guidance
- UIUC Robotics Lab for computational resources
- Team Autoshield members for data collection and collaboration
- PyTorch team for excellent deep learning framework

---

## 📞 Contact

For questions about this implementation:
- **Ansh Bhansali**: anshbhansali5@gmail.com
- **GitHub**: [@ansh1113](https://github.com/ansh1113)

---

**⭐ If you find this helpful, please star the repository!**
