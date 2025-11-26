# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Clone the Repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/ansh1113/ece484-lane-detection.git mp1
```

### 2. Install Dependencies
```bash
# Install system dependencies
sudo apt install ros-humble-cv-bridge \
                 ros-humble-image-transport \
                 ros-humble-sensor-msgs

# Install Python packages
cd ~/ros2_ws/src/mp1
pip install -r requirements.txt
```

### 3. Download Pre-trained Model (Optional)

If available, download checkpoint:
```bash
cd ~/ros2_ws/src/mp1
mkdir -p checkpoints
# Place your .pth file in checkpoints/
```

### 4. Build
```bash
cd ~/ros2_ws
colcon build --packages-select mp1
source install/setup.bash
```

### 5. Run Lane Detection

**Terminal 1: Launch Gazebo**
```bash
ros2 launch mp1 gem_vehicle.launch.py
```

**Terminal 2: Run Lane Detection**
```bash
cd ~/ros2_ws/src/mp1/src
python3 run_lane_detection.py --checkpoint ../checkpoints/simple_enet_checkpoint_epoch_10.pth
```

## 🎓 Training Your Own Model

### Collect Data
```bash
# Terminal 1: Launch sim
ros2 launch mp1 gem_vehicle.launch.py

# Terminal 2: Collect images
cd ~/ros2_ws/src/mp1/scripts
python3 generate_map.py --output_dir ../data/train
```

### Train Model
```bash
cd ~/ros2_ws/src/mp1/scripts
python3 simple_train.py \
    --data_dir ../data \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.001
```

### Evaluate
```bash
python3 eval.py \
    --checkpoint ../checkpoints/simple_enet_checkpoint_epoch_10.pth \
    --visualize
```

## 🎯 Expected Results

- **Training**: Val loss ~0.0014 after 10 epochs
- **Inference**: 30+ FPS on GPU, 5-10 FPS on CPU
- **Accuracy**: Clean binary lane masks

## 🆘 Troubleshooting

**Problem**: CUDA out of memory
- Reduce batch size: `--batch_size 4`
- Use CPU: Add `--device cpu` flag

**Problem**: No lane detected
- Check camera topic: `ros2 topic echo /front_single_camera/image_raw`
- Verify checkpoint path
- Ensure model is trained on similar data

See full [README.md](README.md) for detailed documentation.
