#!/bin/bash
# Training wrapper script

echo "🚀 Starting Lane Detection Training"
echo "===================================="

# Set default values
DATA_DIR="${1:-data}"
EPOCHS="${2:-10}"
BATCH_SIZE="${3:-8}"
LR="${4:-0.001}"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' not found!"
    echo "Please run data collection first or provide correct path."
    exit 1
fi

# Run training
cd scripts
python3 simple_train.py \
    --data_dir ../$DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --checkpoint_dir ../checkpoints

echo ""
echo "✅ Training complete! Checkpoints saved in ./checkpoints/"
