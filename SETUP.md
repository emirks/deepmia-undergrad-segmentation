# Setup Instructions

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate tfuse

# Install additional dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `team_code_transfuser/segmentation/config.py`:

```python
# Set your output directory
SAVE_DIR = "./outputs"  # or your preferred path
```

### 3. Dataset Preparation

#### For CARLA Dataset:
1. Download or generate CARLA dataset
2. Organize data in HDF5 format
3. Update dataset paths in `seg_dataset_carla.py`

#### For A2D2 Dataset:
1. Download A2D2 dataset
2. Update paths in `seg_dataset_a2d2.py`

### 4. CARLA Simulator Setup

1. Download CARLA 0.9.10.1 from [carla.org](https://carla.org/)
2. Extract and set `CARLA_ROOT` environment variable:
   ```bash
   export CARLA_ROOT=/path/to/carla
   ```

3. Set up Python API:
   ```bash
   cd $CARLA_ROOT/PythonAPI/carla
   pip install -e .
   ```

## Training

### Semantic Segmentation

```bash
cd team_code_transfuser/segmentation
python train_seg.py \
    --model-name pidnet_lidar_v2 \
    --num-epoch 50 \
    --batch-size 16 \
    --num-workers 8
```

### Autonomous Driving (TransFuser)

```bash
cd team_code_transfuser
python train.py \
    --batch_size 10 \
    --logdir ./logs \
    --root_dir /path/to/dataset \
    --parallel_training 0
```

## Evaluation

### Segmentation Evaluation

```bash
cd team_code_transfuser/segmentation
python eval_seg.py --load-path <model_checkpoint>
```

### CARLA Evaluation

```bash
# Start CARLA server
./CarlaUE4.sh --world-port=2000 -opengl

# Run evaluation
./leaderboard/scripts/local_evaluation.sh <carla_root> <working_dir>
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Import errors**:
   - Ensure all dependencies are installed
   - Check Python path includes project root

3. **Path errors**:
   - Update `SAVE_DIR` in config.py
   - Ensure dataset paths are correct
   - Check CARLA_ROOT environment variable

4. **CARLA connection issues**:
   - Verify CARLA server is running
   - Check port 2000 is available
   - Ensure CARLA version matches (0.9.10.1)

## System Requirements

- **OS**: Linux (Ubuntu 18.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 100GB+ for datasets
- **Python**: 3.7+
- **CUDA**: 10.2+

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch 1.11.0
- torchvision 0.12.0
- OpenCV
- NumPy
- Matplotlib
- TensorBoard

