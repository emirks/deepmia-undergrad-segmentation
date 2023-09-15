# Evaluation Guide

This guide explains how to evaluate your trained semantic segmentation models.

## Quick Start

### Basic Evaluation

```bash
cd team_code_transfuser/segmentation
python eval_seg.py \
    --model-type pidnet_lidar_v2 \
    --checkpoint-path ./outputs/carla/pretrained/pidnet_lidar_v2-best.pt \
    --dataset carla \
    --town town-1-val
```

### With Visualizations

```bash
python eval_seg.py \
    --model-type pidnet_lidar_v2 \
    --checkpoint-path ./outputs/carla/pretrained/pidnet_lidar_v2-best.pt \
    --dataset carla \
    --save-visualizations \
    --num-vis-samples 20
```

## Arguments

### Required Arguments

- `--checkpoint-path`: Path to your trained model checkpoint (.pt, .pth, or .th file)

### Model Selection

- `--model-type`: Type of model to evaluate
  - `pidnet_lidar_v2` (default): Your custom multi-modal fusion model
  - `pidnet_lidar`: PIDNet with LiDAR fusion
  - `pidnet`: Standard PIDNet
  - `erfnet`: ERFNet baseline
  - `transfuser`: TransFuser-based segmentation

### Dataset Options

- `--dataset`: Dataset to evaluate on
  - `carla`: CARLA dataset
  - `a2d2`: A2D2 dataset

- `--town`: For CARLA dataset, specify town name (e.g., `town-1-val`, `town-2-val`)

### Evaluation Settings

- `--batch-size`: Batch size for evaluation (default: 8)
- `--num-workers`: Number of data loading workers (default: 4)
- `--save-dir`: Custom directory to save results (default: auto-generated)

### Visualization

- `--save-visualizations`: Save visualization images of predictions
- `--num-vis-samples`: Number of samples to visualize (default: 10)

## Output

The evaluation script generates:

1. **evaluation_results.json**: Complete metrics in JSON format
2. **evaluation_summary.txt**: Human-readable summary
3. **confusion_matrix.png**: Confusion matrix visualization
4. **visualizations/**: Directory with prediction visualizations (if enabled)

### Metrics Computed

- **Mean IoU**: Average Intersection over Union across all classes
- **Mean Accuracy**: Average per-class accuracy
- **Overall Accuracy**: Pixel-wise accuracy
- **Per-class IoU**: IoU for each semantic class
- **Per-class Accuracy**: Accuracy for each semantic class
- **Average Loss**: Average cross-entropy loss

### Example Output

```
EVALUATION RESULTS
============================================================
Mean IoU: 0.7234
Mean Accuracy: 0.8567
Overall Accuracy: 0.9123
Average Loss: 0.2345

Per-class IoU:
  background: 0.9234
  pedestrian: 0.6543
  road_line: 0.7123
  road: 0.9234
  sidewalk: 0.7890
  vehicles: 0.8567
  traffic_light: 0.6789
```

## Examples

### Evaluate on CARLA Validation Set

```bash
python eval_seg.py \
    --model-type pidnet_lidar_v2 \
    --checkpoint-path ./outputs/carla/pretrained/pidnet_lidar_v2-best.pt \
    --dataset carla \
    --town town-1-val \
    --batch-size 16 \
    --save-visualizations
```

### Evaluate on A2D2 Dataset

```bash
python eval_seg.py \
    --model-type pidnet_lidar_v2 \
    --checkpoint-path ./outputs/carla/pretrained/pidnet_lidar_v2-best.pt \
    --dataset a2d2 \
    --batch-size 8
```

### Compare Multiple Models

```bash
# Evaluate PIDNetLidarv2
python eval_seg.py \
    --model-type pidnet_lidar_v2 \
    --checkpoint-path ./outputs/pidnet_lidar_v2-best.pt \
    --save-dir ./evaluations/pidnet_lidar_v2

# Evaluate baseline PIDNet
python eval_seg.py \
    --model-type pidnet \
    --checkpoint-path ./outputs/pidnet-best.pt \
    --save-dir ./evaluations/pidnet
```

## Troubleshooting

### Checkpoint Not Found

If you get a warning about checkpoint not found:
- Verify the path is correct
- Check file extension (.pt, .pth, or .th)
- Ensure checkpoint was saved during training

### Out of Memory

If you run out of GPU memory:
- Reduce `--batch-size` (e.g., from 8 to 4)
- Use `--num-workers 0` to reduce memory usage

### Dataset Issues

If dataset loading fails:
- Verify dataset paths in `config.py`
- Check that HDF5 files exist for CARLA dataset
- Ensure A2D2 dataset is properly formatted

### Model Type Mismatch

If you get errors about model architecture:
- Ensure `--model-type` matches the checkpoint
- Check that the model class is imported in `eval_seg.py`

## Integration with Training

The evaluation script uses the same model loading and data processing as training, ensuring consistency. You can evaluate at any point during or after training.

## For Publication

When preparing results for papers or reports:
1. Run evaluation on test set (not validation)
2. Save visualizations for qualitative analysis
3. Export confusion matrix for analysis
4. Compare multiple model variants
5. Report mean IoU and per-class metrics

