# Results Summary

## Autonomous Driving Evaluation (CARLA Longest6 Benchmark)

### Overall Performance
- **Average Driving Score**: 74.49
- **Average Route Completion**: 82.71%
- **Average Infraction Penalty**: 0.894

### Detailed Metrics

#### Infraction Rates
- **Collisions with Pedestrians**: 0.015 (1.5%)
- **Collisions with Vehicles**: 0.064 (6.4%)
- **Collisions with Layout**: 0.000 (0%)
- **Red Light Infractions**: 0.022 (2.2%)
- **Stop Sign Infractions**: 0.143 (14.3%)
- **Off-road Infractions**: 0.000 (0%)
- **Route Deviations**: 0.000 (0%)
- **Route Timeouts**: 0.000 (0%)
- **Agent Blocked**: 0.360 (36.0%)

### Route-by-Route Results

Out of 36 routes evaluated:
- **Completed Successfully**: 20 routes (55.6%)
- **Failed (Agent Blocked)**: 16 routes (44.4%)

Routes with perfect scores (100% completion, no infractions):
- Route 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 24, 25, 27, 28, 29, 30, 31, 34, 35

### Key Observations

1. **High Route Completion**: 82.71% average completion rate demonstrates strong navigation capabilities
2. **Low Collision Rate**: Minimal collisions with pedestrians and vehicles
3. **Traffic Violations**: Some challenges with stop signs (14.3% infraction rate)
4. **Blocking Issues**: 36% of routes had blocking incidents, indicating potential improvements in path planning

### Detailed Results

Full evaluation results are available in `results/autopilot_longest6.json`

## Semantic Segmentation Results

### Training Metrics
- **Loss Function**: CrossEntropyLoss with class weights
- **Evaluation Metric**: Mean Intersection over Union (mIoU)
- **Classes**: 7 semantic classes (pedestrian, road line, road, sidewalk, vehicles, traffic-light, + background)

### Model Variants Evaluated
1. **PIDNetLidarv2** (Custom): Multi-modal fusion with spatial transformers
2. **PIDNetLidar**: LiDAR fusion variant
3. **PIDNetLidarSAM**: SAM-integrated variant
4. **ERFNet**: Baseline
5. **PIDNet**: Baseline

*Note: Detailed segmentation metrics (per-class IoU, mIoU) should be extracted from training logs if available.*

## Visualization

Route visualizations are available in `figures/longest6/` showing:
- Route paths
- Infraction locations
- Town maps with annotations

