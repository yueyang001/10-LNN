## Model Lightweighting Evaluation

This module evaluates the lightweight properties of the AudioCfC model.

### Files
- `evaluate_model.py` - Main evaluation script for model metrics

### Usage
```bash
cd experiments/model_lightweighting/
python evaluate_model.py
```

### Output Metrics
- Model Parameters (M) - Parameter count in millions
- Model Size (MB) - Memory footprint
- FLOPs (MFlops) - Floating point operations
- Latency (ms) - Inference latency
