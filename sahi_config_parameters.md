# SAHI Configuration Parameters - Complete Reference

SAHI (Slicing Aided Hyper Inference) is a framework for detecting small objects in large images. This document lists all configuration parameters used in SAHI with detailed explanations.

## 1. Slicing Parameters

### Basic Slice Parameters
- **`slice_height`** (int): Slice height in pixels
  - Examples: `256`, `512`, `640`
  - Default: Depends on image size

- **`slice_width`** (int): Slice width in pixels
  - Examples: `256`, `512`, `640`
  - Default: Depends on image size

### Overlap Parameters
- **`overlap_height_ratio`** (float): Overlap ratio for height
  - Range: 0.0 - 1.0
  - Examples: `0.1`, `0.2`, `0.3`
  - Default: `0.2`

- **`overlap_width_ratio`** (float): Overlap ratio for width
  - Range: 0.0 - 1.0
  - Examples: `0.1`, `0.2`, `0.3`
  - Default: `0.2`

- **`overlap_ratio`** (float): Single overlap ratio for both height and width
  - Range: 0.0 - 1.0
  - Example: `0.2`

## 2. Model Parameters

### Model Path and Configuration
- **`model_path`** (str): Path to the model file
  - Examples: `"path/to/model.pt"`, `"yolo11n.pt"`

- **`model_config_path`** (str): Path to model configuration file
  - For MMDetection config file path

- **`model_device`** (str): Device for inference
  - Options: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"`
  - Default: `"cpu"`

### Model Confidence Parameters
- **`model_confidence_threshold`** (float): Model confidence threshold
  - Range: 0.0 - 1.0
  - Examples: `0.25`, `0.5`, `0.7`
  - Default: `0.25`

- **`iou_threshold`** (float): IoU (Intersection over Union) threshold
  - Range: 0.0 - 1.0
  - Examples: `0.25`, `0.5`, `0.7`
  - Default: `0.5`

## 3. Postprocess Parameters

### Postprocess Type
- **`postprocess_type`** (str): Type of postprocess merging
  - Options: `"UNIONMERGE"`, `"NMS"`
  - Default: `"UNIONMERGE"`

### Postprocess Matching
- **`postprocess_match_metric`** (str): Matching metric
  - Options:
    - `"IOU"` (Intersection over Union) - Returns ratio of intersection area to union
    - `"IOS"` (Intersection over Smaller) - Returns ratio of intersection area to smaller box's area
  - Default: `"IOU"`

- **`postprocess_match_threshold`** (float): Matching threshold
  - Range: 0.0 - 1.0
  - Examples: `0.5`, `0.6`, `0.7`
  - Default: `0.5`

- **`postprocess_class_agnostic`** (bool): Class-agnostic matching
  - Options: `True`, `False`
  - Default: `False`

## 4. Output and Save Parameters

### Output Directories
- **`project`** (str): Project directory
  - Example: `"runs/predict"`

- **`name`** (str): Run name
  - Examples: `"exp1"`, `"test_run"`

- **`source`** (str): Source image/directory path
  - Examples: `"image.jpg"`, `"images/"`, `"video.mp4"`

### Save Options
- **`save_dir`** (str): Directory to save results

- **`export_pickle`** (bool): Export in pickle format
  - Default: `False`

- **`export_crop`** (bool): Export cropped predictions
  - Default: `False`

- **`visual_bbox_thickness`** (int): Bounding box thickness in visualization
  - Examples: `1`, `2`, `3`

- **`visual_text_size`** (float): Text size in visualization
  - Examples: `0.3`, `0.5`, `0.8`

- **`visual_text_thickness`** (int): Text thickness in visualization
  - Examples: `1`, `2`

## 5. COCO Dataset Parameters

### COCO Slicing
- **`ignore_negative_samples`** (bool): Ignore negative samples
  - Default: `False`

- **`out_dir`** (str): Output directory

- **`train_split_rate`** (float): Training data split rate
  - Range: 0.0 - 1.0
  - Examples: `0.8`, `0.9`

### COCO Evaluation
- **`dataset_json_path`** (str): COCO dataset JSON file path

- **`result_json_path`** (str): Result JSON file path

- **`type`** (str): Evaluation type
  - Options: `"bbox"`, `"segm"`

## 6. Video Inference Parameters

- **`video_path`** (str): Video file path

- **`frame_skip_interval`** (int): Number of frames to skip
  - Examples: `0`, `1`, `5`

## 7. FiftyOne Integration Parameters

- **`dataset_name`** (str): FiftyOne dataset name

- **`launch_fiftyone`** (bool): Launch FiftyOne UI
  - Default: `True`

## 8. Advanced Parameters

### Performance Optimization
- **`auto_slice_resolution`** (bool): Automatic slice resolution
  - Default: `True`

- **`slice_inference_batch_size`** (int): Slice inference batch size
  - Examples: `1`, `4`, `8`

### Debugging
- **`verbose`** (int): Verbose output level
  - Options: `0`, `1`, `2`

- **`return_dict`** (bool): Return results as dictionary
  - Default: `True`

## Usage Examples

### CLI Command
```bash
sahi predict \
    --model_path "yolo11n.pt" \
    --model_confidence_threshold 0.25 \
    --source "test_image.jpg" \
    --slice_height 640 \
    --slice_width 640 \
    --overlap_height_ratio 0.2 \
    --overlap_width_ratio 0.2 \
    --postprocess_type "UNIONMERGE" \
    --postprocess_match_threshold 0.5 \
    --export_pickle \
    --export_crop
```

### Python API
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Load model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolo11n.pt",
    confidence_threshold=0.25,
    device="cpu"
)

# Sliced prediction
result = get_sliced_prediction(
    image="test_image.jpg",
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="UNIONMERGE",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False,
    export_pickle=True,
    export_crop=True
)
```

## Best Practices

### Slice Size Recommendations
- **Small objects (pedestrians, vehicles)**: Use smaller slices (128-256)
- **Medium objects**: Use medium slices (256-512)
- **Large objects**: Use larger slices (512-1024)
- **High resolution images (4K+)**: Start with 512-640 slices

### Overlap Ratio Guidelines
- **Standard detection**: 0.2 overlap ratio works well for most cases
- **Dense objects**: Increase to 0.3-0.4 for better recall
- **Sparse objects**: Can reduce to 0.1 for faster processing
- **Trade-off**: Higher overlap improves accuracy but increases processing time

### Confidence Threshold Tuning
- **High precision needed**: Use 0.5-0.7 threshold
- **High recall needed**: Use 0.1-0.25 threshold
- **Balanced detection**: Use 0.25-0.4 threshold
- **Class-specific**: Different classes may need different thresholds

### Performance Optimization
- **GPU Memory**: Adjust slice size based on available VRAM
  - 4GB VRAM: Use 256-512 slices
  - 8GB VRAM: Use 512-1024 slices
  - 16GB+ VRAM: Use 1024-2048 slices
- **Batch Processing**: Use `slice_inference_batch_size` for faster inference
- **CPU Fallback**: Reduce slice size when using CPU (256-512 recommended)

### Postprocess Settings
- **UNIONMERGE**: Best for general use, handles overlapping predictions well
- **NMS**: Faster but may miss some edge cases
- **IOU Metric**: Good for regular shaped objects
- **IOS Metric**: Better for nested or contained objects

## Important Notes

- Parameters may vary depending on the framework (YOLOv8, MMDetection, etc.)
- Optimal parameter values depend on image size and object type
- Slice sizes should be adjusted based on GPU memory
- Overlap ratios improve detection quality but increase processing time
- Always validate parameter changes with test images before batch processing
- Consider using `auto_slice_resolution` for automatic parameter tuning
