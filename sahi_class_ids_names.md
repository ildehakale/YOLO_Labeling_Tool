# SAHI Class IDs and Names - Complete Reference

SAHI typically uses models trained on the COCO dataset (YOLOv8, YOLOv11, MMDetection, etc.). Therefore, class IDs and names follow the COCO dataset standard.

## COCO Dataset 80 Class List

### People and Animals
| Class ID | Name | Category |
|----------|------|----------|
| 0 | person | Human |
| 15 | bird | Animal |
| 16 | cat | Animal |
| 17 | dog | Animal |
| 18 | horse | Animal |
| 19 | sheep | Animal |
| 20 | cow | Animal |
| 21 | elephant | Animal |
| 22 | bear | Animal |
| 23 | zebra | Animal |
| 24 | giraffe | Animal |

### Vehicles
| Class ID | Name | Category |
|----------|------|----------|
| 1 | bicycle | Vehicle |
| 2 | car | Vehicle |
| 3 | motorcycle | Vehicle |
| 4 | airplane | Vehicle |
| 5 | bus | Vehicle |
| 6 | train | Vehicle |
| 7 | truck | Vehicle |
| 8 | boat | Vehicle |

### Traffic and Outdoor
| Class ID | Name | Category |
|----------|------|----------|
| 9 | traffic light | Traffic |
| 10 | fire hydrant | Outdoor |
| 11 | stop sign | Traffic |
| 12 | parking meter | Traffic |
| 13 | bench | Outdoor |

### Accessories and Items
| Class ID | Name | Category |
|----------|------|----------|
| 24 | backpack | Accessory |
| 25 | umbrella | Accessory |
| 26 | handbag | Accessory |
| 27 | tie | Accessory |
| 28 | suitcase | Accessory |

### Sports Equipment
| Class ID | Name | Category |
|----------|------|----------|
| 29 | frisbee | Sports |
| 30 | skis | Sports |
| 31 | snowboard | Sports |
| 32 | sports ball | Sports |
| 33 | kite | Sports |
| 34 | baseball bat | Sports |
| 35 | baseball glove | Sports |
| 36 | skateboard | Sports |
| 37 | surfboard | Sports |
| 38 | tennis racket | Sports |

### Beverages and Dining
| Class ID | Name | Category |
|----------|------|----------|
| 39 | bottle | Beverage |
| 40 | wine glass | Beverage |
| 41 | cup | Beverage |
| 42 | fork | Dining |
| 43 | knife | Dining |
| 44 | spoon | Dining |
| 45 | bowl | Dining |

### Food
| Class ID | Name | Category |
|----------|------|----------|
| 46 | banana | Fruit |
| 47 | apple | Fruit |
| 48 | sandwich | Food |
| 49 | orange | Fruit |
| 50 | broccoli | Vegetable |
| 51 | carrot | Vegetable |
| 52 | hot dog | Food |
| 53 | pizza | Food |
| 54 | donut | Food |
| 55 | cake | Food |

### Furniture and Home
| Class ID | Name | Category |
|----------|------|----------|
| 56 | chair | Furniture |
| 57 | couch | Furniture |
| 58 | potted plant | Decor |
| 59 | bed | Furniture |
| 60 | dining table | Furniture |
| 61 | toilet | Bathroom |

### Electronics
| Class ID | Name | Category |
|----------|------|----------|
| 62 | tv | Electronics |
| 63 | laptop | Electronics |
| 64 | mouse | Electronics |
| 65 | remote | Electronics |
| 66 | keyboard | Electronics |
| 67 | cell phone | Electronics |

### Appliances
| Class ID | Name | Category |
|----------|------|----------|
| 68 | microwave | Appliance |
| 69 | oven | Appliance |
| 70 | toaster | Appliance |
| 71 | sink | Appliance |
| 72 | refrigerator | Appliance |

### Miscellaneous
| Class ID | Name | Category |
|----------|------|----------|
| 73 | book | Stationery |
| 74 | clock | Decor |
| 75 | vase | Decor |
| 76 | scissors | Stationery |
| 77 | teddy bear | Toy |
| 78 | hair drier | Personal Care |
| 79 | toothbrush | Personal Care |

## Using Classes in SAHI

### Python Code Example
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# COCO class names list
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Load model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolo11n.pt",
    confidence_threshold=0.25,
    device="cpu"
)

# Run prediction
result = get_sliced_prediction(
    image="test_image.jpg",
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# Display results
for object_prediction in result.object_prediction_list:
    class_id = object_prediction.category.id
    class_name = COCO_CLASSES[class_id]
    confidence = object_prediction.score.value
    bbox = object_prediction.bbox

    print(f"Class ID: {class_id}")
    print(f"Class Name: {class_name}")
    print(f"Confidence: {confidence:.2f}")
    print(f"BBox: {bbox}")
    print("---")
```

### Filtering Specific Classes
```python
# Show only person (0) and car (2) classes
target_classes = [0, 2]  # person and car
filtered_predictions = [
    pred for pred in result.object_prediction_list
    if pred.category.id in target_classes
]

# Get class names
for pred in filtered_predictions:
    class_name = COCO_CLASSES[pred.category.id]
    print(f"Detected: {class_name}")
```

## Important Notes

1. **Model Dependency**: Class IDs depend on your model type:
   - YOLOv8/v11: COCO 80 classes
   - Custom models: Depends on training dataset
   - MMDetection: Depends on dataset configuration

2. **Index Start**: COCO class IDs start from 0 (person = 0)

3. **Background Class**: In some implementations, class 0 may be background, making person class 1

4. **Confidence Threshold**: Different confidence thresholds can be used for each class

5. **Custom Classes**: Models trained on your own dataset will have different class lists

This list contains the standard COCO dataset classes used with SAHI. To check your model's class list, inspect the metadata in your model file.
