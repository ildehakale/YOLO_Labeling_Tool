# YOLO Class Labeling Tool

A professional-grade visual labeling tool for YOLO object detection models, featuring SAHI (Sliced Aided Hyper Inference) support, built with PyQt5.

![Python](https://img.shields.io/badge/python-3.8+-green)

## Demo

![Demo](assets/demo.gif)




## 🎯 Overview

The YOLO Class Labeling Tool is a comprehensive solution for creating, editing, and managing object detection datasets in YOLO format. It features an intuitive GUI, automatic detection using YOLO/ONNX models with SAHI support, and efficient batch processing capabilities.

**Key Highlights:**
- Modern PyQt5 interface with gradient designs
- SAHI integration for detecting small objects in large images
- Support for both PyTorch (.pt) and ONNX (.onnx) models
- Real-time visualization of bounding boxes
- Multiple box selection and batch deletion
- Automatic CUDA/CPU device detection
- Class filtering and statistics tracking






## 🚀 Installation



### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/ildehakale/YOLO_Labelling_Tool.git
cd YOLO_Labelling_Tool
```

Or download and extract the ZIP file.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### For LinuxOS (Ubuntu)
1. Install **libxcb** 
```bash
sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0
```
2. Uncomment the lines from **main.py** 

```bash
    # Linux platform settings (uncomment if needed)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
```


### Step 3: Prepare Model Directory

```bash
mkdir models
```

Place your YOLO model files (.pt or .onnx) in the `models/` folder.

### Step 4: Run the Application

```bash
python main.py
```

---

## 📁 Project Structure

```
YOLO_Labelling_Tool/
├── main.py                 # Application entry point
├── application.py          # Main application class
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── config/
│   ├── settings.py        # Application configuration
│   └── embedded_assets.py # Embedded logo
│
├── models/                # Place your .pt or .onnx models here
│   └── (your models)
│
├── controllers/
│   └── labeling_controller.py  # Main business logic
│
├── repositories/
│   ├── interfaces.py      # Repository interfaces
│   └── file_repositories.py  # File-based repositories
│
├── services/
│   └── detection/
│       ├── interfaces.py  # Detection interfaces
│       ├── detectors.py   # YOLO/ONNX/SAHI detectors
│       └── detection_service.py  # Detection logic
│
├── models/                # Data models (not ML models)
│   └── base.py           # BoundingBox, Image classes
│
├── ui/
│   └── components/
│       ├── base.py       # UI components
│       └── image_viewer.py  # Image display widget
│
└── views/
    ├── main_window.py    # Main window UI
    └── main_window_slots.py  # Event handlers
```

---

## ⚙️ Configuration

Edit `config/settings.py` to customize the application:

### Class Configuration

```python
# Allowed classes for detection (COCO format)
ALLOWED_CLASSES = {0, 2}   # 0=person, 2=car

# Class names mapping
class_names = {
    0: "person",
    2: "car"
}
```

### Detection Parameters

```python
# IoU threshold for duplicate suppression
IOU_THRESHOLD = 0.10

# Containment ratio threshold
CONTAIN_RATIO = 0.99

# Only suppress same-class overlaps
SAME_CLASS_ONLY = True

# Default confidence threshold
default_confidence = 0.2

# Default slice dimensions for SAHI
default_slice_height = 256
default_slice_width = 256
```

### UI Settings

```python
# Window dimensions
default_window_width = 1200
default_window_height = 800

# Colors (RGB)
normal_box_color = (255, 107, 107)        # Red for manual labels
selected_box_color = (66, 153, 225)       # Blue for selected
detector_box_color = (16, 185, 129)       # Green for SAHI detections
detector_selected_color = (255, 193, 7)   # Yellow for selected detections
```

---

## 🎮 Usage

### Basic Workflow

1. **Launch the application:**
   ```bash
   python main.py
   ```

2. **Select folders:**
   - Click "📷 Choose image File" to select your images folder
   - Click "🏷️ Choose label File" to select labels folder (auto-created if not exists)

3. **Load a model (optional for AI detection):**
   - Select a model from the "Class Models" list
   - The model will be loaded automatically

4. **Manual labeling:**
   - Left-click and drag to draw bounding boxes
   - Enter the class ID when prompted
   - Box is automatically saved

5. **AI-assisted labeling:**
   - Adjust "Confidence Threshold", "Slice Height", and "Slice Width"
   - Click "✨ APPLY SAHI" to run detection on current image
   - Click "✨ APPLY SAHI (50 IMG)" for batch processing
   - Green boxes show AI detections
   - Click "💾 Save SAHI results" to save them to labels

6. **Navigate:**
   - Use arrow keys or click "Next"/"Prev" buttons
   - Progress is saved automatically

---

## ⌨️ Controls

### Mouse Controls

| Action | Description |
|--------|-------------|
| **Left Click + Drag** | Draw new bounding box |
| **Right Click** | Toggle selection of bounding box (multi-select) |
| **Middle Click + Drag** | Pan the image |
| **Ctrl + Scroll** | Zoom in/out at cursor position |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` (Right Arrow) | Next image |
| `←` (Left Arrow) | Previous image |
| `Delete` | Delete all selected bounding boxes |

### Button Controls

- **📷 Choose image File** - Select images folder
- **🏷️ Choose label File** - Select labels folder
- **✨ APPLY SAHI** - Run detection on current image
- **✨ APPLY SAHI (50 IMG)** - Batch process 50 images
- **💾 Save SAHI results** - Save green boxes to labels
- **🗑️ Delete Selected** - Delete selected boxes
- **🧹 Clean all Boxes** - Delete all labels from current image
- **Fullscreen** - Toggle fullscreen mode
- **Prev/Next** - Navigate images

---

## 🔪 SAHI Detection

SAHI (Sliced Aided Hyper Inference) helps detect small objects in large images by slicing them into smaller patches.

### How It Works

1. Image is divided into overlapping slices
2. Each slice is processed by the YOLO model
3. Results are merged with NMS (Non-Maximum Suppression)
4. Overlapping predictions are filtered

### Parameters

**Confidence Threshold (0.0 - 1.0):**
- Lower values: More detections (may include false positives)
- Higher values: Fewer, more confident detections
- Default: 0.2

**Slice Height (100 - 4096 pixels):**
- Smaller slices: Better for tiny objects, slower
- Larger slices: Faster processing, may miss small objects
- Default: 256

**Slice Width (100 - 4096 pixels):**
- Similar to slice height
- Default: 256

### Tips for Best Results

- **Small objects**: Use smaller slices (128-256)
- **Large images**: Use larger slices (512-1024)
- **High accuracy**: Lower confidence threshold (0.1-0.2)
- **Speed**: Larger slices, higher confidence (0.3-0.5)

### Batch Processing

Process multiple images at once:
1. Click "✨ APPLY SAHI (50 IMG)"
2. Confirm the dialog
3. Wait for progress bar to complete
4. Review detections (green boxes)
5. Click "💾 Save SAHI results" to save all

---

## 📄 File Formats

### Supported Image Formats
- `.jpg`, `.jpeg` - JPEG images
- `.png` - PNG images


### YOLO Label Format

Labels are saved as `.txt` files with the same name as the image:

```
class_id center_x center_y width height
```

**Example:** `labels/image001.txt`
```
0 0.716797 0.395833 0.216406 0.147222
2 0.687500 0.379167 0.255469 0.158333
```

**Coordinate System:**
- All values are normalized (0.0 to 1.0)
- `center_x`, `center_y`: Center point of the box
- `width`, `height`: Dimensions of the box
- Relative to image dimensions

### Model Formats

**Supported Models:**
- `.pt` - PyTorch YOLO models (Ultralytics)
- `.onnx` - ONNX format models

**Model Requirements:**
- Must be trained for object detection
- Output should match YOLO format
- Compatible with Ultralytics or standard ONNX detection

---

## 🐛 Troubleshooting

### Application Won't Start

**Issue:** `ModuleNotFoundError: No module named 'PyQt5'`
**Solution:**
```bash
pip install PyQt5==5.15.11
```

**Issue:** `CUDA not available`
**Solution:** This is just a warning. The app will use CPU. To enable CUDA:
1. Install NVIDIA drivers
2. Install CUDA Toolkit 12.x
3. Reinstall PyTorch with CUDA support

### Images Not Displaying

**Issue:** Images appear blank
**Solutions:**
- Check image file format (must be .jpg, .png)
- Verify image is not corrupted
- Check file permissions

### SAHI Not Working

**Issue:** No detections appear
**Solutions:**
- Ensure a model is loaded (select from list)
- Lower the confidence threshold
- Check that model file exists in `models/` folder
- Verify model format (.pt or .onnx)

**Issue:** "Model state: Waiting..."
**Solution:** Select a model from the "Class Models" dropdown

### Performance Issues

**Issue:** Slow detection
**Solutions:**
- Increase slice size (512 or 1024)
- Use GPU if available
- Close other applications
- Use batch processing instead of single-image

**Issue:** High memory usage
**Solutions:**
- Increase slice size
- Close unnecessary applications
- Process fewer images in batch mode
- Use CPU instead of GPU (less memory)

### Labels Not Saving

**Issue:** Labels disappear after closing
**Solutions:**
- Check labels folder write permissions
- Ensure labels folder path is correct
- Check disk space
- Look for error messages in terminal

---

## 🔥 Advanced Features

### Multi-Box Selection

Select multiple boxes for batch operations:
1. Right-click first box (highlights in color)
2. Right-click additional boxes (adds to selection)
3. Right-click again to deselect
4. Press Delete to remove all selected boxes

The "Chosen" counter shows how many boxes are selected.

### Class Filtering

Filter displayed classes:
1. Look at "🎯 Class Filter" panel
2. Check/uncheck classes to show/hide
3. Filtering is visual only (doesn't delete labels)

### Zoom Functionality

Zoom in for precise labeling:
- Hold `Ctrl` and scroll up/down to zoom
- Zoom is centered at cursor position
- Middle-click and drag to pan
- Supports 0.5x to 5.0x zoom range





### Model Management

Load different models:
1. Add `.pt` or `.onnx` files to `models/` folder
2. Restart application (or refresh)
3. Select model from dropdown
4. Model loads with current slice/confidence settings

### IoU Suppression

The tool automatically prevents duplicate detections:
- SAHI detections overlapping with manual labels are suppressed
- Configurable via `IOU_THRESHOLD` in settings
- Helps maintain clean datasets

---

## 📊 Example Folder Structure

```
my_dataset/
├── images/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── img_0003.png
│
└── labels/
    ├── img_0001.txt
    ├── img_0002.txt
    └── img_0003.txt

YOLO_Labelling_Tool/
├── models/
│   ├── yolov8n.pt
│   └── my_custom_model.onnx
└── (application files)
```







## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Read this README thoroughly first

---


