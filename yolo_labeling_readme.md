# YOLO Labeling Tool

A visual YOLO format labeling tool built with PyQt5. Enables you to label images for object detection and edit existing labels with an intuitive interface.

## Features

- 🖼️ **Image Navigation**: Easy navigation between images in a folder
- 📦 **Bounding Box Drawing**: Left-click and drag to draw rectangles
- 🎯 **Class Labeling**: Assign class IDs to each bounding box
- ✏️ **Label Editing**: Select and delete existing labels
- 💾 **Auto-Save**: Labels are automatically saved in YOLO format
- 🔄 **Live Preview**: Green dashed line preview while drawing

## Requirements

```
Python 3.6+
PyQt5
```

## Installation

```bash
pip install PyQt5
```

## Usage

### Launch from Command Line

```bash
python yolo_labeler.py [image_folder] [labels_folder]
```

**Example:**
```bash
python yolo_labeler.py ./images ./labels
```

### GUI Folder Selection

If folder paths are not specified, the program will automatically open folder selection dialogs:

```bash
python yolo_labeler.py
```

## Controls

### Mouse Controls

| Action | Description |
|--------|-------------|
| **Left Click + Drag** | Draw new bounding box |
| **Right Click** | Select/deselect existing bounding box |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` (Right Arrow) | Next image |
| `←` (Left Arrow) | Previous image |
| `Delete` | Delete selected bounding boxes |
| `O` | Select new image folder |
| `L` | Select new labels folder |

## File Formats

### Supported Image Formats
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`

### YOLO Label Format
Labels are saved in the following format:
```
class_id center_x center_y width height
```

**Example:** `labels/image001.txt`
```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.15 0.25
```

- All coordinates are normalized between 0-1
- `center_x`, `center_y`: Center coordinates of the bounding box
- `width`, `height`: Width and height of the bounding box

## How It Works

1. **Image Loading**: Program lists all images in the specified folder
2. **Label Loading**: Searches for `.txt` files with the same name as each image
3. **Visualization**: Existing labels are displayed with red frames
4. **New Labels**: Left click + drag draws green preview
5. **Class Assignment**: Class ID is requested when drawing is completed
6. **Saving**: Label is automatically added to the corresponding `.txt` file

## Usage Tips

### Effective Labeling
- Try to cover the exact boundaries of objects
- Avoid very small or very large bounding boxes
- Use class IDs consistently

### Quick Workflow
1. Open an image
2. Frame objects with left click + drag
3. Enter class ID (e.g., 0, 1, 2...)
4. Use right arrow key to move to next image
5. Select incorrect labels with right click and delete with Delete key

## Status Bar Information

The status bar at the bottom displays:
- Current image position (e.g., 5/100)
- List of class IDs in the image
- Number of selected bounding boxes

## Example Folder Structure

```
project/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── img003.png
├── labels/
│   ├── img001.txt
│   ├── img002.txt
│   └── img003.txt
└── yolo_labeler.py
```

## Troubleshooting

### Program Won't Start
- Ensure PyQt5 is properly installed: `pip install PyQt5`
- Check that your Python version is 3.6+

### Images Not Displaying
- Ensure image files are in supported formats
- Verify the folder path is correct

### Labels Not Saving
- Ensure the labels folder has write permissions
- Check that class ID is numeric

## Visual Indicators

- **Red Rectangles**: Existing saved labels
- **Blue Rectangles**: Selected labels (thicker border)
- **Green Dashed Rectangle**: Preview while drawing
- **Status Bar**: Current image info and selected count

## Advanced Features

### Selection and Deletion
- Right-click on any bounding box to select it
- Selected boxes appear with blue borders
- Press Delete to remove selected boxes
- Multiple boxes can be selected simultaneously

### Folder Management
- Press `O` to change image folder during runtime
- Press `L` to change labels folder during runtime
- The tool automatically refreshes when folders are changed

## Performance Notes

- Images are automatically scaled to fit the display area
- Coordinate calculations maintain precision for YOLO format
- Large images are handled efficiently with Qt's scaling

## License

This tool is developed for educational and research purposes. Please check appropriate licenses for commercial use.

## Contributing

To contribute to this tool:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Changelog

### Version 1.0
- Initial release with basic labeling functionality
- Support for YOLO format export
- Interactive bounding box drawing
- Label selection and deletion
- Keyboard navigation

## Future Enhancements

- [ ] Support for additional image formats
- [ ] Batch operations
- [ ] Undo/Redo functionality
- [ ] Label validation
- [ ] Export to other formats (Pascal VOC, COCO)
- [ ] Class name management
- [ ] Zoom functionality