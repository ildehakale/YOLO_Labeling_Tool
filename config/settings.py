# config/settings.py
"""Uygulama yapılandırma ayarları"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

ALLOWED_CLASSES = {0, 2}   # person=0, car=2
IOU_THRESHOLD   = 0.10        # IoU >= 0.30 ise bastır
CONTAIN_RATIO   = 0.99        # Det veya GT'nin en az %60'ı içeriliyorsa bastır
SAME_CLASS_ONLY = True        # True: yalnızca aynı sınıfsa bastır (önerilir) 

@dataclass
class ApplicationSettings:
    """Uygulama genel ayarları"""
    app_name: str = "Modern SINIF Etiketleme Aracı"
    app_version: str = "3.0"
    organization: str = "SOLID Class Tools"
    
    # Window settings
    default_window_width: int = 1200
    default_window_height: int = 800
    sidebar_min_width: int = 280
    sidebar_max_width: int = 320
    
    # Paths
    models_folder: Path = Path("models")

    default_labels_subfolder: str = "labels"

@dataclass
class DetectionSettings:
    """Tespit ayarları"""
    default_confidence: float = 0.2
    default_slice_height: int = 256
    default_slice_width: int = 256
    default_overlap_ratio: float = 0.2
    iou_threshold: float = 0.1
    batch_size: int = 50
    
    # Allowed classes for detection
    allowed_classes: set = None


    def __post_init__(self):
        if self.allowed_classes is None:
            self.allowed_classes = {0, 2}

@dataclass
class UISettings:
    """UI görünüm ayarları"""
    # Colors
    primary_color: str = "#667eea"
    secondary_color: str = "#764ba2"
    danger_color: str = "#ff6b6b"
    success_color: str = "#10b981"
    warning_color: str = "#ffc107"
    
    # Fonts
    default_font: str = "Segoe UI"
    title_font_size: int = 16
    normal_font_size: int = 9
    small_font_size: int = 8
    
    # Box drawing
    normal_box_color: Tuple[int, int, int] = (255, 107, 107)
    selected_box_color: Tuple[int, int, int] = (66, 153, 225)
    detector_box_color: Tuple[int, int, int] = (16, 185, 129)
    detector_selected_color: Tuple[int, int, int] = (255, 193, 7)
    
    box_pen_width: int = 2
    selected_pen_width: int = 3
    
    # Zoom settings
    min_zoom: float = 0.1
    max_zoom: float = 10.0
    zoom_factor: float = 1.15

@dataclass
class ClassMapping:
    """Sınıf eşlemeleri"""
    class_names: Dict[int, str] = None

    def __post_init__(self):
        if self.class_names is None:
            # COCO dataset class names - only person and car
            self.class_names = {
                0: "person",
                2: "car"
            }
    
    def get_name(self, class_id: int) -> str:
        """Sınıf adını getir"""
        return self.class_names.get(class_id, f"Class_{class_id}")

# Global settings instances
app_settings = ApplicationSettings()
detection_settings = DetectionSettings()
ui_settings = UISettings()
class_mapping = ClassMapping()