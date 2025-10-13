# models/base.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from enum import Enum

class ClassType(Enum):
    PERSON = 0
    VEHICLE = 2
    DRONE = 3
    UAV = 4
    HELICOPTER = 5

@dataclass
class BoundingBox:
    """Tek bir bounding box'ı temsil eder"""
    class_id: int
    x: float
    y: float
    width: float
    height: float
    
    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """YOLO formatına dönüştür"""
        center_x = (self.x + self.width / 2) / img_width
        center_y = (self.y + self.height / 2) / img_height
        norm_width = self.width / img_width
        norm_height = self.height / img_height
        return f"{self.class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
    
    @classmethod
    def from_yolo_format(cls, yolo_line: str, img_width: int, img_height: int) -> 'BoundingBox':
        """YOLO formatından oluştur"""
        parts = yolo_line.strip().split()
        class_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x = (xc - bw/2) * img_width
        y = (yc - bh/2) * img_height
        w = bw * img_width
        h = bh * img_height
        return cls(class_id, x, y, w, h)
    
    def calculate_iou(self, other: 'BoundingBox') -> float:
        """IoU hesapla"""
        x1_a, y1_a = self.x, self.y
        x2_a, y2_a = self.x + self.width, self.y + self.height
        
        x1_b, y1_b = other.x, other.y
        x2_b, y2_b = other.x + other.width, other.y + other.height
        
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)
        
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area_a = self.width * self.height
        area_b = other.width * other.height
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0

@dataclass
class Image:
    """Tek bir görüntüyü temsil eder"""
    path: str
    filename: str
    width: int
    height: int
    boxes: List[BoundingBox]
    detector_boxes: List[BoundingBox]
    
    def get_all_boxes(self) -> List[BoundingBox]:
        """Tüm kutuları döndür"""
        return self.boxes + self.detector_boxes
    
    def get_visible_boxes(self, visible_classes: Set[int]) -> List[BoundingBox]:
        """Görünür sınıflardaki kutuları döndür"""
        return [b for b in self.get_all_boxes() if b.class_id in visible_classes]

@dataclass
class DetectionResult:
    """Tespit sonuçlarını temsil eder"""
    image_filename: str
    boxes: List[BoundingBox]
    confidence: float
    slice_height: int
    slice_width: int