# repositories/file_repositories.py
import os
from pathlib import Path
from typing import List, Optional
from PyQt5.QtGui import QPixmap
import sys
sys.path.append('..')
from models.base import Image, BoundingBox
from repositories.interfaces import IImageRepository, ILabelRepository, IModelRepository

class FileImageRepository(IImageRepository):
    """Dosya sisteminden görüntü yükleme"""
    
    def __init__(self):
        self.images: List[Image] = []
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    
    def load_images_from_folder(self, folder_path: Path) -> List[Image]:
        """Klasörden görüntüleri yükle"""
        self.images.clear()
        
        if not folder_path.exists():
            return []
        
        image_files = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in self.supported_formats
        ])
        
        for img_file in image_files:
            pixmap = QPixmap(str(img_file))
            if not pixmap.isNull():
                image = Image(
                    path=str(img_file),
                    filename=img_file.name,
                    width=pixmap.width(),
                    height=pixmap.height(),
                    boxes=[],
                    detector_boxes=[]
                )
                self.images.append(image)
        
        return self.images
    
    def get_image(self, index: int) -> Optional[Image]:
        """Belirli bir görüntüyü getir"""
        if 0 <= index < len(self.images):
            return self.images[index]
        return None
    
    def get_total_count(self) -> int:
        """Toplam görüntü sayısını döndür"""
        return len(self.images)

class YoloLabelRepository(ILabelRepository):
    """YOLO formatında etiket okuma/yazma"""
    
    def load_labels(self, label_path: Path, img_width: int, img_height: int) -> List[BoundingBox]:
        """YOLO formatındaki etiketleri yükle"""
        boxes = []
        
        if not label_path.exists():
            return boxes
        
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            
        for line in lines:
            try:
                box = BoundingBox.from_yolo_format(line, img_width, img_height)
                boxes.append(box)
            except (ValueError, IndexError):
                continue
        
        return boxes
    
    def save_labels(self, label_path: Path, boxes: List[BoundingBox], img_width: int, img_height: int):
        """Etiketleri YOLO formatında kaydet"""
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(box.to_yolo_format(img_width, img_height) + '\n')
    
    def append_labels(self, label_path: Path, boxes: List[BoundingBox], img_width: int, img_height: int):
        """Mevcut etiketlere yeni kutular ekle"""
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(label_path, 'a') as f:
            for box in boxes:
                f.write(box.to_yolo_format(img_width, img_height) + '\n')

class ModelFileRepository(IModelRepository):
    """Model dosyaları yönetimi"""
    
    def __init__(self):
        self.supported_formats = ('.pt', '.onnx')
    
    def get_available_models(self, model_folder: Path) -> List[str]:
        """Kullanılabilir model dosyalarını listele"""
        if not model_folder.exists():
            return []
        
        return sorted([
            f.name for f in model_folder.iterdir()
            if f.suffix.lower() in self.supported_formats
        ])
    
    def validate_model_path(self, model_path: Path) -> bool:
        """Model dosyasının geçerliliğini kontrol et"""
        return model_path.exists() and model_path.suffix.lower() in self.supported_formats