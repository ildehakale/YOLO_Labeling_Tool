# repositories/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import sys
sys.path.append('..')
from models.base import Image, BoundingBox

class IImageRepository(ABC):
    """Görüntü depolama işlemleri için arayüz"""
    
    @abstractmethod
    def load_images_from_folder(self, folder_path: Path) -> List[Image]:
        """Klasörden görüntüleri yükle"""
        pass
    
    @abstractmethod
    def get_image(self, index: int) -> Optional[Image]:
        """Belirli bir görüntüyü getir"""
        pass
    
    @abstractmethod
    def get_total_count(self) -> int:
        """Toplam görüntü sayısını döndür"""
        pass

class ILabelRepository(ABC):
    """Etiket depolama işlemleri için arayüz"""
    
    @abstractmethod
    def load_labels(self, label_path: Path, img_width: int, img_height: int) -> List[BoundingBox]:
        """YOLO formatındaki etiketleri yükle"""
        pass
    
    @abstractmethod
    def save_labels(self, label_path: Path, boxes: List[BoundingBox], img_width: int, img_height: int):
        """Etiketleri YOLO formatında kaydet"""
        pass
    
    @abstractmethod
    def append_labels(self, label_path: Path, boxes: List[BoundingBox], img_width: int, img_height: int):
        """Mevcut etiketlere yeni kutular ekle"""
        pass

class IModelRepository(ABC):
    """Model dosyaları için arayüz"""
    
    @abstractmethod
    def get_available_models(self, model_folder: Path) -> List[str]:
        """Kullanılabilir model dosyalarını listele"""
        pass
    
    @abstractmethod
    def validate_model_path(self, model_path: Path) -> bool:
        """Model dosyasının geçerliliğini kontrol et"""
        pass