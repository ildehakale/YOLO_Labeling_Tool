# services/detection/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import sys
sys.path.append('../..')
from models.base import BoundingBox, DetectionResult

class IDetector(ABC):
    """Nesne tespiti için temel arayüz"""
    
    @abstractmethod
    def detect(self, image_path: str, confidence: float) -> List[BoundingBox]:
        """Görüntüde nesne tespiti yap"""
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float):
        """Güven eşiğini ayarla"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Model bilgilerini döndür"""
        pass

class IDetectorFactory(ABC):
    """Detector oluşturma fabrikası"""
    
    @abstractmethod
    def create_detector(self, model_path: Path, device: str = 'cpu') -> IDetector:
        """Model tipine göre uygun detector oluştur"""
        pass

class IDetectionService(ABC):
    """Tespit servisi arayüzü"""
    
    @abstractmethod
    def detect_single_image(self, image_path: str) -> DetectionResult:
        """Tek görüntüde tespit yap"""
        pass
    
    @abstractmethod
    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """Toplu tespit yap"""
        pass
    
    @abstractmethod
    def filter_duplicate_detections(
        self, 
        new_boxes: List[BoundingBox], 
        existing_boxes: List[BoundingBox], 
        iou_threshold: float = 0.1
    ) -> List[BoundingBox]:
        """Duplike tespitleri filtrele"""
        pass