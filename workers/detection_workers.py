# workers/detection_workers.py
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from typing import List, Dict
import sys
sys.path.append('..')
from models.base import BoundingBox
from services.detection.interfaces import IDetectionService

class SingleDetectionWorker(QThread):
    """Tek görüntü tespiti için worker"""
    
    finished = pyqtSignal(list)  # List[BoundingBox]
    error_occurred = pyqtSignal(str)
    
    def __init__(self, 
                 image_path: str,
                 detection_service: IDetectionService,
                 confidence: float):
        super().__init__()
        self.image_path = image_path
        self.detection_service = detection_service
        self.confidence = confidence
    
    def run(self):
        """Thread çalıştır"""
        try:
            result = self.detection_service.detect_single_image(
                self.image_path,
                self.confidence
            )
            self.finished.emit(result.boxes)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit([])

class BatchDetectionWorker(QThread):
    """Toplu tespit için worker"""
    
    progress_updated = pyqtSignal(int, int)  # current, total
    image_processed = pyqtSignal(str, list)  # filename, boxes
    finished = pyqtSignal(dict)  # all results
    error_occurred = pyqtSignal(str)
    
    def __init__(self,
                 image_paths: List[str],
                 detection_service: IDetectionService,
                 confidence: float,
                 existing_boxes: Dict[str, List[BoundingBox]]):
        super().__init__()
        self.image_paths = image_paths
        self.detection_service = detection_service
        self.confidence = confidence
        self.existing_boxes = existing_boxes
        self.mutex = QMutex()
        self.results = {}
    
    def run(self):
        """Thread çalıştır"""
        total = len(self.image_paths)
        
        try:
            for i, image_path in enumerate(self.image_paths):
                if self.isInterruptionRequested():
                    break
                
                # Tespit yap
                result = self.detection_service.detect_single_image(
                    image_path,
                    self.confidence
                )
                
                # Mevcut kutularla filtrele
                filename = image_path.split('/')[-1]
                existing = self.existing_boxes.get(filename, [])
                
                filtered = self.detection_service.filter_duplicate_detections(
                    result.boxes,
                    existing,
                    iou_threshold=0.1
                )
                
                # Sonuçları sakla
                self.mutex.lock()
                self.results[filename] = filtered
                self.mutex.unlock()
                
                # Sinyalleri gönder
                self.image_processed.emit(filename, filtered)
                self.progress_updated.emit(i + 1, total)
            
            self.finished.emit(self.results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class ModelLoadingWorker(QThread):
    """Model yükleme için worker"""
    
    finished = pyqtSignal(bool)  # success
    status_updated = pyqtSignal(str)  # status message
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path: str, detector_factory, device: str = 'cpu'):
        super().__init__()
        self.model_path = model_path
        self.detector_factory = detector_factory
        self.device = device
    
    def run(self):
        """Model yükle"""
        try:
            self.status_updated.emit(f"Model yükleniyor: {self.model_path}")
            
            from pathlib import Path
            detector = self.detector_factory.create_detector(
                Path(self.model_path),
                device=self.device
            )
            
            if detector:
                self.status_updated.emit("Model başarıyla yüklendi")
                self.finished.emit(True)
            else:
                self.error_occurred.emit("Model yüklenemedi")
                self.finished.emit(False)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit(False)