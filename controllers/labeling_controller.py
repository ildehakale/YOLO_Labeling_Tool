# controllers/labeling_controller.py
from pathlib import Path
from typing import List, Optional, Set
from PyQt5.QtCore import QObject, pyqtSignal
import sys
sys.path.append('..')
from models.base import Image, BoundingBox, DetectionResult
from repositories.interfaces import IImageRepository, ILabelRepository, IModelRepository
from services.detection.interfaces import IDetectionService, IDetectorFactory
from config.settings import detection_settings
from config.settings import ALLOWED_CLASSES, IOU_THRESHOLD, CONTAIN_RATIO, SAME_CLASS_ONLY
class LabelingController(QObject):
    """Ana kontrol mantığı - MVC Controller"""
    
    # Signals
    image_changed = pyqtSignal(object)  # Image
    stats_updated = pyqtSignal(dict)  # stats dictionary
    detection_completed = pyqtSignal(list)  # detection results
    batch_progress = pyqtSignal(int, int)  # current, total
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, 
                 image_repo: IImageRepository,
                 label_repo: ILabelRepository,
                 model_repo: IModelRepository,
                 detector_factory: IDetectorFactory):
        super().__init__()
        
        # Repositories
        self.image_repo = image_repo
        self.label_repo = label_repo
        self.model_repo = model_repo
        self.detector_factory = detector_factory
        
        # Services
        self.detection_service: Optional[IDetectionService] = None

        # State
        self.current_index = 0
        self.current_image: Optional[Image] = None
        self.image_folder = Path()
        self.labels_folder = Path()
        self.temp_detections = {}  # image_name -> List[BoundingBox]
        self.current_model_path: Optional[Path] = None  # Store current model path
        
        # Settings
        self.confidence_threshold = 0.2
        self.slice_height = 256
        self.slice_width = 256
        self.use_sahi = True

        # Device detection
        self.device = self._detect_device()
    
    def set_image_folder(self, folder: str):
        """Görüntü klasörünü ayarla"""
        self.image_folder = Path(folder)
        
        # Otomatik labels klasörü oluştur
        if not self.labels_folder:
            self.labels_folder = self.image_folder / "labels"
            self.labels_folder.mkdir(exist_ok=True)
        
        # Görüntüleri yükle
        self.image_repo.load_images_from_folder(self.image_folder)
        self.current_index = 0
        self._load_current_image()
    
    def set_labels_folder(self, folder: str):
        """Etiket klasörünü ayarla"""
        self.labels_folder = Path(folder)
        self.labels_folder.mkdir(exist_ok=True)
        self._load_current_image()
    
    def load_model(self, model_name: str):
        """Model yükle"""
        model_path = Path("models") / model_name

        if not self.model_repo.validate_model_path(model_path):
            self.error_occurred.emit(f"Geçersiz model: {model_path}")
            return False

        try:
            detector = self.detector_factory.create_detector(
                model_path,
                device=self.device,
                use_sahi=self.use_sahi,
                slice_height=self.slice_height,
                slice_width=self.slice_width
            )

            from services.detection.detection_service import DetectionService
            self.detection_service = DetectionService(detector)
            self.current_model_path = model_path  # Store model path for reloading
            return True

        except Exception as e:
            self.error_occurred.emit(f"Model yüklenemedi: {e}")
            return False
    
    def _load_current_image(self):
        """Mevcut görüntüyü yükle"""
        self.current_image = self.image_repo.get_image(self.current_index)
        
        if self.current_image:
            # Etiketleri yükle
            label_file = self.labels_folder / f"{Path(self.current_image.filename).stem}.txt"
            boxes = self.label_repo.load_labels(
                label_file,
                self.current_image.width,
                self.current_image.height
            )
            self.current_image.boxes = boxes
            
            # Geçici tespitleri yükle
            if self.current_image.filename in self.temp_detections:
                self.current_image.detector_boxes = self.temp_detections[self.current_image.filename]
            
            self.image_changed.emit(self.current_image)
            self._update_stats()
    
    def next_image(self):
        """Sonraki görüntüye geç"""
        total = self.image_repo.get_total_count()
        if self.current_index < total - 1:
            self.current_index += 1
            self._load_current_image()
    
    def previous_image(self):
        """Önceki görüntüye geç"""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()
    
    def add_manual_box(self, class_id: int, x: float, y: float, w: float, h: float):
        """Manuel kutu ekle"""
        if not self.current_image:
            return
        
        box = BoundingBox(class_id, x, y, w, h)
        self.current_image.boxes.append(box)
        
        # Hemen kaydet
        self._save_current_labels()
        self.image_changed.emit(self.current_image)
        self._update_stats()
    
    def delete_selected_boxes(self, indices: Set[int], is_detector: bool = False):
        """Seçili kutuları sil"""
        if not self.current_image:
            return
        
        if is_detector:
            # Detector kutularını sil
            self.current_image.detector_boxes = [
                box for i, box in enumerate(self.current_image.detector_boxes)
                if i not in indices
            ]
            # Temp'ten de sil
            if self.current_image.filename in self.temp_detections:
                self.temp_detections[self.current_image.filename] = self.current_image.detector_boxes
        else:
            # Normal kutuları sil
            self.current_image.boxes = [
                box for i, box in enumerate(self.current_image.boxes)
                if i not in indices
            ]
            self._save_current_labels()
        
        self.image_changed.emit(self.current_image)
        self._update_stats()
    
    def clear_all_boxes(self):
        """Tüm kutuları temizle"""
        if not self.current_image:
            return
        
        self.current_image.boxes.clear()
        self._save_current_labels()
        self.image_changed.emit(self.current_image)
        self._update_stats()
    
    def run_detection(self):
        if not self.current_image or not self.detection_service:
            return

        # Clear previous detections for this image to start fresh
        self.temp_detections[self.current_image.filename] = []

        result = self.detection_service.detect_single_image(
            self.current_image.path,
            self.confidence_threshold
        )

        # 1) kırmızı etiketlere göre güçlü bastırma (IoU/containment)
        filtered = self.detection_service.suppress_by_manual(
            det_boxes=result.boxes,
            manual_boxes=self.current_image.boxes,
            iou_thr=IOU_THRESHOLD,
            contain_ratio=CONTAIN_RATIO,
            same_class_only=SAME_CLASS_ONLY,
        )

        # 2) kaydet + yayınla
        self.temp_detections[self.current_image.filename] = filtered
        self.current_image.detector_boxes = filtered

        self.detection_completed.emit(filtered)
        self.image_changed.emit(self.current_image)
        self._update_stats()

    
    def run_batch_detection(self, count: int = 50):
        if not self.detection_service:
            return

        total = self.image_repo.get_total_count()
        end_index = min(self.current_index + count, total)

        for i in range(self.current_index, end_index):
            image = self.image_repo.get_image(i)
            if not image:
                continue

            # 1) tespit
            result = self.detection_service.detect_single_image(
                image.path,
                self.confidence_threshold
            )

            # 2) mevcut etiketleri yükle
            label_file = self.labels_folder / f"{Path(image.filename).stem}.txt"
            existing = self.label_repo.load_labels(
                label_file,
                image.width,
                image.height
            )

            # 3) kırmızı etiketlere göre güçlü bastırma (IoU/containment)
            filtered = self.detection_service.suppress_by_manual(
                det_boxes=result.boxes,
                manual_boxes=existing,
                iou_thr=IOU_THRESHOLD,
                contain_ratio=CONTAIN_RATIO,
                same_class_only=SAME_CLASS_ONLY,
            )

            # 4) temp’e kaydet
            self.temp_detections[image.filename] = filtered

            # progress
            self.batch_progress.emit(i - self.current_index + 1, count)

        self._load_current_image()

    
    def save_temp_detections(self):
        """Geçici tespitleri kalıcı olarak kaydet"""
        saved_count = 0

        for filename, boxes in self.temp_detections.items():
            if not boxes:
                continue

            # Görüntü bilgilerini repository'den al (QPixmap yüklemeden)
            image = None
            for i in range(self.image_repo.get_total_count()):
                img = self.image_repo.get_image(i)
                if img and img.filename == filename:
                    image = img
                    break

            if not image:
                continue

            # Label dosyasına ekle (Image nesnesindeki width/height kullan)
            label_file = self.labels_folder / f"{Path(filename).stem}.txt"
            self.label_repo.append_labels(
                label_file,
                boxes,
                image.width,
                image.height
            )

            saved_count += len(boxes)

        # Temp'i temizle
        self.temp_detections.clear()

        # Mevcut görüntüyü yenile
        self._load_current_image()

        return saved_count
    
    def _save_current_labels(self):
        """Mevcut etiketleri kaydet"""
        if not self.current_image:
            return
        
        label_file = self.labels_folder / f"{Path(self.current_image.filename).stem}.txt"
        self.label_repo.save_labels(
            label_file,
            self.current_image.boxes,
            self.current_image.width,
            self.current_image.height
        )
    
    def _update_stats(self):
        """İstatistikleri güncelle"""
        total_images = self.image_repo.get_total_count()
        current_image_num = self.current_index + 1 if total_images > 0 else 0


        if self.current_image:
            existing_labels = len([b for b in self.current_image.boxes if b.class_id in ALLOWED_CLASSES])
            detector_added = len([b for b in self.current_image.detector_boxes if b.class_id in ALLOWED_CLASSES])
        else:
            existing_labels = 0
            detector_added = 0

        total_labels = existing_labels + detector_added

        stats = {
            'toplam_goruntu': total_images,
            'mevcut_goruntu': current_image_num,
            'secili': 0,  # This will be updated from UI
            'mevcut_etiket': existing_labels,
            'sahi_eklenen': detector_added,
            'toplam_etiket': total_labels
        }

        self.stats_updated.emit(stats)
    
    def update_settings(self, confidence: float, slice_height: int, slice_width: int, use_sahi: bool):
        """Ayarları güncelle"""
        # Check if slice parameters changed
        slice_changed = (self.slice_height != slice_height or
                        self.slice_width != slice_width or
                        self.use_sahi != use_sahi)

        self.confidence_threshold = confidence
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.use_sahi = use_sahi

        # If slice parameters changed and we have a loaded model, recreate detector
        if slice_changed and self.current_model_path and self.detection_service:
            try:
                detector = self.detector_factory.create_detector(
                    self.current_model_path,
                    device=self.device,
                    use_sahi=self.use_sahi,
                    slice_height=self.slice_height,
                    slice_width=self.slice_width
                )
                from services.detection.detection_service import DetectionService
                self.detection_service = DetectionService(detector)
                # Set confidence on the new detector
                self.detection_service.detector.set_confidence_threshold(confidence)
            except Exception as e:
                self.error_occurred.emit(f"Ayarlar uygulanamadı: {e}")
        elif self.detection_service:
            # Only update confidence threshold if detector wasn't recreated
            self.detection_service.detector.set_confidence_threshold(confidence)

    def _detect_device(self) -> str:
        """Detect if CUDA is available, otherwise use CPU"""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda:0'
            else:
                return 'cpu'
        except ImportError:
            # If torch is not available, default to CPU
            return 'cpu'