import sys
import os
import cv2
import numpy as np
import warnings
import gc 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QInputDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLineEdit, QProgressBar, QFrame, QSplitter, QScrollArea,
    QGroupBox, QSpinBox, QMessageBox, QDialog, QSizePolicy, QDoubleSpinBox
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QColor, QPalette, QImage
from PyQt5.QtCore import Qt, QRectF, QPointF, QSize, QThread, pyqtSignal, QMutex

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass
os.environ['YOLO_VERBOSE'] = 'False'
warnings.filterwarnings("ignore", message="Unable to automatically guess model task")

GKST_AVAILABLE = False
try:
    import sahi
    import ultralytics
    import torch
    from ultralytics import YOLO
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_sliced_prediction
    GKST_AVAILABLE = True
except ImportError as e:
    pass

class ONNXDetectionModel:
    def __init__(self, model_path, confidence_threshold=0.20, device='cuda'):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_onnx = False
        self.precision_info = "FP16"
        
        if ONNX_AVAILABLE:
            self.ort_session = self._create_onnx_session()
            if self.ort_session is not None:
                self.use_onnx = True
                self.input_name = self.ort_session.get_inputs()[0].name
                self.input_shape = self.ort_session.get_inputs()[0].shape
        
        if not self.use_onnx:
            self.model = YOLO(model_path, task='detect')
        
        self.category_names = None
        self.category_mapping = None
    
    def _create_onnx_session(self, ):
        try:
            providers = []
            
            if self.device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
                cuda_provider_options = {
                    'device_id': int(self.device.split(':')[1]) if ':' in self.device else 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                }
                providers.append(('CUDAExecutionProvider', cuda_provider_options))
                self.precision_info = "FP16 (CUDA)"
            else:
                providers.append('CPUExecutionProvider')
                self.precision_info = "FP32 (CPU)"
            
            session = ort.InferenceSession(self.model_path, providers=providers)
            return session
            
        except Exception as e:
            print(f"ONNX Runtime session oluşturulamadı: {e}")
            return None
    
    def _preprocess_image(self, image_path):
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_size = 640 if len(self.input_shape) > 2 else max(self.input_shape[2:])
        image = cv2.resize(image, (target_size, target_size))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        try:
            input_type = self.ort_session.get_inputs()[0].type
            if 'float16' in input_type:
                image = image.astype(np.float16)
                self.precision_info = "FP16 (ONNX)"
            else:
                image = image.astype(np.float32)
                self.precision_info = "FP32 (ONNX)"
        except:
            image = image.astype(np.float32)
            self.precision_info = "FP32 (ONNX)"
        
        return image
    
    def _postprocess_outputs(self, outputs, orig_img_shape):
        predictions = outputs[0]
        
        if predictions.shape[1] == 8400:
            predictions = np.squeeze(predictions, axis=0)
            mask = predictions[:, 4] >= self.confidence_threshold
            filtered_predictions = predictions[mask]
            
            if len(filtered_predictions) == 0:
                return []
                
            orig_h, orig_w = orig_img_shape[:2]
            target_size = 640
            scale_x = orig_w / target_size
            scale_y = orig_h / target_size
            
            results = []
            for pred in filtered_predictions:
                x_center, y_center, width, height, conf = pred[:5]
                class_scores = pred[5:]
                class_id = np.argmax(class_scores)
                
                if class_id not in [0, 2]:
                    continue
                
                x1 = (x_center - width/2) * scale_x
                y1 = (y_center - height/2) * scale_y
                x2 = (x_center + width/2) * scale_x
                y2 = (y_center + height/2) * scale_y
                
                results.append((class_id, x1, y1, x2, y2))
                
        elif predictions.shape[2] == 8400:
            predictions = np.transpose(np.squeeze(predictions, axis=0), (1, 0))
            mask = predictions[:, 4] >= self.confidence_threshold
            filtered_predictions = predictions[mask]
            
            if len(filtered_predictions) == 0:
                return []
                
            orig_h, orig_w = orig_img_shape[:2]
            target_size = 640
            scale_x = orig_w / target_size
            scale_y = orig_h / target_size
            
            results = []
            for pred in filtered_predictions:
                x_center, y_center, width, height, conf = pred[:5]
                class_scores = pred[5:]
                class_id = np.argmax(class_scores)
                
                if class_id not in [0, 2]:
                    continue
                
                x1 = (x_center - width/2) * scale_x
                y1 = (y_center - height/2) * scale_y
                x2 = (x_center + width/2) * scale_x
                y2 = (y_center + height/2) * scale_y
                
                results.append((class_id, x1, y1, x2, y2))
        else:
            print(f"Bilinmeyen ONNX çıktı formatı: {predictions.shape}")
            return []

        return results
    
    def perform_inference(self, image_path):
        if self.use_onnx:
            try:
                if isinstance(image_path, str):
                    orig_img = cv2.imread(image_path)
                else:
                    orig_img = image_path
                
                preprocessed = self._preprocess_image(image_path)
                
                outputs = self.ort_session.run(None, {self.input_name: preprocessed})
                
                detections = self._postprocess_outputs(outputs, orig_img.shape)
                
                return self._convert_to_sahi_format(detections)
                
            except Exception as e:
                print(f"ONNX inference hatası: {e}")
                if hasattr(self, 'model'):
                    return self._ultralytics_inference(image_path)
                else:
                    raise e
        else:
            return self._ultralytics_inference(image_path)
    
    def _ultralytics_inference(self, image_path):
        results = self.model.predict(
            image_path, 
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        return self._convert_ultralytics_to_sahi_format(results[0])
    
    def _convert_ultralytics_to_sahi_format(self, result):
        object_predictions = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls in zip(boxes, scores, classes):
                if score >= self.confidence_threshold and cls in [0, 2]:
                    prediction = self._create_mock_prediction(box, cls, score)
                    object_predictions.append(prediction)
        
        return self._create_mock_detection_result(object_predictions)
    
    def _convert_to_sahi_format(self, detections):
        object_predictions = []
        
        for class_id, x1, y1, x2, y2 in detections:
            score = 1.0
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            prediction = self._create_mock_prediction(bbox, int(class_id), float(score))
            object_predictions.append(prediction)
        
        return self._create_mock_detection_result(object_predictions)
    
    def _create_mock_prediction(self, bbox, category_id, score):
        class MockPrediction:
            def __init__(self, bbox, category_id, score):
                self.bbox = bbox
                self.category_id = int(category_id)
                self.score = MockScore(float(score))
                self.category = MockCategory(int(category_id))
        
        class MockScore:
            def __init__(self, value):
                self.value = value
        
        class MockCategory:
            def __init__(self, category_id):
                self.id = category_id
        
        return MockPrediction(bbox, category_id, score)
    
    def _create_mock_detection_result(self, predictions):
        class MockDetectionResult:
            def __init__(self, predictions):
                self.object_prediction_list = predictions
        
        return MockDetectionResult(predictions)

class ModernButton(QPushButton):
    def __init__(self, text, color="primary"):
        super().__init__(text)
        self.setMinimumHeight(35)
        self.setFont(QFont("Segoe UI", 9, QFont.Medium))

        if color == "primary":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5a67d8, stop:1 #6b46c1);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4c51bf, stop:1 #553c9a);
                }
            """)
        elif color == "secondary":
            self.setStyleSheet("""
                QPushButton {
                    background: #f7fafc;
                    color: #4a5568;
                    border: 1px solid #e2e8f0;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #edf2f7;
                    border-color: #cbd5e0;
                }
                QPushButton:pressed {
                    background: #e2e8f0;
                }
            """)
        elif color == "danger":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ff6b6b, stop:1 #ee5a52);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ff5252, stop:1 #e53e3e);
                }
            """)

class ModernGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: 600;
                color: #4a5568;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                margin: 10px 0px;
                padding-top: 15px;
                background: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background: #fafafa;
            }
        """)

class StatsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)

        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
                color: white;
            }
        """)

        stats = [
            ("Toplam Görüntü", "0"),
            ("Mevcut Görüntü", "0"),
            ("Seçili", "0"),
            ("Mevcut Etiket", "0"),
            ("GKST Eklenen", "0"),
            ("Toplam Etiket", "0")
        ]

        self.stat_labels = {}
        for i, (title, value) in enumerate(stats):
            stat_widget = QWidget()
            stat_layout = QVBoxLayout(stat_widget)
            stat_layout.setAlignment(Qt.AlignCenter)

            value_label = QLabel(value)
            value_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
            value_label.setAlignment(Qt.AlignCenter)

            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 9))
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("color: rgba(255, 255, 255, 0.8);")

            stat_layout.addWidget(value_label)
            stat_layout.addWidget(title_label)

            layout.addWidget(stat_widget)
            self.stat_labels[title.lower().replace(" ", "_")] = value_label

            if i < len(stats) - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.VLine)
                separator.setStyleSheet("color: rgba(255, 255, 255, 0.3);")
                layout.addWidget(separator)

    def update_stats(self, total_images=0, current_image=0, selected=0, existing_labels=0, gkdt_added=0, total_labels=0):
        self.stat_labels["toplam_görüntü"].setText(str(total_images))
        self.stat_labels["mevcut_görüntü"].setText(str(current_image))
        self.stat_labels["seçili"].setText(str(selected))
        self.stat_labels["mevcut_etiket"].setText(str(existing_labels))
        self.stat_labels["gkst_eklenen"].setText(str(gkdt_added))
        self.stat_labels["toplam_etiket"].setText(str(total_labels))

class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if hasattr(self.parent_window, 'on_mouse_press'):
            self.parent_window.on_mouse_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self.parent_window, 'on_mouse_move'):
            self.parent_window.on_mouse_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self.parent_window, 'on_mouse_release'):
            self.parent_window.on_mouse_release(event)
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        if hasattr(self.parent_window, 'on_wheel_event'):
            self.parent_window.on_wheel_event(event)
        else:
            super().wheelEvent(event)

class GkstThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, image_path, gkdt_predictor, confidence, slice_height, slice_width, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.gkdt_predictor = gkdt_predictor
        self.confidence = confidence
        self.slice_height = slice_height
        self.slice_width = slice_width

    def run(self):
        try:
            if isinstance(self.gkdt_predictor, ONNXDetectionModel):
                self.gkdt_predictor.confidence_threshold = self.confidence
                result = self.gkdt_predictor.perform_inference(self.image_path)
            else:
                self.gkdt_predictor.confidence_threshold = self.confidence
                result = get_sliced_prediction(
                    self.image_path,
                    self.gkdt_predictor,
                    slice_height=self.slice_height,
                    slice_width=self.slice_width,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )
            
            gkdt_results = []
            for detection in result.object_prediction_list:
                if hasattr(detection, 'bbox') and hasattr(detection.bbox, 'to_voc_bbox'):
                    bbox = detection.bbox.to_voc_bbox()
                    class_id = detection.category.id
                else:
                    bbox = detection.bbox
                    class_id = detection.category_id
                
                gkdt_results.append((class_id, bbox[0], bbox[1], bbox[2], bbox[3]))
                
            self.finished.emit(gkdt_results)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([]) 
            
class BatchGkdtWorker(QThread):
    progress_updated = pyqtSignal(int, int) 
    finished = pyqtSignal(dict) 
    error_occurred = pyqtSignal(str)
    image_processed_in_batch = pyqtSignal(str, list)

    def __init__(self, parent_app, image_paths_to_process):
        super().__init__()
        self.parent_app = parent_app
        self.image_paths_to_process = image_paths_to_process
        self.total_count = len(image_paths_to_process)
        self.mutex = QMutex()
        self.batch_results = {} 
        self.confidence = self.parent_app.confidence_input.value()
        self.slice_height = self.parent_app.slice_height_input.value()
        self.slice_width = self.parent_app.slice_width_input.value()

    def run(self):
        try:
            for i, image_filename in enumerate(self.image_paths_to_process):
                if self.isInterruptionRequested():
                    break
                
                current_image_path = os.path.join(self.parent_app.image_folder, image_filename)
                
                try:
                    # Raw GKST results
                    if isinstance(self.parent_app.gkdt_predictor, ONNXDetectionModel):
                        self.parent_app.gkdt_predictor.confidence_threshold = self.confidence
                        result = self.parent_app.gkdt_predictor.perform_inference(current_image_path)
                    else:
                        self.parent_app.gkdt_predictor.confidence_threshold = self.confidence
                        result = get_sliced_prediction(
                            current_image_path,
                            self.parent_app.gkdt_predictor,
                            slice_height=self.slice_height,
                            slice_width=self.slice_width,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2
                        )
                    
                    gkdt_results = []
                    for detection in result.object_prediction_list:
                        if hasattr(detection, 'bbox') and hasattr(detection.bbox, 'to_voc_bbox'):
                            bbox = detection.bbox.to_voc_bbox()
                            class_id = detection.category.id
                        else:
                            bbox = detection.bbox
                            class_id = detection.category_id
                        
                        gkdt_results.append((class_id, bbox[0], bbox[1], bbox[2], bbox[3]))

                    img = QPixmap(current_image_path)
                    image_size = img.size()
                    
                    # TEKİL İŞLEMDEKİ AYNI FONKSİYONU KULLAN
                    processed_results = self.parent_app.process_gkdt_results(
                        gkdt_results,
                        image_size.width(),
                        image_size.height(),
                        image_filename 
                    )
                    
                    self.batch_results[image_filename] = processed_results
                    
                    # Hemen ana uygulamaya bildir ki update_image çağrılsın
                    self.image_processed_in_batch.emit(image_filename, processed_results)

                except Exception as e:
                    self.error_occurred.emit(f"Resim '{image_filename}' için hata: {e}")
                
                self.progress_updated.emit(i + 1, self.total_count)
            
            self.finished.emit(self.batch_results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

    def calculate_iou(self, box1, box2):
        """IoU hesaplama - tekli işlemle aynı"""
        x1_a, y1_a, w_a, h_a = box1
        x2_a = x1_a + w_a
        y2_a = y1_a + h_a
        
        x1_b, y1_b, w_b, h_b = box2
        x2_b = x1_b + w_b
        y2_b = y1_b + h_b

        x1_intersection = max(x1_a, x1_b)
        y1_intersection = max(y1_a, y1_b)
        x2_intersection = min(x2_a, x2_b)
        y2_intersection = min(y2_a, y2_b)

        intersection_width = max(0, x2_intersection - x1_intersection)
        intersection_height = max(0, y2_intersection - y1_intersection)

        intersection_area = intersection_width * intersection_height

        area_a = w_a * h_a
        area_b = w_b * h_b

        union_area = area_a + area_b - intersection_area

        if union_area == 0:
            return 0
        return intersection_area / union_area

    def process_gkdt_results_for_batch(self, gkdt_results, image_width, image_height, image_name):
        """Tekli işlemle tamamen aynı mantık"""
        allowed_classes = {0, 2}
        unique_gkdt_boxes = []
        iou_threshold = 0.1 
        
        # Mevcut etiketleri oku
        existing_boxes = []
        base = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.parent_app.labels_folder, base + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                for line in lines:
                    parts = line.split()
                    cid, xc, yc, bw, bh = map(float, parts[:5])
                    x = (xc - bw/2) * image_width
                    y = (yc - bh/2) * image_height
                    w = bw * image_width
                    h = bh * image_height
                    existing_boxes.append((int(cid), x, y, w, h))

        # Önceki GKST sonuçlarını kontrol et
        existing_gkdt_boxes = []
        # Ana uygulamadaki all_gkdt_boxes'tan al
        if hasattr(self.parent_app, 'all_gkdt_boxes') and image_name in self.parent_app.all_gkdt_boxes:
            existing_gkdt_boxes = self.parent_app.all_gkdt_boxes[image_name]
        
        # Bu toplu işlemde önceden işlenmiş resimleri de kontrol et
        for processed_img_name, boxes in self.batch_results.items():
            if processed_img_name != image_name:  # Kendisi hariç
                existing_gkdt_boxes.extend(boxes)

        # Duplikasyon kontrolü - tekli işlemle aynı
        for class_id, x1, y1, x2, y2 in gkdt_results:
            if class_id in allowed_classes:
                w = x2 - x1
                h = y2 - y1
                
                is_duplicate = False
                # Hem mevcut etiketlerle hem de önceki GKST sonuçlarıyla karşılaştır
                for existing_cid, ex, ey, ew, eh in existing_boxes + existing_gkdt_boxes:
                    if class_id == existing_cid:
                        iou = self.calculate_iou((x1, y1, w, h), (ex, ey, ew, eh))
                        if iou > iou_threshold:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_gkdt_boxes.append((class_id, x1, y1, w, h))
        
        return unique_gkdt_boxes

class ModernImageNavigator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_folder = ""
        self.labels_folder = ""
        self.image_paths = []
        self.index = 0
        self.labels = []
        self.boxes = []
        self.selected = set()
        self.selected_gkdt = set() 
        self.is_drawing = False
        self.start_pos = QPointF()
        self.end_pos = QPointF()
        self.is_fullscreen = False

        self.original_pix = QPixmap()
        self.scaled_pix = QPixmap()
        self.original_image_size = QSize()
        self.sf_w = 1.0
        self.sf_h = 1.0
        self.x_off = 0.0
        self.y_off = 0.0
        
        self.zoom_level = 1.0
        self.is_panning = False
        self.pan_start_pos = QPointF()
        self.pan_offset = QPointF()
        self.gkdt_thread = None 
        self.batch_worker = None
        
        self.all_gkdt_boxes = {}
        self.gkdt_added_count = 0

        self.device = 'cuda:0' if GKST_AVAILABLE and torch.cuda.is_available() else 'cpu'

        self.gkdt_predictor = None

        self.class_names = {
            0: "person",
            2: "vehicle"
        }

        self.init_ui()
        self.apply_styles()
        self.update_image()

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.load_default_models()

    def load_gkdt_model(self):
        if not GKST_AVAILABLE:
            self.gkdt_status_label.setText("GKST ve/veya Ultralytics kütüphanesi eksik!")
            self.gkdt_btn.setEnabled(False)
            self.batch_gkdt_btn.setEnabled(False)
            self.save_gkdt_btn.setEnabled(False)
            return

        if not self.model_list.currentItem():
            self.gkdt_status_label.setText("Lütfen listeden bir model dosyası seçin.")
            self.gkdt_btn.setEnabled(False)
            self.batch_gkdt_btn.setEnabled(False)
            self.save_gkdt_btn.setEnabled(False)
            return

        model_path = os.path.join(self.model_folder, self.model_list.currentItem().text())

        if not os.path.exists(model_path):
            self.gkdt_status_label.setText(f"Hata: {model_path} dosyası bulunamadı!")
            self.gkdt_btn.setEnabled(False)
            self.batch_gkdt_btn.setEnabled(False)
            self.save_gkdt_btn.setEnabled(False)
            return

        try:
            self.gkdt_status_label.setText(f"GKST Ultralytics modeli yükleniyor... ({self.device})")
            
            model_extension = model_path.lower().split('.')[-1]
            
            if model_extension == 'onnx':
                self.gkdt_predictor = ONNXDetectionModel(
                    model_path=model_path,
                    confidence_threshold=self.confidence_input.value(),
                    device=self.device
                )
                precision_info = getattr(self.gkdt_predictor, 'precision_info', 'FP32')
                runtime_info = "ONNX Runtime" if getattr(self.gkdt_predictor, 'use_onnx', False) else "Ultralytics"
                self.gkdt_status_label.setText(f"GKST ONNX modeli hazır. Cihaz: {self.device} ({precision_info}) - {runtime_info}")
                
            elif model_extension == 'pt':
                self.gkdt_predictor = UltralyticsDetectionModel(
                    model_path=model_path,
                    confidence_threshold=self.confidence_input.value(),
                    device=self.device
                )
                
                if self.device.startswith('cuda'):
                    try:
                        self.gkdt_predictor.model.model.half()
                        self.gkdt_status_label.setText(f"GKST PyTorch modeli hazır. Cihaz: {self.device} (FP16)")
                    except Exception as half_error:
                        self.gkdt_status_label.setText(f"GKST PyTorch modeli hazır. Cihaz: {self.device} (FP32)")
                else:
                    self.gkdt_status_label.setText(f"GKST PyTorch modeli hazır. Cihaz: {self.device} (FP32)")
            else:
                raise ValueError(f"Desteklenmeyen model formatı: {model_extension}")

            self.gkdt_btn.setEnabled(True)
            self.batch_gkdt_btn.setEnabled(True)
            self.save_gkdt_btn.setEnabled(True)
            
        except Exception as e:
            self.gkdt_status_label.setText(f"GKST Ultralytics modeli yüklenemedi: {e}")
            self.gkdt_btn.setEnabled(False)
            self.batch_gkdt_btn.setEnabled(False)
            self.save_gkdt_btn.setEnabled(False)
            print(f"Model yükleme hatası detayları: {e}")

    def load_default_models(self):
        self.model_folder = os.path.join(os.path.dirname(__file__), "models")
        if os.path.exists(self.model_folder):
            self.update_model_list(self.model_folder)
        else:
            self.gkdt_status_label.setText("`models` klasörü bulunamadı. Lütfen bir model klasörü oluşturun ve içine model dosyalarınızı koyun.")
            self.gkdt_btn.setEnabled(False)
            self.batch_gkdt_btn.setEnabled(False)
            self.save_gkdt_btn.setEnabled(False)

    def update_model_list(self, folder):
        self.model_list.clear()
        model_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.pt', '.onnx'))
        ])
        for model_path in model_files:
            self.model_list.addItem(model_path)
        
        if model_files:
            self.model_list.setCurrentRow(0)
            self.load_gkdt_model()
        else:
            self.gkdt_status_label.setText("Seçilen klasörde model dosyası (.pt veya .onnx) bulunamadı.")
            self.gkdt_btn.setEnabled(False)
            self.batch_gkdt_btn.setEnabled(False)
            self.save_gkdt_btn.setEnabled(False)

    def update_gkdt_confidence(self, value):
        if self.gkdt_predictor:
            if hasattr(self.gkdt_predictor, 'confidence_threshold'):
                self.gkdt_predictor.confidence_threshold = value

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        splitter = QSplitter(Qt.Horizontal)

        sidebar = self.create_sidebar()
        sidebar.setMaximumWidth(320)
        sidebar.setMinimumWidth(280)

        main_area = self.create_main_area()

        splitter.addWidget(sidebar)
        splitter.addWidget(main_area)
        splitter.setSizes([320, 800])

        main_layout.addWidget(splitter)

        self.setWindowTitle("🎯 Modern SINIF Etiketleme Aracı")
        self.setGeometry(100, 100, 1200, 800)

    def create_sidebar(self):
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(15)

        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaledToWidth(150, Qt.SmoothTransformation))
        else:
            logo_label.setText("logo.png bulunamadı")
            logo_label.setStyleSheet("color: red; font-size: 10px; font-style: italic;")
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        title = QLabel("🎯 SINIF Etiketleyici")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #4a5568; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        folder_group = ModernGroupBox("📁 Klasör Ayarları")
        folder_layout = QVBoxLayout(folder_group)

        self.select_images_btn = ModernButton("📷 Görüntü Klasörü Seç")
        self.select_labels_btn = ModernButton("🏷️ Etiket Klasörü Seç")

        folder_layout.addWidget(self.select_images_btn)
        folder_layout.addWidget(self.select_labels_btn)
        layout.addWidget(folder_group)

        tools_group = ModernGroupBox("🔧 Araçlar")
        tools_layout = QVBoxLayout(tools_group)

        model_list_layout = QVBoxLayout()
        model_list_label = QLabel("SINIF Modelleri:")
        model_list_label.setFont(QFont("Segoe UI", 9))
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(100)
        self.model_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)

        model_list_layout.addWidget(model_list_label)
        model_list_layout.addWidget(self.model_list)
        tools_layout.addLayout(model_list_layout)
        
        gkdt_settings_layout = QHBoxLayout()
        gkdt_confidence_label = QLabel("GKST Güvenilirlik:")
        gkdt_confidence_label.setFont(QFont("Segoe UI", 9))
        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setMinimum(0.0)
        self.confidence_input.setMaximum(1.0)
        self.confidence_input.setValue(0.2)
        self.confidence_input.setStyleSheet("""
            QDoubleSpinBox {
                padding: 6px;
                border: 2px solid #e2e8f0;
                border-radius: 4px;
                background: white;
            }
            QDoubleSpinBox:focus {
                border-color: #667eea;
            }
        """)
        gkdt_settings_layout.addWidget(gkdt_confidence_label)
        gkdt_settings_layout.addWidget(self.confidence_input)
        tools_layout.addLayout(gkdt_settings_layout)

        slice_height_layout = QHBoxLayout()
        slice_height_label = QLabel("Dilim Yüksekliği:")
        slice_height_label.setFont(QFont("Segoe UI", 9))
        self.slice_height_input = QSpinBox()
        self.slice_height_input.setMinimum(100)
        self.slice_height_input.setMaximum(4096)
        self.slice_height_input.setValue(256)
        self.slice_height_input.setStyleSheet("""
            QSpinBox {
                padding: 6px;
                border: 2px solid #e2e8f0;
                border-radius: 4px;
                background: white;
            }
            QSpinBox:focus {
                border-color: #667eea;
            }
        """)
        slice_height_layout.addWidget(slice_height_label)
        slice_height_layout.addWidget(self.slice_height_input)
        tools_layout.addLayout(slice_height_layout)

        slice_width_layout = QHBoxLayout()
        slice_width_label = QLabel("Dilim Genişliği:")
        slice_width_label.setFont(QFont("Segoe UI", 9))
        self.slice_width_input = QSpinBox()
        self.slice_width_input.setMinimum(100)
        self.slice_width_input.setMaximum(4096)
        self.slice_width_input.setValue(256)
        self.slice_width_input.setStyleSheet("""
            QSpinBox {
                padding: 6px;
                border: 2px solid #e2e8f0;
                border-radius: 4px;
                background: white;
            }
            QSpinBox:focus {
                border-color: #667eea;
            }
        """)
        slice_width_layout.addWidget(slice_width_label)
        slice_width_layout.addWidget(self.slice_width_input)
        tools_layout.addLayout(slice_width_layout)

        self.gkdt_btn = ModernButton("✨ GKST Uygula", "primary")
        self.gkdt_btn.setEnabled(False)
        tools_layout.addWidget(self.gkdt_btn)

        self.batch_gkdt_btn = ModernButton("✨ GKST Uygula (50 Resim)", "primary")
        self.batch_gkdt_btn.setEnabled(False)
        tools_layout.addWidget(self.batch_gkdt_btn)

        self.save_gkdt_btn = ModernButton("💾 GKST Sonuçlarını Kaydet", "secondary")
        self.save_gkdt_btn.setEnabled(False)
        tools_layout.addWidget(self.save_gkdt_btn)

        self.gkdt_status_label = QLabel(f"GKST durumu: Cihaz: {self.device}")
        self.gkdt_status_label.setFont(QFont("Segoe UI", 8))
        self.gkdt_status_label.setStyleSheet("color: #718096; font-style: italic;")
        tools_layout.addWidget(self.gkdt_status_label)

        tools_layout.addSpacing(10)
        self.delete_btn = ModernButton("🗑️ Seçili Sil", "danger")
        self.clear_btn = ModernButton("🧹 Tümünü Temizle", "secondary")
        tools_layout.addWidget(self.delete_btn)
        tools_layout.addWidget(self.clear_btn)
        layout.addWidget(tools_group)

        class_group = ModernGroupBox("🎯 Sınıf Filtresi")
        class_layout = QVBoxLayout(class_group)

        class_list_label = QLabel("Görüntülenecek sınıflar:")
        class_list_label.setFont(QFont("Segoe UI", 9))
        
        self.class_filter_list = QListWidget()
        self.class_filter_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        self.class_filter_list.itemChanged.connect(self.update_image)

        class_layout.addWidget(class_list_label)
        class_layout.addWidget(self.class_filter_list)
        layout.addWidget(class_group)
        
        layout.addStretch()

        self.select_images_btn.clicked.connect(self.select_image_folder)
        self.select_labels_btn.clicked.connect(self.select_labels_folder)
        self.delete_btn.clicked.connect(self.delete_selected_boxes)
        self.clear_btn.clicked.connect(self.clear_all_boxes)
        self.gkdt_btn.clicked.connect(self.apply_gkdt_detection)
        self.batch_gkdt_btn.clicked.connect(self.start_batch_gkdt_process)
        self.save_gkdt_btn.clicked.connect(self.save_gkdt_results)
        self.confidence_input.valueChanged.connect(self.update_gkdt_confidence)
        self.model_list.itemSelectionChanged.connect(self.load_gkdt_model)

        return sidebar

    def create_main_area(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)

        header_layout = QHBoxLayout()

        self.file_label = QLabel("Dosya seçilmedi")
        self.file_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        self.file_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
            }
        """)
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        header_layout.addWidget(self.file_label)
        header_layout.addStretch()

        nav_layout = QHBoxLayout()
        self.fullscreen_btn = ModernButton("Tam Ekran")
        self.prev_btn = ModernButton("⬅️ Önceki")
        self.next_btn = ModernButton("Sonraki ➡️")

        nav_layout.addWidget(self.fullscreen_btn)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        header_layout.addLayout(nav_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 8px;
                text-align: center;
                background: #e2e8f0;
                height: 8px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
            }
        """)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(8)

        image_container = QFrame()
        image_container.setStyleSheet("""
            QFrame {
                border: 2px dashed #cbd5e0;
                border-radius: 10px;
                background: #f8fafc;
            }
        """)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(1, 1, 1, 1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background: white; border-radius: 6px;")

        scroll_area.setWidget(self.image_label)
        image_layout.addWidget(scroll_area)

        self.stats_widget = StatsWidget()

        layout.addLayout(header_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(image_container, 1)
        layout.addWidget(self.stats_widget)

        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn.clicked.connect(self.next_image)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)

        return main_widget

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.showNormal()
            self.is_fullscreen = False
        else:
            self.showFullScreen()
            self.is_fullscreen = True
        self.update_image()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f7fafc, stop:1 #edf2f7);
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

    def load_images(self):
        if self.image_folder and os.path.exists(self.image_folder):
            self.image_paths = sorted([
                f for f in os.listdir(self.image_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            if not self.image_paths:
                self.file_label.setText("Görüntü klasöründe desteklenen dosya bulunamadı.")
                self.image_label.clear()
            else:
                self.index = 0
                self.update_image()
            self.update_progress()
            self.update_stats()
        else:
            self.image_paths = []
            self.index = 0
            self.update_image()

    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Görüntü Klasörü Seçin", self.image_folder
        )
        if folder:
            self.image_folder = folder
            if not self.labels_folder:
                self.labels_folder = os.path.join(self.image_folder, "labels")
                os.makedirs(self.labels_folder, exist_ok=True)
            self.load_images()

    def select_labels_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Etiket Klasörü Seçin", self.labels_folder
        )
        if folder:
            self.labels_folder = folder
            if self.image_paths:
                self.update_image()

    def load_labels(self):
        if not self.image_paths or not self.image_folder or not self.labels_folder:
            self.labels.clear()
            self.boxes.clear()
            return

        path = self.image_paths[self.index]
        img_path = os.path.join(self.image_folder, path)
        base = os.path.splitext(path)[0]
        self.label_path = os.path.join(self.labels_folder, base + '.txt')

        self.labels.clear()
        self.boxes.clear()

        if os.path.exists(img_path):
            self.original_pix = QPixmap(img_path)
            self.original_image_size = self.original_pix.size()
            iw, ih = self.original_pix.width(), self.original_pix.height()

            if os.path.exists(self.label_path):
                with open(self.label_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = parts[0]
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x = (xc - bw/2) * iw
                    y = (yc - bh/2) * ih
                    w = bw * iw
                    h = bh * ih
                    self.labels.append(line)
                    self.boxes.append((cid, x, y, w, h))
        
        self.update_class_filter_list()
    
    def calculate_iou(self, box1, box2):
        x1_a, y1_a, w_a, h_a = box1
        x2_a = x1_a + w_a
        y2_a = y1_a + h_a
        
        x1_b, y1_b, w_b, h_b = box2
        x2_b = x1_b + w_b
        y2_b = y1_b + h_b

        # Kesişim alanını bul
        x1_intersection = max(x1_a, x1_b)
        y1_intersection = max(y1_a, y1_b)
        x2_intersection = min(x2_a, x2_b)
        y2_intersection = min(y2_a, y2_b)

        # Kesişim alanının genişlik ve yüksekliğini hesapla
        intersection_width = max(0, x2_intersection - x1_intersection)
        intersection_height = max(0, y2_intersection - y1_intersection)

        intersection_area = intersection_width * intersection_height

        # Bireysel kutu alanlarını hesapla
        area_a = w_a * h_a
        area_b = w_b * h_b

        # Birleşim alanını hesapla
        union_area = area_a + area_b - intersection_area

        if union_area == 0:
            return 0
        return intersection_area / union_area

    def process_gkdt_results(self, gkdt_results, image_width, image_height, image_name):
        allowed_classes = {0, 2}
        unique_gkdt_boxes = []
        iou_threshold = 0.1 
        
        existing_boxes = []
        base = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.labels_folder, base + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                for line in lines:
                    parts = line.split()
                    cid, xc, yc, bw, bh = map(float, parts[:5])
                    x = (xc - bw/2) * image_width
                    y = (yc - bh/2) * image_height
                    w = bw * image_width
                    h = bh * image_height
                    existing_boxes.append((int(cid), x, y, w, h))

        existing_gkdt_boxes = self.all_gkdt_boxes.get(image_name, [])

        for class_id, x1, y1, x2, y2 in gkdt_results:
            if class_id in allowed_classes:
                w = x2 - x1
                h = y2 - y1
                
                is_duplicate = False
                for existing_cid, ex, ey, ew, eh in existing_boxes + existing_gkdt_boxes:
                    if class_id == existing_cid:
                        iou = self.calculate_iou((x1, y1, w, h), (ex, ey, ew, eh))
                        if iou > iou_threshold:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_gkdt_boxes.append((class_id, x1, y1, w, h))
        
        return unique_gkdt_boxes

    def apply_gkdt_detection(self):
        if not GKST_AVAILABLE:
            QMessageBox.critical(self, "Hata", "GKST veya Ultralytics kütüphanesi eksik.")
            return

        if not self.gkdt_predictor:
            QMessageBox.warning(self, "Uyarı", "GKST modeli henüz hazır değil. Lütfen bekleyin.")
            return

        if not self.image_paths:
            QMessageBox.information(self, "Uyarı", "Lütfen önce bir görüntü klasörü seçin.")
            return
        
        if self.gkdt_thread and self.gkdt_thread.isRunning():
            QMessageBox.warning(self, "Uyarı", "Başka bir GKST işlemi zaten devam ediyor.")
            return

        self.gkdt_status_label.setText(f"GKST algılama başlatılıyor... Cihaz: {self.device}")
        self.gkdt_btn.setEnabled(False)
        self.batch_gkdt_btn.setEnabled(False)
        self.save_gkdt_btn.setEnabled(False)
        
        current_image_path = os.path.join(self.image_folder, self.image_paths[self.index])

        confidence = self.confidence_input.value()
        slice_height = self.slice_height_input.value()
        slice_width = self.slice_width_input.value()
        self.gkdt_thread = GkstThread(current_image_path, self.gkdt_predictor, confidence, slice_height, slice_width)
        self.gkdt_thread.finished.connect(self.on_gkdt_finished)
        self.gkdt_thread.error.connect(self.on_gkdt_error)
        self.gkdt_thread.start()

    def on_gkdt_finished(self, gkdt_results):
            self.gkdt_status_label.setText("GKST algılama tamamlandı. Yeşil kutuları kaydetmek için butona basın.")
            self.gkdt_btn.setEnabled(True)
            self.batch_gkdt_btn.setEnabled(True)
            self.save_gkdt_btn.setEnabled(True)
            
            current_image_name = self.image_paths[self.index]
            
            # Sadece tekil işlemde, ilgili resmin önceki GKST kutularını temizle
            if current_image_name in self.all_gkdt_boxes:
                del self.all_gkdt_boxes[current_image_name]

            processed_results = self.process_gkdt_results(
                gkdt_results,
                self.original_image_size.width(),
                self.original_image_size.height(),
                current_image_name
            )
            
            # all_gkdt_boxes'a yeni işlenmiş sonuçları ekle
            self.all_gkdt_boxes[current_image_name] = processed_results
            self.gkdt_added_count = len(processed_results)

            self.update_image()
            self.update_class_filter_list()
            self.gkdt_thread.quit()
            self.gkdt_thread.wait()

    def on_gkdt_error(self, message):
        self.gkdt_status_label.setText(f"GKST algılama hatası: {message}")
        self.gkdt_btn.setEnabled(True)
        self.batch_gkdt_btn.setEnabled(True)
        self.save_gkdt_btn.setEnabled(False)
        QMessageBox.critical(self, "GKST Algılama Hatası", message)
        self.gkdt_thread.quit()
        self.gkdt_thread.wait()

    def save_gkdt_results(self):
            print(f"save_gkdt_results çağrıldı")
            print(f"all_gkdt_boxes içeriği: {len(self.all_gkdt_boxes)} resim")
            print(f"labels_folder: {self.labels_folder}")
            
            if not self.all_gkdt_boxes:
                QMessageBox.information(self, "Bilgi", "Kaydedilecek GKST sonucu bulunamadı.")
                return

            try:
                # Labels klasörünün var olduğundan emin ol
                os.makedirs(self.labels_folder, exist_ok=True)
                saved_count = 0
                
                # Tüm toplu ve tekil işlem sonuçlarını kaydet
                for image_name, gkdt_boxes in self.all_gkdt_boxes.items():
                    print(f"İşleniyor: {image_name}, {len(gkdt_boxes)} kutu")
                    
                    if not gkdt_boxes:
                        print(f"  {image_name} için kutu yok, atlanıyor")
                        continue

                    img_path = os.path.join(self.image_folder, image_name)
                    if not os.path.exists(img_path):
                        print(f"  Resim dosyası bulunamadı: {img_path}")
                        continue

                    # Resim boyutlarını al
                    pixmap = QPixmap(img_path)
                    img_w, img_h = pixmap.width(), pixmap.height()
                    print(f"  Resim boyutları: {img_w} x {img_h}")

                    base = os.path.splitext(image_name)[0]
                    label_path = os.path.join(self.labels_folder, base + '.txt')
                    print(f"  Hedef txt dosyası: {label_path}")

                    # Append modunda dosyaya yaz (mevcut etiketlerin üzerine ekleme)
                    try:
                        with open(label_path, 'a', encoding='utf-8') as f:
                            for cid, x, y, w, h in gkdt_boxes:
                                # Koordinatlar zaten piksel formatında geliyor (x, y, w, h)
                                # YOLO formatına dönüştür: (center_x, center_y, width, height) normalized
                                center_x = (x + w / 2) / img_w
                                center_y = (y + h / 2) / img_h
                                norm_width = w / img_w
                                norm_height = h / img_h
                                
                                # YOLO formatında kaydet
                                line = f"{int(cid)} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                                print(f"    Yazılan satır: {line}")
                                f.write(line + "\n")
                                saved_count += 1
                        print(f"  {label_path} dosyasına başarıyla yazıldı")
                    except Exception as file_error:
                        print(f"  Dosya yazma hatası {label_path}: {file_error}")
                        continue
                
                if saved_count > 0:
                    # Kaydettikten sonra geçici belleği temizle
                    self.all_gkdt_boxes.clear()
                    
                    QMessageBox.information(self, "Başarılı", f"{saved_count} GKST sonucu YOLO formatında kaydedildi.")
                    
                    # Gkdt eklenen sayacını sıfırla
                    self.gkdt_added_count = 0   
                    
                    # Etiketleri yeniden yükle ve ekranı güncelle
                    self.load_labels()
                    self.update_image()
                    self.update_class_filter_list()
                else:
                    QMessageBox.warning(self, "Uyarı", "Hiçbir GKST sonucu kaydedilemedi.")
                
            except Exception as e:
                error_msg = f"GKST sonuçları kaydedilirken bir hata oluştu: {e}"
                QMessageBox.critical(self, "Hata", error_msg)
                print(f"Detaylı hata: {e}")
                import traceback
                traceback.print_exc()
        


    def start_batch_gkdt_process(self):
        if not self.gkdt_predictor:
            QMessageBox.warning(self, "Uyarı", "GKST modeli henüz hazır değil. Lütfen bekleyin.")
            return

        if not self.image_paths:
            QMessageBox.information(self, "Bilgi", "Lütfen önce bir görüntü klasörü seçin.")
            return
        
        if self.batch_worker and self.batch_worker.isRunning():
            QMessageBox.warning(self, "Uyarı", "Başka bir toplu işlem zaten devam ediyor.")
            return

        start_index = self.index
        num_images_to_process = min(50, len(self.image_paths) - start_index)

        if num_images_to_process <= 0:
            QMessageBox.information(self, "Bilgi", "Klasördeki tüm resimler zaten işlendi.")
            return

        reply = QMessageBox.question(
            self, 'Toplu İşlem Onayı',
            f'Mevcut resimden başlayarak sonraki {num_images_to_process} resme toplu GKST uygulamak istediğinizden emin misiniz? Sonuçlar geçici olarak saklanacaktır.',
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.gkdt_status_label.setText(f"Toplu işlem başlatılıyor... {num_images_to_process} resim işleniyor.")
            
            self.gkdt_btn.setEnabled(False)
            self.batch_gkdt_btn.setEnabled(False)
            self.save_gkdt_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(num_images_to_process)
            
            images_to_process = self.image_paths[start_index:start_index + num_images_to_process]
            self.batch_worker = BatchGkdtWorker(self, images_to_process)
            self.batch_worker.progress_updated.connect(self.update_batch_progress)
            self.batch_worker.finished.connect(self.on_batch_finished)
            self.batch_worker.image_processed_in_batch.connect(self.on_batch_image_processed)
            self.batch_worker.error_occurred.connect(self.on_batch_error)
            self.batch_worker.start()

    def update_batch_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.gkdt_status_label.setText(f"İşleniyor: Resim {current} / {total}")

    def on_batch_finished(self, batch_results):
        self.gkdt_status_label.setText("Toplu GKST işlemi tamamlandı. Sonuçlar geçici belleğe aktarıldı.")
        
        # Tüm batch sonuçlarını all_gkdt_boxes'a ekle
        self.all_gkdt_boxes.update(batch_results)
        
        # Şu anki resmin kutularını hesapla ve güncelle
        current_image_name = self.image_paths[self.index] if self.image_paths else None
        if current_image_name and current_image_name in self.all_gkdt_boxes:
            self.gkdt_added_count = len(self.all_gkdt_boxes[current_image_name])
        else:
            self.gkdt_added_count = 0
        
        # Ekranı güncelle
        self.update_image()
        self.update_class_filter_list()  # Bu önemli - sınıf filtresini güncelle
        
        QMessageBox.information(self, "Başarılı", "Toplu GKST işlemi başarıyla tamamlandı. Artık resimler arasında gezinerek yeşil etiketleri görebilirsiniz.")
        self.gkdt_btn.setEnabled(True)
        self.batch_gkdt_btn.setEnabled(True)
        self.save_gkdt_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def on_batch_image_processed(self, image_name, processed_results):
        # Bu fonksiyon her resim işlendiğinde çağrılır
        
        # all_gkdt_boxes'a hemen ekle
        if image_name not in self.all_gkdt_boxes:
            self.all_gkdt_boxes[image_name] = []
        self.all_gkdt_boxes[image_name] = processed_results
        
        # Eğer güncellenen resim şu an ekranda olan resimse, anlık olarak güncelle
        if self.image_paths[self.index] == image_name:
            self.gkdt_added_count = len(processed_results)
            self.update_image()
            self.update_class_filter_list()

    def on_batch_error(self, message):
        self.gkdt_status_label.setText("Toplu işlem sırasında bir hata oluştu.")
        QMessageBox.critical(self, "Hata", f"Toplu işlem sırasında bir hata oluştu: {message}")
        self.gkdt_btn.setEnabled(True)
        self.batch_gkdt_btn.setEnabled(True)
        self.save_gkdt_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def update_scaling_factors(self):
        scaled_width = int(self.original_image_size.width() * self.zoom_level)
        scaled_height = int(self.original_image_size.height() * self.zoom_level)
        
        self.sf_w = scaled_width / self.original_image_size.width() if self.original_image_size.width() > 0 else 1.0
        self.sf_h = scaled_height / self.original_image_size.height() if self.original_image_size.height() > 0 else 1.0

    def update_progress(self):
        if self.image_paths:
            progress = int((self.index + 1) / len(self.image_paths) * 100)
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setValue(0)

    def update_stats(self):
        total_images = len(self.image_paths)
        current_image = self.index + 1 if self.image_paths else 0
        
        current_image_name = self.image_paths[self.index] if self.image_paths and self.index < len(self.image_paths) else None
        current_gkdt_boxes = self.all_gkdt_boxes.get(current_image_name, [])
        
        self.gkdt_added_count = len(current_gkdt_boxes)
        selected = len(self.selected) + len(self.selected_gkdt)
        existing_labels = len(self.boxes)
        
        total_labels = existing_labels + self.gkdt_added_count

        self.stats_widget.update_stats(total_images, current_image, selected, existing_labels, self.gkdt_added_count, total_labels)
    
    def update_class_filter_list(self):
        current_checked = {}
        for i in range(self.class_filter_list.count()):
            item = self.class_filter_list.item(i)
            text = item.text()
            try:
                cid = int(text.split('(')[-1].strip(')'))
                current_checked[cid] = (item.checkState() == Qt.Checked)
            except (ValueError, IndexError):
                pass
        
        self.class_filter_list.clear()
        
        unique_classes = set()
        for cid, _, _, _, _ in self.boxes:
            unique_classes.add(int(cid))
        
        for gkdt_boxes in self.all_gkdt_boxes.values():
            for cid, _, _, _, _ in gkdt_boxes:
                unique_classes.add(int(cid))

        sorted_classes = sorted(list(unique_classes))
        
        for cid in sorted_classes:
            cid_int = int(cid)
            class_name = self.class_names.get(cid_int, f"C{cid_int}")

            item = QListWidgetItem(f"{class_name} ({cid})")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            
            if cid in current_checked:
                item.setCheckState(Qt.Checked if current_checked[cid] else Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
            
            self.class_filter_list.addItem(item)

    def on_mouse_press(self, event):
        if not self.scaled_pix.isNull():
            if event.button() == Qt.LeftButton:
                self.is_drawing = True
                self.start_pos = event.pos()
                self.end_pos = event.pos()
                
            elif event.button() == Qt.RightButton:
                x_im = event.pos().x() / self.sf_w
                y_im = event.pos().y() / self.sf_h

                gkdt_selected = False
                
                current_image_name = self.image_paths[self.index]
                boxes_to_check = self.all_gkdt_boxes.get(current_image_name, [])

                for idx, (_, x, y, w, h) in enumerate(boxes_to_check):
                    if x <= x_im <= x + w and y <= y_im <= y + h:
                        if idx in self.selected_gkdt:
                            self.selected_gkdt.remove(idx)
                        else:
                            self.selected_gkdt.add(idx)
                        gkdt_selected = True
                        break
                
                if not gkdt_selected:
                    for idx, (_, bx, by, bw, bh) in enumerate(self.boxes):
                        if bx <= x_im <= bx+bw and by <= y_im <= by+bh:
                            if idx in self.selected:
                                self.selected.remove(idx)
                            else:
                                self.selected.add(idx)
                            break
                self.update_image()
                
            elif event.button() == Qt.MidButton:
                self.is_panning = True
                self.pan_start_pos = event.pos()
                QApplication.setOverrideCursor(Qt.ClosedHandCursor)

    def on_mouse_move(self, event):
        if self.is_drawing:
            self.end_pos = event.pos()
            self.update_image()
            
        elif self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            
            scroll_area = self.image_label.parent().parent()
            h_bar = scroll_area.horizontalScrollBar()
            v_bar = scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())

    def on_mouse_release(self, event):
        if self.is_drawing and event.button() == Qt.LeftButton:
            self.is_drawing = False

            x1 = self.start_pos.x()
            y1 = self.start_pos.y()
            x2 = self.end_pos.x()
            y2 = self.end_pos.y()

            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)

            if rw > 10 and rh > 10:
                ox = rx / self.sf_w
                oy = ry / self.sf_h
                ow = rw / self.sf_w
                oh = rh / self.sf_h

                dialog = QInputDialog(self)
                dialog.setWindowTitle("Sınıf ID Girin")
                dialog.setLabelText("Sınıf ID:")
                dialog.setTextValue("0")

                if dialog.exec_() == QDialog.Accepted:
                    class_id = dialog.textValue()
                    if class_id.isdigit():
                        class_id_int = int(class_id)
                        
                        self.boxes.append((str(class_id_int), ox, oy, ow, oh))
                        self.labels.append("") 
                        self.update_image()
            
            self.selected.clear()
            self.selected_gkdt.clear()
            self.update_image()
        
        elif self.is_panning and event.button() == Qt.MidButton:
            self.is_panning = False
            QApplication.restoreOverrideCursor()

    def on_wheel_event(self, event):
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            zoom_factor = 1.15 if delta > 0 else 1 / 1.15

            scroll_area = self.image_label.parent().parent()
            
            old_scroll_x = scroll_area.horizontalScrollBar().value()
            old_scroll_y = scroll_area.verticalScrollBar().value()
            
            mouse_x = event.pos().x()
            mouse_y = event.pos().y()
            
            image_x = (mouse_x + old_scroll_x) / self.sf_w
            image_y = (mouse_y + old_scroll_y) / self.sf_h

            new_zoom_level = max(0.1, min(10.0, self.zoom_level * zoom_factor))
            
            if new_zoom_level != self.zoom_level:
                self.zoom_level = new_zoom_level
                
                self.update_image()
                
                new_scroll_x = image_x * self.sf_w - mouse_x
                new_scroll_y = image_y * self.sf_h - mouse_y
                
                h_scrollbar = scroll_area.horizontalScrollBar()
                v_scrollbar = scroll_area.verticalScrollBar()
                
                new_scroll_x = max(h_scrollbar.minimum(), 
                                min(h_scrollbar.maximum(), int(new_scroll_x)))
                new_scroll_y = max(v_scrollbar.minimum(), 
                                min(v_scrollbar.maximum(), int(new_scroll_y)))
                
                h_scrollbar.setValue(new_scroll_x)
                v_scrollbar.setValue(new_scroll_y)
            
            event.accept()
        else:
            super().wheelEvent(event)

    def update_image(self):
        if not self.image_paths or not self.image_folder:
            self.file_label.setText("Görüntü klasörü seçin.")
            self.image_label.clear()
            self.scaled_pix = QPixmap()
            self.update_progress()
            self.update_stats()
            return

        path = self.image_paths[self.index]
        img_path = os.path.join(self.image_folder, path)

        if not os.path.exists(img_path):
            self.file_label.setText(f"Hata: {path} dosyası bulunamadı.")
            return

        self.file_label.setText(path)
        self.load_labels()
        
        visible_classes = set()
        for i in range(self.class_filter_list.count()):
            item = self.class_filter_list.item(i)
            if item.checkState() == Qt.Checked:
                text = item.text()
                try:
                    cid = int(text.split('(')[-1].strip(')'))
                    visible_classes.add(cid)
                except (ValueError, IndexError):
                    pass

        if self.original_pix.isNull():
            self.file_label.setText(f"Hata: {path} dosyası yüklenemedi.")
            self.image_label.clear()
            return
        
        scaled_width = int(self.original_image_size.width() * self.zoom_level)
        scaled_height = int(self.original_image_size.height() * self.zoom_level)

        scaled = self.original_pix.scaled(
            scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.scaled_pix = scaled

        self.image_label.setFixedSize(scaled.size())

        self.sf_w = scaled_width / self.original_image_size.width() if self.original_image_size.width() > 0 else 1.0
        self.sf_h = scaled_height / self.original_image_size.height() if self.original_image_size.height() > 0 else 1.0
        
        canvas = scaled.copy()
        painter = QPainter(canvas)

        for idx, (cid, x, y, w, h) in enumerate(self.boxes):
            class_id_int = int(cid)
            if class_id_int in visible_classes:
                class_name = self.class_names.get(class_id_int, f"C{class_id_int}")
            
                if idx in self.selected:
                    pen = QPen(QColor(66, 153, 225), 3)
                else:
                    pen = QPen(QColor(255, 107, 107), 2)

                painter.setPen(pen)
                r = QRectF(x*self.sf_w, y*self.sf_h, w*self.sf_w, h*self.sf_h)
                painter.drawRect(r)

                label_x = int(x * self.sf_w)
                label_y = max(0, int(y * self.sf_h - 20))
            
                font_metrics = painter.fontMetrics()
                label_width = font_metrics.width(class_name) + 10
                label_height = font_metrics.height() + 4

                painter.fillRect(
                    label_x, label_y, label_width, label_height,
                    QColor(66, 153, 225) if idx in self.selected else QColor(255, 107, 107)
                )
                painter.setPen(QPen(Qt.white))
                painter.drawText(
                    label_x + 5, label_y + 12, class_name
                )
        
        current_image_name = self.image_paths[self.index] if self.image_paths and self.index < len(self.image_paths) else None
        gkdt_boxes_to_draw = self.all_gkdt_boxes.get(current_image_name, [])

        for idx, (cid, x, y, w, h) in enumerate(gkdt_boxes_to_draw):
            if int(cid) in visible_classes:
                class_name = self.class_names.get(cid, f"C{cid}")
            
                if idx in self.selected_gkdt:
                    gkdt_pen = QPen(QColor(255, 193, 7), 3)
                else:
                    gkdt_pen = QPen(QColor(16, 185, 129), 2)
            
                painter.setPen(gkdt_pen)
                r = QRectF(x*self.sf_w, y*self.sf_h, w*self.sf_w, h*self.sf_h)
                painter.drawRect(r)

                gkdt_label_x = int(x * self.sf_w)
                gkdt_label_y = max(0, int(y * self.sf_h - 20))

                font_metrics = painter.fontMetrics()
                label_width = font_metrics.width(class_name) + 10
                label_height = font_metrics.height() + 4
            
                bg_color = QColor(255, 193, 7) if idx in self.selected_gkdt else QColor(16, 185, 129)
                painter.fillRect(
                    gkdt_label_x, gkdt_label_y, label_width, label_height, bg_color
                )
                painter.setPen(QPen(Qt.white))
                painter.drawText(
                    gkdt_label_x + 5, gkdt_label_y + 12, class_name
                )

        if self.is_drawing:
            preview = QPen(QColor(16, 185, 129), 2, Qt.DashLine)
            painter.setPen(preview)
            
            x1 = self.start_pos.x()
            y1 = self.start_pos.y()
            x2 = self.end_pos.x()
            y2 = self.end_pos.y()

            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            
            painter.drawRect(QRectF(rx, ry, rw, rh))

        painter.end()
        self.image_label.setPixmap(canvas)
        self.update_progress()
        self.update_stats()
        self.setFocus()

    def delete_selected_gkdt_boxes(self):
        if not self.selected_gkdt:
            return

        current_image_name = self.image_paths[self.index]
        new_gkdt_boxes = [
            box for i, box in enumerate(self.all_gkdt_boxes.get(current_image_name, []))
            if i not in self.selected_gkdt
        ]
        self.all_gkdt_boxes[current_image_name] = new_gkdt_boxes

        self.selected_gkdt.clear()
        self.update_image()
        
    def delete_selected_boxes(self):
        if not self.selected:
            return

        reply = QMessageBox.question(
            self, 'Onay',
            f'{len(self.selected)} etiketi silmek istediğinizden emin misiniz?',
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            new_labels = [
                label for i, label in enumerate(self.labels)
                if i not in self.selected
            ]
            new_boxes = [
                box for i, box in enumerate(self.boxes)
                if i not in self.selected
            ]
            self.boxes = new_boxes
            self.labels = new_labels

            if os.path.exists(self.label_path):
                with open(self.label_path, 'w') as f:
                    if new_labels:
                        f.write("\n".join(new_labels) + "\n")

            self.selected.clear()
            self.update_image()

    def clear_all_boxes(self):
        if not self.boxes:
            return

        reply = QMessageBox.question(
            self, 'Onay',
            'Tüm etiketleri silmek istediğinizden emin misiniz?',
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if os.path.exists(self.label_path):
                with open(self.label_path, 'w') as f:
                    f.write("")

            self.selected.clear()
            self.load_labels()
            self.update_image()

    def previous_image(self):
        if self.image_paths and self.index > 0:
            self.index -= 1
            self.selected.clear()
            self.selected_gkdt.clear()
            self.zoom_level = 1.0
            self.pan_offset = QPointF(0, 0)
            self.update_image()

    def next_image(self):
        if self.image_paths and self.index < len(self.image_paths) - 1:
            self.index += 1
            self.selected.clear()
            self.selected_gkdt.clear()
            self.zoom_level = 1.0
            self.pan_offset = QPointF(0, 0)
            self.update_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key_Delete:
            self.delete_selected_gkdt_boxes()
            self.delete_selected_boxes()
        elif event.key() == Qt.Key_O:
            self.select_image_folder()
        elif event.key() == Qt.Key_L:
            self.select_labels_folder()
        elif event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key_Escape:
            self.selected.clear()
            self.selected_gkdt.clear()
            self.update_image()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()

class ModernYOLOApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setStyle('Fusion')
        self.setup_palette()
        self.window = ModernImageNavigator()
        self.window.show()

    def setup_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(247, 250, 252))
        palette.setColor(QPalette.WindowText, QColor(74, 85, 104))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(247, 250, 252))
        palette.setColor(QPalette.Text, QColor(45, 55, 72))
        palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(226, 232, 240))
        palette.setColor(QPalette.ButtonText, QColor(74, 85, 104))
        palette.setColor(QPalette.Highlight, QColor(102, 126, 234))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)

def main():
    app = ModernYOLOApp(sys.argv)
    app.setApplicationName("Modern GKST Etiketleme Aracı")
    app.setApplicationVersion("2.7")
    app.setOrganizationName("CLass Tools")
    return app.exec_()

if __name__ == '__main__':
    main()