# services/detection/detectors.py
import numpy as np
import cv2
from typing import List, Optional
from pathlib import Path
import sys
sys.path.append('../..')
from models.base import BoundingBox
from services.detection.interfaces import IDetector

class YoloDetector(IDetector):
    """YOLO PyTorch model detector"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='detect')
        self.device = device
        self.confidence_threshold = 0.2
        self.allowed_classes = {0, 2, 3, 4, 5}
        
    def detect(self, image_path: str, confidence: float) -> List[BoundingBox]:
        results = self.model.predict(
            image_path,
            conf=confidence,
            device=self.device,
            verbose=False
        )
        
        boxes = []
        if results[0].boxes is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls in zip(xyxy, scores, classes):
                if cls in self.allowed_classes:
                    x1, y1, x2, y2 = box
                    boxes.append(BoundingBox(
                        class_id=int(cls),
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1)
                    ))
        
        return boxes
    
    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = threshold
    
    def get_model_info(self) -> dict:
        return {
            'type': 'YOLO PyTorch',
            'device': self.device,
            'confidence': self.confidence_threshold,
            'precision': 'FP32'
        }

class ONNXDetector(IDetector):
    """ONNX model detector"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        import onnxruntime as ort
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = 0.2
        self.allowed_classes = {0, 2, 3, 4, 5}
        self.session = self._create_session()
        
    def _create_session(self):
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        
        if self.device.startswith('cuda'):
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider']
        
        return ort.InferenceSession(self.model_path, providers=providers)
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image
    
    def detect(self, image_path: str, confidence: float) -> List[BoundingBox]:
        # Orijinal görüntü boyutlarını al
        orig_img = cv2.imread(image_path)
        orig_h, orig_w = orig_img.shape[:2]
        
        # Preprocess ve inference
        preprocessed = self._preprocess_image(image_path)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: preprocessed})
        
        # Postprocess
        predictions = outputs[0]
        boxes = []
        
        if predictions.shape[1] == 8400:
            predictions = np.squeeze(predictions, axis=0)
        elif predictions.shape[2] == 8400:
            predictions = np.transpose(np.squeeze(predictions, axis=0), (1, 0))
        
        mask = predictions[:, 4] >= confidence
        filtered = predictions[mask]
        
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        
        for pred in filtered:
            x_center, y_center, width, height, conf = pred[:5]
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            
            if class_id in self.allowed_classes:
                x1 = (x_center - width/2) * scale_x
                y1 = (y_center - height/2) * scale_y
                w = width * scale_x
                h = height * scale_y
                
                boxes.append(BoundingBox(
                    class_id=int(class_id),
                    x=float(x1),
                    y=float(y1),
                    width=float(w),
                    height=float(h)
                ))
        
        return boxes
    
    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = threshold
    
    def get_model_info(self) -> dict:
        return {
            'type': 'ONNX',
            'device': self.device,
            'confidence': self.confidence_threshold,
            'precision': 'FP16' if self.device.startswith('cuda') else 'FP32'
        }

class SAHIDetector(IDetector):
    """SAHI (Sliced Aided Hyper Inference) wrapper"""
    
    def __init__(self, base_detector: IDetector, slice_height: int = 256, slice_width: int = 256):
        self.base_detector = base_detector
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = 0.2
        
    def detect(self, image_path: str, confidence: float) -> List[BoundingBox]:
        try:
            from sahi.predict import get_sliced_prediction
            from sahi.models.ultralytics import UltralyticsDetectionModel
            
            # SAHI için model wrapper oluştur
            if hasattr(self.base_detector, 'model'):
                # YOLO model için
                detection_model = UltralyticsDetectionModel(
                    model_path=self.base_detector.model.model_name,
                    confidence_threshold=confidence,
                    device=self.base_detector.device
                )
            else:
                # ONNX için custom wrapper kullan
                return self.base_detector.detect(image_path, confidence)
            
            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio
            )
            
            boxes = []
            for detection in result.object_prediction_list:
                bbox = detection.bbox.to_voc_bbox()
                boxes.append(BoundingBox(
                    class_id=detection.category.id,
                    x=bbox[0],
                    y=bbox[1],
                    width=bbox[2] - bbox[0],
                    height=bbox[3] - bbox[1]
                ))
            
            return boxes
            
        except ImportError:
            # SAHI yoksa normal detection yap
            return self.base_detector.detect(image_path, confidence)
    
    def set_confidence_threshold(self, threshold: float):
        self.base_detector.set_confidence_threshold(threshold)
    
    def get_model_info(self) -> dict:
        info = self.base_detector.get_model_info()
        info['sahi'] = True
        info['slice_size'] = f"{self.slice_width}x{self.slice_height}"
        return info