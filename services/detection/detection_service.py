# services/detection/detection_service.py
from pathlib import Path
from typing import List, Any, Iterable
import sys
sys.path.append('../..')

from models.base import BoundingBox, DetectionResult
from services.detection.interfaces import IDetector, IDetectorFactory, IDetectionService
from services.detection.detectors import YoloDetector, ONNXDetector, SAHIDetector
from config.settings import (
    ALLOWED_CLASSES,    # örn: {0,2,3,4,5}
    IOU_THRESHOLD,      # 0.10
    CONTAIN_RATIO,      # 0.60
    SAME_CLASS_ONLY     # True
)

def _get_w(b: BoundingBox) -> float:
    # width alanını hem w hem width olarak destekle
    return float(getattr(b, "w", getattr(b, "width", 0.0)))

def _get_h(b: BoundingBox) -> float:
    return float(getattr(b, "h", getattr(b, "height", 0.0)))

class DetectorFactory(IDetectorFactory):
    def create_detector(
        self,
        model_path: Path,
        device: str = 'cpu',
        use_sahi: bool = False,
        slice_height: int = 256,
        slice_width: int = 256
    ) -> IDetector:
        model_str = str(model_path)
        if model_path.suffix.lower() == '.onnx':
            base_detector = ONNXDetector(model_str, device)
        elif model_path.suffix.lower() == '.pt':
            base_detector = YoloDetector(model_str, device)
        else:
            raise ValueError(f"Desteklenmeyen model formatı: {model_path.suffix}")
        return SAHIDetector(base_detector, slice_height, slice_width) if use_sahi else base_detector


class DetectionService(IDetectionService):
    def __init__(self, detector: IDetector):
        self.detector = detector

    # ---------- geometri ----------
    @staticmethod
    def _area(b: BoundingBox) -> float:
        return max(0.0, _get_w(b)) * max(0.0, _get_h(b))

    @staticmethod
    def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
        ax2, ay2 = a.x + _get_w(a), a.y + _get_h(a)
        bx2, by2 = b.x + _get_w(b), b.y + _get_h(b)
        iw = max(0.0, min(ax2, bx2) - max(a.x, b.x))
        ih = max(0.0, min(ay2, by2) - max(a.y, b.y))
        return iw * ih

    @staticmethod
    def _iou_xywh(a: BoundingBox, b: BoundingBox) -> float:
        ax1, ay1 = float(a.x), float(a.y)
        ax2, ay2 = ax1 + _get_w(a), ay1 + _get_h(a)
        bx1, by1 = float(b.x), float(b.y)
        bx2, by2 = bx1 + _get_w(b), by1 + _get_h(b)

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter   = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a  = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b  = max(1e-6, (bx2 - bx1) * (by2 - by1))
        union   = max(1e-6, area_a + area_b - inter)
        return inter / union

    # Dedektörü uygun yöntem adıyla çağır
    def _call_detector(self, image_path: str, confidence: float) -> Any:
        for name in ("predict", "detect", "run", "infer", "detect_single_image"):
            if hasattr(self.detector, name):
                fn = getattr(self.detector, name)
                try:
                    return fn(image_path, confidence)
                except TypeError:
                    return fn(image_path)
        raise AttributeError("Detector has no supported inference method (predict/detect/run/infer/detect_single_image).")

    # Çıktıyı BoundingBox listesine normalize et
    def _normalize_result(self, raw: Any) -> List[BoundingBox]:
        boxes: List[BoundingBox] = []

        # Zaten DetectionResult ise
        if isinstance(raw, DetectionResult):
            candidates = raw.boxes

        # Doğrudan BoundingBox listesi
        elif isinstance(raw, list) and (not raw or isinstance(raw[0], BoundingBox)):
            candidates = raw

        # SAHI benzeri sonuç
        elif hasattr(raw, "object_prediction_list"):
            candidates = []
            for p in raw.object_prediction_list:
                if hasattr(p, "bbox"):
                    if hasattr(p.bbox, "to_voc_bbox"):
                        x1, y1, x2, y2 = p.bbox.to_voc_bbox()
                    else:
                        x1, y1, x2, y2 = p.bbox
                    w, h = float(x2 - x1), float(y2 - y1)
                    cid = int(getattr(getattr(p, "category", None), "id", getattr(p, "category_id", -1)))
                    candidates.append(BoundingBox(cid, float(x1), float(y1), w, h))
        else:
            candidates = []

        # İzinli sınıf filtresi + alan adlarını normalize et
        for b in candidates:
            cid = int(b.class_id)
            if cid in ALLOWED_CLASSES:
                boxes.append(BoundingBox(cid, float(b.x), float(b.y), _get_w(b), _get_h(b)))
        return boxes

    # Tek görüntü tespiti
    def detect_single_image(self, image_path: str, confidence: float) -> DetectionResult:
        raw = self._call_detector(image_path, confidence)
        boxes = self._normalize_result(raw)

        # SAHI kullanılıyorsa bu öznitelikler detektörde mevcut olur; yoksa 0 veriyoruz
        slice_h = int(getattr(self.detector, "slice_height", 0) or 0)
        slice_w = int(getattr(self.detector, "slice_width", 0) or 0)

        return DetectionResult(
            image_filename=Path(image_path).name,
            boxes=boxes,
            confidence=float(confidence),
            slice_height=slice_h,
            slice_width=slice_w,
        )

    def _filter_by_allowlist(self, boxes: Iterable[BoundingBox]) -> List[BoundingBox]:
        return [b for b in boxes if int(b.class_id) in ALLOWED_CLASSES]

    def _suppress_with_existing(self,
                                candidates: Iterable[BoundingBox],
                                existing: Iterable[BoundingBox],
                                iou_thr: float) -> List[BoundingBox]:
        """Aynı sınıftaki mevcut (kırmızı) kutularla IoU>=eşik olan adayları at."""
        keep: List[BoundingBox] = []
        for c in candidates:
            clash = False
            for e in existing:
                if int(c.class_id) != int(e.class_id):
                    continue
                if self._iou_xywh(c, e) >= iou_thr:
                    clash = True
                    break
            if not clash:
                keep.append(c)
        return keep

    def _nms_same_source(self, boxes: List[BoundingBox], iou_thr: float = 0.5) -> List[BoundingBox]:
        """SAHI kendi içindeki yakın çiftleri bastırmak için basit NMS."""
        boxes = sorted(boxes, key=lambda b: getattr(b, "score", 1.0), reverse=True)
        kept: List[BoundingBox] = []
        for b in boxes:
            if all(self._iou_xywh(b, kb) < iou_thr or int(b.class_id) != int(kb.class_id) for kb in kept):
                kept.append(b)
        return kept

    # Toplu tespit
    def detect_batch(self, image_paths: list[str], confidence: float = 0.2) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        slice_h = int(getattr(self.detector, "slice_height", 0) or 0)
        slice_w = int(getattr(self.detector, "slice_width", 0) or 0)

        for p in image_paths:
            raw = self._call_detector(p, confidence)
            boxes = self._normalize_result(raw)
            results.append(
                DetectionResult(
                    image_filename=Path(p).name,
                    boxes=boxes,
                    confidence=float(confidence),
                    slice_height=slice_h,
                    slice_width=slice_w,
                )
            )
        return results

    # Kırmızı kutulara göre yeşilleri bastır (IoU veya içerilme)
    def suppress_by_manual(
        self,
        det_boxes: List[BoundingBox],
        manual_boxes: List[BoundingBox],
        iou_thr: float = IOU_THRESHOLD,
        contain_ratio: float = CONTAIN_RATIO,
        same_class_only: bool = SAME_CLASS_ONLY,
    ) -> List[BoundingBox]:
        if not manual_boxes or not det_boxes:
            return det_boxes
        kept: List[BoundingBox] = []
        for d in det_boxes:
            suppress = False
            for m in manual_boxes:
                if same_class_only and int(d.class_id) != int(m.class_id):
                    continue
                inter = self._intersection_area(d, m)
                if inter <= 0:
                    continue
                d_area = self._area(d)
                m_area = self._area(m)
                union = d_area + m_area - inter if (d_area + m_area - inter) > 0 else 0.0
                iou = inter / union if union > 0 else 0.0
                d_cont = inter / d_area if d_area > 0 else 0.0
                m_cont = inter / m_area if m_area > 0 else 0.0
                if (iou >= iou_thr) or (d_cont >= contain_ratio) or (m_cont >= contain_ratio):
                    suppress = True
                    break
            if not suppress:
                kept.append(d)
        return kept

    # Dupe temizliği (geriye uyumlu)
    def filter_duplicate_detections(
        self,
        det_boxes: List[BoundingBox],
        other_boxes: List[BoundingBox],
        iou_threshold: float = IOU_THRESHOLD,
        same_class_only: bool = SAME_CLASS_ONLY,
        contain_ratio: float = CONTAIN_RATIO,
    ) -> List[BoundingBox]:
        if not other_boxes or not det_boxes:
            return det_boxes
        kept: List[BoundingBox] = []
        for d in det_boxes:
            duplicate = False
            for o in other_boxes:
                if same_class_only and int(d.class_id) != int(o.class_id):
                    continue
                inter = self._intersection_area(d, o)
                if inter <= 0:
                    continue
                d_area = self._area(d)
                o_area = self._area(o)
                union = d_area + o_area - inter if (d_area + o_area - inter) > 0 else 0.0
                iou = inter / union if union > 0 else 0.0
                d_cont = inter / d_area if d_area > 0 else 0.0
                o_cont = inter / o_area if o_area > 0 else 0.0
                if (iou >= iou_threshold):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(d)
        return kept
