# ui/components/image_viewer.py
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from typing import List, Set, Tuple, Optional
from PyQt5.QtCore import QRectF
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtGui import QPixmap
import sys
sys.path.append('../..')
from models.base import BoundingBox
from config.settings import ALLOWED_CLASSES

class ImageViewer(QLabel):
    """Görüntü görüntüleme ve etkileşim widget'ı"""
    
    # Signals
    box_drawn = pyqtSignal(float, float, float, float)  # x, y, w, h
    box_selected = pyqtSignal(int, bool)  # index, is_detector_box
    zoom_changed = pyqtSignal(float)  # zoom_level
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: white; border-radius: 6px;")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        # Image data
        self.original_pixmap = QPixmap()
        self.scaled_pixmap = QPixmap()
        self._pixmap_orig = QPixmap()
        # Zoom & pan
        self.zoom_level = 1.0
        self.zoom_offset_x = 0.0
        self.zoom_offset_y = 0.0
        self.is_panning = False
        self.pan_start_pos = QPointF()
        
        # Drawing
        self.is_drawing = False
        self.draw_start_pos = QPointF()
        self.draw_end_pos = QPointF()
        
        # Boxes
        self.boxes: List[BoundingBox] = []
        self.detector_boxes: List[BoundingBox] = []
        self.selected_boxes: Set[int] = set()
        self.selected_detector_boxes: Set[int] = set()
        self.visible_classes: Set[int] = set()
        
        # Scaling factors
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.sf_w = 1.0  # width scaling factor
        self.sf_h = 1.0  # height scaling factor

        # Class names - COCO dataset classes (only person and car)
        self.class_names = {
            0: "person",
            2: "car"
        }

        # Scroll area reference for zoom-to-cursor
        self.scroll_area = None

    def set_scroll_area(self, scroll_area):
        """Set reference to parent scroll area for zoom functionality"""
        self.scroll_area = scroll_area
    
    def set_image(self, pixmap: QPixmap):
        """Dışarıdan gelen ham pixmap'i iç duruma koyup ekrana hazırla."""
        if isinstance(pixmap, QPixmap):
            # orijinali sakla (copy ile kopyalamak istersen: pixmap.copy())
            self._pixmap_orig = pixmap
        else:
            self._pixmap_orig = QPixmap()  # güvenli fallback

        # Reset zoom when changing images
        self.reset_zoom()
        self.update_display()

    def reset_zoom(self):
        """Reset zoom level to default"""
        self.zoom_level = 1.0
        self.zoom_offset_x = 0.0
        self.zoom_offset_y = 0.0

        # Reset scroll area to resizable mode
        if self.scroll_area:
            self.scroll_area.setWidgetResizable(True)

        # Reset size constraints
        self.setMinimumSize(400, 300)
        self.setMaximumSize(16777215, 16777215)  # QWIDGETSIZE_MAX
    
    def set_boxes(self, boxes: List[BoundingBox], detector_boxes: List[BoundingBox]):
        """Kutuları ayarla"""
        self.boxes = boxes
        self.detector_boxes = detector_boxes
        self.update_display()
    
    def set_visible_classes(self, classes: Set[int]):
        """Görünür sınıfları ayarla"""
        self.visible_classes = classes
        self.update_display()

    def _normalize_box(self, box):
        """
        Bir kutuyu (tuple/list/BoundingBox) (cid, x, y, w, h) formatına normalizer.
        Desteklenen varyantlar:
        - (cid, x, y, w, h)
        - (cid, x1, y1, x2, y2)  -> w = x2-x1, h = y2-y1
        - obj.class_id / obj.cid + (x,y,w,h) veya (x1,y1,x2,y2)
        """
        # 1) Tuple / list
        if isinstance(box, (tuple, list)):
            if len(box) == 5:
                cid, x, y, w, h = box
                return int(cid), float(x), float(y), float(w), float(h)
            if len(box) == 6:
                # bazı akışlarda (cid, x1, y1, x2, y2, score) gibi olabilir
                cid, x1, y1, x2, y2, *_ = box
                return int(cid), float(x1), float(y1), float(x2 - x1), float(y2 - y1)
            if len(box) == 4:
                # (x,y,w,h) verilmiş ise sınıfı 0 kabul et
                x, y, w, h = box
                return 0, float(x), float(y), float(w), float(h)

        # 2) Nesne (BoundingBox benzeri)
        cid = getattr(box, "class_id", getattr(box, "cid", 0))

        if all(hasattr(box, a) for a in ("x", "y", "w", "h")):
            return int(cid), float(box.x), float(box.y), float(box.w), float(box.h)

        if all(hasattr(box, a) for a in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = float(box.x1), float(box.y1), float(box.x2), float(box.y2)
            return int(cid), x1, y1, (x2 - x1), (y2 - y1)

        # son çare: hataya düşmemek için 0 kutu döndür
        return 0, 0.0, 0.0, 0.0, 0.0
    
    def _suppress_for_drawing(self, det_boxes, manual_boxes, iou_thr=0.30, contain_ratio=0.60, same_class=True):
        kept = []
        def cid(b): return int(getattr(b, "class_id", getattr(b, "cid", -1)))

        for d in det_boxes:
            dx, dy = float(d.x), float(d.y)
            dw, dh = float(d.width), float(d.height)
            d_area = max(0.0, dw) * max(0.0, dh)
            d_cid = cid(d)
            suppress = False
            for m in manual_boxes:
                if same_class and (d_cid != cid(m)):
                    continue
                mx, my = float(m.x), float(m.y)
                mw, mh = float(m.width), float(m.height)
                # kesişim
                dx2, dy2 = dx + dw, dy + dh
                mx2, my2 = mx + mw, my + mh
                iw = max(0.0, min(dx2, mx2) - max(dx, mx))
                ih = max(0.0, min(dy2, my2) - max(dy, my))
                inter = iw * ih
                if inter <= 0.0:
                    continue
                # IoU
                union = d_area + (mw*mh) - inter
                iou = inter/union if union > 0 else 0.0
                # içerilme
                d_cont = inter/d_area if d_area > 0 else 0.0
                m_area = mw*mh
                m_cont = inter/m_area if m_area > 0 else 0.0
                if (iou >= iou_thr) or (d_cont >= contain_ratio) or (m_cont >= contain_ratio):
                    suppress = True
                    break
            if not suppress:
                kept.append(d)
        return kept

    
    def update_display(self):
        # 1) Kaynak görüntü var mı?
        src = self._pixmap_orig                     # <--- ESKİ: getattr(self, "pixmap", None)
        if not isinstance(src, QPixmap) or src.isNull():
            # QLabel isen boş göster
            if hasattr(self, "setPixmap"):
                self.setPixmap(QPixmap())
            return

        # 2) Ölçekli görseli hazırla (apply zoom)
        target_w = max(1, self.width())
        target_h = max(1, self.height())

        # Apply zoom to target size
        zoomed_w = int(target_w * self.zoom_level)
        zoomed_h = int(target_h * self.zoom_level)

        scaled = src.scaled(zoomed_w, zoomed_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 3) Ölçekleri güncelle
        img_w, img_h = src.width(), src.height()
        self.scale_x = scaled.width() / img_w if img_w else 1.0
        self.scale_y = scaled.height() / img_h if img_h else 1.0

        # 4) Canvas ve painter
        canvas = scaled.copy()
        painter = QPainter(canvas)

        # 5) Kutu listelerini al (gerekirse bastırma/filtre uygula)
        manual_boxes = list(getattr(self, "boxes", []))
        det_boxes    = list(getattr(self, "detector_boxes", []))
        
        det_boxes = self._suppress_for_drawing(
            det_boxes, manual_boxes,
            iou_thr=0.30, contain_ratio=0.60, same_class=True
        )
        if hasattr(self, "_suppress_for_drawing"):
            det_boxes = self._suppress_for_drawing(det_boxes, manual_boxes, iou_thr=0.35, same_class=True)

        allowed = getattr(self, "allowed_class_ids", None)
        if allowed is not None:
            def _cid(b): return int(getattr(b, "class_id", getattr(b, "cid", -1)))
            manual_boxes = [b for b in manual_boxes if _cid(b) in allowed]
            det_boxes    = [b for b in det_boxes    if _cid(b) in allowed]

        try:
            # Önce kırmızı, sonra yeşil (ya da tam tersi, tercihine göre)
            self._draw_boxes(painter, manual_boxes, self.selected_boxes, is_detector=False)
            self._draw_boxes(painter, det_boxes,    self.selected_detector_boxes, is_detector=True)

            # Draw preview rectangle if drawing
            if self.is_drawing:
                self._draw_preview_rect(painter)
        finally:
            painter.end()

        # 7) Ekrana bas
        if hasattr(self, "setPixmap"):
            self.setPixmap(canvas)

    
    def _draw_boxes(self, painter, boxes, selected_set, is_detector: bool = False):
        """
        boxes: BoundingBox listesi (veya benzer alan adlarına sahip objeler)
            (x,y,width,height,class_id[/cid])
        selected_set: seçili indeksleri içeren set
        is_detector: True => yeşil (detector), False => kırmızı (manuel)
        """
        # Renkler
        base_color   = QColor(16, 185, 129) if is_detector else QColor(255, 107, 107)  # yeşil / kırmızı
        sel_color    = QColor(255, 193, 7)  if is_detector else QColor(66, 153, 225)   # sarı / mavi
        pen_width    = 2

        # class adını yazarken kullanacağımız helper
        def _class_name(cid: int) -> str:
            mapping = getattr(self, "class_names", {
                0:"person", 2:"car"
            })
            # Only return name if class is in ALLOWED_CLASSES
            if cid in ALLOWED_CLASSES:
                return mapping.get(cid, f"Class_{cid}")
            # Don't show any label for other classes
            return None

        for idx, b in enumerate(boxes):
            # kutu değerlerini normalleştir
            cid = int(getattr(b, "class_id", getattr(b, "cid", -1)))

            # Skip boxes that are not in ALLOWED_CLASSES (both red and green)
            if cid not in ALLOWED_CLASSES:
                continue

            # Apply visible_classes filter if set
            if self.visible_classes and cid not in self.visible_classes:
                continue

            x   = float(getattr(b, "x", getattr(b, "x1", 0.0)))
            y   = float(getattr(b, "y", getattr(b, "y1", 0.0)))
            w   = float(getattr(b, "width", getattr(b, "w",  getattr(b, "x2", 0.0) - getattr(b, "x1", 0.0))))
            h   = float(getattr(b, "height", getattr(b, "h", getattr(b, "y2", 0.0) - getattr(b, "y1", 0.0))))

            # ölçeği uygula
            sx = self.scale_x or 1.0
            sy = self.scale_y or 1.0
            rx, ry, rw, rh = x*sx, y*sy, w*sx, h*sy

            # çizim rengi (seçili mi?)
            color = sel_color if (idx in selected_set) else base_color
            pen   = QPen(color, 3 if (idx in selected_set) else pen_width)
            painter.setPen(pen)

            # dikdörtgen
            painter.drawRect(QRectF(rx, ry, rw, rh))

            # etiket
            label = _class_name(cid)
            if label:  # Draw label if we have one
                self._draw_label(painter, label, rx, ry, color)
    
    def _draw_label(self, painter, text: str, x: float, y: float, bg_color: QColor):
        """
        Kutunun sol-üst köşesine etiket çizer.
        x, y: kutunun sol-üstü (float olabilir)
        """
        fm = painter.fontMetrics()
        # horizontalAdvance: PyQt5'te genişlik için daha doğru yöntem
        label_width = fm.horizontalAdvance(text) + 10
        label_height = fm.height() + 4

        # Metni kutunun üstüne taşı (20px yukarı), ekrandan taşmayı engelle
        label_y = max(0, y - 20)

        # QRectF kullanarak float koordinatlarla doldur
        bg_rect = QRectF(x, label_y, float(label_width), float(label_height))
        painter.fillRect(bg_rect, bg_color)

        # Metin rengi beyaz
        painter.setPen(QPen(Qt.white))
        # Yazının dikey konumunu güzel ayarlamak için ascent kullan
        text_x = x + 5
        text_y = label_y + fm.ascent() + 2
        painter.drawText(int(text_x), int(text_y), text)
    
    def _draw_preview_rect(self, painter: QPainter):
        """Çizim önizleme dikdörtgeni"""
        painter.setPen(QPen(QColor(66, 153, 225), 2, Qt.DashLine))

        # Get the current pixmap to calculate offset
        current_pixmap = self.pixmap()
        if current_pixmap and not current_pixmap.isNull():
            # Calculate the offset due to centered alignment
            widget_w = self.width()
            widget_h = self.height()
            pixmap_w = current_pixmap.width()
            pixmap_h = current_pixmap.height()

            offset_x = (widget_w - pixmap_w) / 2.0
            offset_y = (widget_h - pixmap_h) / 2.0

            # Adjust positions for the offset - the preview should be drawn relative to the scaled pixmap
            x1 = self.draw_start_pos.x() - offset_x
            y1 = self.draw_start_pos.y() - offset_y
            x2 = self.draw_end_pos.x() - offset_x
            y2 = self.draw_end_pos.y() - offset_y

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            painter.drawRect(QRectF(x, y, w, h))

    def clear_all_selections(self):
        """Normal ve detector seçimlerini temizle ve görüntüyü yenile."""
        if hasattr(self, "selected_boxes"):
            self.selected_boxes.clear()
        if hasattr(self, "selected_detector_boxes"):
            self.selected_detector_boxes.clear()
        self.update_display()  # <-- update_display kullan    

    def mousePressEvent(self, event):
        """Mouse tıklama olayı"""
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.draw_start_pos = event.pos()
            self.draw_end_pos = event.pos()
                
        elif event.button() == Qt.RightButton:
            # Multiple selection support - toggle boxes instead of clearing
            pos = event.pos()

            # Get the current pixmap to calculate offset
            current_pixmap = self.pixmap()
            if current_pixmap and not current_pixmap.isNull():
                # Calculate the offset due to centered alignment
                widget_w = self.width()
                widget_h = self.height()
                pixmap_w = current_pixmap.width()
                pixmap_h = current_pixmap.height()

                offset_x = (widget_w - pixmap_w) / 2.0
                offset_y = (widget_h - pixmap_h) / 2.0

                # Adjust position for offset and scale
                sx = self.scale_x if getattr(self, "scale_x", 0) else 1.0
                sy = self.scale_y if getattr(self, "scale_y", 0) else 1.0

                x_im = (pos.x() - offset_x) / sx
                y_im = (pos.y() - offset_y) / sy

                # 1) Detector kutuları (BoundingBox nesneleri)
                for idx, box in enumerate(getattr(self, "detector_boxes", [])):
                    if (box.x <= x_im <= box.x + box.width) and (box.y <= y_im <= box.y + box.height):
                        # Toggle selection instead of clearing
                        if idx in self.selected_detector_boxes:
                            self.selected_detector_boxes.remove(idx)
                        else:
                            self.selected_detector_boxes.add(idx)
                        # seçim sinyali gönder ki "Seçili" sayısı güncellensin
                        self.box_selected.emit(idx, True)
                        self.setFocus()
                        self.update_display()
                        return

                # 2) Normal kutular
                for idx, box in enumerate(getattr(self, "boxes", [])):
                    if (box.x <= x_im <= box.x + box.width) and (box.y <= y_im <= box.y + box.height):
                        # Toggle selection instead of clearing
                        if idx in self.selected_boxes:
                            self.selected_boxes.remove(idx)
                        else:
                            self.selected_boxes.add(idx)
                        self.box_selected.emit(idx, False)
                        self.setFocus()
                        self.update_display()
                        return

            # If clicked on empty space, do nothing (keep current selections)
            self.update_display()

                        
        elif event.button() == Qt.MidButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()
            QApplication.setOverrideCursor(Qt.ClosedHandCursor)
            
    def mouseMoveEvent(self, event):
        """Mouse hareket olayı"""
        if self.is_drawing:
            self.draw_end_pos = event.pos()
            self.update_display()
            
        elif self.is_panning:
            # Pan handling is done by parent
            pass
    
    def mouseReleaseEvent(self, event):
        """Mouse bırakma olayı"""
        if self.is_drawing and event.button() == Qt.LeftButton:
            self.is_drawing = False

            # Get the current pixmap to calculate offset
            current_pixmap = self.pixmap()
            if current_pixmap and not current_pixmap.isNull():
                # Calculate the offset due to centered alignment
                widget_w = self.width()
                widget_h = self.height()
                pixmap_w = current_pixmap.width()
                pixmap_h = current_pixmap.height()

                offset_x = (widget_w - pixmap_w) / 2.0
                offset_y = (widget_h - pixmap_h) / 2.0

                # Calculate box coordinates with offset correction
                x1, y1 = self.draw_start_pos.x() - offset_x, self.draw_start_pos.y() - offset_y
                x2, y2 = self.draw_end_pos.x() - offset_x, self.draw_end_pos.y() - offset_y

                x = min(x1, x2) / self.scale_x
                y = min(y1, y2) / self.scale_y
                w = abs(x2 - x1) / self.scale_x
                h = abs(y2 - y1) / self.scale_y

                if w > 10 and h > 10:  # Minimum size check
                    self.box_drawn.emit(x, y, w, h)

            self.draw_start_pos = QPointF()
            self.draw_end_pos = QPointF()
            self.update_display()
            
        elif self.is_panning and event.button() == Qt.MidButton:
            self.is_panning = False
            QApplication.restoreOverrideCursor()
    
    def wheelEvent(self, event):
        """Mouse tekerleği olayı - Zoom at cursor position"""
        if event.modifiers() == Qt.ControlModifier:
            if not self.scroll_area or self._pixmap_orig.isNull():
                event.accept()
                return

            # Get cursor position in scroll area coordinates
            cursor_pos = event.pos()

            # Calculate zoom factor (reduced for smoother, safer zooming)
            delta = event.angleDelta().y()
            zoom_factor = 1.08 if delta > 0 else 1 / 1.08  # Reduced from 1.15 to 1.08

            old_zoom = self.zoom_level
            new_zoom = max(0.5, min(5.0, self.zoom_level * zoom_factor))  # Range: 0.5x to 5.0x

            if new_zoom != old_zoom:
                # Get scroll bar positions before zoom
                h_scroll = self.scroll_area.horizontalScrollBar()
                v_scroll = self.scroll_area.verticalScrollBar()

                # Calculate cursor position relative to current scroll position
                cursor_x_on_widget = cursor_pos.x() + h_scroll.value()
                cursor_y_on_widget = cursor_pos.y() + v_scroll.value()

                # Disable widget resizable to allow manual size control
                if self.scroll_area.widgetResizable():
                    self.scroll_area.setWidgetResizable(False)

                # Update zoom level first
                self.zoom_level = new_zoom

                # Calculate new size immediately
                target_w = max(1, self.scroll_area.viewport().width())
                target_h = max(1, self.scroll_area.viewport().height())
                zoomed_w = int(target_w * self.zoom_level)
                zoomed_h = int(target_h * self.zoom_level)

                # Scale pixmap with new zoom
                scaled = self._pixmap_orig.scaled(zoomed_w, zoomed_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Update scaling factors
                img_w, img_h = self._pixmap_orig.width(), self._pixmap_orig.height()
                self.scale_x = scaled.width() / img_w if img_w else 1.0
                self.scale_y = scaled.height() / img_h if img_h else 1.0

                # Set exact widget size before drawing
                self.setFixedSize(scaled.width(), scaled.height())

                # Redraw with boxes
                self.update_display()

                # Calculate new scroll position to keep cursor at same image point
                new_cursor_x = cursor_x_on_widget * (new_zoom / old_zoom)
                new_cursor_y = cursor_y_on_widget * (new_zoom / old_zoom)

                # Set scroll position immediately
                h_scroll.setValue(int(new_cursor_x - cursor_pos.x()))
                v_scroll.setValue(int(new_cursor_y - cursor_pos.y()))

                # Emit signal
                self.zoom_changed.emit(self.zoom_level)

            event.accept()
        else:
            super().wheelEvent(event)
    
    def _handle_box_selection(self, pos: QPointF):
        """Kutu seçim işlemi"""
        x = pos.x() / self.scale_x
        y = pos.y() / self.scale_y
        
        # Check detector boxes first
        for idx, box in enumerate(self.detector_boxes):
            if self._point_in_box(x, y, box):
                if idx in self.selected_detector_boxes:
                    self.selected_detector_boxes.remove(idx)
                else:
                    self.selected_detector_boxes.add(idx)
                self.box_selected.emit(idx, True)
                self.update_display()
                return
        
        # Check normal boxes
        for idx, box in enumerate(self.boxes):
            if self._point_in_box(x, y, box):
                if idx in self.selected_boxes:
                    self.selected_boxes.remove(idx)
                else:
                    self.selected_boxes.add(idx)
                self.box_selected.emit(idx, False)
                self.update_display()
                return
    
    def _point_in_box(self, x: float, y: float, box: BoundingBox) -> bool:
        """Noktanın kutunun içinde olup olmadığını kontrol et"""
        return (box.x <= x <= box.x + box.width and 
                box.y <= y <= box.y + box.height)
    def get_selected_indices(self):
        """Normal kutular için seçili indeks listesi."""
        return list(getattr(self, "selected_boxes", []))

    def get_selected_detector_indices(self):
        """Detector (yeşil) kutular için seçili indeks listesi."""
        return list(getattr(self, "selected_detector_boxes", []))