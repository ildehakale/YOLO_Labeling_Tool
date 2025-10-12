# -*- coding: utf-8 -*-
"""
ui/components/base.py

Temel ve yeniden-kullanılabilir UI bileşenleri.
- ModernButton
- ModernGroupBox
- StatsWidget
- ClickableLabel
- StyleSheetFactory (yardımcı)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPen, QPainter, QPixmap, QPalette
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFrame,
    QGroupBox,
    QListWidget,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
)


__all__ = [
    "StyleSheetFactory",
    "ColorScheme",
    "ModernButton",
    "ModernGroupBox",
    "StatsWidget",
    "ClickableLabel",
]


# ---------------------------
# Stil / Tema Yardımcıları
# ---------------------------

@dataclass(frozen=True)
class ColorScheme:
    """Uygulama genelinde kullanılan ana renkler."""
    primary_start: str = "#667eea"
    primary_end: str = "#764ba2"
    primary_hover_start: str = "#5a67d8"
    primary_hover_end: str = "#6b46c1"
    primary_pressed_start: str = "#4c51bf"
    primary_pressed_end: str = "#553c9a"

    secondary_bg: str = "#f7fafc"
    secondary_fg: str = "#4a5568"
    secondary_bd: str = "#e2e8f0"
    secondary_hover_bg: str = "#edf2f7"
    secondary_hover_bd: str = "#cbd5e0"
    secondary_pressed_bg: str = "#e2e8f0"

    danger_start: str = "#ff6b6b"
    danger_end: str = "#ee5a52"
    danger_hover_start: str = "#ff5252"
    danger_hover_end: str = "#e53e3e"

    slate_text: str = "#4a5568"
    slate_muted: str = "#718096"

    success: str = "#10b981"
    warning: str = "#ffc107"
    error: str = "#ef4444"

    panel_bd: str = "#e2e8f0"
    panel_bg: str = "#fafafa"


class StyleSheetFactory:
    """Bileşenler için tekrar-eden stil dizelerini üretir (SRP)."""
    @staticmethod
    def gradient_button_css(
        start: str, end: str, hover_start: str, hover_end: str, pressed_start: str, pressed_end: str
    ) -> str:
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {start}, stop:1 {end});
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-height: 35px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {hover_start}, stop:1 {hover_end});
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {pressed_start}, stop:1 {pressed_end});
            }}
            QPushButton:disabled {{
                opacity: 0.6;
            }}
        """

    @staticmethod
    def outline_button_css(bg: str, fg: str, bd: str, hover_bg: str, hover_bd: str, pressed_bg: str) -> str:
        return f"""
            QPushButton {{
                background: {bg};
                color: {fg};
                border: 1px solid {bd};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-height: 35px;
            }}
            QPushButton:hover {{
                background: {hover_bg};
                border-color: {hover_bd};
            }}
            QPushButton:pressed {{
                background: {pressed_bg};
            }}
            QPushButton:disabled {{
                opacity: 0.6;
            }}
        """

    @staticmethod
    def group_box_css(title_color: str, bd: str, bg: str) -> str:
        return f"""
            QGroupBox {{
                font-size: 12px;
                font-weight: 600;
                color: {title_color};
                border: 2px solid {bd};
                border-radius: 8px;
                margin: 10px 0px;
                padding-top: 15px;
                background: {bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background: {bg};
            }}
        """

    @staticmethod
    def list_widget_css() -> str:
        return """
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """

    @staticmethod
    def spin_css() -> str:
        return """
            QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 2px solid #e2e8f0;
                border-radius: 4px;
                background: white;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #667eea;
            }
        """

    @staticmethod
    def stats_container_css(start: str, end: str) -> str:
        return f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {start}, stop:1 {end});
                border-radius: 8px;
                color: white;
            }}
        """


# ---------------------------
# ModernButton
# ---------------------------

class ModernButton(QPushButton):
    """
    Renk varyantları ve boyut seçenekleriyle modern buton.
    Open/Closed: Varyant eklemek için yalnızca init parametreleri yeterli.
    """
    def __init__(
        self,
        text: str,
        variant: str = "primary",
        *,
        scheme: Optional[ColorScheme] = None,
        minimum_height: int = 35,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self._scheme = scheme or ColorScheme()
        self.setFont(QFont("Segoe UI", 9, QFont.Medium))
        self.setMinimumHeight(minimum_height)
        self.apply_variant(variant)

    def apply_variant(self, variant: str) -> None:
        s = self._scheme
        if variant == "primary":
            self.setStyleSheet(
                StyleSheetFactory.gradient_button_css(
                    s.primary_start, s.primary_end,
                    s.primary_hover_start, s.primary_hover_end,
                    s.primary_pressed_start, s.primary_pressed_end,
                )
            )
        elif variant == "secondary":
            self.setStyleSheet(
                StyleSheetFactory.outline_button_css(
                    s.secondary_bg, s.secondary_fg, s.secondary_bd,
                    s.secondary_hover_bg, s.secondary_hover_bd, s.secondary_pressed_bg,
                )
            )
        elif variant == "danger":
            self.setStyleSheet(
                StyleSheetFactory.gradient_button_css(
                    s.danger_start, s.danger_end,
                    s.danger_hover_start, s.danger_hover_end,
                    s.danger_hover_start, s.danger_hover_end,
                )
            )
        elif variant == "ghost":
            # Metin ön planda, arka plan şeffaf
            self.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #4a5568;
                    border: 1px dashed #e2e8f0;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 500;
                    min-height: 35px;
                }
                QPushButton:hover {
                    background: #f7fafc;
                }
                QPushButton:disabled {
                    opacity: 0.6;
                }
            """)
        else:
            # Varsayılan primary
            self.apply_variant("primary")


# ---------------------------
# ModernGroupBox
# ---------------------------

class ModernGroupBox(QGroupBox):
    """Başlık ve panel stili tutarlı olan grup kutusu."""
    def __init__(self, title: str, *, scheme: Optional[ColorScheme] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        s = scheme or ColorScheme()
        self.setStyleSheet(StyleSheetFactory.group_box_css("#4a5568", s.panel_bd, s.panel_bg))


# ---------------------------
# StatsWidget
# ---------------------------

class StatsWidget(QWidget):
    """
    Üst bilgi istatistik paneli.
    SRP: Sadece istatistik gösterir ve günceller.
    """
    DEFAULT_ITEMS: Tuple[Tuple[str, str], ...] = (
        ("Total IMG", "0"),
        ("Current IMG", "0"),
        ("Chosen", "0"),
        ("Current Label", "0"),
        ("SAHI Added", "0"),
        ("Total Label", "0"),
    )

    def __init__(self, *, scheme: Optional[ColorScheme] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scheme = scheme or ColorScheme()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        self.setStyleSheet(StyleSheetFactory.stats_container_css(self._scheme.primary_start, self._scheme.primary_end))

        self._labels: Dict[str, QLabel] = {}

        for i, (title, value) in enumerate(self.DEFAULT_ITEMS):
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
            self._labels[self._key_from_title(title)] = value_label

            # Dikey ayraç
            if i < len(self.DEFAULT_ITEMS) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet("color: rgba(255, 255, 255, 0.3);")
                layout.addWidget(sep)

    @staticmethod
    def _key_from_title(title: str) -> str:
        return title.lower().replace(" ", "_")

    # Kolay erişim için iki API: isimli argümanlar veya dict
    def update_stats(
        self,
        total_images: int = 0,
        current_image: int = 0,
        selected: int = 0,
        existing_labels: int = 0,
        detector_added: int = 0,
        total_labels: int = 0,
    ) -> None:
        self._set_text("total_img", total_images)
        self._set_text("current_img", current_image)
        self._set_text("chosen", selected)
        self._set_text("current_label", existing_labels)
        self._set_text("sahi_added", detector_added)
        self._set_text("total_label", total_labels)

    def update_stats_from_dict(self, values: Dict[str, int]) -> None:
        for key, val in values.items():
            if key in self._labels:
                self._labels[key].setText(str(val))

    def _set_text(self, key: str, value: int) -> None:
        if key in self._labels:
            self._labels[key].setText(str(value))
    _ALIASES = {
        "toplam_goruntu": "total_img",
        "toplam_görüntü": "total_img",  # Turkish character variant
        "mevcut_goruntu": "current_img",
        "mevcut_görüntü": "current_img",  # Turkish character variant
        "secili": "chosen",
        "seçili": "chosen",  # Turkish character variant
        "mevcut_etiket": "current_label",
        "sahi_eklenen": "sahi_added",
        "toplam_etiket": "total_label",
    }

    def update_stats_from_dict(self, values: dict) -> None:
        fixed = {}
        for k, v in values.items():
            k2 = self._ALIASES.get(k, k)
            fixed[k2] = v
        for key, val in fixed.items():
            if key in self._labels:
                self._labels[key].setText(str(val))

# ---------------------------
# ClickableLabel
# ---------------------------

class ClickableLabel(QLabel):
    """
    Tıklanabilir ve zoom/draw/pan etkileşimini sinyal olarak yayınlayan etiket.

    Sinyaller:
        leftClicked(QPoint)
        rightClicked(QPoint)
        middleClicked(QPoint)
        moved(QPoint)
        released(QPoint)
        wheelZoom(int, QPoint)  # delta, pos
        drawPreview(QRectF)     # çizim sırasında önizleme dikdörtgeni
    """
    leftClicked = pyqtSignal(QPoint)
    rightClicked = pyqtSignal(QPoint)
    middleClicked = pyqtSignal(QPoint)
    moved = pyqtSignal(QPoint)
    released = pyqtSignal(QPoint)
    wheelZoom = pyqtSignal(int, QPoint)
    drawPreview = pyqtSignal(QRectF)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

        # Çizim önizleme durumu
        self._is_drawing: bool = False
        self._start_pos: Optional[QPoint] = None
        self._end_pos: Optional[QPoint] = None

        # Pan bilgisi (opsiyonel)
        self._is_panning: bool = False
        self._pan_start: Optional[QPoint] = None

        # Geriye dönük uyumluluk için controller metotları
        self._parent_has_handlers = {
            "on_mouse_press": hasattr(parent, "on_mouse_press"),
            "on_mouse_move": hasattr(parent, "on_mouse_move"),
            "on_mouse_release": hasattr(parent, "on_mouse_release"),
            "on_wheel_event": hasattr(parent, "on_wheel_event"),
        }

    # ---- Mouse Events ----

    def mousePressEvent(self, event):
        btn = event.button()

        # Çizim başlangıcı (sol tık)
        if btn == Qt.LeftButton:
            self._is_drawing = True
            self._start_pos = event.pos()
            self._end_pos = event.pos()
            self.leftClicked.emit(event.pos())

        elif btn == Qt.RightButton:
            self.rightClicked.emit(event.pos())

        elif btn == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start = event.pos()
            self.middleClicked.emit(event.pos())

        # Geriye dönük çağrı (opsiyonel)
        if self._parent_has_handlers["on_mouse_press"]:
            # type: ignore[attr-defined]
            self.parent().on_mouse_press(event)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_drawing and self._start_pos is not None:
            self._end_pos = event.pos()
            rx = min(self._start_pos.x(), self._end_pos.x())
            ry = min(self._start_pos.y(), self._end_pos.y())
            rw = abs(self._end_pos.x() - self._start_pos.x())
            rh = abs(self._end_pos.y() - self._start_pos.y())
            self.drawPreview.emit(QRectF(rx, ry, rw, rh))

        if self._is_panning and self._pan_start is not None:
            # Pan mantığını controller üstlensin (sinyal)
            pass

        self.moved.emit(event.pos())

        if self._parent_has_handlers["on_mouse_move"]:
            # type: ignore[attr-defined]
            self.parent().on_mouse_move(event)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.released.emit(event.pos())

        if event.button() == Qt.LeftButton:
            self._is_drawing = False
            self._start_pos = None
            self._end_pos = None
            # drawPreview sinyalini "boş" vermiyoruz; controller son çizimi kendisi temizleyebilir.

        elif event.button() == Qt.MiddleButton:
            self._is_panning = False
            self._pan_start = None

        if self._parent_has_handlers["on_mouse_release"]:
            # type: ignore[attr-defined]
            self.parent().on_mouse_release(event)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Ctrl + tekerlek: zoom
        if event.modifiers() == Qt.ControlModifier:
            self.wheelZoom.emit(event.angleDelta().y(), event.pos())
            event.accept()
        else:
            if self._parent_has_handlers["on_wheel_event"]:
                # type: ignore[attr-defined]
                self.parent().on_wheel_event(event)
            else:
                super().wheelEvent(event)


# ---------------------------
# Küçük yardımcılar (opsiyonel)
# ---------------------------

def apply_default_list_style(widget: QListWidget) -> None:
    widget.setStyleSheet(StyleSheetFactory.list_widget_css())


def apply_default_spin_style(spin: QSpinBox | QDoubleSpinBox) -> None:
    spin.setStyleSheet(StyleSheetFactory.spin_css())
    # Varsayılanları okunur hâle getir
    spin.setAlignment(Qt.AlignRight)
