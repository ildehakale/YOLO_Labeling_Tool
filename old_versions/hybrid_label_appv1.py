import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QInputDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLineEdit, QProgressBar, QFrame, QSplitter, QScrollArea,
    QGroupBox, QSpinBox, QMessageBox, QDialog
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QRectF, QPointF, QSize

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
                    background: qlineagradient(x1:0, y1:0, x2:0, y2:1,
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
            ("Mevcut", "0"),
            ("Etiketler", "0"),
            ("Seçili", "0")
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
    
    def update_stats(self, total_images=0, current=0, labels=0, selected=0):
        self.stat_labels["toplam_görüntü"].setText(str(total_images))
        self.stat_labels["mevcut"].setText(str(current))
        self.stat_labels["etiketler"].setText(str(labels))
        self.stat_labels["seçili"].setText(str(selected))

class ClassDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sınıf ID Girin")
        self.setFixedSize(300, 150)
        self.setStyleSheet("""
            QDialog {
                background: white;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        label = QLabel("Sınıf ID:")
        label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Örn: 0, 1, 2...")
        self.class_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #667eea;
            }
        """)
        
        button_layout = QHBoxLayout()
        self.ok_button = ModernButton("✓ Onayla", "primary")
        self.cancel_button = ModernButton("✗ İptal", "secondary")
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        
        layout.addWidget(label)
        layout.addWidget(self.class_input)
        layout.addLayout(button_layout)
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.class_input.returnPressed.connect(self.accept)
        
        self.class_input.setFocus()
    
    def get_class_id(self):
        return self.class_input.text()

class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent

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
        self.is_drawing = False
        self.start_pos = QPointF()
        self.end_pos = QPointF()
        
        self.scaled_pix = QPixmap()
        self.original_image_size = QSize()
        self.sf_w = 1.0
        self.sf_h = 1.0
        self.x_off = 0.0
        self.y_off = 0.0
        
        self.init_ui()
        self.apply_styles()
        self.update_image()

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
        
        self.setWindowTitle("🎯 Modern YOLO Etiketleme Aracı")
        self.setGeometry(100, 100, 1200, 800)
        
    def create_sidebar(self):
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(15)
        
        title = QLabel("🎯 YOLO Etiketleyici")
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
        
        class_group = ModernGroupBox("🎯 Sınıf Ayarları")
        class_layout = QVBoxLayout(class_group)
        
        class_input_layout = QHBoxLayout()
        class_label = QLabel("Varsayılan Sınıf:")
        class_label.setFont(QFont("Segoe UI", 9))
        self.class_input = QSpinBox()
        self.class_input.setMinimum(0)
        self.class_input.setMaximum(99)
        self.class_input.setStyleSheet("""
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
        
        class_input_layout.addWidget(class_label)
        class_input_layout.addWidget(self.class_input)
        class_layout.addLayout(class_input_layout)
        
        labels_group = ModernGroupBox("📋 Mevcut Etiketler")
        labels_layout = QVBoxLayout(labels_group)
        
        self.labels_list = QListWidget()
        self.labels_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
                alternate-background-color: #f7fafc;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f1f5f9;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
            }
        """)
        self.labels_list.setMaximumHeight(150)
        labels_layout.addWidget(self.labels_list)
        
        tools_group = ModernGroupBox("🔧 Araçlar")
        tools_layout = QVBoxLayout(tools_group)
        
        self.delete_btn = ModernButton("🗑️ Seçili Sil", "danger")
        self.clear_btn = ModernButton("🧹 Tümünü Temizle", "secondary")
        
        tools_layout.addWidget(self.delete_btn)
        tools_layout.addWidget(self.clear_btn)
        
        help_group = ModernGroupBox("💡 Kısayollar")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QLabel("""
        • Sol tık: Kutu çiz
        • Sağ tık: Kutu seç/kaldır
        • ← →: Görüntü değiştir
        • Del: Seçili kutuları sil
        • O: Görüntü klasörü seç
        • L: Etiket klasörü seç
        """)
        help_text.setFont(QFont("Segoe UI", 8))
        help_text.setStyleSheet("color: #718096; line-height: 1.4;")
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        
        layout.addWidget(folder_group)
        layout.addWidget(class_group)
        layout.addWidget(labels_group)
        layout.addWidget(tools_group)
        layout.addWidget(help_group)
        layout.addStretch()
        
        self.select_images_btn.clicked.connect(self.select_image_folder)
        self.select_labels_btn.clicked.connect(self.select_labels_folder)
        self.delete_btn.clicked.connect(self.delete_selected_boxes)
        self.clear_btn.clicked.connect(self.clear_all_boxes)
        
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
        # Bu satır, etiketin kopyalanabilir olmasını sağlar
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        header_layout.addWidget(self.file_label)
        header_layout.addStretch()
        
        nav_layout = QHBoxLayout()
        self.prev_btn = ModernButton("⬅️ Önceki")
        self.next_btn = ModernButton("Sonraki ➡️")
        
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
        
        return main_widget
    
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
            self.update_labels_list()
            return
            
        path = self.image_paths[self.index]
        img_path = os.path.join(self.image_folder, path)
        base = os.path.splitext(path)[0]
        self.label_path = os.path.join(self.labels_folder, base + '.txt')
        
        self.labels.clear()
        self.boxes.clear()
        
        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.original_image_size = pix.size()
            iw, ih = pix.width(), pix.height()

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

        self.update_labels_list()

    def update_labels_list(self):
        self.labels_list.clear()
        for i, (cid, _, _, _, _) in enumerate(self.boxes):
            item = QListWidgetItem(f"Sınıf {cid}")
            if i in self.selected:
                item.setSelected(True)
            self.labels_list.addItem(item)
    
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
        
        self.file_label.setText(path)
        self.load_labels()

        if not os.path.exists(img_path):
            return

        orig = QPixmap(img_path)
        max_size = self.image_label.size()
        
        if max_size.width() < 100 or max_size.height() < 100:
             max_size = QSize(800, 600)
            
        scaled = orig.scaled(
            max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.scaled_pix = scaled
        
        sw, sh = scaled.width(), scaled.height()
        iw, ih = orig.width(), orig.height()
        self.sf_w = sw / iw if iw > 0 else 1.0
        self.sf_h = sh / ih if ih > 0 else 1.0
        
        lw, lh = self.image_label.size().width(), self.image_label.size().height()
        self.x_off = (lw - sw) / 2
        self.y_off = (lh - sh) / 2

        canvas = scaled.copy()
        painter = QPainter(canvas)
        
        for idx, (cid, x, y, w, h) in enumerate(self.boxes):
            if idx in self.selected:
                pen = QPen(QColor(66, 153, 225), 3)
            else:
                pen = QPen(QColor(255, 107, 107), 2)
            
            painter.setPen(pen)
            r = QRectF(x*self.sf_w, y*self.sf_h, w*self.sf_w, h*self.sf_h)
            painter.drawRect(r)
            
            label_x = int(x * self.sf_w)
            label_y = int(y * self.sf_h - 20)
            label_width = 40
            label_height = 18
            
            painter.fillRect(
                label_x, label_y, label_width, label_height,
                QColor(66, 153, 225) if idx in self.selected else QColor(255, 107, 107)
            )
            painter.setPen(QPen(Qt.white))
            painter.drawText(
                label_x + 5, label_y + 12, f"C{cid}"
            )
        
        if self.is_drawing:
            preview = QPen(QColor(16, 185, 129), 2, Qt.DashLine)
            painter.setPen(preview)
            x1, y1 = self.start_pos.x(), self.start_pos.y()
            x2, y2 = self.end_pos.x(), self.end_pos.y()
            rx = min(x1, x2); ry = min(y1, y2)
            rw = abs(x2 - x1); rh = abs(y2 - y1)
            painter.drawRect(QRectF(rx, ry, rw, rh))
        
        painter.end()
        self.image_label.setPixmap(canvas)
        
        self.update_progress()
        self.update_stats()
        
    def update_progress(self):
        if self.image_paths:
            progress = int((self.index + 1) / len(self.image_paths) * 100)
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setValue(0)
    
    def update_stats(self):
        total_images = len(self.image_paths)
        current = self.index + 1 if self.image_paths else 0
        labels = len(self.boxes)
        selected = len(self.selected)
        
        self.stats_widget.update_stats(total_images, current, labels, selected)

    def on_mouse_press(self, event):
        if not self.scaled_pix.isNull():
            pos = event.pos()
            x_adj = pos.x() - self.x_off
            y_adj = pos.y() - self.y_off
            
            x_adj = max(0.0, min(float(self.scaled_pix.width()), x_adj))
            y_adj = max(0.0, min(float(self.scaled_pix.height()), y_adj))
            
            if event.button() == Qt.LeftButton:
                self.is_drawing = True
                self.start_pos = QPointF(x_adj, y_adj)
                self.end_pos = QPointF(x_adj, y_adj)
            elif event.button() == Qt.RightButton:
                x_im = x_adj / self.sf_w
                y_im = y_adj / self.sf_h
                for idx, (_, bx, by, bw, bh) in enumerate(self.boxes):
                    if bx <= x_im <= bx+bw and by <= y_im <= by+bh:
                        if idx in self.selected:
                            self.selected.remove(idx)
                        else:
                            self.selected.add(idx)
                        break
                self.update_image()

    def on_mouse_move(self, event):
        if self.is_drawing:
            pos = event.pos()
            x_adj = pos.x() - self.x_off
            y_adj = pos.y() - self.y_off
            
            x_adj = max(0.0, min(float(self.scaled_pix.width()), x_adj))
            y_adj = max(0.0, min(float(self.scaled_pix.height()), y_adj))
            
            self.end_pos = QPointF(x_adj, y_adj)
            self.update_image()

    def on_mouse_release(self, event):
        if self.is_drawing and event.button() == Qt.LeftButton:
            self.is_drawing = False
            
            sw, sh = self.scaled_pix.width(), self.scaled_pix.height()
            
            x1 = max(0, min(sw, self.start_pos.x()))
            y1 = max(0, min(sh, self.start_pos.y()))
            x2 = max(0, min(sw, self.end_pos.x()))
            y2 = max(0, min(sh, self.end_pos.y()))
            
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            
            if rw > 10 and rh > 10:
                ox = rx / self.sf_w
                oy = ry / self.sf_h
                ow = rw / self.sf_w
                oh = rh / self.sf_h
                
                nx = (ox + ow / 2) / self.original_image_size.width()
                ny = (oy + oh / 2) / self.original_image_size.height()
                nw = ow / self.original_image_size.width()
                nh = oh / self.original_image_size.height()
                
                dialog = QInputDialog(self)
                dialog.setWindowTitle("Sınıf ID Girin")
                dialog.setLabelText("Sınıf ID:")
                dialog.setTextValue(str(self.class_input.value()))
                
                if dialog.exec_() == QDialog.Accepted:
                    class_id = dialog.textValue()
                    if class_id.isdigit():
                        line = f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
                        os.makedirs(self.labels_folder, exist_ok=True)
                        with open(self.label_path, 'a') as f:
                            f.write(line + "\n")
                        self.load_labels()
            
            self.selected.clear()
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
            
            if os.path.exists(self.label_path):
                with open(self.label_path, 'w') as f:
                    if new_labels:
                        f.write("\n".join(new_labels) + "\n")
            
            self.selected.clear()
            self.load_labels()
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
            self.update_image()
    
    def next_image(self):
        if self.image_paths and self.index < len(self.image_paths) - 1:
            self.index += 1
            self.selected.clear()
            self.update_image()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key_Delete:
            self.delete_selected_boxes()
        elif event.key() == Qt.Key_O:
            self.select_image_folder()
        elif event.key() == Qt.Key_L:
            self.select_labels_folder()
        elif event.key() == Qt.Key_Escape:
            self.selected.clear()
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
    app.setApplicationName("Modern YOLO Labeling Tool")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("YOLO Tools")
    
    splash_msg = QMessageBox()
    splash_msg.setWindowTitle("Modern YOLO Etiketleyici")
    splash_msg.setText("""
🎯 Modern YOLO Etiketleme Aracı v2.0

✨ Özellikler:
• Modern ve kullanıcı dostu arayüz
• Sürükle-bırak ile kutu çizimi
• Çoklu seçim ve toplu silme
• Gerçek zamanlı istatistikler
• Klavye kısayolları
• Otomatik kaydetme

📝 Kullanım:
• Sol tık: Kutu çiz
• Sağ tık: Kutu seç/kaldır
• ← →: Görüntü değiştir
• Del: Seçili kutuları sil

İyi çalışmalar! 🚀
    """)
    splash_msg.setIcon(QMessageBox.Information)
    splash_msg.exec_()
    
    return app.exec_()

if __name__ == '__main__':
    main()