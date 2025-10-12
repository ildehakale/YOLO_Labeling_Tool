# views/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFileDialog, QMessageBox, QInputDialog, QDialog,
    QProgressBar, QScrollArea, QFrame, QLabel, QListWidget,
    QListWidgetItem, QSpinBox, QDoubleSpinBox,QAction,
)
from PyQt5.QtCore import Qt, pyqtSlot, QByteArray
from PyQt5.QtGui import QFont, QPixmap, QKeySequence
from pathlib import Path
from typing import Optional
import sys
import base64
sys.path.append('..')
from ui.components.base import ModernButton, ModernGroupBox, StatsWidget
from ui.components.image_viewer import ImageViewer
from models.base import Image, BoundingBox
from controllers.labeling_controller import LabelingController
from config.embedded_assets import LOGO_BASE64

class MainWindow(QMainWindow):
    """Ana uygulama penceresi"""
    
    def __init__(self, controller: LabelingController):
        super().__init__()
        self.controller = controller
        self.is_fullscreen = False
        
        # UI bileşenleri
        self.image_viewer: Optional[ImageViewer] = None
        self.scroll_area: Optional[QScrollArea] = None
        self.stats_widget: Optional[StatsWidget] = None
        self.file_label: Optional[QLabel] = None
        self.progress_bar: Optional[QProgressBar] = None
        
        # Model listesi
        self.model_list: Optional[QListWidget] = None
        self.detector_status_label: Optional[QLabel] = None
        
        # Settings
        self.confidence_input: Optional[QDoubleSpinBox] = None
        self.slice_height_input: Optional[QSpinBox] = None
        self.slice_width_input: Optional[QSpinBox] = None
        
        # Class filter
        self.class_filter_list: Optional[QListWidget] = None

        # Global keyboard shortcuts
        self._actDelete = QAction(self)
        self._actDelete.setShortcut(QKeySequence.Delete)
        self.addAction(self._actDelete)
        self._actDelete.triggered.connect(self._on_delete_pressed)

        # Arrow key navigation - global shortcuts
        self._actPrevImage = QAction(self)
        self._actPrevImage.setShortcut(Qt.Key_Left)
        self.addAction(self._actPrevImage)
        self._actPrevImage.triggered.connect(self.controller.previous_image)

        self._actNextImage = QAction(self)
        self._actNextImage.setShortcut(Qt.Key_Right)
        self.addAction(self._actNextImage)
        self._actNextImage.triggered.connect(self.controller.next_image)

        self._init_ui()
        self._connect_signals()
        self._load_default_models()
    
    def _init_ui(self):
        """UI'yi başlat"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Sidebar
        sidebar = self._create_sidebar()
        sidebar.setMaximumWidth(320)
        sidebar.setMinimumWidth(280)
        
        # Main area
        main_area = self._create_main_area()
        
        splitter.addWidget(sidebar)
        splitter.addWidget(main_area)
        splitter.setSizes([320, 800])
        
        main_layout.addWidget(splitter)
        
        # Window settings
        self.setWindowTitle("YOLO Class Labeling Tool")
        self.setGeometry(100, 100, 1200, 800)
        self._apply_styles()
    
    def _create_sidebar(self) -> QWidget:
        """Sidebar oluştur"""
        # Make sidebar scrollable to prevent overlap in fullscreen
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)

        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(10)  # Reduced from 15 to 10 for tighter layout
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Logo (embedded)
        logo_label = QLabel()
        try:
            logo_bytes = base64.b64decode(LOGO_BASE64)
            logo_pixmap = QPixmap()
            logo_pixmap.loadFromData(QByteArray(logo_bytes))
            logo_label.setPixmap(logo_pixmap.scaledToWidth(150, Qt.SmoothTransformation))
        except Exception as e:
            logo_label.setText("Logo")
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)
        
        # Title
        title = QLabel("🎯 Class Labeling")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #4a5568; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Folder group
        folder_group = ModernGroupBox("📁 File Settings")
        folder_layout = QVBoxLayout(folder_group)
        
        self.select_images_btn = ModernButton("📷 Choose image File")
        self.select_labels_btn = ModernButton("🏷️ Choose label File")
        
        folder_layout.addWidget(self.select_images_btn)
        folder_layout.addWidget(self.select_labels_btn)
        layout.addWidget(folder_group)
        
        # Tools group
        tools_group = ModernGroupBox("🔧 Models")
        tools_layout = QVBoxLayout(tools_group)
        
        # Model list
        model_label = QLabel("Class Models:")
        model_label.setFont(QFont("Segoe UI", 9))
        self.model_list = QListWidget()
        self.model_list.setFixedHeight(115)  # Fixed height to prevent overlap with controls below
        # Prevent arrow keys from getting stuck on this widget
        self.model_list.setFocusPolicy(Qt.ClickFocus)
        
        tools_layout.addWidget(model_label)
        tools_layout.addWidget(self.model_list)
        
        # Confidence
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        conf_label.setFont(QFont("Segoe UI", 9))
        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0.0, 1.0)
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setValue(0.2)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.confidence_input)
        tools_layout.addLayout(conf_layout)
        
        # Slice height
        height_layout = QHBoxLayout()
        height_label = QLabel("Slice Height:")
        height_label.setFont(QFont("Segoe UI", 9))
        self.slice_height_input = QSpinBox()
        self.slice_height_input.setRange(100, 4096)
        self.slice_height_input.setValue(256)
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.slice_height_input)
        tools_layout.addLayout(height_layout)
        
        # Slice width
        width_layout = QHBoxLayout()
        width_label = QLabel("Slice Width:")
        width_label.setFont(QFont("Segoe UI", 9))
        self.slice_width_input = QSpinBox()
        self.slice_width_input.setRange(100, 4096)
        self.slice_width_input.setValue(256)
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.slice_width_input)
        tools_layout.addLayout(width_layout)
        
        # Buttons
        self.detector_btn = ModernButton("✨ APPLY SAHI", "primary")
        self.detector_btn.setEnabled(False)
        
        self.batch_detector_btn = ModernButton("✨ APPLY SAHI (50 IMG)", "primary")
        self.batch_detector_btn.setEnabled(False)
        
        self.save_detector_btn = ModernButton("💾 Save SAHI results", "secondary")
        self.save_detector_btn.setEnabled(False)
        
        tools_layout.addWidget(self.detector_btn)
        tools_layout.addWidget(self.batch_detector_btn)
        tools_layout.addWidget(self.save_detector_btn)
        
        # Status
        self.detector_status_label = QLabel("Model state: Waiting...")
        self.detector_status_label.setFont(QFont("Segoe UI", 8))
        self.detector_status_label.setStyleSheet("color: #718096; font-style: italic;")
        tools_layout.addWidget(self.detector_status_label)
        
        # Delete buttons
        tools_layout.addSpacing(10)
        self.delete_btn = ModernButton("🗑️ Delete Selected", "danger")
        self.clear_btn = ModernButton("🧹 Clean all Boxes", "secondary")
        tools_layout.addWidget(self.delete_btn)
        tools_layout.addWidget(self.clear_btn)
        
        layout.addWidget(tools_group)
        
        # Class filter
        class_group = ModernGroupBox("🎯 Class Filter")
        class_layout = QVBoxLayout(class_group)
        
        class_label = QLabel("Displayed Classes:")
        class_label.setFont(QFont("Segoe UI", 9))

        self.class_filter_list = QListWidget()
        # Prevent arrow keys from getting stuck on this widget
        self.class_filter_list.setFocusPolicy(Qt.ClickFocus)

        class_layout.addWidget(class_label)
        class_layout.addWidget(self.class_filter_list)
        layout.addWidget(class_group)

        # Set sidebar as scroll area content and return scroll area
        scroll_area.setWidget(sidebar)
        return scroll_area
    
    def _create_main_area(self) -> QWidget:
        """Ana alan oluştur"""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        
        self.file_label = QLabel("File not selected")
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
        
        header_layout.addWidget(self.file_label)
        header_layout.addStretch()
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.fullscreen_btn = ModernButton("Fullscreen")
        self.prev_btn = ModernButton("Prev")
        self.next_btn = ModernButton("Next")
        
        nav_layout.addWidget(self.fullscreen_btn)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        
        header_layout.addLayout(nav_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(8)
        
        # Image container
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
        
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.image_viewer = ImageViewer()
        self.image_viewer.set_scroll_area(self.scroll_area)  # Pass scroll area reference
        self.scroll_area.setWidget(self.image_viewer)
        image_layout.addWidget(self.scroll_area)
        
        # Stats widget
        self.stats_widget = StatsWidget()
        
        layout.addLayout(header_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(image_container, 1)
        layout.addWidget(self.stats_widget)
        
        return main_widget
    
    def _connect_signals(self):
        """Connect Signals"""
        # Controller signals
        self.controller.image_changed.connect(self._on_image_changed)
        self.controller.stats_updated.connect(self._on_stats_updated)
        self.controller.error_occurred.connect(self._on_error)
        self.controller.batch_progress.connect(self._on_batch_progress)
        
        # Button clicks
        self.select_images_btn.clicked.connect(self._select_image_folder)
        self.select_labels_btn.clicked.connect(self._select_labels_folder)
        self.prev_btn.clicked.connect(self.controller.previous_image)
        self.next_btn.clicked.connect(self.controller.next_image)
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        
        self.detector_btn.clicked.connect(self._run_detection)
        self.batch_detector_btn.clicked.connect(self._run_batch_detection)
        self.save_detector_btn.clicked.connect(self._save_detections)
        
        self.delete_btn.clicked.connect(self._on_delete_pressed)
        self.clear_btn.clicked.connect(self._clear_all)
        
        # Image viewer signals
        self.image_viewer.box_drawn.connect(self._on_box_drawn)
        
        # Model selection
        self.model_list.itemSelectionChanged.connect(self._on_model_selected)
        
        # Class filter
        self.class_filter_list.itemChanged.connect(self._update_class_filter)
        
        # Settings
        self.confidence_input.valueChanged.connect(self._update_settings)
        self.slice_height_input.valueChanged.connect(self._update_settings)
        self.slice_width_input.valueChanged.connect(self._update_settings)
    
    def _load_default_models(self):
        """Varsayılan modelleri yükle"""
        model_folder = Path("models")
        if not model_folder.exists():
            self.detector_status_label.setText("'models' file not found")
            return
        
        models = sorted([
            f.name for f in model_folder.iterdir()
            if f.suffix.lower() in ('.pt', '.onnx')
        ])
        
        for model in models:
            self.model_list.addItem(model)
        
        if models:
            self.model_list.setCurrentRow(0)
    
