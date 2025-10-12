# application.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

# Import all necessary components
from repositories.file_repositories import FileImageRepository, YoloLabelRepository, ModelFileRepository
from services.detection.detection_service import DetectorFactory
from controllers.labeling_controller import LabelingController
from views.main_window import MainWindow
from views.main_window_slots import MainWindowSlots

class ModernLabelingApp(QApplication):
    """Modern etiketleme uygulaması"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.setStyle('Fusion')
        self._setup_palette()
        self._initialize_components()
        
    def _setup_palette(self):
        """Uygulama paleti ayarla"""
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
    
    def _initialize_components(self):
        """Bileşenleri başlat - Dependency Injection"""
        
        # Create repositories
        image_repo = FileImageRepository()
        label_repo = YoloLabelRepository()
        model_repo = ModelFileRepository()
        
        # Create factory
        detector_factory = DetectorFactory()
        
        # Create controller
        controller = LabelingController(
            image_repo=image_repo,
            label_repo=label_repo,
            model_repo=model_repo,
            detector_factory=detector_factory
        )
        
        # Create combined main window with slots
        class MainWindowWithSlots(MainWindow, MainWindowSlots):
            pass
        
        # Create and show main window
        self.main_window = MainWindowWithSlots(controller)
        self.main_window.show()
    
    def run(self):
        """Uygulamayı çalıştır"""
        return self.exec_()