#!/usr/bin/env python3
# main.py
"""
Modern Sınıf Etiketleme Aracı
SOLID prensipleriyle yeniden yapılandırılmış versiyon
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from application import ModernLabelingApp

def check_dependencies():
    """Bağımlılıkları kontrol et"""
    required = {
        'PyQt5': 'PyQt5',
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    

    
    missing_required = []
    missing_optional = []
    
    # Check required
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(package)
    

    
    if missing_required:
        print("HATA: Gerekli paketler eksik:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nKurulum için: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print("UYARI: Opsiyonel paketler eksik (bazı özellikler çalışmayabilir):")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nTüm özellikleri kullanmak için: pip install " + " ".join(missing_optional))
        print()
    
    return True

def setup_environment():
    """Ortam değişkenlerini ayarla"""
    # Suppress verbose outputs
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # Linux platform settings (uncomment if needed)
    # os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
    # os.environ["QT_QPA_PLATFORM"] = "xcb"

def main():
    """Ana fonksiyon"""
    print("YOLO Class Labeling Tool ")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Create required directories
    Path("models").mkdir(exist_ok=True)

    
    # Run application
    app = ModernLabelingApp(sys.argv)
    app.setApplicationName("YOLO Class Labeling Tool")
    
    sys.exit(app.run())

if __name__ == '__main__':
    main()