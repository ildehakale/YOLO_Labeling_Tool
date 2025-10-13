import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from application import ModernLabelingApp

def setup_environment():
    """Ortam değişkenlerini ayarla"""
    # Suppress verbose outputs
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # Linux platform settings (uncomment if needed)
    #os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
    #os.environ["QT_QPA_PLATFORM"] = "xcb"

def main():

    # Setup environment
    setup_environment()
            
    # Run application
    app = ModernLabelingApp(sys.argv)
    app.setApplicationName("YOLO Class Labeling Tool")
    
    sys.exit(app.run())

if __name__ == '__main__':
    main()