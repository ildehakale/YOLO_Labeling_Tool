# views/main_window_slots.py
# Bu dosya main_window.py'nin devamı - slot metodları

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QDialog, QListWidgetItem, QApplication
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
from typing import Set
from pathlib import Path

class MainWindowSlots:
    """MainWindow için slot metodları - Mixin class"""
    
    @pyqtSlot(object)
    def _on_image_changed(self, image):
        """Görüntü değiştiğinde"""
        if not image:
            self.file_label.setText("No File Selected")
            self.image_viewer.clear()
            self.clear_btn.setEnabled(False)
            return

        # Dosya adını göster
        self.file_label.setText(image.filename)

        # Görüntüyü yükle
        pixmap = QPixmap(image.path)
        self.image_viewer.set_image(pixmap)

        # Kutuları ayarla
        self.image_viewer.set_boxes(image.boxes, image.detector_boxes)

        # Sınıf filtresini güncelle
        self._update_class_list(image)

        # Progress güncelle
        total = self.controller.image_repo.get_total_count()
        current = self.controller.current_index + 1
        progress = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)

        # Görüntü yüklendikten sonra:
        self.image_viewer.clear_all_selections()

        # Bu satırların bir kez bağlandığından emin ol; tekrar tekrar bağlama:
        self.image_viewer.box_selected.connect(self._on_box_selected)

        # seçili sayısını sıfırla
        self.stats_widget.update_stats_from_dict({'seçili': 0})

        self.delete_btn.clicked.connect(self._on_delete_pressed)
        self._actDelete.triggered.connect(self._on_delete_pressed)

        # Enable clear button when image is loaded
        self.clear_btn.setEnabled(True)
    @pyqtSlot(dict)
    def _on_stats_updated(self, stats):
        """İstatistikler güncellendiğinde"""
        self.stats_widget.update_stats_from_dict(stats)
    
    @pyqtSlot(str)
    def _on_error(self, message):
        """Hata oluştuğunda"""
        QMessageBox.critical(self, "Hata", message)
    
    @pyqtSlot(int, int)
    def _on_batch_progress(self, current, total):
        """Toplu işlem ilerlemesi"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.detector_status_label.setText(f"Processing: {current}/{total}")
    
    @pyqtSlot(float, float, float, float)
    def _on_box_drawn(self, x, y, w, h):
        """Kutu çizildiğinde"""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Class ID")
        dialog.setLabelText("Class ID:")
        dialog.setTextValue("0")
        
        if dialog.exec_() == QDialog.Accepted:
            class_id = dialog.textValue()
            if class_id.isdigit():
                self.controller.add_manual_box(int(class_id), x, y, w, h)


    @pyqtSlot(int, bool)
    def _on_box_selected(self, idx: int, is_detector: bool):
        # toplam seçili (normal + detector)
        selected_count = len(self.image_viewer.get_selected_indices()) + \
                        len(self.image_viewer.get_selected_detector_indices())
        self.stats_widget.update_stats_from_dict({'seçili': selected_count})
    
    def _select_image_folder(self):
        """Görüntü klasörü seç"""
        folder = QFileDialog.getExistingDirectory(
            self, "Görüntü Klasörü Seçin", str(self.controller.image_folder)
        )
        if folder:
            self.controller.set_image_folder(folder)
    
    def _select_labels_folder(self):
        """Etiket klasörü seç"""
        folder = QFileDialog.getExistingDirectory(
            self, "Etiket Klasörü Seçin", str(self.controller.labels_folder)
        )
        if folder:
            self.controller.set_labels_folder(folder)
    
    def _toggle_fullscreen(self):
        """Tam ekran geçiş"""
        if self.is_fullscreen:
            self.showNormal()
            self.is_fullscreen = False
        else:
            self.showFullScreen()
            self.is_fullscreen = True
    
    def _on_model_selected(self):
        """Model seçildiğinde"""
        current_item = self.model_list.currentItem()
        if not current_item:
            return
        
        model_name = current_item.text()
        success = self.controller.load_model(model_name)
        
        if success:
            self.detector_btn.setEnabled(True)
            self.batch_detector_btn.setEnabled(True)
            self.save_detector_btn.setEnabled(True)
            self.detector_status_label.setText(f"Model uploaded: {model_name}")
        else:
            self.detector_btn.setEnabled(False)
            self.batch_detector_btn.setEnabled(False)
            self.save_detector_btn.setEnabled(False)
    
    def _run_detection(self):
        """Tespit çalıştır"""
        self.detector_btn.setEnabled(False)
        device_name = self.controller.device.upper() if self.controller.device == 'cpu' else 'CUDA'
        self.detector_status_label.setText(f"SAHI applies, device: {device_name}")
        QApplication.processEvents()  # Force UI update before detection starts

        try:
            self.controller.run_detection()
            self.detector_status_label.setText("Detection completed")
        finally:
            self.detector_btn.setEnabled(True)
    
    def _run_batch_detection(self):
        """Toplu tespit çalıştır"""
        reply = QMessageBox.question(
            self, 'Batch Processing',
            'The next 50 images will be used for detection. Do you want to continue?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.batch_detector_btn.setEnabled(False)
            self.detector_btn.setEnabled(False)
            
            try:
                self.controller.run_batch_detection(50)
                QMessageBox.information(self, "Başarılı", "Toplu tespit tamamlandı")
            finally:
                self.batch_detector_btn.setEnabled(True)
                self.detector_btn.setEnabled(True)
    
    def _save_detections(self):
        """Tespitleri kaydet"""
        count = self.controller.save_temp_detections()
        if count > 0:
            QMessageBox.information(self, "Başarılı", f"{count} tespit kaydedildi")
        else:
            QMessageBox.information(self, "Bilgi", "Kaydedilecek tespit yok")
    

    @pyqtSlot()
    def _on_delete_pressed(self):
        # 1) O an seçili olan indeksleri al
        det_idx = self.image_viewer.get_selected_detector_indices()
        nor_idx = self.image_viewer.get_selected_indices()

        # 2) Hiç seçim yoksa çık
        if not det_idx and not nor_idx:
            return

        # 3) Seçili kutuları sil (artık çoklu seçimi destekliyor)
        if det_idx:
            # Convert list to set and delete all selected detector boxes
            self.controller.delete_selected_boxes(set(det_idx), is_detector=True)
        if nor_idx:
            # Convert list to set and delete all selected normal boxes
            self.controller.delete_selected_boxes(set(nor_idx), is_detector=False)

        # 4) Silme sonrası kesinlikle seçim kalmasın
        self.image_viewer.clear_all_selections()
        self.stats_widget.update_stats_from_dict({'seçili': 0})

    def _clear_all(self):
        """Tüm kutuları temizle"""
        reply = QMessageBox.question(
            self, 'Onay',
            'Tüm etiketler silinecek. Devam?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.controller.clear_all_boxes()
    
    def _update_settings(self):
        """Ayarları güncelle"""
        self.controller.update_settings(
            self.confidence_input.value(),
            self.slice_height_input.value(),
            self.slice_width_input.value(),
            use_sahi=True
        )
    
    def _update_class_list(self, image):
        """Sınıf listesini güncelle"""
        from config.settings import ALLOWED_CLASSES

        # Mevcut seçimleri sakla
        current_checked = {}
        for i in range(self.class_filter_list.count()):
            item = self.class_filter_list.item(i)
            text = item.text()
            try:
                cid = int(text.split('(')[-1].strip(')'))
                current_checked[cid] = (item.checkState() == Qt.Checked)
            except (ValueError, IndexError):
                pass

        # Listeyi temizle
        self.class_filter_list.clear()

        # Benzersiz sınıfları topla
        unique_classes = set()
        for box in image.boxes:
            unique_classes.add(box.class_id)
        for box in image.detector_boxes:
            unique_classes.add(box.class_id)

        # Only show ALLOWED_CLASSES in the filter list
        unique_classes = unique_classes.intersection(ALLOWED_CLASSES)

        # Sıralı liste oluştur
        for class_id in sorted(unique_classes):
            class_name = self.image_viewer.class_names.get(class_id, f"Class_{class_id}")
            item = QListWidgetItem(f"{class_name} ({class_id})")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            self.class_filter_list.addItem(item)
            if class_id in current_checked:
                item.setCheckState(Qt.Checked if current_checked[class_id] else Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
    
    def _update_class_filter(self):
        """Sınıf filtresini uygula"""
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
        
        self.image_viewer.set_visible_classes(visible_classes)

    def _apply_styles(self):
        """Genel stilleri uygula"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f7fafc, stop:1 #edf2f7);
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 2px solid #e2e8f0;
                border-radius: 4px;
                background: white;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #667eea;
            }
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