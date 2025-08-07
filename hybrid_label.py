import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog,
    QInputDialog, QWidget, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF

class ClickableLabel(QLabel):
    def mousePressEvent(self, event):
        win = self.window()
        if hasattr(win, 'on_mouse_press'):
            win.on_mouse_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        win = self.window()
        if hasattr(win, 'on_mouse_move'):
            win.on_mouse_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        win = self.window()
        if hasattr(win, 'on_mouse_release'):
            win.on_mouse_release(event)
        super().mouseReleaseEvent(event)

class ImageNavigator(QMainWindow):
    def __init__(self, image_folder, labels_folder):
        super().__init__()
        self.image_folder = image_folder
        self.labels_folder = labels_folder
        self.image_paths = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
        ])
        self.index = 0
        self.labels = []
        self.boxes = []  # (cid, x, y, w, h) in pixel coords
        self.selected = set()
        # drawing state
        self.drawing = False
        self.start_pos = QPointF()
        self.end_pos = QPointF()

        # filename label
        self.titleLabel = QLabel()
        self.titleLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.titleLabel.setMaximumHeight(20)
        self.titleLabel.setStyleSheet("font-size:10px; background:transparent;")

        # image display
        self.imageLabel = ClickableLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)

        # layout
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.imageLabel)
        self.setCentralWidget(container)

        self.statusBar()
        self.setWindowTitle("YOLO Labeling Tool")
        self.resize(800,600)
        self.update_image()

    def load_labels(self):
        img = self.image_paths[self.index]
        base = os.path.splitext(os.path.basename(img))[0]
        self.label_path = os.path.join(self.labels_folder, base + '.txt')
        self.labels.clear()
        self.boxes.clear()
        if os.path.exists(self.label_path):
            with open(self.label_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            pix = QPixmap(img)
            iw, ih = pix.width(), pix.height()
            for ln in lines:
                cid, xc, yc, bw, bh = ln.split()
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                x = (xc - bw/2) * iw
                y = (yc - bh/2) * ih
                w = bw * iw
                h = bh * ih
                self.labels.append(ln)
                self.boxes.append((cid, x, y, w, h))

    def update_image(self):
        if not self.image_paths:
            return
        img = self.image_paths[self.index]
        self.titleLabel.setText(os.path.basename(img))
        self.load_labels()

        orig = QPixmap(img)
        scaled = orig.scaled(
            self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.scaled_pix = scaled
        sw, sh = scaled.width(), scaled.height()
        iw, ih = orig.width(), orig.height()
        # compute scale and offsets
        self.sf_w = sw / iw
        self.sf_h = sh / ih
        lw, lh = self.imageLabel.size().width(), self.imageLabel.size().height()
        self.x_off = (lw - sw) // 2
        self.y_off = (lh - sh) // 2

        canvas = scaled.copy()
        painter = QPainter(canvas)
        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)
        # draw existing boxes
        for idx, (cid, x, y, w, h) in enumerate(self.boxes):
            x_c = x * self.sf_w
            y_c = y * self.sf_h
            w_c = w * self.sf_w
            h_c = h * self.sf_h
            painter.drawRect(QRectF(x_c, y_c, w_c, h_c))
            if idx in self.selected:
                sel = QPen(Qt.blue)
                sel.setWidth(3)
                painter.setPen(sel)
                painter.drawRect(QRectF(x_c, y_c, w_c, h_c))
                painter.setPen(pen)
        # draw preview
        if self.drawing:
            preview = QPen(Qt.green, 2, Qt.DashLine)
            painter.setPen(preview)
            # compute pixmap coords from widget positions
            x1 = self.start_pos.x() - self.x_off
            y1 = self.start_pos.y() - self.y_off
            x2 = self.end_pos.x() - self.x_off
            y2 = self.end_pos.y() - self.y_off
            # clamp within pixmap
            x1 = max(0, min(sw, x1)); y1 = max(0, min(sh, y1))
            x2 = max(0, min(sw, x2)); y2 = max(0, min(sh, y2))
            rx = min(x1, x2); ry = min(y1, y2)
            rw = abs(x2 - x1); rh = abs(y2 - y1)
            painter.drawRect(QRectF(rx, ry, rw, rh))
        painter.end()
        self.imageLabel.setPixmap(canvas)
        cls = [l.split()[0] for l in self.labels]
        self.statusBar().showMessage(
            f"{self.index+1}/{len(self.image_paths)} | Classes: {','.join(cls)} | Selected: {len(self.selected)}"
        )

    def widget_to_image_coords(self, pos):
        x_im = (pos.x() - self.x_off) / self.sf_w
        y_im = (pos.y() - self.y_off) / self.sf_h
        return x_im, y_im

    def on_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pos = QPointF(event.pos())
            self.end_pos = QPointF(event.pos())
        elif event.button() == Qt.RightButton:
            # toggle selection
            x_im, y_im = self.widget_to_image_coords(event.pos())
            for idx, (_, bx, by, bw, bh) in enumerate(self.boxes):
                if bx <= x_im <= bx + bw and by <= y_im <= by + bh:
                    if idx in self.selected:
                        self.selected.remove(idx)
                    else:
                        self.selected.add(idx)
                    break
            self.update_image()

    def on_mouse_move(self, event):
        if self.drawing:
            self.end_pos = QPointF(event.pos())
            self.update_image()

    def on_mouse_release(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            # compute normalized YOLO coords from pixmap preview
            sw, sh = self.scaled_pix.width(), self.scaled_pix.height()
            x1 = max(0, min(sw, self.start_pos.x() - self.x_off))
            y1 = max(0, min(sh, self.start_pos.y() - self.y_off))
            x2 = max(0, min(sw, self.end_pos.x() - self.x_off))
            y2 = max(0, min(sh, self.end_pos.y() - self.y_off))
            rx = min(x1, x2); ry = min(y1, y2)
            rw = abs(x2 - x1); rh = abs(y2 - y1)
            nx = (rx + rw / 2) / sw
            ny = (ry + rh / 2) / sh
            nw = rw / sw
            nh = rh / sh
            cid, ok = QInputDialog.getText(self, "Class ID", "Enter class ID:")
            if ok and cid.isdigit():
                line = f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
                with open(self.label_path, 'a') as f:
                    f.write(line + "\n")
            self.selected.clear()
            self.update_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right and self.index < len(self.image_paths) - 1:
            self.index += 1
            self.selected.clear()
            self.update_image()
        elif event.key() == Qt.Key_Left and self.index > 0:
            self.index -= 1
            self.selected.clear()
            self.update_image()
        elif event.key() == Qt.Key_Delete and self.selected:
            new = [l for i, l in enumerate(self.labels) if i not in self.selected]
            with open(self.label_path, 'w') as f:
                f.write("\n".join(new) + ("\n" if new else ""))
            self.selected.clear()
            self.update_image()
        elif event.key() == Qt.Key_O:
            f = QFileDialog.getExistingDirectory(self, "Select Image Folder", self.image_folder)
            if f:
                self.image_folder = f
                self.image_paths = sorted([
                    os.path.join(f, fn) for fn in os.listdir(f)
                    if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
                ])
                self.index = 0
                self.selected.clear()
                self.update_image()
        elif event.key() == Qt.Key_L:
            f = QFileDialog.getExistingDirectory(self, "Select Labels Folder", self.labels_folder)
            if f:
                self.labels_folder = f
                self.selected.clear()
                self.update_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if len(sys.argv) >= 3:
        img_folder, lbl_folder = sys.argv[1], sys.argv[2]
    else:
        img_folder = QFileDialog.getExistingDirectory(None, "Select Image Folder", os.getcwd())
        lbl_folder = QFileDialog.getExistingDirectory(None, "Select Labels Folder", os.getcwd())
    window = ImageNavigator(img_folder, lbl_folder)
    window.show()
    sys.exit(app.exec_())