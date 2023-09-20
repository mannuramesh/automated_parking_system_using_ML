import sys
import cv2
import pytesseract
import numpy as np
import imutils
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class LicensePlateApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI Elements
        self.setWindowTitle("License Plate Detector")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.text_edit = QLineEdit(self)
        layout.addWidget(self.text_edit)

        self.load_button = QPushButton("Load Image", self)
        layout.addWidget(self.load_button)

        self.save_button = QPushButton("Save Result (Check for license_log.txt)", self)
        layout.addWidget(self.save_button)

        # Connect actions
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_result)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "All Files (*);;Images (*.png;*.jpg)",
                                                   options=options)
        if file_name:
            detected_text, license_img = self.detect_license_plate(file_name)
            height, width, channel = license_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(license_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
            self.text_edit.setText(detected_text)

    def detect_license_plate(self, img_path):
        img = cv2.imread(img_path)
        img = imutils.resize(img, width=500)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 170, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        NumberPlateCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                NumberPlateCnt = approx
                break
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(new_image, config=config)
        return text.strip(), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    def save_result(self):
        with open("license_log.txt", "a") as log_file:
            log_file.write(self.text_edit.text() + "\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = LicensePlateApp()
    mainWin.show()
    sys.exit(app.exec_())
