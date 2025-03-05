from PyQt5.QtWidgets import QDialog, QLabel, QComboBox, QPushButton
from PyQt5.QtCore import Qt
from PyQt5 import uic
import os


class CameraSettingsDialog(QDialog):
    """카메라 설정 다이얼로그"""

    def __init__(self, camera_manager, parent=None):
        super().__init__(parent)

        # UI 파일 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, 'ui_files', 'camera_settings.ui')
        uic.loadUi(ui_file, self)

        self.camera_manager = camera_manager

        # UI 요소 접근
        self.cameraCombo.addItems(self.camera_manager.available_cameras.keys())
        self.resolutionCombo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])

        # 버튼 연결
        self.connectButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

    def get_camera_settings(self):
        return {
            'camera': self.cameraCombo.currentText(),
            'resolution': self.resolutionCombo.currentText()
        }