#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QComboBox, QPushButton, QGroupBox,
                            QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class CameraSelector(QWidget):
    """카메라 선택 및 설정 위젯"""

    # 시그널 정의
    camera_connected = pyqtSignal(str, str)  # 카메라 이름, 해상도
    camera_disconnected = pyqtSignal()

    def __init__(self, camera_manager, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 그룹 박스
        group_box = QGroupBox("카메라 설정")
        grid_layout = QGridLayout()

        # 카메라 선택
        grid_layout.addWidget(QLabel("카메라:"), 0, 0)
        self.camera_combo = QComboBox()
        self.load_cameras()
        grid_layout.addWidget(self.camera_combo, 0, 1)

        # 해상도 선택
        grid_layout.addWidget(QLabel("해상도:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])
        grid_layout.addWidget(self.resolution_combo, 1, 1)

        group_box.setLayout(grid_layout)
        layout.addWidget(group_box)

        # 버튼
        button_layout = QHBoxLayout()

        self.connect_button = QPushButton("연결")
        self.connect_button.clicked.connect(self.on_connect_clicked)
        button_layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton("연결 해제")
        self.disconnect_button.clicked.connect(self.on_disconnect_clicked)
        self.disconnect_button.setEnabled(False)
        button_layout.addWidget(self.disconnect_button)

        layout.addLayout(button_layout)

    def load_cameras(self):
        """사용 가능한 카메라 목록 로드"""
        self.camera_combo.clear()
        cameras = list(self.camera_manager.available_cameras.keys())
        self.camera_combo.addItems(cameras)

    def on_connect_clicked(self):
        """연결 버튼 클릭 처리"""
        camera = self.camera_combo.currentText()
        resolution = self.resolution_combo.currentText()

        # 연결 시도
        success, message = self.camera_manager.connect_camera(camera, resolution)

        if success:
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            # 시그널 발생
            self.camera_connected.emit(camera, resolution)

        return success, message

    def on_disconnect_clicked(self):
        """연결 해제 버튼 클릭 처리"""
        self.camera_manager.disconnect_camera()
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        # 시그널 발생
        self.camera_disconnected.emit()

    def update_status(self, connected):
        """카메라 연결 상태 업데이트"""
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)

    def update_font_sizes(self, font_size):
        """UI 요소의 폰트 크기 업데이트"""
        font = QFont()
        font.setPointSize(font_size)

        self.setFont(font)

        for widget in self.findChildren(QLabel):
            widget.setFont(font)

        for widget in self.findChildren(QPushButton):
            widget.setFont(font)

        for widget in self.findChildren(QComboBox):
            widget.setFont(font)