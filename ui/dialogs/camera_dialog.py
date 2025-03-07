#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import uic

# 로거 설정
logger = logging.getLogger(__name__)

# UI 파일 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "ui", "ui_files")
CAMERA_DIALOG_UI = os.path.join(UI_DIR, "camera_dialog.ui")


class CameraDialog(QDialog):
    """카메라 선택 및 설정 다이얼로그"""

    # 시그널 정의
    camera_connect_requested = pyqtSignal(str, str, int)  # 카메라 이름, 해상도, FPS
    camera_disconnect_requested = pyqtSignal()

    def __init__(self, camera_manager, parent=None):
        # 부모를 None으로 설정하여 독립적인 창으로 작동
        super().__init__(None)

        # 윈도우 플래그 설정
        self.setWindowFlags(Qt.Window)
        self.setWindowModality(Qt.NonModal)

        # UI 파일 로드 (한 번만 수행)
        if os.path.exists(CAMERA_DIALOG_UI):
            uic.loadUi(CAMERA_DIALOG_UI, self)
        else:
            raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {CAMERA_DIALOG_UI}")

        self.camera_manager = camera_manager
        self.parent_window = parent  # 부모 창 참조 저장

        # 해상도 목록 설정
        self.resolutionCombo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])

        # 시그널 연결
        self.connectButton.clicked.connect(self.on_connect_clicked)
        self.disconnectButton.clicked.connect(self.on_disconnect_clicked)
        self.refreshButton.clicked.connect(self.load_cameras)
        self.closeButton.clicked.connect(self.close)

        # 부모 윈도우 핸들러와 시그널 연결
        if self.parent_window:
            self.camera_connect_requested.connect(self.parent_window.handlers.on_camera_connect_requested)
            self.camera_disconnect_requested.connect(self.parent_window.handlers.on_camera_disconnect_requested)

        # 카메라 목록 로드
        self.load_cameras()

        # 현재 카메라 상태 업데이트
        self.update_camera_state(self.camera_manager.is_connected())

    def load_cameras(self):
        """사용 가능한 카메라 목록 로드"""
        # 현재 선택된 카메라 저장
        current_camera = self.cameraCombo.currentText() if self.cameraCombo.count() > 0 else ""

        # 카메라 콤보박스 업데이트
        self.cameraCombo.clear()
        cameras = list(self.camera_manager.available_cameras.keys())

        # 카메라가 없으면 기본 항목 추가
        if not cameras:
            cameras = ["카메라를 찾을 수 없음"]
            self.connectButton.setEnabled(False)
        else:
            self.connectButton.setEnabled(True)

        self.cameraCombo.addItems(cameras)

        # 이전 선택 복원
        if current_camera and current_camera in cameras:
            index = self.cameraCombo.findText(current_camera)
            if index >= 0:
                self.cameraCombo.setCurrentIndex(index)

    def on_connect_clicked(self):
        """연결 버튼 클릭 처리"""
        camera = self.cameraCombo.currentText()
        resolution = self.resolutionCombo.currentText()
        fps = self.fpsSpinBox.value()

        # 카메라가 선택되었는지 확인
        if not camera or camera == "카메라를 찾을 수 없음":
            QMessageBox.warning(self, "카메라 연결 오류", "연결할 카메라를 선택해주세요.")
            return

        # 이미 연결되어 있는지 확인
        if self.camera_manager.is_connected():
            logger.warning("이미 카메라가 연결되어 있습니다. 먼저 연결을 해제합니다.")
            self.camera_manager.disconnect_camera()

        # 연결 시도
        logger.info(f"카메라 '{camera}' 연결 시도 (해상도: {resolution}, FPS: {fps})")
        success, message = self.camera_manager.connect_camera(camera, resolution)

        if success:
            # 버튼 상태 업데이트
            self.update_camera_state(True)

            # 시그널 발생 (한 번만 발생)
            self.camera_connect_requested.emit(camera, resolution, fps)

            # 설정 저장 (옵션)
            if self.saveSettingsCheckBox.isChecked():
                self.save_camera_settings(camera, resolution, fps)

            # 연결 성공 후 다이얼로그 닫기
            self.close()

            logger.info(f"카메라 '{camera}' 연결 성공 (해상도: {resolution}, FPS: {fps})")
        else:
            # 오류 메시지 표시
            error_msg = f"카메라 연결에 실패했습니다: {message}"
            logger.error(error_msg)
            QMessageBox.critical(self, "카메라 연결 오류", error_msg)

    def on_disconnect_clicked(self):
        """연결 해제 버튼 클릭 처리"""
        self.camera_manager.disconnect_camera()

        # 버튼 상태 업데이트
        self.update_camera_state(False)

        # 시그널 발생
        self.camera_disconnect_requested.emit()

    def update_camera_state(self, connected):
        """카메라 연결 상태 업데이트"""
        # 버튼 상태 업데이트
        self.connectButton.setEnabled(not connected)
        self.disconnectButton.setEnabled(connected)

        # 설정 위젯 활성화/비활성화
        self.cameraCombo.setEnabled(not connected)
        self.resolutionCombo.setEnabled(not connected)
        self.fpsSpinBox.setEnabled(not connected)
        self.refreshButton.setEnabled(not connected)
        self.saveSettingsCheckBox.setEnabled(not connected)

    def save_camera_settings(self, camera, resolution, fps):
        """카메라 설정 저장"""
        from core.processing.config_manager import ConfigManager

        config = ConfigManager.get_instance()
        settings = {
            "camera": camera,
            "resolution": resolution,
            "fps": fps
        }

        config.save_settings("camera", settings)
        print("카메라 설정 저장 완료")

    @staticmethod
    def load_last_settings():
        """최근 저장된 카메라 설정 로드"""
        from core.processing.config_manager import ConfigManager

        config = ConfigManager.get_instance()
        settings = config.get_setting("camera", default={})

        # 기본값 설정
        if not settings:
            settings = {
                'camera': '',
                'resolution': '640x480',
                'fps': 30
            }

        return settings

