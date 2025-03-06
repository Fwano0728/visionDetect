#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
메인 윈도우 초기화 모듈
"""

import os
import logging
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon

# UI 자동 생성 파일 가져오기
from ui.generated.ui_main_window import Ui_MainWindow

logger = logging.getLogger(__name__)


class MainWindowInit:
    """메인 윈도우 초기화 담당 클래스"""

    def __init__(self, main_window):
        """
        초기화

        Args:
            main_window: 메인 윈도우 인스턴스
        """
        self.window = main_window
        self.ui = Ui_MainWindow()
        self.settings = QSettings("VisionDetect", "ObjectDetection")

    def setup_ui(self):
        """UI 초기화"""
        # UI 설정
        self.ui.setupUi(self.window)

        # 윈도우 설정
        self.setup_window()

        # 기타 UI 설정
        self.setup_controls()

        # 설정 적용
        self.load_settings()

        # 윈도우 표시 후 추가 설정
        self.post_show_setup()

        logger.info("메인 윈도우 UI 초기화 완료")

    def setup_window(self):
        """윈도우 기본 설정"""
        # 아이콘 설정 (아이콘 파일이 있는 경우)
        icon_path = os.path.join('resources', 'icons', 'app_icon.png')
        if os.path.exists(icon_path):
            self.window.setWindowIcon(QIcon(icon_path))

        # 윈도우 상태 및 위치 복원 (저장된 경우)
        geometry = self.settings.value("window/geometry")
        if geometry:
            self.window.restoreGeometry(geometry)
        else:
            # 기본 위치 및 크기
            self.window.resize(1280, 800)
            self.window.setMinimumSize(1024, 768)

        # 스플리터 크기 설정
        self.ui.contentSplitter.setSizes([800, 400])

    def setup_controls(self):
        """컨트롤 초기화"""
        # 툴바 설정
        self.ui.toolBar.setIconSize(Qt.QSize(24, 24))

        # 해상도 콤보박스 초기화
        resolutions = ["640x480", "800x600", "1280x720", "1920x1080"]
        self.ui.resolutionComboBox.addItems(resolutions)

        # YOLO 관련 위젯 초기 상태 설정
        self.ui.detectionStackedWidget.setCurrentIndex(0)  # HOG 페이지 기본 표시

        # 결과 패널 초기화
        self.ui.resultsTextLabel.setText("카메라가 연결되면 탐지된 객체가 여기에 표시됩니다.")

        # 초기 상태 설정
        self.ui.action_Disconnect.setEnabled(False)
        self.ui.action_StartDetection.setEnabled(False)
        self.ui.action_StopDetection.setEnabled(False)
        self.ui.disconnectButton.setEnabled(False)
        self.ui.detectionCheckBox.setEnabled(False)
        self.ui.detectionModeComboBox.setEnabled(False)
        self.ui.applyDetectorButton.setEnabled(False)

    def load_settings(self):
        """설정 로드 및 적용"""
        # 저장 경로
        save_path = self.settings.value("system/save_path", os.path.join(os.path.expanduser("~"), "VisionDetect"))
        self.ui.savePathEdit.setText(save_path)

        # 로그 보관 일수
        log_retention = self.settings.value("system/log_retention", 30, type=int)
        self.ui.logRetentionSpinBox.setValue(log_retention)

        # 체크박스 설정
        self.ui.startMinimizedCheckBox.setChecked(self.settings.value("system/start_minimized", False, type=bool))
        self.ui.autoStartCheckBox.setChecked(self.settings.value("system/auto_start", False, type=bool))
        self.ui.autoConnectCameraCheckBox.setChecked(
            self.settings.value("system/auto_connect_camera", False, type=bool))

        logger.debug("사용자 설정 로드 완료")

    def post_show_setup(self):
        """윈도우가 표시된 후 추가 설정"""
        # 자동 연결 설정이 활성화되어 있으면 카메라 연결 시도
        if self.settings.value("system/auto_connect_camera", False, type=bool):
            # 지연 실행 (UI가 완전히 로드된 후)
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, self.window.connect_last_camera)

    def save_settings(self):
        """현재 설정 저장"""
        # 윈도우 상태 저장
        self.settings.setValue("window/geometry", self.window.saveGeometry())

        # 시스템 설정 저장
        self.settings.setValue("system/save_path", self.ui.savePathEdit.text())
        self.settings.setValue("system/log_retention", self.ui.logRetentionSpinBox.value())
        self.settings.setValue("system/start_minimized", self.ui.startMinimizedCheckBox.isChecked())
        self.settings.setValue("system/auto_start", self.ui.autoStartCheckBox.isChecked())
        self.settings.setValue("system/auto_connect_camera", self.ui.autoConnectCameraCheckBox.isChecked())

        # 마지막 카메라 설정 저장
        last_camera = self.window.last_connected_camera
        if last_camera:
            self.settings.setValue("camera/last_camera", last_camera.get("name", ""))
            self.settings.setValue("camera/last_resolution", last_camera.get("resolution", "640x480"))
            self.settings.setValue("camera/last_fps", last_camera.get("fps", 30))

        # 마지막 탐지기 설정 저장
        last_detector = self.window.last_detector_settings
        if last_detector:
            self.settings.setValue("detector/type", last_detector.get("type", ""))
            self.settings.setValue("detector/version", last_detector.get("version", ""))
            self.settings.setValue("detector/model", last_detector.get("model", ""))
            self.settings.setValue("detector/confidence", last_detector.get("confidence", 0.5))

        logger.debug("사용자 설정 저장 완료")