#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
메인 윈도우 초기화 모듈
"""

import os
import logging
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt, QSettings, QSize
from PyQt5.QtGui import QIcon
from PyQt5 import uic

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
        self.ui = None
        self.settings = QSettings("VisionDetect", "ObjectDetection")

    def setup_ui(self):
        """UI 초기화"""
        try:
            # UI 파일 경로
            ui_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui_files', 'main_window.ui')

            # UI 파일이 없는 경우 오류 표시
            if not os.path.exists(ui_file):
                logger.error(f"UI 파일을 찾을 수 없음: {ui_file}")
                # 기본 빈 UI 설정
                self.window.setWindowTitle("객체 탐지 시스템")
                self.window.resize(1280, 800)
                self.window.statusBar().showMessage("UI 파일을 찾을 수 없습니다.")
                return

            # UI 파일 직접 로드
            uic.loadUi(ui_file, self.window)

            # UI 참조 저장 (loadUi가 window에 직접 로드하므로 window를 UI로 사용)
            self.ui = self.window

            # 윈도우 설정
            self.setup_window()

            # 기타 UI 설정
            self.setup_controls()

            # 설정 적용
            self.load_settings()

            # 윈도우 표시 후 추가 설정
            self.post_show_setup()

            logger.info("메인 윈도우 UI 초기화 완료")

        except Exception as e:
            logger.error(f"UI 초기화 오류: {str(e)}")
            # 기본 UI 설정
            self.window.setWindowTitle("객체 탐지 시스템 (오류)")
            self.window.resize(1280, 800)
            self.window.statusBar().showMessage(f"UI 초기화 오류: {str(e)}")

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
            self.window.setMinimumSize(1024, 720)

        # 스플리터 크기 설정
        self.ui.contentSplitter.setSizes([800, 400])

    def setup_controls(self):
        """컨트롤 초기화"""
        # 해상도 콤보박스 초기화
        resolutions = ["640x480", "800x600", "1280x720", "1920x1080"]
        self.ui.resolutionComboBox.addItems(resolutions)

        # 초기 상태 설정
        self.ui.disconnectButton.setEnabled(False)
        self.ui.detectionCheckBox.setEnabled(False)
        self.ui.detectionModeComboBox.setEnabled(False)
        self.ui.modelSettingsButton.setEnabled(False)

    def load_settings(self):
        """설정 로드 및 적용"""
        # 저장 경로 (이제 UI에 표시되지 않지만 설정은 유지)
        self.save_path = self.settings.value("system/save_path", os.path.join(os.path.expanduser("~"), "VisionDetect"))

        # 체크박스 설정 (이제 UI에 표시되지 않지만 설정은 유지)
        self.start_minimized = self.settings.value("system/start_minimized", False, type=bool)
        self.auto_connect_camera = self.settings.value("system/auto_connect_camera", False, type=bool)

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
        self.settings.setValue("system/save_path", self.save_path)
        self.settings.setValue("system/start_minimized", self.start_minimized)
        self.settings.setValue("system/auto_connect_camera", self.auto_connect_camera)

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