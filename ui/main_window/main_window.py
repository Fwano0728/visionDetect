#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QStatusBar
from PyQt5.QtCore import Qt, QTimer

from ui.error_handling import ErrorHandler
from ui.main_window.main_window_init import MainWindowInit
from ui.main_window.main_window_handlers import MainWindowHandlers
from ui.main_window.main_window_updates import MainWindowUpdates


class MainWindow(QMainWindow):
    """메인 윈도우 클래스"""

    def __init__(self, camera_manager=None, detector_manager=None):
        super().__init__()

        # 기본 윈도우 설정
        self.setWindowTitle("객체 탐지 시스템")
        self.setMinimumSize(1280, 800)
        self.resize(1280, 800)

        # 매니저 초기화
        if camera_manager is None:
            from models.camera_manager import CameraManager
            self.camera_manager = CameraManager()
        else:
            self.camera_manager = camera_manager

        if detector_manager is None:
            from models.detector_manager import DetectorManager
            self.detector_manager = DetectorManager()
        else:
            self.detector_manager = detector_manager

        # 에러 핸들러 초기화
        self.error_handler = ErrorHandler(self)

        # 기능 클래스 초기화
        self.handlers = MainWindowHandlers(self)
        self.updaters = MainWindowUpdates(self)

        # 사이드바 매니저는 init_ui에서 초기화
        self.sidebar_manager = None

        # UI 초기화
        MainWindowInit.init_ui(self)

        # 시그널 연결
        MainWindowInit.connect_signals(self)

        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("시스템 준비")

        # 시스템 상태 업데이트 타이머
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.updaters.update_system_status)
        self.status_timer.start(1000)  # 1초마다 상태 업데이트

    def closeEvent(self, event):
        """프로그램 종료 시 처리"""
        # 카메라 연결 해제
        if hasattr(self, 'camera_manager'):
            self.camera_manager.disconnect_camera()

        # 타이머 중지
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()

        # 기본 종료 이벤트 처리
        super().closeEvent(event)