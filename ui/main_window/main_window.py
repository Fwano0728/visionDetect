#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
메인 윈도우 클래스
UI 생성, 이벤트 처리 및 업데이트 로직을 모듈화하여 관리
"""

import os
import sys
import logging
import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt, pyqtSlot, QSettings

# 내부 모듈
from ui.main_window.main_window_init import MainWindowInit
from ui.main_window.main_window_handlers import MainWindowHandlers
from ui.main_window.main_window_updates import MainWindowUpdates
from ui.error_handling import ErrorHandler

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """객체 탐지 애플리케이션의 메인 윈도우 클래스"""

    def __init__(self, camera_manager, detector_manager):
        """
        메인 윈도우 초기화

        Args:
            camera_manager: 카메라 관리 객체
            detector_manager: 탐지기 관리 객체
        """
        super().__init__()

        # 매니저 설정
        self.camera_manager = camera_manager
        self.detector_manager = detector_manager

        # 상태 변수
        self.current_frame = None
        self.last_connected_camera = None
        self.last_detector_settings = None

        # UI 컴포넌트 참조 변수
        self.ui = None

        # 관리자 클래스 초기화
        self.init_manager = MainWindowInit(self)
        self.handlers = MainWindowHandlers(self)
        self.updaters = MainWindowUpdates(self)

        # UI 초기화
        self.init_ui()

        # 이벤트 연결
        self.connect_signals()

        logger.info("메인 윈도우 초기화 완료")

    def init_ui(self):
        """UI 초기화"""
        try:
            # UI 설정
            self.init_manager.setup_ui()
            self.ui = self.init_manager.ui

        except Exception as e:
            ErrorHandler.handle_exception(self, e, "UI 초기화")
            sys.exit(1)  # 초기화 실패 시 종료

    def connect_signals(self):
        """이벤트 시그널 연결"""
        try:
            # 이벤트 핸들러 연결
            self.handlers.connect_signals()

            # 카메라 매니저 시그널 연결
            self.camera_manager.frame_updated.connect(self.updaters.update_frame)
            self.camera_manager.fps_updated.connect(self.updaters.update_fps)
            self.camera_manager.system_status_signal.connect(self.updaters.update_system_status)

            # 탐지 매니저 시그널 연결
            self.detector_manager.detection_result_signal.connect(self.updaters.update_detection_result)
            self.detector_manager.detection_complete_signal.connect(self.updaters.handle_detection_complete)

        except Exception as e:
            ErrorHandler.handle_exception(self, e, "시그널 연결")

    def connect_last_camera(self):
        """마지막으로 연결했던 카메라 연결 시도"""
        try:
            settings = QSettings("VisionDetect", "ObjectDetection")
            last_camera = settings.value("camera/last_camera", "")

            if last_camera:
                # 콤보박스에서 카메라 찾기
                index = self.ui.cameraComboBox.findText(last_camera)
                if index >= 0:
                    self.ui.cameraComboBox.setCurrentIndex(index)

                    # 해상도 설정
                    last_resolution = settings.value("camera/last_resolution", "640x480")
                    res_index = self.ui.resolutionComboBox.findText(last_resolution)
                    if res_index >= 0:
                        self.ui.resolutionComboBox.setCurrentIndex(res_index)

                    # FPS 설정
                    last_fps = settings.value("camera/last_fps", 30, type=int)
                    self.ui.fpsSpinBox.setValue(last_fps)

                    # 연결 버튼 클릭
                    QApplication.processEvents()  # UI 업데이트
                    self.handlers.on_connect_camera()

                    logger.info(f"마지막 카메라 자동 연결 시도: {last_camera}")

        except Exception as e:
            logger.error(f"마지막 카메라 연결 시도 중 오류: {str(e)}")
            # 자동 연결 실패는 치명적이지 않으므로 오류 대화상자 표시하지 않음

    def closeEvent(self, event):
        """윈도우 닫기 이벤트"""
        try:
            # 설정 저장
            self.init_manager.save_settings()

            # 카메라 연결 해제
            if self.camera_manager.is_connected():
                self.camera_manager.disconnect_camera()

            # 탐지기 해제
            if self.detector_manager.is_active():
                self.detector_manager.set_detector(None)

            logger.info("프로그램 종료")
            event.accept()

        except Exception as e:
            logger.error(f"프로그램 종료 중 오류: {str(e)}")
            event.accept()  # 오류가 있어도 종료