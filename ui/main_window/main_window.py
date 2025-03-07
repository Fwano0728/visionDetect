#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/main_window/main_window.py

import os
import sys
import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QStatusBar, QToolBar,
                             QAction, QDialog, QVBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon

from ui.error_handling import ErrorHandler
from ui.main_window.main_window_init import MainWindowInit
from ui.main_window.main_window_handlers import MainWindowHandlers
from ui.main_window.main_window_updates import MainWindowUpdates

# 다이얼로그 클래스들 임포트
from ui.dialogs.camera_dialog import CameraDialog
from ui.dialogs.model_dialog import ModelDialog
from ui.dialogs.tools_dialog import ToolsDialog

# 로거 설정
logger = logging.getLogger(__name__)


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
            from detection.detector_manager import DetectorManager
            self.detector_manager = DetectorManager()
        else:
            self.detector_manager = detector_manager

        # 에러 핸들러 초기화
        self.error_handler = ErrorHandler(self)

        # 다이얼로그 참조 저장용 변수
        self.camera_dialog = None
        self.model_dialog = None
        self.tools_dialog = None

        try:
            # UI 초기화 (MainWindowInit 사용)
            MainWindowInit.init_ui(self)

            # 기능 클래스 초기화 (UI 초기화 후)
            self.handlers = MainWindowHandlers(self)
            self.updaters = MainWindowUpdates(self)

            # 시그널 연결
            MainWindowInit.connect_signals(self)
            self.connect_toolbar_actions()
        except Exception as e:
            # show_dialog 파라미터를 제거하거나 ErrorHandler에서 지원하는 파라미터만 사용
            self.error_handler.log_error(f"UI 초기화 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()

        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("시스템 준비")

        # 시스템 상태 업데이트 타이머
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.updaters.update_system_status)
        self.status_timer.start(1000)  # 1초마다 상태 업데이트

    def setup_icons(self):
        """아이콘 설정"""
        # 아이콘 디렉토리 경로
        icon_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icons")
        os.makedirs(icon_dir, exist_ok=True)

        # 테마 아이콘 사용 (아이콘 파일이 없는 경우)
        if not QIcon.hasThemeIcon("camera-photo"):
            # 필요한 경우 아이콘 테마 설정
            QIcon.setThemeName("breeze")

    def connect_toolbar_actions(self):
        """툴바 액션 연결"""
        # 빠른 연결 액션
        if hasattr(self, 'quickStartAction'):
            self.quickStartAction.triggered.connect(self.quick_connect)

        # 빠른 탐지 액션
        if hasattr(self, 'quickDetectionAction'):
            self.quickDetectionAction.triggered.connect(self.quick_detection)

        # 카메라 설정 액션
        if hasattr(self, 'cameraAction'):
            self.cameraAction.triggered.connect(self.show_camera_dialog)

        # 모델 설정 액션
        if hasattr(self, 'modelAction'):
            self.modelAction.triggered.connect(self.show_model_dialog)

        # 시작/중지 액션
        if hasattr(self, 'startStopAction'):
            self.startStopAction.triggered.connect(self.toggle_detection)

        # 도구 액션
        if hasattr(self, 'toolsAction'):
            self.toolsAction.triggered.connect(self.show_tools_dialog)

    def quick_connect(self):
        """최근 설정으로 빠른 연결"""
        # 최근 설정 로드
        settings = CameraDialog.load_last_settings()

        if not settings['camera']:
            QMessageBox.information(self, "정보", "저장된 카메라 설정이 없습니다. 카메라 설정 다이얼로그를 이용해주세요.")
            self.show_camera_dialog()
            return

        # 연결 시도
        success, message = self.camera_manager.connect_camera(settings['camera'], settings['resolution'])

        if success:
            # UI 상태 업데이트
            self.update_ui_state(camera_connected=True)
            self.statusBar.showMessage(
                f"카메라 '{settings['camera']}' 연결됨, 해상도: {settings['resolution']}, FPS: {settings['fps']}")
        else:
            self.error_handler.show_warning(f"카메라 연결 실패: {message}")
            self.show_camera_dialog()

    def quick_detection(self, checked):
        """기본 설정으로 탐지 시작/중지"""
        if checked:
            # 카메라가 연결되어 있는지 확인
            if not self.camera_manager.is_connected():
                QMessageBox.warning(self, "경고", "카메라가 연결되어 있지 않습니다. 먼저 카메라를 연결해주세요.")
                self.quickDetectionAction.setChecked(False)
                return

            # 최근 설정 로드
            settings = ModelDialog.load_last_settings()

            if not settings['detector_type']:
                QMessageBox.information(self, "정보", "저장된 탐지기 설정이 없습니다. 모델 설정 다이얼로그를 이용해주세요.")
                self.show_model_dialog()
                self.quickDetectionAction.setChecked(False)
                return

            # 탐지기 설정
            if settings['detector_type'] == "YOLO":
                params = {
                    'conf_threshold': settings['confidence'],
                    'nms_threshold': settings['nms_threshold']
                }
                success, message = self.detector_manager.set_detector(
                    settings['detector_type'], settings['yolo_version'], settings['yolo_model'], **params)
            else:
                success, message = self.detector_manager.set_detector(settings['detector_type'])

            if success:
                self.statusBar.showMessage("객체 탐지 시작")
                self.startStopAction.setChecked(True)
                self.startStopAction.setText("탐지 중지")
            else:
                self.error_handler.show_warning(f"탐지기 설정 실패: {message}")
                self.quickDetectionAction.setChecked(False)
                self.show_model_dialog()
        else:
            # 탐지 중지
            self.detector_manager.set_detector(None)
            self.statusBar.showMessage("객체 탐지 중지")
            self.startStopAction.setChecked(False)
            self.startStopAction.setText("탐지 시작")

    def show_camera_dialog(self):
        """카메라 설정 다이얼로그 표시"""
        self.camera_dialog = CameraDialog(self.camera_manager, parent=self)
        self.camera_dialog.camera_connect_requested.connect(self.handlers.on_camera_connect_requested)
        self.camera_dialog.camera_disconnect_requested.connect(self.handlers.on_camera_disconnect_requested)
        self.camera_dialog.exec_()  # 모달 방식으로 표시

    def show_model_dialog(self):
        """모델 설정 다이얼로그 표시"""
        if hasattr(self, 'model_dialog') and self.model_dialog is not None:
            # 기존 다이얼로그가 열려있으면 활성화만 시킴
            self.model_dialog.activateWindow()
            return

        # 새 다이얼로그 생성
        from ui.dialogs.model_dialog import ModelDialog
        self.model_dialog = ModelDialog(self.detector_manager, parent=self)

        # 다이얼로그가 닫힐 때 참조 제거하는 시그널 연결
        self.model_dialog.finished.connect(lambda: setattr(self, 'model_dialog', None))

        # 시그널 연결
        self.model_dialog.detection_toggled.connect(self.handlers.on_detection_toggled)
        self.model_dialog.detector_settings_changed.connect(self.handlers.on_detector_settings_changed)

        # 다이얼로그 표시
        self.model_dialog.exec_()

    def show_tools_dialog(self):
        """도구 다이얼로그 표시"""
        self.tools_dialog = ToolsDialog(parent=self)
        self.tools_dialog.object_selection_requested.connect(self.handlers.show_object_selection_dialog)
        self.tools_dialog.exec_()  # 모달 방식으로 표시

    def toggle_detection(self, checked):
        """탐지 시작/중지 토글"""
        self.handlers.on_detection_toggled(checked)

        # 빠른 탐지 버튼도 함께 업데이트
        if hasattr(self, 'quickDetectionAction'):
            self.quickDetectionAction.setChecked(checked)

        # 액션 텍스트 변경
        if checked:
            self.startStopAction.setText("탐지 중지")
        else:
            self.startStopAction.setText("탐지 시작")

    def update_ui_state(self, camera_connected=None, detection_active=None):
        """UI 상태 업데이트"""
        if camera_connected is not None:
            # 카메라 연결 상태에 따른 UI 업데이트
            if hasattr(self, 'startStopAction'):
                self.startStopAction.setEnabled(camera_connected)

            if hasattr(self, 'quickDetectionAction'):
                self.quickDetectionAction.setEnabled(camera_connected)

            if not camera_connected:
                if hasattr(self, 'startStopAction'):
                    self.startStopAction.setChecked(False)
                    self.startStopAction.setText("탐지 시작")

                if hasattr(self, 'quickDetectionAction'):
                    self.quickDetectionAction.setChecked(False)

        if detection_active is not None:
            # 탐지 활성화 상태에 따른 UI 업데이트
            if hasattr(self, 'startStopAction'):
                self.startStopAction.setChecked(detection_active)

                if detection_active:
                    self.startStopAction.setText("탐지 중지")
                else:
                    self.startStopAction.setText("탐지 시작")

            if hasattr(self, 'quickDetectionAction'):
                self.quickDetectionAction.setChecked(detection_active)

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