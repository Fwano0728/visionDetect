#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5 import uic

# UI 파일 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "ui_files")
MAIN_WINDOW_UI = os.path.join(UI_DIR, "main_window.ui")


class MainWindowInit:
    """메인 윈도우 초기화 담당 클래스"""

    @staticmethod
    def init_ui(window):
        """UI 초기화"""
        # 메인 윈도우 UI 로드
        if os.path.exists(MAIN_WINDOW_UI):
            uic.loadUi(MAIN_WINDOW_UI, window)
        else:
            raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {MAIN_WINDOW_UI}")

        # 사이드바 초기화
        MainWindowInit.setup_sidebar(window)

    @staticmethod
    def setup_sidebar(window):
        """사이드바 설정"""
        # 사이드바 매니저 초기화
        from ui.sidebar.sidebar_manager import SidebarManager

        window.sidebar_manager = SidebarManager(window)
        window.sidebar_manager.setup_sidebar(
            window.sidebarFrame,
            window.camera_manager,
            window.detector_manager
        )

    @staticmethod
    def connect_signals(window):
        """시그널 연결"""
        # 카메라 매니저 시그널
        window.camera_manager.frame_updated.connect(window.updaters.update_frame)
        window.camera_manager.fps_updated.connect(window.updaters.update_fps)
        window.camera_manager.system_status_signal.connect(window.updaters.update_system_status)

        # 탐지 매니저 시그널
        window.detector_manager.detection_result_signal.connect(window.updaters.update_detection_result)

        # 사이드바 매니저 시그널
        window.sidebar_manager.camera_connect_requested.connect(window.handlers.on_camera_connect_requested)
        window.sidebar_manager.camera_disconnect_requested.connect(window.handlers.on_camera_disconnect_requested)
        window.sidebar_manager.detection_toggled.connect(window.handlers.on_detection_toggled)
        window.sidebar_manager.detector_settings_changed.connect(window.handlers.on_detector_settings_changed)