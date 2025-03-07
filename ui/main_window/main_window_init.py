#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/main_window/main_window_init.py

import os
import sys
import logging
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon  # QIcon import 추가
from PyQt5 import uic

# 로거 설정
logger = logging.getLogger(__name__)

# UI 파일 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "ui", "ui_files")
MAIN_WINDOW_UI = os.path.join(UI_DIR, "main_window.ui")


class MainWindowInit:
    """메인 윈도우 초기화 담당 클래스"""

    @staticmethod
    def init_ui(window):
        """UI 초기화"""
        try:
            # 메인 윈도우 UI 로드
            if os.path.exists(MAIN_WINDOW_UI):
                uic.loadUi(MAIN_WINDOW_UI, window)
                logger.info(f"UI 파일 로드 완료: {MAIN_WINDOW_UI}")

                # 아이콘 설정 (UI 로드 후)
                MainWindowInit._setup_icons(window)
            else:
                error_msg = f"UI 파일을 찾을 수 없습니다: {MAIN_WINDOW_UI}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # 사이드바 초기화 시도
            try:
                if hasattr(window, 'sidebarFrame'):
                    MainWindowInit.setup_sidebar(window)
            except Exception as e:
                logger.warning(f"사이드바 설정을 건너뜁니다: {str(e)}")

            # 시각화 위젯 설정 (UI 로드 이후)
            try:
                MainWindowInit.setup_visualization(window)
            except Exception as e:
                logger.warning(f"시각화 위젯 설정을 건너뜁니다: {str(e)}")

            # 결과 텍스트 레이블 스타일 및 속성 설정
            if hasattr(window, 'resultsText'):
                window.resultsText.setWordWrap(True)  # 텍스트 자동 줄바꿈 활성화
                window.resultsText.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 텍스트 정렬 변경

                # 스타일시트 수정
                style = """
                    QLabel#resultsText {
                        background-color: #f8f8f8;
                        color: #333333;
                        border-radius: 4px;
                        border: 1px solid #dddddd;
                        padding: 10px;
                        font-family: 'Nanum Gothic', sans-serif;
                        min-height: 80px;
                    }
                """
                window.resultsText.setStyleSheet(style)

        except Exception as e:
            logger.error(f"UI 초기화 중 오류 발생: {str(e)}", exc_info=True)
            raise


    @staticmethod
    def setup_visualization(window):
        """시각화 위젯 설정"""
        # 결과 그룹박스 확인
        if not hasattr(window, 'resultsGroupBox'):
            logger.warning("결과 그룹박스가 없습니다. 시각화 위젯을 추가하지 않습니다.")
            return

        try:
            # 시각화 위젯 가져오기
            from ui.visualization.detection_visualizer import DetectionResultsWidget

            # 시각화 위젯 생성
            window.results_visualizer = DetectionResultsWidget(window.resultsGroupBox)

            # 레이아웃 확인
            layout = window.resultsGroupBox.layout()

            if layout is not None:
                # 기존 결과 텍스트가 있다면 숨김
                for i in range(layout.count()):
                    widget = layout.itemAt(i).widget()
                    if widget and hasattr(widget, 'objectName') and widget.objectName() == "resultsText":
                        widget.setVisible(False)

                # 시각화 위젯 추가
                layout.addWidget(window.results_visualizer)
                logger.info("시각화 위젯 추가 완료")
            else:
                # 레이아웃이 없으면 새로 생성
                from PyQt5.QtWidgets import QVBoxLayout
                new_layout = QVBoxLayout(window.resultsGroupBox)

                # 시각화 위젯 추가
                new_layout.addWidget(window.results_visualizer)
                logger.info("새 레이아웃에 시각화 위젯 추가 완료")

            # 트렌드 추적 시작
            window.results_visualizer.start_trend_tracking()

        except ImportError as e:
            logger.warning(f"시각화 모듈을 가져올 수 없습니다: {str(e)}")

            # PyQtChart 관련 오류인 경우
            if "PyQt5.QtChart" in str(e):
                logger.warning("PyQtChart 모듈이 설치되지 않았습니다. pip install PyQtChart 명령으로 설치하세요.")

                # 기존 텍스트 레이블에 메시지 표시
                if hasattr(window, 'resultsText'):
                    window.resultsText.setText("차트 시각화를 위해 PyQtChart 모듈을 설치하세요: pip install PyQtChart")
            raise
        except Exception as e:
            logger.error(f"시각화 위젯 설정 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def connect_signals(window):
        """시그널 연결"""
        try:
            # 카메라 매니저 시그널
            if hasattr(window, 'camera_manager') and hasattr(window, 'updaters'):
                window.camera_manager.frame_updated.connect(window.updaters.update_frame)
                window.camera_manager.fps_updated.connect(window.updaters.update_fps)
                window.camera_manager.system_status_signal.connect(window.updaters.update_system_status)
                logger.debug("카메라 매니저 시그널 연결 완료")

            # 탐지 매니저 시그널
            if hasattr(window, 'detector_manager') and hasattr(window, 'updaters'):
                window.detector_manager.detection_result_signal.connect(window.updaters.update_detection_result)
                logger.debug("탐지 매니저 시그널 연결 완료")

            # 사이드바 매니저 시그널
            if (hasattr(window, 'sidebar_manager') and window.sidebar_manager is not None and
                    hasattr(window, 'handlers')):
                window.sidebar_manager.camera_connect_requested.connect(window.handlers.on_camera_connect_requested)
                window.sidebar_manager.camera_disconnect_requested.connect(
                    window.handlers.on_camera_disconnect_requested)
                window.sidebar_manager.detection_toggled.connect(window.handlers.on_detection_toggled)
                window.sidebar_manager.detector_settings_changed.connect(window.handlers.on_detector_settings_changed)
                logger.debug("사이드바 매니저 시그널 연결 완료")

            logger.info("모든 시그널 연결 완료")
        except Exception as e:
            logger.error(f"시그널 연결 중 오류 발생: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def _setup_icons(window):
        """아이콘 설정 - 상단 툴바 아이콘 제거"""
        try:
            # 아이콘 비우기
            empty_icon = QIcon()

            # 툴바 액션 아이콘 제거
            if hasattr(window, 'quickStartAction'):
                window.quickStartAction.setIcon(empty_icon)
            if hasattr(window, 'quickDetectionAction'):
                window.quickDetectionAction.setIcon(empty_icon)
            if hasattr(window, 'cameraAction'):
                window.cameraAction.setIcon(empty_icon)
            if hasattr(window, 'modelAction'):
                window.modelAction.setIcon(empty_icon)
            if hasattr(window, 'startStopAction'):
                window.startStopAction.setIcon(empty_icon)
            if hasattr(window, 'toolsAction'):
                window.toolsAction.setIcon(empty_icon)

            logger.info("툴바 아이콘 제거 완료")
        except Exception as e:
            logger.warning(f"아이콘 설정 중 오류 발생: {str(e)}")
            # 아이콘 설정 오류는 심각한 문제가 아니므로 예외를 다시 발생시키지 않음