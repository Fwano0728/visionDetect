#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSplitter, QFrame, QStatusBar)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

from ui.camera_selector import CameraSelector
from ui.model_selector import ModelSelector
from models.camera_manager import CameraManager
from models.detector_manager import DetectorManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 기본 윈도우 설정
        self.setWindowTitle("객체 탐지 시스템")
        self.setMinimumSize(1280, 800)
        self.resize(1280, 800)

        # 매니저 초기화
        self.camera_manager = CameraManager()
        self.detector_manager = DetectorManager()

        # UI 요소 초기화
        self.init_ui()

        # 시그널 연결
        self.connect_signals()

        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("시스템 준비")

    def init_ui(self):
        """UI 컴포넌트 초기화"""
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)

        # 상단 패널 (카메라 및 모델 선택)
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)

        # 카메라 선택기
        self.camera_selector = CameraSelector(self.camera_manager)
        top_layout.addWidget(self.camera_selector)

        # 모델 선택기
        self.model_selector = ModelSelector(self.detector_manager)
        top_layout.addWidget(self.model_selector)

        main_layout.addWidget(top_panel)

        # 비디오 및 결과 영역
        content_panel = QSplitter(Qt.Horizontal)

        # 비디오 디스플레이 영역
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setFrameStyle(QFrame.StyledPanel)
        self.video_frame.setMinimumSize(640, 480)
        self.video_frame.setText("카메라가 연결되지 않았습니다.")

        # 결과 디스플레이 영역
        self.results_text = QLabel("탐지된 객체가 없습니다.")
        self.results_text.setAlignment(Qt.AlignCenter)
        self.results_text.setFrameStyle(QFrame.StyledPanel)

        content_panel.addWidget(self.video_frame)
        content_panel.addWidget(self.results_text)
        content_panel.setSizes([800, 400])  # 초기 크기 설정

        main_layout.addWidget(content_panel, 1)  # 전체 스페이스 차지

    def connect_signals(self):
        """시그널 연결"""
        # 카메라 매니저 시그널
        self.camera_manager.frame_updated.connect(self.update_frame)
        self.camera_manager.fps_updated.connect(self.update_fps)
        self.camera_manager.system_status_signal.connect(self.update_system_status)

        # 탐지 매니저 시그널
        self.detector_manager.detection_result_signal.connect(self.update_detection_result)

        # 카메라 선택기 시그널
        self.camera_selector.camera_connected.connect(self.on_camera_connected)
        self.camera_selector.camera_disconnected.connect(self.on_camera_disconnected)

        # 모델 선택기 시그널
        self.model_selector.detector_activated.connect(self.on_detector_activated)
        self.model_selector.detector_deactivated.connect(self.on_detector_deactivated)

    def update_frame(self, frame):
        """카메라 프레임 업데이트"""
        try:
            if frame is None:
                print("받은 프레임이 None입니다.")
                return

            # 객체 탐지 실행 (탐지 매니저에 프레임 전달)
            if self.detector_manager.is_active():
                frame, _, _ = self.detector_manager.detect(frame)

            # OpenCV BGR 형식을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape

            # Qt 이미지 변환
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 라벨에 이미지 표시 (비율 유지하며 크기 조정)
            pixmap = QPixmap.fromImage(qt_image)

            # 비디오 프레임 위젯의 실제 크기
            frame_width = self.video_frame.width()
            frame_height = self.video_frame.height()

            # 이미지 크기 조정 (비율 유지)
            scaled_pixmap = pixmap.scaled(frame_width, frame_height,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)

            self.video_frame.setPixmap(scaled_pixmap)
            self.video_frame.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(f"프레임 업데이트 오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_fps(self, fps):
        """FPS 업데이트 처리"""
        # FPS 정보 저장 (상태 표시줄에서 사용)
        self.current_fps = fps

    def update_system_status(self, status):
        """시스템 상태 정보 업데이트"""
        fps = status.get('fps', 0)
        cpu = status.get('cpu', 0)
        memory = status.get('memory', 0)
        gpu = status.get('gpu', 0)

        # 상태 바에 정보 표시
        status_text = f"FPS: {fps:.1f} | CPU: {cpu:.1f}% | MEM: {memory:.1f}%"
        if gpu > 0:
            status_text += f" | GPU: {gpu:.1f}%"

        # 현재 사용 중인 탐지기 정보
        detector_info = self.detector_manager.get_current_detector_info()
        if detector_info.get("status") == "설정됨":
            detector_type = detector_info.get("type", "")
            version = detector_info.get("version", "")
            model = detector_info.get("model", "")

            if detector_type:
                status_text += f" | 탐지기: {detector_type}"
                if version:
                    status_text += f" {version}"
                if model:
                    status_text += f" ({model})"

        self.statusBar.showMessage(status_text)

    def update_detection_result(self, result_text):
        """객체 탐지 결과 업데이트"""
        self.results_text.setText(result_text)

    def on_camera_connected(self, camera_name, resolution):
        """카메라 연결 성공 시 처리"""
        self.statusBar.showMessage(f"카메라 '{camera_name}' 연결됨, 해상도: {resolution}")
        # 모델 선택기 활성화
        self.model_selector.enable_detection_controls(True)

    def on_camera_disconnected(self):
        """카메라 연결 해제 시 처리"""
        self.video_frame.clear()
        self.video_frame.setText("카메라가 연결되지 않았습니다.")
        self.results_text.setText("탐지된 객체가 없습니다.")

        # 탐지기도 비활성화
        self.detector_manager.set_detector(None)
        self.model_selector.detection_checkbox.setChecked(False)
        self.model_selector.enable_detection_controls(False)

        self.statusBar.showMessage("카메라 연결 해제됨")

    def on_detector_activated(self, detector_type, version, model, params):
        """탐지기 활성화 시 처리"""
        self.statusBar.showMessage(f"탐지기 활성화: {detector_type} {version} {model}")

    def on_detector_deactivated(self):
        """탐지기 비활성화 시 처리"""
        self.results_text.setText("탐지된 객체가 없습니다.")