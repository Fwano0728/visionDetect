#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class MainWindowUpdates:
    """메인 윈도우 UI 업데이트 담당 클래스"""

    def __init__(self, main_window):
        self.main_window = main_window

    def update_frame(self, frame):
        """카메라 프레임 업데이트"""
        try:
            if frame is None:
                self.main_window.error_handler.log_error("받은 프레임이 None입니다.")
                return

            # 객체 탐지 실행 (탐지 매니저에 프레임 전달)
            if self.main_window.detector_manager.is_active():
                frame, detections, result_text = self.main_window.detector_manager.detect(frame)
                self.update_detection_result(result_text)

            # OpenCV BGR 형식을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape

            # Qt 이미지 변환
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 라벨에 이미지 표시 (비율 유지하며 크기 조정)
            pixmap = QPixmap.fromImage(qt_image)

            # 비디오 프레임 위젯의 실제 크기
            frame_width = self.main_window.videoFrame.width()
            frame_height = self.main_window.videoFrame.height()

            # 이미지 크기 조정 (비율 유지)
            scaled_pixmap = pixmap.scaled(frame_width, frame_height,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)

            self.main_window.videoFrame.setPixmap(scaled_pixmap)
            self.main_window.videoFrame.setAlignment(Qt.AlignCenter)
        except Exception as e:
            self.main_window.error_handler.log_error(f"프레임 업데이트 오류: {str(e)}")

    def update_fps(self, fps):
        """FPS 업데이트 처리"""
        # 사이드바의 FPS 레이블 업데이트
        try:
            sidebar_widget = self.main_window.sidebar_manager.sidebar_widget
            if hasattr(sidebar_widget, 'previewFpsLabel'):
                sidebar_widget.previewFpsLabel.setText(f"FPS: {fps:.1f}")
        except AttributeError:
            # 아직 사이드바가 초기화되지 않은 경우
            pass

    def update_detection_result(self, result_text):
        """객체 탐지 결과 업데이트"""
        self.main_window.resultsText.setText(result_text)

    def update_system_status(self, status=None):
        """시스템 상태 정보 업데이트"""
        if status is None and not hasattr(self.main_window.camera_manager, 'get_system_load'):
            return

        if status is None:
            system_load = self.main_window.camera_manager.get_system_load()
            fps = getattr(self.main_window.camera_manager, 'current_fps', 0)
            cpu = system_load.get('cpu', 0)
            memory = system_load.get('memory', 0)
            gpu = system_load.get('gpu', 0)
        else:
            fps = status.get('fps', 0)
            cpu = status.get('cpu', 0)
            memory = status.get('memory', 0)
            gpu = status.get('gpu', 0)

        # 상태 바에 정보 표시
        status_text = f"FPS: {fps:.1f} | CPU: {cpu:.1f}% | MEM: {memory:.1f}%"
        if gpu > 0:
            status_text += f" | GPU: {gpu:.1f}%"

        # 현재 사용 중인 탐지기 정보
        detector_info = self.main_window.detector_manager.get_current_detector_info()
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

        self.main_window.statusBar.showMessage(status_text)