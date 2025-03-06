#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
메인 윈도우 UI 업데이트 모듈
"""

import logging
import cv2
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from ui.error_handling import ErrorHandler

logger = logging.getLogger(__name__)


class MainWindowUpdates:
    """메인 윈도우 UI 업데이트 담당 클래스"""

    def __init__(self, main_window):
        """
        초기화

        Args:
            main_window: 메인 윈도우 인스턴스
        """
        self.window = main_window
        self.ui = main_window.ui

    @pyqtSlot(object)
    def update_frame(self, frame):
        """
        카메라 프레임 업데이트

        Args:
            frame (numpy.ndarray): 카메라 프레임
        """
        try:
            if frame is None:
                logger.warning("받은 프레임이 None입니다.")
                return

            # 프레임 저장 (스크린샷 등에 사용)
            self.window.current_frame = frame.copy()

            # 객체 탐지 활성화 여부에 따라 처리
            detect_enabled = self.ui.detectionCheckBox.isChecked() and self.ui.action_StartDetection.isChecked()

            if detect_enabled and self.window.detector_manager.is_active():
                # 객체 탐지 수행 (탐지 매니저에 프레임 전달)
                # 실제 탐지 결과 처리는 handle_detection_complete에서 수행
                self.window.detector_manager.detect(frame)
                # 여기서는 프레임을 바로 표시하지 않음
            else:
                # 탐지 없이 원본 프레임 표시
                self.display_frame(frame)

        except Exception as e:
            logger.error(f"프레임 업데이트 오류: {str(e)}")
            # 모든 프레임에서 예외가 발생하면 대화상자 폭탄이 생길 수 있으므로
            # 여기서는 ErrorHandler를 사용하지 않고 로그만 남김

    def display_frame(self, frame):
        """
        프레임을 UI에 표시

        Args:
            frame (numpy.ndarray): 표시할 프레임
        """
        try:
            # OpenCV BGR 형식을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape

            # Qt 이미지 변환
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 라벨에 이미지 표시 (비율 유지하며 크기 조정)
            pixmap = QPixmap.fromImage(qt_image)

            # 비디오 프레임 위젯의 실제 크기
            frame_width = self.ui.videoFrame.width()
            frame_height = self.ui.videoFrame.height()

            # 이미지 크기 조정 (비율 유지)
            scaled_pixmap = pixmap.scaled(
                frame_width,
                frame_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.ui.videoFrame.setPixmap(scaled_pixmap)
            self.ui.videoFrame.setAlignment(Qt.AlignCenter)

        except Exception as e:
            logger.error(f"프레임 표시 오류: {str(e)}")

    @pyqtSlot(float)
    def update_fps(self, fps):
        """
        FPS 업데이트 처리

        Args:
            fps (float): 현재 FPS
        """
        # FPS 정보 저장 (다른 곳에서 사용할 수 있음)
        self.window.current_fps = fps

    @pyqtSlot(dict)
    def update_system_status(self, status):
        """
        시스템 상태 정보 업데이트

        Args:
            status (dict): 시스템 상태 정보
        """
        try:
            fps = status.get('fps', 0)
            cpu = status.get('cpu', 0)
            memory = status.get('memory', 0)
            gpu = status.get('gpu', 0)

            # 상태 바에 정보 표시
            status_text = f"FPS: {fps:.1f} | CPU: {cpu:.1f}% | MEM: {memory:.1f}%"
            if gpu > 0:
                status_text += f" | GPU: {gpu:.1f}%"

            # 현재 사용 중인 탐지기 정보
            detector_info = self.window.detector_manager.get_current_detector_info()
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

            self.window.statusBar().showMessage(status_text)

        except Exception as e:
            logger.error(f"시스템 상태 업데이트 오류: {str(e)}")

    @pyqtSlot(str)
    def update_detection_result(self, result_text):
        """
        객체 탐지 결과 텍스트 업데이트

        Args:
            result_text (str): 결과 텍스트
        """
        try:
            self.ui.resultsTextLabel.setText(result_text)
        except Exception as e:
            logger.error(f"탐지 결과 업데이트 오류: {str(e)}")

    @pyqtSlot(object, list, str)
    def handle_detection_complete(self, result_frame, detections, result_text):
        """
        탐지 완료 시 처리

        Args:
            result_frame (numpy.ndarray): 탐지 결과가 표시된 프레임
            detections (list): 탐지된 객체 목록
            result_text (str): 결과 텍스트
        """
        try:
            # 결과 프레임 표시
            self.display_frame(result_frame)

            # 결과 텍스트 업데이트는 이미 update_detection_result에서 처리됨
        except Exception as e:
            logger.error(f"탐지 결과 처리 오류: {str(e)}")