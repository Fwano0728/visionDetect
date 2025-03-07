#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage


class MainWindowUpdates:
    """메인 윈도우 UI 업데이트 담당 클래스"""

    def __init__(self, main_window):
        self.main_window = main_window

        # 프레임 처리 관련 변수
        self.frame_count = 0
        self.last_frame = None

        # FPS 관련 변수
        self.camera_fps = 0.0
        self.detection_fps = 0.0

        # 화면 FPS 측정 변수
        self.display_fps = 0.0
        self.display_frame_count = 0
        self.display_fps_timer = time.time()

        # UI 업데이트 관련 변수
        self.last_ui_update_time = time.time()
        self.ui_update_interval = 1.0 / 10.0  # 10fps로 제한 (0.1초마다 업데이트)

        # 상태 표시 관련 변수
        self.last_status_update_time = time.time()
        self.status_update_interval = 1.0  # 상태바 1초마다 업데이트

        # 별도의 QTimer를 통한 정기적인 화면 업데이트
        self.update_timer = QTimer(main_window)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 100ms = 0.1초 = 10FPS

    def update_display(self):
        """타이머에 의한 정기적인 화면 업데이트"""
        try:
            # 탐지기 활성화 확인
            if not hasattr(self.main_window, 'detector_manager'):
                return

            # 현재 처리된 프레임 가져오기
            detector_info = self.main_window.detector_manager.get_current_detector_info()

            if self.main_window.detector_manager.is_active() and "result_frame" in detector_info:
                # 최신 탐지 결과 프레임 표시
                self._display_frame(detector_info["result_frame"])
            elif self.last_frame is not None:
                # 탐지기가 비활성화된 경우 원본 프레임 표시
                self._display_frame(self.last_frame)

            # 화면 FPS 측정
            self.display_frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.display_fps_timer

            # 1초마다 FPS 계산
            if elapsed >= 1.0:
                self.display_fps = self.display_frame_count / elapsed
                self.display_frame_count = 0
                self.display_fps_timer = current_time

            # 상태 정보 주기적 업데이트
            if current_time - self.last_status_update_time >= self.status_update_interval:
                self.last_status_update_time = current_time
                self.update_system_status()

        except Exception as e:
            if hasattr(self.main_window, 'error_handler'):
                self.main_window.error_handler.log_error(f"화면 업데이트 오류: {str(e)}")
            print(f"화면 업데이트 오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_frame(self, frame):
        """카메라 프레임 업데이트 및 탐지 처리 (화면 갱신 없음)"""
        try:
            if frame is None:
                return

            # 최신 프레임 저장
            self.last_frame = frame.copy()  # 여기서는 복사 필요 (참조용)

            # 프레임 카운트 증가
            self.frame_count += 1

            # 탐지기가 활성화된 경우 프레임 전달
            if hasattr(self.main_window, 'detector_manager') and self.main_window.detector_manager.is_active():
                # 탐지 처리를 위해 프레임 전달만 수행 (화면 갱신은 타이머에서 처리)
                self.main_window.detector_manager.process_frame(frame)

                # 탐지 FPS 업데이트 (표시는 하지 않음)
                if hasattr(self.main_window.detector_manager, 'get_detection_fps'):
                    self.detection_fps = self.main_window.detector_manager.get_detection_fps()

        except Exception as e:
            if hasattr(self.main_window, 'error_handler'):
                self.main_window.error_handler.log_error(f"프레임 업데이트 오류: {str(e)}")
            print(f"프레임 업데이트 오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def _display_frame(self, frame):
        """화면에 프레임 표시 (최적화된 방식)"""
        try:
            if not hasattr(self.main_window, 'videoFrame'):
                return

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

            # 이미지 크기 조정 (비율 유지, 빠른 변환 사용)
            scaled_pixmap = pixmap.scaled(frame_width, frame_height,
                                          Qt.KeepAspectRatio,
                                          Qt.FastTransformation)

            # 프레임 표시
            self.main_window.videoFrame.setPixmap(scaled_pixmap)
            self.main_window.videoFrame.setAlignment(Qt.AlignCenter)

        except Exception as e:
            if hasattr(self.main_window, 'error_handler'):
                self.main_window.error_handler.log_error(f"프레임 표시 오류: {str(e)}")
            print(f"프레임 표시 오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_fps(self, fps):
        """카메라 FPS 업데이트 처리"""
        self.camera_fps = fps

        # 사이드바 업데이트 (있는 경우)
        try:
            if hasattr(self.main_window, 'sidebar_manager') and self.main_window.sidebar_manager is not None:
                sidebar_widget = self.main_window.sidebar_manager.sidebar_widget
                if hasattr(sidebar_widget, 'previewFpsLabel'):
                    sidebar_widget.previewFpsLabel.setText(f"카메라 FPS: {fps:.1f}")
        except AttributeError:
            pass

    def update_detection_result(self, result_text):
        """객체 탐지 결과 텍스트 업데이트"""
        if hasattr(self.main_window, 'resultsText'):
            self.main_window.resultsText.setText(result_text)

        # 시각화 위젯 업데이트 (있는 경우)
        if hasattr(self.main_window, 'results_visualizer'):
            if hasattr(self.main_window, 'detector_manager'):
                detections = self.main_window.detector_manager.get_current_detector_info().get("last_detections", [])
                self.main_window.results_visualizer.update_results(detections, result_text)
                self.main_window.results_visualizer.add_to_history()

    def update_system_status(self, status=None):
        """시스템 상태 정보 업데이트 (CPU, 메모리, GPU 사용량 등)"""
        if (status is None and not hasattr(self.main_window, 'camera_manager') or
                not hasattr(self.main_window.camera_manager, 'get_system_load')):
            return

        # 시스템 자원 사용량 정보 가져오기
        if status is None:
            system_load = self.main_window.camera_manager.get_system_load()
            cpu = system_load.get('cpu', 0)
            memory = system_load.get('memory', 0)
            gpu = system_load.get('gpu', 0)
        else:
            cpu = status.get('cpu', 0)
            memory = status.get('memory', 0)
            gpu = status.get('gpu', 0)

        # 탐지 FPS 가져오기
        if hasattr(self.main_window, 'detector_manager') and hasattr(self.main_window.detector_manager,
                                                                     'get_detection_fps'):
            self.detection_fps = self.main_window.detector_manager.get_detection_fps()

        # 상태 바에 정보 표시 - 카메라, 처리, 화면 FPS 모두 표시
        process_type = "GPU" if gpu > 0 else "CPU"
        status_text = f"카메라: {self.camera_fps:.1f} FPS | 처리: {self.detection_fps:.1f} FPS ({process_type}) | 화면: {self.display_fps:.1f} FPS | CPU: {cpu:.1f}% | MEM: {memory:.1f}%"

        # GPU 정보가 있으면 추가
        if gpu > 0:
            status_text += f" | GPU: {gpu:.1f}%"

        # 현재 사용 중인 탐지기 정보
        if hasattr(self.main_window, 'detector_manager'):
            detector_info = self.main_window.detector_manager.get_current_detector_info()
            if detector_info.get("status") == "설정됨":
                detector_type = detector_info.get("type", "")
                version = detector_info.get("version", "")
                model = detector_info.get("model", "")

                # 입력 크기 정보 추가 (있는 경우)
                input_size = ""
                if "params" in detector_info and "input_size" in detector_info["params"]:
                    input_size = f", 입력: {detector_info['params']['input_size']}"

                if detector_type:
                    status_text += f" | 탐지기: {detector_type}"
                    if version:
                        status_text += f" {version}"
                    if model:
                        status_text += f" ({model}{input_size})"

        # 상태 바에 표시
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar.showMessage(status_text)