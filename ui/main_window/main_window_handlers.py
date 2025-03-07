#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/main_window/main_window_handlers.py

import os
import sys
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt


class MainWindowHandlers:
    """메인 윈도우 이벤트 핸들러 담당 클래스"""

    def __init__(self, main_window):
        self.main_window = main_window

    def on_camera_connect_requested(self, camera, resolution, fps=30):
        """카메라 연결 요청 처리"""
        # 연결 시도
        success, message = self.main_window.camera_manager.connect_camera(camera, resolution)

        if success:
            # UI 상태 업데이트
            self.main_window.update_ui_state(camera_connected=True)
            self.main_window.statusBar.showMessage(f"카메라 '{camera}' 연결됨, 해상도: {resolution}, FPS: {fps}")
        else:
            self.main_window.error_handler.show_warning(f"카메라 연결 실패: {message}")

    def on_camera_disconnect_requested(self):
        """카메라 연결 해제 요청 처리"""
        self.main_window.camera_manager.disconnect_camera()

        # 비디오 프레임 초기화
        self.main_window.videoFrame.clear()
        self.main_window.videoFrame.setText("카메라가 연결되지 않았습니다.")
        self.main_window.resultsText.setText("탐지된 객체가 없습니다.")

        # 탐지기도 비활성화
        self.main_window.detector_manager.set_detector(None)

        # UI 상태 업데이트
        self.main_window.update_ui_state(camera_connected=False, detection_active=False)
        self.main_window.statusBar.showMessage("카메라 연결 해제됨")

    def on_detection_toggled(self, enabled):
        """탐지 활성화/비활성화 처리"""
        if not enabled:
            # 탐지 비활성화
            self.main_window.detector_manager.set_detector(None)
            self.main_window.resultsText.setText("탐지된 객체가 없습니다.")
            self.main_window.statusBar.showMessage("객체 탐지 비활성화됨")
        else:
            # 활성화 시 현재 설정된 탐지기 확인
            detector_info = self.main_window.detector_manager.get_current_detector_info()
            if detector_info.get("status") == "미설정":
                # 탐지기가 설정되지 않은 경우 모델 다이얼로그 열기
                self.main_window.show_model_dialog()
            else:
                self.main_window.statusBar.showMessage("객체 탐지 활성화됨")

        # UI 상태 업데이트
        self.main_window.update_ui_state(detection_active=enabled)

    def on_detector_settings_changed(self, detector_type, version, model, params):
        """탐지 설정 변경 처리"""
        # 탐지기 설정
        if detector_type == "YOLO":
            success, message = self.main_window.detector_manager.set_detector(
                detector_type, version, model, **params)
        else:
            success, message = self.main_window.detector_manager.set_detector(detector_type)

        if success:
            self.main_window.statusBar.showMessage(f"탐지기 설정 적용: {detector_type}")
        else:
            self.main_window.error_handler.show_warning(f"탐지기 설정 실패: {message}")

    def show_object_selection_dialog(self):
        """객체 선택 대화상자 표시"""
        # TODO: 객체 선택 대화상자 구현
        QMessageBox.information(self.main_window, "기능 준비 중", "객체 선택 기능은 아직 구현 중입니다.")

    def show_recording_settings_dialog(self):
        """녹화 설정 대화상자 표시"""
        # TODO: 녹화 설정 대화상자 구현
        QMessageBox.information(self.main_window, "기능 준비 중", "녹화 설정 기능은 아직 구현 중입니다.")

    def show_object_selection_dialog(self):
        """객체 선택 대화상자 표시"""
        from ui.dialogs.object_selection_dialog import ObjectSelectionDialog

        dialog = ObjectSelectionDialog(parent=self.main_window)
        dialog.settings_changed.connect(self.on_object_settings_changed)
        dialog.exec_()

    def on_object_settings_changed(self, settings):
        """객체 탐지 설정 변경 처리"""
        # 설정 적용
        self.main_window.detector_manager.set_object_detection_settings(settings)