#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
메인 윈도우 이벤트 핸들러 모듈
"""

import os
import logging
import time
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage

from ui.error_handling import ErrorHandler
#from ui.dashboard.dashboard_widget import DashboardWidget
from PyQt5.QtWidgets import QMessageBox


logger = logging.getLogger(__name__)


class MainWindowHandlers:
    """메인 윈도우 이벤트 처리 담당 클래스"""

    def __init__(self, main_window):
        """
        초기화

        Args:
            main_window: 메인 윈도우 인스턴스
        """
        self.window = main_window
        self.ui = main_window.ui
        self.dashboard = None
        self.model_settings_dialog = None
        self.system_settings_dialog = None

    def connect_signals(self):
        """이벤트 시그널 연결"""
        try:
            # 카메라 제어 연결
            self.ui.refreshCameraButton.clicked.connect(self.on_refresh_cameras)
            self.ui.connectButton.clicked.connect(self.on_connect_camera)
            self.ui.disconnectButton.clicked.connect(self.on_disconnect_camera)

            # 탐지 제어 연결
            self.ui.detectionCheckBox.stateChanged.connect(self.on_detection_toggled)
            self.ui.detectionModeComboBox.currentIndexChanged.connect(self.on_detection_mode_changed)
            self.ui.modelSettingsButton.clicked.connect(self.on_model_settings)
            self.ui.dashboardButton.clicked.connect(self.on_show_dashboard)
            self.ui.systemSettingsButton.clicked.connect(self.on_system_settings)

            # 결과 패널 버튼 연결
            self.ui.saveResultsButton.clicked.connect(self.on_save_results)
            self.ui.clearResultsButton.clicked.connect(self.on_clear_results)
            self.ui.screenshotButton.clicked.connect(self.on_take_screenshot)

            # 비디오 프레임 컨텍스트 메뉴
            self.ui.videoFrame.setContextMenuPolicy(Qt.CustomContextMenu)
            self.ui.videoFrame.customContextMenuRequested.connect(self.on_video_context_menu)

            logger.debug("이벤트 시그널 연결 완료")
        except Exception as e:
            logger.error(f"시그널 연결 중 오류: {str(e)}")
            ErrorHandler.handle_exception(self.window, e, "시그널 연결")

    # 카메라 관련 핸들러
    def on_refresh_cameras(self):
        """카메라 목록 새로고침"""
        try:
            # 카메라 매니저에서 카메라 검색
            available_cameras = self.window.camera_manager.scan_cameras()

            # 콤보박스 업데이트
            self.ui.cameraComboBox.clear()
            for camera_name in available_cameras.keys():
                self.ui.cameraComboBox.addItem(camera_name)

            camera_count = len(available_cameras)
            if camera_count > 0:
                logger.info(f"{camera_count}개의 카메라를 찾았습니다.")
                self.window.statusBar().showMessage(f"{camera_count}개의 카메라를 찾았습니다.")
            else:
                logger.warning("연결된 카메라가 없습니다.")
                self.window.statusBar().showMessage("연결된 카메라가 없습니다.")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "카메라 검색")

    def on_connect_camera(self):
        """카메라 연결"""
        try:
            if self.window.camera_manager.is_connected():
                # 이미 연결됨
                logger.info("이미 카메라가 연결되어 있습니다.")
                return

            # 카메라 정보 가져오기
            camera_name = self.ui.cameraComboBox.currentText()
            resolution = self.ui.resolutionComboBox.currentText()
            fps = self.ui.fpsSpinBox.value()

            if not camera_name:
                ErrorHandler.show_error(self.window, "오류", "카메라를 선택하세요.", "warning")
                return

            # 상태 메시지 표시
            self.window.statusBar().showMessage(f"카메라 '{camera_name}' 연결 중...")

            # 카메라 연결 시도
            success, message = self.window.camera_manager.connect_camera(
                camera_name, resolution, fps
            )

            if success:
                # 연결 성공 시 UI 업데이트
                self.ui.connectButton.setEnabled(False)
                self.ui.disconnectButton.setEnabled(True)
                self.ui.detectionCheckBox.setEnabled(True)

                # 마지막 연결 카메라 정보 저장
                self.window.last_connected_camera = {
                    "name": camera_name,
                    "resolution": resolution,
                    "fps": fps
                }

                self.window.statusBar().showMessage(f"카메라 '{camera_name}' 연결됨, 해상도: {resolution}, FPS: {fps}")
                logger.info(f"카메라 '{camera_name}' 연결됨, 해상도: {resolution}, FPS: {fps}")
            else:
                # 연결 실패 시 오류 표시
                ErrorHandler.show_error(self.window, "연결 실패", message, "error")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "카메라 연결")

    def on_disconnect_camera(self):
        """카메라 연결 해제"""
        try:
            # 탐지 중지
            if self.ui.detectionCheckBox.isChecked():
                self.ui.detectionCheckBox.setChecked(False)

            # 카메라 연결 해제
            self.window.camera_manager.disconnect_camera()

            # UI 업데이트
            self.ui.connectButton.setEnabled(True)
            self.ui.disconnectButton.setEnabled(False)
            self.ui.detectionCheckBox.setEnabled(False)
            self.ui.detectionModeComboBox.setEnabled(False)
            self.ui.modelSettingsButton.setEnabled(False)

            # 비디오 프레임 초기화
            self.ui.videoFrame.clear()
            self.ui.videoFrame.setText("카메라가 연결되지 않았습니다.")

            # 탐지 결과 초기화
            self.ui.resultsTextLabel.setText("탐지된 객체가 없습니다.")

            self.window.statusBar().showMessage("카메라 연결 해제됨")
            logger.info("카메라 연결 해제됨")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "카메라 연결 해제")

    # 탐지 관련 핸들러
    def on_detection_toggled(self, state):
        """객체 탐지 활성화/비활성화"""
        try:
            enabled = (state == Qt.Checked)

            # UI 요소 활성화/비활성화
            self.ui.detectionModeComboBox.setEnabled(enabled)
            self.ui.modelSettingsButton.setEnabled(enabled)

            if enabled:
                # 모델 설정 다이얼로그 표시
                self.on_model_settings()
            else:
                # 탐지 비활성화 시 탐지기 해제
                self.window.detector_manager.set_detector(None)
                self.window.statusBar().showMessage("객체 탐지가 비활성화되었습니다.")

            logger.info(f"객체 탐지 {'활성화' if enabled else '비활성화'}")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "탐지 설정 변경")

    def on_detection_mode_changed(self, index):
        """탐지 모드 변경 시 처리"""
        try:
            # 현재 선택된 모드
            mode = self.ui.detectionModeComboBox.currentText()

            # 모드 별 설정 변경
            if self.ui.detectionCheckBox.isChecked():
                # 모델 설정 다이얼로그 표시
                self.on_model_settings()

            logger.debug(f"탐지 모드 변경: {mode}")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "탐지 모드 변경")

    def on_model_settings(self):
        """모델 설정 다이얼로그 표시"""
        try:
            # 현재 탐지 모드 가져오기
            detector_type = self.ui.detectionModeComboBox.currentText()

            # 모델 설정 다이얼로그 생성 (간단한 버전)
            if not self.model_settings_dialog:
                from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
                from PyQt5.QtWidgets import QSlider, QPushButton, QGroupBox, QFormLayout

                self.model_settings_dialog = QDialog(self.window)
                self.model_settings_dialog.setWindowTitle("모델 설정")
                self.model_settings_dialog.resize(500, 300)

                # 메인 레이아웃
                layout = QVBoxLayout(self.model_settings_dialog)

                # YOLO 설정 그룹
                self.yolo_group = QGroupBox("YOLO 설정")
                yolo_layout = QFormLayout()

                # YOLO 버전 선택
                self.yolo_version_combo = QComboBox()
                yolo_layout.addRow("버전:", self.yolo_version_combo)

                # YOLO 모델 선택
                self.yolo_model_combo = QComboBox()
                yolo_layout.addRow("모델:", self.yolo_model_combo)

                # 신뢰도 임계값
                self.confidence_slider = QSlider(Qt.Horizontal)
                self.confidence_slider.setRange(1, 99)
                self.confidence_slider.setValue(50)
                self.confidence_value_label = QLabel("0.50")

                confidence_layout = QHBoxLayout()
                confidence_layout.addWidget(self.confidence_slider)
                confidence_layout.addWidget(self.confidence_value_label)
                yolo_layout.addRow("신뢰도 임계값:", confidence_layout)

                self.yolo_group.setLayout(yolo_layout)
                layout.addWidget(self.yolo_group)

                # HOG 설정 그룹
                self.hog_group = QGroupBox("HOG 설정")
                hog_layout = QFormLayout()

                # HOG에는 설정 옵션이 없음
                hog_label = QLabel("OpenCV HOG에는 추가 설정이 없습니다.")
                hog_layout.addRow(hog_label)

                self.hog_group.setLayout(hog_layout)
                layout.addWidget(self.hog_group)

                # 버튼
                button_layout = QHBoxLayout()

                self.apply_button = QPushButton("적용")
                self.apply_button.clicked.connect(self.on_apply_model_settings)
                button_layout.addWidget(self.apply_button)

                self.cancel_button = QPushButton("취소")
                self.cancel_button.clicked.connect(self.model_settings_dialog.reject)
                button_layout.addWidget(self.cancel_button)

                layout.addLayout(button_layout)

                # 시그널 연결
                self.yolo_version_combo.currentIndexChanged.connect(self.on_yolo_version_changed)
                self.confidence_slider.valueChanged.connect(self.on_confidence_changed)

            # 다이얼로그 초기화
            if detector_type == "YOLO":
                self.yolo_group.setVisible(True)
                self.hog_group.setVisible(False)

                # YOLO 버전 초기화
                self.yolo_version_combo.clear()
                versions = self.window.detector_manager.get_available_versions("YOLO")
                self.yolo_version_combo.addItems(versions)

                # 버전 선택 시 모델 목록 업데이트
                if versions:
                    self.on_yolo_version_changed(0)
            else:
                # HOG 설정
                self.yolo_group.setVisible(False)
                self.hog_group.setVisible(True)

            # 다이얼로그 표시
            result = self.model_settings_dialog.exec_()

            # 취소 버튼 클릭 시
            if result == QDialog.Rejected:
                # 탐지가 활성화되어 있지만 설정이 없는 경우, 탐지 비활성화
                if self.ui.detectionCheckBox.isChecked() and not self.window.detector_manager.is_active():
                    self.ui.detectionCheckBox.setChecked(False)

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "모델 설정")

    def on_yolo_version_changed(self, index):
        """YOLO 버전 변경 시 모델 목록 업데이트"""
        try:
            if index < 0:
                return

            self.yolo_model_combo.clear()

            # 현재 선택된 버전
            version = self.yolo_version_combo.currentText()

            # 해당 버전의 모델 목록 가져오기
            models = self.window.detector_manager.get_available_models("YOLO", version)

            if models:
                self.yolo_model_combo.addItems(models)
            else:
                logger.warning(f"YOLO {version}에 사용 가능한 모델이 없습니다.")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "YOLO 모델 로드")

    def on_confidence_changed(self, value):
        """신뢰도 슬라이더 값 변경 시 처리"""
        # 슬라이더 값을 0.0-1.0 범위로 변환
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")

    def on_apply_model_settings(self):
        """모델 설정 적용"""
        try:
            # 현재 모드
            detector_type = self.ui.detectionModeComboBox.currentText()
            params = {}

            if detector_type == "YOLO":
                # YOLO 설정
                version = self.yolo_version_combo.currentText()
                model = self.yolo_model_combo.currentText()
                conf_threshold = self.confidence_slider.value() / 100.0

                params['conf_threshold'] = conf_threshold
                params['nms_threshold'] = 0.4  # 기본값

                # 탐지기 설정
                success, message = self.window.detector_manager.set_detector(
                    detector_type, version, model, **params
                )

                # 마지막 탐지기 설정 저장
                if success:
                    self.window.last_detector_settings = {
                        "type": detector_type,
                        "version": version,
                        "model": model,
                        "confidence": conf_threshold
                    }
            else:
                # HOG 등 다른 탐지기
                success, message = self.window.detector_manager.set_detector(detector_type)

                # 마지막 탐지기 설정 저장
                if success:
                    self.window.last_detector_settings = {
                        "type": detector_type,
                        "version": "Default",
                        "model": "Default"
                    }

            # 결과 처리
            if success:
                self.window.statusBar().showMessage(message)
                logger.info(f"탐지기 설정 완료: {message}")
                self.model_settings_dialog.accept()
            else:
                ErrorHandler.show_error(self.window, "탐지기 설정 실패", message, "error")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "탐지기 설정")

    def on_system_settings(self):
        """시스템 설정 다이얼로그 표시"""
        try:
            # 시스템 설정 다이얼로그 생성 (Ubuntu 환경에 맞게 조정)
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
            from PyQt5.QtWidgets import QCheckBox, QPushButton, QGroupBox, QFormLayout, QSpinBox

            if not self.system_settings_dialog:
                self.system_settings_dialog = QDialog(self.window)
                self.system_settings_dialog.setWindowTitle("시스템 설정")
                self.system_settings_dialog.resize(500, 300)

                # 메인 레이아웃
                layout = QVBoxLayout(self.system_settings_dialog)

                # 설정 그룹
                settings_group = QGroupBox("일반 설정")
                settings_layout = QFormLayout()

                # 저장 경로
                self.save_path_edit = QLineEdit()
                self.save_path_edit.setText(self.window.init_manager.save_path)

                path_layout = QHBoxLayout()
                path_layout.addWidget(self.save_path_edit)

                browse_button = QPushButton("찾아보기...")
                browse_button.clicked.connect(self.on_browse_save_path)
                path_layout.addWidget(browse_button)

                settings_layout.addRow("저장 경로:", path_layout)

                # 로그 보관 일수
                self.log_retention_spin = QSpinBox()
                self.log_retention_spin.setRange(1, 365)
                self.log_retention_spin.setValue(30)
                settings_layout.addRow("로그 보관 일수:", self.log_retention_spin)

                # 자동 연결 설정
                self.auto_connect_check = QCheckBox("프로그램 시작 시 자동으로 카메라 연결")
                self.auto_connect_check.setChecked(self.window.init_manager.auto_connect_camera)
                settings_layout.addRow(self.auto_connect_check)

                # 최소화 시작
                self.start_minimized_check = QCheckBox("최소화 상태로 시작")
                self.start_minimized_check.setChecked(self.window.init_manager.start_minimized)
                settings_layout.addRow(self.start_minimized_check)

                settings_group.setLayout(settings_layout)
                layout.addWidget(settings_group)

                # Ubuntu 특화 설정
                ubuntu_group = QGroupBox("Linux 설정")
                ubuntu_layout = QFormLayout()

                # 여기에 Linux/Ubuntu 특화 설정 추가 가능
                # 예: 카메라 권한, 자동 시작 등

                ubuntu_group.setLayout(ubuntu_layout)
                layout.addWidget(ubuntu_group)

                # 버튼
                button_layout = QHBoxLayout()

                save_button = QPushButton("저장")
                save_button.clicked.connect(self.on_save_system_settings)
                button_layout.addWidget(save_button)

                cancel_button = QPushButton("취소")
                cancel_button.clicked.connect(self.system_settings_dialog.reject)
                button_layout.addWidget(cancel_button)

                layout.addLayout(button_layout)

            # 다이얼로그 초기화
            self.save_path_edit.setText(self.window.init_manager.save_path)
            self.auto_connect_check.setChecked(self.window.init_manager.auto_connect_camera)
            self.start_minimized_check.setChecked(self.window.init_manager.start_minimized)

            # 다이얼로그 표시
            self.system_settings_dialog.exec_()

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "시스템 설정")

    def on_browse_save_path(self):
        """저장 경로 선택 다이얼로그"""
        try:
            current_path = self.save_path_edit.text()

            # 폴더 선택 다이얼로그
            directory = QFileDialog.getExistingDirectory(
                self.system_settings_dialog,
                "저장 폴더 선택",
                current_path,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )

            if directory:
                self.save_path_edit.setText(directory)

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "폴더 선택")

    def on_save_system_settings(self):
        """시스템 설정 저장"""
        try:
            # 설정 저장
            self.window.init_manager.save_path = self.save_path_edit.text()
            self.window.init_manager.auto_connect_camera = self.auto_connect_check.isChecked()
            self.window.init_manager.start_minimized = self.start_minimized_check.isChecked()

            # 설정 파일에 저장
            self.window.init_manager.save_settings()

            self.window.statusBar().showMessage("설정이 저장되었습니다.")
            logger.info("시스템 설정 저장됨")

            self.system_settings_dialog.accept()

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "설정 저장")

    def on_show_dashboard(self):
        """대시보드 창 표시"""
        try:
            # 임시 메시지
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self.window, "안내", "대시보드 기능은 아직 구현되지 않았습니다.")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "대시보드 표시")

    # 결과 관련 핸들러
    def on_save_results(self):
        """탐지 결과 저장"""
        try:
            # 저장할 내용이 없는 경우
            if self.ui.resultsTextLabel.text() == "탐지된 객체가 없습니다.":
                ErrorHandler.show_error(self.window, "알림", "저장할 탐지 결과가 없습니다.", "info")
                return

            # 저장 경로 가져오기
            save_path = self.window.init_manager.save_path
            if not save_path or not os.path.exists(save_path):
                save_path = os.path.join(os.path.expanduser("~"), "VisionDetect")
                os.makedirs(save_path, exist_ok=True)

            # 파일 이름 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_path, f"detection_results_{timestamp}.txt")

            # 결과 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"객체 탐지 결과 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.ui.resultsTextLabel.text())

            self.window.statusBar().showMessage(f"결과가 저장되었습니다: {file_path}")
            logger.info(f"탐지 결과 저장됨: {file_path}")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "결과 저장")

    def on_clear_results(self):
        """탐지 결과 지우기"""
        self.ui.resultsTextLabel.setText("탐지된 객체가 없습니다.")

    def on_take_screenshot(self):
        """현재 화면 스크린샷 저장"""
        try:
            # 카메라가 연결되지 않은 경우
            if not self.window.camera_manager.is_connected():
                ErrorHandler.show_error(self.window, "알림", "카메라가 연결되지 않았습니다.", "info")
                return

            # 현재 프레임 가져오기
            frame = self.window.current_frame
            if frame is None:
                ErrorHandler.show_error(self.window, "오류", "현재 프레임을 가져올 수 없습니다.", "warning")
                return

            # 저장 경로 가져오기
            save_path = self.window.init_manager.save_path
            if not save_path or not os.path.exists(save_path):
                save_path = os.path.join(os.path.expanduser("~"), "VisionDetect")
                os.makedirs(save_path, exist_ok=True)

            # 파일 이름 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_path, f"screenshot_{timestamp}.png")

            # 이미지 저장
            cv2.imwrite(file_path, frame)

            self.window.statusBar().showMessage(f"스크린샷이 저장되었습니다: {file_path}")
            logger.info(f"스크린샷 저장됨: {file_path}")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "스크린샷 저장")

    # 기타 핸들러
    def on_video_context_menu(self, position):
        """비디오 프레임 컨텍스트 메뉴"""
        try:
            # 메뉴 생성
            from PyQt5.QtWidgets import QMenu

            context_menu = QMenu(self.window)

            # 항목 추가
            save_action = context_menu.addAction("스크린샷 저장")

            # 카메라 연결 상태에 따른 활성화/비활성화
            is_connected = self.window.camera_manager.is_connected()
            save_action.setEnabled(is_connected)

            # 메뉴 표시 및 선택 처리
            action = context_menu.exec_(self.ui.videoFrame.mapToGlobal(position))

            if action == save_action:
                self.on_take_screenshot()

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "컨텍스트 메뉴")