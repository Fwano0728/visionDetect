#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5 import uic

# UI 파일 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(CURRENT_DIR, "ui_files")
SIDEBAR_UI = os.path.join(UI_DIR, "sidebar.ui")


class SidebarManager(QObject):
    """사이드바 관리 클래스"""

    # 시그널 정의
    camera_connect_requested = pyqtSignal(str, str, int)
    camera_disconnect_requested = pyqtSignal()
    detection_toggled = pyqtSignal(bool)
    detector_settings_changed = pyqtSignal(str, str, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.sidebar_widget = None
        self.sidebar_expanded = True

    def setup_sidebar(self, sidebar_frame, camera_manager=None, detector_manager=None):
        """사이드바 설정"""
        self.camera_manager = camera_manager
        self.detector_manager = detector_manager

        # 사이드바 UI 로드
        if not os.path.exists(SIDEBAR_UI):
            raise FileNotFoundError(f"사이드바 UI 파일을 찾을 수 없습니다: {SIDEBAR_UI}")

        # 사이드바 컨테이너에 레이아웃 설정
        layout = QVBoxLayout(sidebar_frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 사이드바 위젯 생성 및 로드
        self.sidebar_widget = QWidget()
        uic.loadUi(SIDEBAR_UI, self.sidebar_widget)
        layout.addWidget(self.sidebar_widget)

        # 시그널 연결
        self.connect_signals()

        # 초기 상태 설정
        self.refresh_cameras()
        self.setup_yolo_settings()
        self.sidebar_widget.yoloGroup.setVisible(False)

        return self.sidebar_widget

    def connect_signals(self):
        """시그널 연결"""
        sidebar = self.sidebar_widget

        # 토글 버튼
        sidebar.toggleButton.clicked.connect(self.toggle_sidebar)

        # 아이콘 버튼들
        sidebar.cameraButton.clicked.connect(lambda: self.show_page(0))
        sidebar.modelButton.clicked.connect(lambda: self.show_page(1))
        sidebar.toolsButton.clicked.connect(lambda: self.show_page(2))

        # 카메라 페이지
        sidebar.refreshButton.clicked.connect(self.refresh_cameras)
        sidebar.connectButton.clicked.connect(self.connect_camera)
        sidebar.disconnectButton.clicked.connect(self.disconnect_camera)

        # 모델 페이지
        sidebar.detectionCheckBox.stateChanged.connect(self.on_detection_toggled)
        sidebar.modeCombo.currentIndexChanged.connect(self.on_detection_mode_changed)
        sidebar.versionCombo.currentIndexChanged.connect(self.on_yolo_version_changed)
        sidebar.confidenceSlider.valueChanged.connect(self.on_confidence_changed)
        sidebar.applyButton.clicked.connect(self.apply_detector_settings)

    def show_page(self, index):
        """스택 위젯 페이지 변경"""
        sidebar = self.sidebar_widget

        # 스택 위젯 페이지 변경
        sidebar.stackedWidget.setCurrentIndex(index)

        # 버튼 선택 상태 업데이트
        sidebar.cameraButton.setChecked(index == 0)
        sidebar.modelButton.setChecked(index == 1)
        sidebar.toolsButton.setChecked(index == 2)

    def toggle_sidebar(self):
        """사이드바 토글"""
        sidebar = self.sidebar_widget

        if self.sidebar_expanded:
            # 축소
            sidebar.contentWidget.setVisible(False)
            sidebar.parent().setMaximumWidth(50)
            sidebar.toggleButton.setText("▶")
        else:
            # 확장
            sidebar.contentWidget.setVisible(True)
            sidebar.parent().setMaximumWidth(250)
            sidebar.toggleButton.setText("◀")

        self.sidebar_expanded = not self.sidebar_expanded

    def refresh_cameras(self):
        """카메라 목록 업데이트"""
        if not self.camera_manager:
            return

        sidebar = self.sidebar_widget

        # 현재 선택된 카메라 저장
        current_camera = sidebar.cameraCombo.currentText()

        # 카메라 콤보박스 업데이트
        sidebar.cameraCombo.clear()
        cameras = list(self.camera_manager.available_cameras.keys())
        sidebar.cameraCombo.addItems(cameras)

        # 이전 선택 복원
        if current_camera and current_camera in cameras:
            index = sidebar.cameraCombo.findText(current_camera)
            if index >= 0:
                sidebar.cameraCombo.setCurrentIndex(index)

        # 해상도 콤보박스 업데이트
        if not sidebar.resolutionCombo.count():
            sidebar.resolutionCombo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])

    def setup_yolo_settings(self):
        """YOLO 설정 초기화"""
        if not self.detector_manager:
            return

        sidebar = self.sidebar_widget

        # YOLO 버전 콤보박스 업데이트
        versions = self.detector_manager.get_available_versions("YOLO")
        sidebar.versionCombo.clear()
        sidebar.versionCombo.addItems(versions)

        # 첫 번째 버전 선택 시 모델 업데이트
        if versions:
            self.on_yolo_version_changed(0)

    def connect_camera(self):
        """카메라 연결"""
        sidebar = self.sidebar_widget

        camera = sidebar.cameraCombo.currentText()
        resolution = sidebar.resolutionCombo.currentText()
        fps = sidebar.fpsSpinBox.value()

        # 신호 발생 (메인 윈도우에서 처리)
        self.camera_connect_requested.emit(camera, resolution, fps)

    def disconnect_camera(self):
        """카메라 연결 해제"""
        # 신호 발생 (메인 윈도우에서 처리)
        self.camera_disconnect_requested.emit()

    def on_detection_toggled(self, state):
        """탐지 활성화 상태 변경"""
        sidebar = self.sidebar_widget
        enabled = (state == Qt.Checked)

        # UI 요소 활성화/비활성화
        sidebar.modeCombo.setEnabled(enabled)
        sidebar.applyButton.setEnabled(enabled)

        # YOLO 설정 그룹 표시 여부
        mode = sidebar.modeCombo.currentText()
        sidebar.yoloGroup.setVisible(enabled and mode == "YOLO")

        # 신호 발생
        self.detection_toggled.emit(enabled)

    def on_detection_mode_changed(self, index):
        """탐지 모드 변경"""
        sidebar = self.sidebar_widget
        mode = sidebar.modeCombo.currentText()

        # YOLO 설정 그룹 표시 여부
        is_yolo = (mode == "YOLO")
        sidebar.yoloGroup.setVisible(is_yolo and sidebar.detectionCheckBox.isChecked())

        if is_yolo:
            # YOLO 설정 업데이트
            self.setup_yolo_settings()

    def on_yolo_version_changed(self, index):
        """YOLO 버전 변경"""
        if not self.detector_manager or index < 0:
            return

        sidebar = self.sidebar_widget
        version = sidebar.versionCombo.currentText()

        # 모델 콤보박스 업데이트
        models = self.detector_manager.get_available_models("YOLO", version)
        sidebar.modelCombo.clear()
        sidebar.modelCombo.addItems(models)

    def on_confidence_changed(self, value):
        """신뢰도 슬라이더 값 변경"""
        sidebar = self.sidebar_widget
        confidence = value / 100.0
        sidebar.confidenceValueLabel.setText(f"{confidence:.2f}")

    def apply_detector_settings(self):
        """탐지 설정 적용"""
        sidebar = self.sidebar_widget

        # 설정 파라미터 수집
        detection_mode = sidebar.modeCombo.currentText()
        params = {}

        if detection_mode == "YOLO":
            yolo_version = sidebar.versionCombo.currentText()
            yolo_model = sidebar.modelCombo.currentText()
            confidence = sidebar.confidenceSlider.value() / 100.0

            params = {
                'conf_threshold': confidence,
                'nms_threshold': 0.4
            }

            # 신호 발생
            self.detector_settings_changed.emit(
                detection_mode, yolo_version, yolo_model, params)
        else:
            # OpenCV-HOG 등 다른 탐지기
            self.detector_settings_changed.emit(detection_mode, "", "", {})

    def update_camera_state(self, connected, camera_name=None, resolution=None):
        """카메라 연결 상태 업데이트"""
        sidebar = self.sidebar_widget

        sidebar.connectButton.setEnabled(not connected)
        sidebar.disconnectButton.setEnabled(connected)
        sidebar.detectionCheckBox.setEnabled(connected)

    def update_detector_info(self, info):
        """탐지기 정보 업데이트"""
        sidebar = self.sidebar_widget

        if info.get("status") == "미설정":
            sidebar.currentSettingsLabel.setText("탐지기가 설정되지 않았습니다.")
            return

        # 정보 텍스트 구성
        detector_type = info.get("type", "")
        version = info.get("version", "")
        model = info.get("model", "")
        params = info.get("params", {})

        text = f"탐지기: {detector_type}"
        if version:
            text += f"\n버전: {version}"
        if model:
            text += f"\n모델: {model}"

        # 매개변수 표시
        if params:
            text += "\n매개변수:"
            for key, value in params.items():
                text += f"\n - {key}: {value}"

        sidebar.currentSettingsLabel.setText(text)