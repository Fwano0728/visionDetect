#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QGroupBox,
                             QGridLayout, QCheckBox, QSlider)
from PyQt5.QtCore import Qt, pyqtSignal


class ModelSelector(QWidget):
    """모델 선택 및 설정 위젯"""

    # 시그널 정의
    detector_activated = pyqtSignal(str, str, str, dict)  # 탐지기 유형, 버전, 모델, 매개변수
    detector_deactivated = pyqtSignal()

    def __init__(self, detector_manager, parent=None):
        super().__init__(parent)
        self.detector_manager = detector_manager
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 객체 탐지 설정 그룹
        detection_group = QGroupBox("객체 탐지 설정")
        detection_layout = QVBoxLayout()

        # 사람 탐지 활성화 체크박스
        self.detection_checkbox = QCheckBox("사람 탐지 활성화")
        self.detection_checkbox.stateChanged.connect(self.toggle_detection)
        detection_layout.addWidget(self.detection_checkbox)

        # 탐지 모드 설정
        detection_mode_layout = QHBoxLayout()
        detection_mode_layout.addWidget(QLabel("탐지 모드:"))
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems(["OpenCV-HOG", "YOLO"])
        self.detection_mode_combo.setEnabled(False)
        self.detection_mode_combo.currentTextChanged.connect(self.on_detection_mode_changed)
        detection_mode_layout.addWidget(self.detection_mode_combo)
        detection_layout.addLayout(detection_mode_layout)

        # YOLO 설정 (처음에는 숨김)
        self.yolo_settings = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_settings)

        # YOLO 버전
        yolo_version_layout = QHBoxLayout()
        yolo_version_layout.addWidget(QLabel("YOLO 버전:"))
        self.yolo_version_combo = QComboBox()
        self.yolo_version_combo.addItems(self.detector_manager.get_available_versions("YOLO"))
        self.yolo_version_combo.currentTextChanged.connect(self.on_yolo_version_changed)
        yolo_version_layout.addWidget(self.yolo_version_combo)
        yolo_layout.addLayout(yolo_version_layout)

        # YOLO 모델
        yolo_model_layout = QHBoxLayout()
        yolo_model_layout.addWidget(QLabel("모델:"))
        self.yolo_model_combo = QComboBox()
        yolo_model_layout.addWidget(self.yolo_model_combo)
        yolo_layout.addLayout(yolo_model_layout)

        # 신뢰도 임계값 슬라이더
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("신뢰도 임계값:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 99)
        self.confidence_slider.setValue(40)  # 기본값 0.4
        self.confidence_slider.valueChanged.connect(self.update_confidence_value)
        confidence_layout.addWidget(self.confidence_slider)

        self.confidence_value_label = QLabel("0.40")
        confidence_layout.addWidget(self.confidence_value_label)
        yolo_layout.addLayout(confidence_layout)

        # YOLO 설정 초기에는 숨김
        self.yolo_settings.setVisible(False)
        detection_layout.addWidget(self.yolo_settings)

        # 적용 버튼
        self.apply_button = QPushButton("설정 적용")
        self.apply_button.clicked.connect(self.apply_detector_settings)
        self.apply_button.setEnabled(False)
        detection_layout.addWidget(self.apply_button)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # 현재 설정 정보
        current_settings_group = QGroupBox("현재 설정")
        current_settings_layout = QVBoxLayout()

        self.current_settings_label = QLabel("탐지기가 설정되지 않았습니다.")
        current_settings_layout.addWidget(self.current_settings_label)

        current_settings_group.setLayout(current_settings_layout)
        layout.addWidget(current_settings_group)

        # 초기화
        self.on_detection_mode_changed(self.detection_mode_combo.currentText())
        self.update_detector_info()

        # 처음에는 컨트롤 비활성화
        self.enable_detection_controls(False)

    def on_detection_mode_changed(self, mode):
        """탐지 모드 변경 시 처리"""
        # YOLO 설정 표시 여부
        self.yolo_settings.setVisible(mode == "YOLO")

        # YOLO 버전 목록 업데이트
        if mode == "YOLO":
            self.yolo_version_combo.clear()
            versions = self.detector_manager.get_available_versions("YOLO")
            self.yolo_version_combo.addItems(versions)

            # 첫 번째 버전 선택 및 모델 업데이트
            if versions:
                self.on_yolo_version_changed(versions[0])

    def on_yolo_version_changed(self, version):
        """YOLO 버전 변경 시 모델 목록 업데이트"""
        self.yolo_model_combo.clear()
        models = self.detector_manager.get_available_models("YOLO", version)
        self.yolo_model_combo.addItems(models)

    def update_confidence_value(self, value):
        """신뢰도 슬라이더 값 변경 시 레이블 업데이트"""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")

    def toggle_detection(self, state):
        """객체 탐지 활성화/비활성화"""
        enabled = (state == Qt.Checked)

        # UI 요소 활성화/비활성화
        self.detection_mode_combo.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)

        # YOLO 설정 표시 여부
        if enabled and self.detection_mode_combo.currentText() == "YOLO":
            self.yolo_settings.setVisible(True)
        else:
            self.yolo_settings.setVisible(False)

        # 탐지 비활성화 시 탐지기 해제
        if not enabled:
            self.detector_manager.set_detector(None)
            self.update_detector_info()
            self.detector_deactivated.emit()

    def apply_detector_settings(self):
        """탐지기 설정 적용"""
        # 매개변수 수집
        detector_type = self.detection_mode_combo.currentText()
        params = {}

        if detector_type == "YOLO":
            # YOLO 설정
            version = self.yolo_version_combo.currentText()
            model = self.yolo_model_combo.currentText()
            conf_threshold = self.confidence_slider.value() / 100.0

            params['conf_threshold'] = conf_threshold
            params['nms_threshold'] = 0.4  # 기본값

            # 탐지기 설정
            success, message = self.detector_manager.set_detector(
                detector_type, version, model, **params)
        else:
            # HOG 등 다른 탐지기
            success, message = self.detector_manager.set_detector(detector_type)

        # 결과 표시
        if success:
            self.update_detector_info()
        else:
            # 오류 처리 (실제 구현 시 메시지 박스 등으로 표시)
            print(f"탐지기 설정 실패: {message}")

        # 시그널 발생
        if detector_type == "YOLO":
            self.detector_activated.emit(
                detector_type,
                self.yolo_version_combo.currentText(),
                self.yolo_model_combo.currentText(),
                params
            )
        else:
            self.detector_activated.emit(detector_type, "", "", {})

    def update_detector_info(self):
        """현재 설정된 탐지기 정보 업데이트"""
        info = self.detector_manager.get_current_detector_info()

        if info.get("status") == "미설정":
            self.current_settings_label.setText("탐지기가 설정되지 않았습니다.")
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

        self.current_settings_label.setText(text)

    def enable_detection_controls(self, enabled):
        """탐지 컨트롤 활성화/비활성화"""
        self.detection_checkbox.setEnabled(enabled)
        # 체크박스가 체크되어 있지 않으면 다른 컨트롤은 비활성화
        should_enable = enabled and self.detection_checkbox.isChecked()
        self.detection_mode_combo.setEnabled(should_enable)
        self.apply_button.setEnabled(should_enable)
        self.yolo_settings.setEnabled(should_enable)