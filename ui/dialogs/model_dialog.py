#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import uic

# UI 파일 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "ui", "ui_files")
MODEL_DIALOG_UI = os.path.join(UI_DIR, "model_dialog.ui")


class ModelDialog(QDialog):
    """모델 선택 및 설정 다이얼로그"""

    # 시그널 정의
    detection_toggled = pyqtSignal(bool)  # 탐지 활성화 상태
    detector_settings_changed = pyqtSignal(str, str, str, dict)  # 탐지기 유형, 버전, 모델, 매개변수

    def __init__(self, detector_manager, parent=None):
        # 부모를 None으로 설정하여 독립적인 창으로 작동
        super().__init__(None)

        # 윈도우 플래그 설정
        self.setWindowFlags(Qt.Window)
        self.setWindowModality(Qt.NonModal)

        # UI 파일 로드
        if os.path.exists(MODEL_DIALOG_UI):
            uic.loadUi(MODEL_DIALOG_UI, self)
        else:
            raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {MODEL_DIALOG_UI}")

        self.detector_manager = detector_manager
        self.parent_window = parent  # 부모 창 참조 저장

        # 시그널 연결
        self.detectionCheckBox.stateChanged.connect(self.on_detection_toggled)
        self.modeCombo.currentIndexChanged.connect(self.on_detection_mode_changed)
        self.versionCombo.currentIndexChanged.connect(self.on_yolo_version_changed)
        self.confidenceSlider.valueChanged.connect(self.on_confidence_changed)
        self.applyButton.clicked.connect(self.apply_detector_settings)
        self.closeButton.clicked.connect(self.close)

        # 부모 윈도우 핸들러와 시그널 연결
        if self.parent_window:
            self.detection_toggled.connect(self.parent_window.handlers.on_detection_toggled)
            self.detector_settings_changed.connect(self.parent_window.handlers.on_detector_settings_changed)

        # 초기 값 설정
        self.on_detection_mode_changed(self.modeCombo.currentIndex())
        self.setup_yolo_settings()

        # 현재 탐지기 정보 업데이트
        self.update_detector_info()

    def on_detection_toggled(self, state):
        """탐지 활성화 상태 변경"""
        enabled = (state == Qt.Checked)

        # UI 요소 활성화/비활성화
        self.modeCombo.setEnabled(enabled)
        self.applyButton.setEnabled(enabled)

        # YOLO 설정 그룹 표시 여부
        mode = self.modeCombo.currentText()
        self.yoloGroup.setVisible(enabled and mode == "YOLO")

        # 시그널 발생
        self.detection_toggled.emit(enabled)

    def on_detection_mode_changed(self, index):
        """탐지 모드 변경"""
        if index < 0:
            return

        mode = self.modeCombo.currentText()

        # YOLO 설정 그룹 표시 여부
        is_yolo = (mode == "YOLO")
        is_enabled = self.detectionCheckBox.isChecked()
        self.yoloGroup.setVisible(is_yolo and is_enabled)

        if is_yolo:
            # YOLO 설정 업데이트
            self.setup_yolo_settings()

    def on_yolo_version_changed(self, index):
        """YOLO 버전 변경"""
        if index < 0:
            return

        version = self.versionCombo.currentText()
        print(f"선택된 YOLO 버전: {version}")  # 디버깅용

        # 모델 콤보박스 업데이트
        try:
            models = self.detector_manager.get_available_models("YOLO", version)
            print(f"가져온 YOLO 모델: {models}")  # 디버깅용
        except Exception as e:
            print(f"모델 목록 가져오기 오류: {str(e)}")
            models = []

        # 모델이 없으면 기본값 추가
        if not models:
            models = ["YOLOv4-tiny", "YOLOv4"]
            print("YOLO 모델을 찾을 수 없어 기본값을 사용합니다.")

        self.modelCombo.clear()
        self.modelCombo.addItems(models)

    def on_confidence_changed(self, value):
        """신뢰도 슬라이더 값 변경"""
        confidence = value / 100.0
        self.confidenceValueLabel.setText(f"{confidence:.2f}")

    def setup_yolo_settings(self):
        """YOLO 설정 초기화"""
        # YOLO 버전 콤보박스 업데이트
        versions = self.detector_manager.get_available_versions("YOLO")
        print(f"가져온 YOLO 버전: {versions}")  # 디버깅용
        current_version = self.versionCombo.currentText() if self.versionCombo.count() > 0 else ""

        # 버전이 없으면 기본값 추가
        if not versions:
            versions = ["v4"]
            print("YOLO 버전을 찾을 수 없어 기본값 'v4'를 사용합니다.")

        self.versionCombo.clear()
        self.versionCombo.addItems(versions)

        # 이전 버전 선택 복원
        if current_version and current_version in versions:
            index = self.versionCombo.findText(current_version)
            if index >= 0:
                self.versionCombo.setCurrentIndex(index)
        elif versions and versions[0] != "버전을 찾을 수 없음":
            # 첫 번째 버전 선택 시 모델 업데이트
            self.on_yolo_version_changed(0)

    def apply_detector_settings(self):
        """탐지 설정 적용"""
        # 설정 파라미터 수집
        detection_mode = self.modeCombo.currentText()
        params = {}

        if detection_mode == "YOLO":
            yolo_version = self.versionCombo.currentText()
            yolo_model = self.modelCombo.currentText()

            # 유효한 모델인지 확인
            if yolo_version == "버전을 찾을 수 없음" or yolo_model == "모델을 찾을 수 없음":
                QMessageBox.warning(self, "설정 오류", "유효한 YOLO 버전과 모델을 선택해주세요.")
                return

            confidence = self.confidenceSlider.value() / 100.0
            input_size = self.inputSizeCombo.currentText()  # 입력 크기 가져오기

            params = {
                'conf_threshold': confidence,
                'nms_threshold': 0.4,
                'input_size': input_size  # 입력 크기 파라미터 추가
            }

            # 시그널 발생
            self.detector_settings_changed.emit(
                detection_mode, yolo_version, yolo_model, params)
        else:
            # OpenCV-HOG 등 다른 탐지기
            self.detector_settings_changed.emit(detection_mode, "", "", {})

        # 설정 저장
        self.save_detector_settings(detection_mode, params)

        # 다이얼로그 닫기
        self.close()

    def update_detector_info(self):
        """탐지기 정보 업데이트"""
        info = self.detector_manager.get_current_detector_info()

        if info.get("status") == "미설정":
            self.currentSettingsLabel.setText("탐지기가 설정되지 않았습니다.")
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

        self.currentSettingsLabel.setText(text)

        # 체크박스 상태 업데이트
        self.detectionCheckBox.setChecked(detector_type != "")

        # 모드 콤보박스 업데이트
        if detector_type:
            index = self.modeCombo.findText(detector_type)
            if index >= 0:
                self.modeCombo.setCurrentIndex(index)

    def save_detector_settings(self, detector_type, params):
        """탐지기 설정 저장"""
        # 설정 파일 디렉토리 확인
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "config")
        os.makedirs(config_dir, exist_ok=True)

        # 설정 파일 경로
        config_file = os.path.join(config_dir, "detector_settings.json")

        # 설정 저장
        settings = {
            "detector_type": detector_type
        }

        if detector_type == "YOLO":
            settings.update({
                "yolo_version": self.versionCombo.currentText(),
                "yolo_model": self.modelCombo.currentText(),
                "confidence": params.get('conf_threshold', 0.4),
                "nms_threshold": params.get('nms_threshold', 0.4),
                "input_size": params.get('input_size', '416x416')  # 입력 크기 저장
            })

        # JSON으로 저장
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

        print(f"탐지기 설정 저장 완료: {config_file}")

    @staticmethod
    def load_last_settings():
        """최근 저장된 탐지기 설정 로드"""
        # 설정 파일 경로
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "config")
        config_file = os.path.join(config_dir, "detector_settings.txt")

        # 기본 설정
        settings = {
            'detector_type': '',
            'yolo_version': '',
            'yolo_model': '',
            'confidence': 0.4,
            'nms_threshold': 0.4
        }

        # 설정 파일이 있으면 로드
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key in ['confidence', 'nms_threshold']:
                                settings[key] = float(value)
                            else:
                                settings[key] = value
            except Exception as e:
                print(f"설정 로드 중 오류 발생: {str(e)}")

        return settings