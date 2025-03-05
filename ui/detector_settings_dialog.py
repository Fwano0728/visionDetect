from PyQt5.QtWidgets import (QDialog, QRadioButton, QComboBox, QSlider,
                             QLabel, QPushButton, QStackedWidget, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5 import uic
import os


class DetectorSettingsDialog(QDialog):
    """탐지 설정 다이얼로그"""

    def __init__(self, detector_manager, parent=None):
        super().__init__(parent)

        # UI 파일 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, 'ui_files', 'detector_settings.ui')
        uic.loadUi(ui_file, self)

        # 창 속성 설정 (독립적인 창으로 작동)
        self.setWindowFlags(Qt.Window | Qt.WindowSystemMenuHint |
                            Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

        self.detector_manager = detector_manager

        # 현재 설정값 로드
        self.load_current_settings()

        # 시그널 연결
        self.connect_signals()

    def load_current_settings(self):
        """현재 탐지기 설정 불러오기"""
        # 현재 설정 정보 가져오기
        info = self.detector_manager.get_current_detector_info()

        # 탐지기 유형에 따라 라디오 버튼 선택
        detector_type = info.get("type", "YOLO")
        if detector_type == "OpenCV-HOG":
            self.opencvRadio.setChecked(True)
            self.settingsStack.setCurrentIndex(0)
        elif detector_type == "YOLO":
            self.yoloRadio.setChecked(True)
            self.settingsStack.setCurrentIndex(1)
        elif detector_type == "DeepStream":
            self.deepstreamRadio.setChecked(True)
            self.settingsStack.setCurrentIndex(2)

        # YOLO 설정 로드
        yolo_version = info.get("version", "v4")
        yolo_model = info.get("model", "YOLOv4-tiny")
        params = info.get("params", {})

        # YOLO 버전 선택
        index = self.yoloVersionCombo.findText(yolo_version)
        if index >= 0:
            self.yoloVersionCombo.setCurrentIndex(index)

        # 버전에 맞는 모델 목록 업데이트
        self.update_yolo_models(yolo_version)

        # YOLO 모델 선택
        index = self.yoloModelCombo.findText(yolo_model)
        if index >= 0:
            self.yoloModelCombo.setCurrentIndex(index)

        # 신뢰도 임계값 설정
        conf_threshold = params.get("conf_threshold", 0.4)
        self.yoloConfidenceSlider.setValue(int(conf_threshold * 100))
        self.yoloConfidenceLabel.setText(f"{conf_threshold:.2f}")

        # NMS 임계값 설정
        nms_threshold = params.get("nms_threshold", 0.4)
        self.yoloNmsSlider.setValue(int(nms_threshold * 100))
        self.yoloNmsLabel.setText(f"{nms_threshold:.2f}")

        # 입력 크기 설정
        input_size = params.get("input_size", "416x416")
        index = self.yoloInputSizeCombo.findText(input_size)
        if index >= 0:
            self.yoloInputSizeCombo.setCurrentIndex(index)

    def connect_signals(self):
        """시그널 연결"""
        # 라디오 버튼 상태 변경 시 스택 위젯 페이지 변경
        self.opencvRadio.toggled.connect(self.on_detector_type_changed)
        self.yoloRadio.toggled.connect(self.on_detector_type_changed)
        self.deepstreamRadio.toggled.connect(self.on_detector_type_changed)

        # YOLO 버전 변경 시 모델 목록 업데이트
        self.yoloVersionCombo.currentTextChanged.connect(self.update_yolo_models)

        # 슬라이더 값 변경 시 레이블 업데이트
        self.yoloConfidenceSlider.valueChanged.connect(self.update_confidence_label)
        self.yoloNmsSlider.valueChanged.connect(self.update_nms_label)
        self.deepstreamConfidenceSlider.valueChanged.connect(self.update_deepstream_confidence_label)

        # 객체 선택 버튼
        self.objectSelectionButton.clicked.connect(self.on_object_selection_clicked)

        # 적용/취소 버튼
        self.applyButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

    def on_detector_type_changed(self):
        """탐지기 유형 변경 시 처리"""
        if self.opencvRadio.isChecked():
            self.settingsStack.setCurrentIndex(0)
        elif self.yoloRadio.isChecked():
            self.settingsStack.setCurrentIndex(1)
        elif self.deepstreamRadio.isChecked():
            self.settingsStack.setCurrentIndex(2)

    def update_yolo_models(self, version):
        """YOLO 버전에 따라 모델 목록 업데이트"""
        # 현재 선택된 모델 저장
        current_model = self.yoloModelCombo.currentText()

        # 콤보박스 초기화
        self.yoloModelCombo.clear()

        # 버전에 따른 모델 목록 가져오기
        models = self.detector_manager.get_available_models("YOLO", version)

        if models:
            self.yoloModelCombo.addItems(models)
            # 이전에 선택된 모델이 있으면 선택 복원
            index = self.yoloModelCombo.findText(current_model)
            if index >= 0:
                self.yoloModelCombo.setCurrentIndex(index)
        else:
            # 모델이 없는 경우 기본값 추가
            if version == "v4":
                self.yoloModelCombo.addItems(["YOLOv4-tiny", "YOLOv4"])
            elif version == "v5":
                self.yoloModelCombo.addItems(["YOLOv5n", "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"])
            elif version == "v7":
                self.yoloModelCombo.addItems(["YOLOv7-tiny", "YOLOv7"])
            elif version == "v8":
                self.yoloModelCombo.addItems(["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])

    def update_confidence_label(self, value):
        """YOLO 신뢰도 임계값 레이블 업데이트"""
        confidence = value / 100.0
        self.yoloConfidenceLabel.setText(f"{confidence:.2f}")

    def update_nms_label(self, value):
        """YOLO NMS 임계값 레이블 업데이트"""
        nms = value / 100.0
        self.yoloNmsLabel.setText(f"{nms:.2f}")

    def update_deepstream_confidence_label(self, value):
        """DeepStream 신뢰도 임계값 레이블 업데이트"""
        confidence = value / 100.0
        self.deepstreamConfidenceLabel.setText(f"{confidence:.2f}")

    def on_object_selection_clicked(self):
        """객체 선택 버튼 클릭 처리"""
        # 객체 선택 다이얼로그가 구현되어 있다면 호출
        # 여기서는 부모 창의 show_object_selection 메서드를 호출
        if hasattr(self.parent(), 'show_object_selection'):
            self.parent().show_object_selection()

    def get_detector_settings(self):
        """현재 설정된 탐지기 설정 반환"""
        settings = {}

        # 탐지기 유형
        if self.opencvRadio.isChecked():
            settings['detector_type'] = "OpenCV-HOG"
            settings['version'] = ""
            settings['model'] = self.opencvModelCombo.currentText()
            settings['params'] = {
                'scale': self.opencvScaleSpinBox.value(),
                'min_neighbors': self.opencvMinNeighborsSpinBox.value()
            }
        elif self.yoloRadio.isChecked():
            settings['detector_type'] = "YOLO"
            settings['version'] = self.yoloVersionCombo.currentText()
            settings['model'] = self.yoloModelCombo.currentText()
            settings['params'] = {
                'conf_threshold': self.yoloConfidenceSlider.value() / 100.0,
                'nms_threshold': self.yoloNmsSlider.value() / 100.0,
                'input_size': self.yoloInputSizeCombo.currentText()
            }
        elif self.deepstreamRadio.isChecked():
            settings['detector_type'] = "DeepStream"
            settings['version'] = ""
            settings['model'] = self.deepstreamModelCombo.currentText()
            settings['params'] = {
                'conf_threshold': self.deepstreamConfidenceSlider.value() / 100.0,
                'batch_size': self.deepstreamBatchSizeSpinBox.value()
            }

        return settings