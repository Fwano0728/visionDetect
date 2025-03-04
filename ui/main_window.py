# ui/main_window.py
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSplitter, QFrame, QStatusBar, QSizePolicy,
                             QPushButton, QComboBox, QGroupBox, QGridLayout,
                             QCheckBox, QSlider)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont

from ui.camera_selector import CameraSelector
from ui.model_selector import ModelSelector
from ui.camera_manager import CameraManager
from detector_manager import DetectorManager
import cv2
import time


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

        # 마지막 프레임 초기화
        self.last_frame = None

        # UI 요소 초기화
        self.init_ui()

        # 시그널 연결
        self.connect_signals()

    def init_ui(self):
        """UI 컴포넌트 초기화"""
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃 - 세로 레이아웃
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)

        # 상단 패널 영역 (카메라 설정 + 객체 탐지 설정)
        top_panel = QWidget()
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # 1. 카메라 설정 (가로로 넓게)
        camera_group = QGroupBox("카메라 설정")
        camera_layout = QHBoxLayout()  # 가로 레이아웃으로 변경
        camera_layout.setContentsMargins(5, 5, 5, 5)

        # 카메라 설정 왼쪽 부분 (라벨 + 콤보박스)
        camera_left_layout = QVBoxLayout()

        # 카메라 선택 콤보박스
        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("카메라:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(self.camera_manager.available_cameras.keys())
        camera_select_layout.addWidget(self.camera_combo)
        camera_left_layout.addLayout(camera_select_layout)

        # 해상도 선택 콤보박스
        resolution_select_layout = QHBoxLayout()
        resolution_select_layout.addWidget(QLabel("해상도:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])
        resolution_select_layout.addWidget(self.resolution_combo)
        camera_left_layout.addLayout(resolution_select_layout)

        camera_layout.addLayout(camera_left_layout, 3)  # 비율 3

        # 카메라 설정 오른쪽 부분 (버튼)
        button_layout = QVBoxLayout()
        self.connect_button = QPushButton("연결")
        self.connect_button.clicked.connect(self.on_connect_camera)
        button_layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton("연결 해제")
        self.disconnect_button.clicked.connect(self.on_disconnect_camera)
        self.disconnect_button.setEnabled(False)
        button_layout.addWidget(self.disconnect_button)

        camera_layout.addLayout(button_layout, 1)  # 비율 1
        camera_group.setLayout(camera_layout)

        # 2. 객체 탐지 설정 (세로로 유동적)
        detector_group = QGroupBox("객체 탐지 설정")
        detector_layout = QVBoxLayout()
        detector_layout.setContentsMargins(5, 5, 5, 5)

        # 사람 탐지 활성화 체크박스
        self.detection_checkbox = QCheckBox("사람 탐지 활성화")
        self.detection_checkbox.stateChanged.connect(self.toggle_detection)
        detector_layout.addWidget(self.detection_checkbox)

        # 탐지 모드 설정
        detection_mode_layout = QHBoxLayout()
        detection_mode_layout.addWidget(QLabel("탐지 모드:"))
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems(["OpenCV-HOG", "YOLO"])
        self.detection_mode_combo.setEnabled(False)
        self.detection_mode_combo.currentTextChanged.connect(self.on_detection_mode_changed)
        detection_mode_layout.addWidget(self.detection_mode_combo)
        detector_layout.addLayout(detection_mode_layout)

        # YOLO 설정
        self.yolo_settings = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_settings)
        yolo_layout.setContentsMargins(0, 0, 0, 0)

        # YOLO 버전
        yolo_version_layout = QHBoxLayout()
        yolo_version_layout.addWidget(QLabel("YOLO 버전:"))
        self.yolo_version_combo = QComboBox()
        yolo_version_combo_versions = self.detector_manager.get_available_versions("YOLO")
        self.yolo_version_combo.addItems(yolo_version_combo_versions)
        self.yolo_version_combo.currentTextChanged.connect(self.on_yolo_version_changed)
        yolo_version_layout.addWidget(self.yolo_version_combo)
        yolo_layout.addLayout(yolo_version_layout)

        # YOLO 모델
        yolo_model_layout = QHBoxLayout()
        yolo_model_layout.addWidget(QLabel("모델:"))
        self.yolo_model_combo = QComboBox()
        if yolo_version_combo_versions:
            initial_models = self.detector_manager.get_available_models("YOLO", yolo_version_combo_versions[0])
            self.yolo_model_combo.addItems(initial_models)
            self.yolo_model_combo.setEnabled(len(initial_models) > 0)
        else:
            self.yolo_model_combo.addItem("사용 가능한 모델 없음")
            self.yolo_model_combo.setEnabled(False)
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

        # YOLO 설정 초기 표시
        self.yolo_settings.setVisible(False)
        detector_layout.addWidget(self.yolo_settings)

        # 적용 버튼
        self.apply_button = QPushButton("설정 적용")
        self.apply_button.clicked.connect(self.apply_detector_settings)
        self.apply_button.setEnabled(False)
        detector_layout.addWidget(self.apply_button)

        detector_group.setLayout(detector_layout)

        # 상단 패널에 추가
        top_layout.addWidget(camera_group, 1)  # 비율 1
        top_layout.addWidget(detector_group, 1)  # 비율 1
        top_panel.setLayout(top_layout)

        # 상단 패널을 메인 레이아웃에 추가
        main_layout.addWidget(top_panel)

        # 현재 설정 정보 패널
        self.current_detector_info = QLabel("탐지기가 설정되지 않았습니다.")
        self.current_detector_info.setAlignment(Qt.AlignCenter)
        self.current_detector_info.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.current_detector_info.setMinimumHeight(30)
        self.current_detector_info.setMaximumHeight(40)
        main_layout.addWidget(self.current_detector_info)

        # 중앙 컨텐츠 영역 (비디오 + 결과)
        content_panel = QSplitter(Qt.Horizontal)

        # 비디오 프레임 (좌측)
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setFrameStyle(QFrame.StyledPanel)
        self.video_frame.setMinimumSize(640, 480)
        self.video_frame.setText("카메라가 연결되지 않았습니다.")

        # 결과 패널 (우측)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 탐지 결과
        self.results_text = QLabel("탐지된 객체가 없습니다.")
        self.results_text.setAlignment(Qt.AlignCenter)
        self.results_text.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.results_text.setMinimumHeight(150)
        right_layout.addWidget(self.results_text)

        # 하단 빈 공간 (확장 가능)
        empty_space = QWidget()
        empty_space.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(empty_space)

        # 컨텐츠 패널에 추가
        content_panel.addWidget(self.video_frame)
        content_panel.addWidget(right_panel)
        content_panel.setSizes([700, 300])  # 비디오:결과 비율

        # 컨텐츠 패널을 메인 레이아웃에 추가
        main_layout.addWidget(content_panel, 1)  # 스트레치 비율 1

        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("시스템 준비")

    def connect_signals(self):
        """시그널 연결"""
        # 카메라 매니저 시그널
        self.camera_manager.frame_updated.connect(self.update_frame)
        self.camera_manager.fps_updated.connect(self.update_fps)
        self.camera_manager.system_status_signal.connect(self.update_system_status)

        # 탐지 매니저 시그널
        self.detector_manager.detection_result_signal.connect(self.update_detection_result)

    def update_frame(self, frame):
        """카메라 프레임 업데이트 - 해상도에 따라 텍스트 크기 조정"""
        try:
            if frame is None:
                print("받은 프레임이 None입니다.")
                return

            # 마지막 프레임 저장
            self.last_frame = frame.copy()

            # 객체 탐지 실행 (탐지 매니저에 프레임 전달)
            if self.detector_manager.is_active():
                # 현재 해상도에 따라 텍스트 크기 조정
                resolution = self.resolution_combo.currentText()
                text_scale = self.get_text_scale_for_resolution(resolution)

                # 텍스트 스케일을 탐지 함수에 전달
                frame, _, _ = self.detector_manager.detect(frame, text_scale=text_scale)

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

    def get_text_scale_for_resolution(self, resolution):
        """해상도에 따른 텍스트 스케일 반환"""
        if resolution == "1920x1080":
            return 1.5  # 큰 해상도에서는 더 큰 텍스트
        elif resolution == "1280x720":
            return 1.2
        elif resolution == "800x600":
            return 1.0
        else:  # 640x480 또는 기타
            return 0.8

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

    def on_connect_camera(self):
        """카메라 연결 버튼 클릭 처리"""
        camera = self.camera_combo.currentText()
        resolution = self.resolution_combo.currentText()

        # 연결 시도
        success, message = self.camera_manager.connect_camera(camera, resolution)

        if success:
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            # 탐지기 컨트롤 활성화
            self.enable_detection_controls(True)
            self.statusBar.showMessage(f"카메라 '{camera}' 연결됨, 해상도: {resolution}")
        else:
            self.statusBar.showMessage(f"카메라 연결 실패: {message}")

    def on_disconnect_camera(self):
        """카메라 연결 해제 버튼 클릭 처리"""
        self.camera_manager.disconnect_camera()
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.video_frame.clear()
        self.video_frame.setText("카메라가 연결되지 않았습니다.")
        self.results_text.setText("탐지된 객체가 없습니다.")

        # 탐지기 비활성화
        self.detector_manager.set_detector(None)
        self.detection_checkbox.setChecked(False)
        self.enable_detection_controls(False)
        self.statusBar.showMessage("카메라 연결 해제됨")

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

        if models:
            self.yolo_model_combo.addItems(models)
            self.yolo_model_combo.setEnabled(True)
        else:
            self.yolo_model_combo.addItem("사용 가능한 모델 없음")
            self.yolo_model_combo.setEnabled(False)

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
            self.results_text.setText("탐지된 객체가 없습니다.")

    def apply_detector_settings(self):
        """탐지기 설정 적용"""
        # 매개변수 수집
        detector_type = self.detection_mode_combo.currentText()
        params = {}

        if detector_type == "YOLO":
            # YOLO 설정
            version = self.yolo_version_combo.currentText()
            model = self.yolo_model_combo.currentText()

            # 사용 가능한 모델이 없는 경우 처리
            if model == "사용 가능한 모델 없음":
                self.statusBar.showMessage(f"사용 가능한 모델이 없습니다.")
                return

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
            self.statusBar.showMessage(f"탐지기 설정 적용됨: {message}")
        else:
            self.statusBar.showMessage(f"탐지기 설정 실패: {message}")

    def enable_detection_controls(self, enabled):
        """탐지 컨트롤 활성화/비활성화"""
        self.detection_checkbox.setEnabled(enabled)

        # 체크박스가 체크되어 있지 않으면 다른 컨트롤은 비활성화
        should_enable = enabled and self.detection_checkbox.isChecked()
        self.detection_mode_combo.setEnabled(should_enable)
        self.apply_button.setEnabled(should_enable)
        self.yolo_settings.setEnabled(should_enable)

    def resizeEvent(self, event):
        """창 크기 조정 시 이벤트 처리"""
        super().resizeEvent(event)

        # 이전 프레임이 있으면 크기에 맞게 다시 그리기
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            self.update_frame(self.last_frame)

    def update_detector_info(self):
        """현재 설정된 탐지기 정보 업데이트"""
        info = self.detector_manager.get_current_detector_info()

        if info.get("status") == "미설정":
            self.current_detector_info.setText("탐지기가 설정되지 않았습니다.")
            return

        # 정보 텍스트 구성
        detector_type = info.get("type", "")
        version = info.get("version", "")
        model = info.get("model", "")
        params = info.get("params", {})

        text = f"현재 탐지 모드: {detector_type}"
        if version:
            text += f" {version}"
        if model:
            text += f" - {model}"

        # 매개변수 표시
        if params:
            conf = params.get('conf_threshold', 0)
            if conf:
                text += f" (신뢰도 임계값: {conf:.2f})"

        self.current_detector_info.setText(text)
