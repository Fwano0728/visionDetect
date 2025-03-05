# ui/main_window.py
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSplitter, QFrame, QStatusBar, QSizePolicy,
                             QPushButton, QComboBox, QAction, QDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSlot, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5 import uic
import os
import cv2
import time
import numpy as np

# 다이얼로그 클래스 임포트
from ui.camera_settings_dialog import CameraSettingsDialog
from ui.detector_settings_dialog import DetectorSettingsDialog
from ui.object_selection_dialog import ObjectSelectionDialog

# 매니저 클래스 임포트
from ui.camera_manager import CameraManager
from detector_manager import DetectorManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI 파일 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, 'ui_files', 'main_window.ui')
        uic.loadUi(ui_file, self)

        # 매니저 초기화
        self.camera_manager = CameraManager()
        self.detector_manager = DetectorManager()

        # 마지막 프레임 초기화
        self.last_frame = None

        # 프레임 처리 관련 변수
        self.processing_frame = False
        self.frame_queue = []
        self.max_queue_size = 2  # 최대 2개 프레임만 큐에 유지

        # YOLO 설정 저장 변수 초기화
        self.saved_detection_mode = "YOLO"
        self.saved_yolo_version = "v4"
        self.saved_yolo_model = "YOLOv4-tiny"
        self.saved_confidence = 40

        # 객체 선택 상태 (기본적으로 사람만 선택)
        self.selected_objects = {}
        try:
            with open("models/yolo/coco.names", "r") as f:
                coco_classes = f.read().strip().split("\n")
                self.selected_objects = {cls: (cls == "person") for cls in coco_classes}
        except:
            # 파일을 찾을 수 없으면 기본값으로 person만 선택
            self.selected_objects = {"person": True}

        # 객체별 색상 지정 (초기 색상 맵)
        self.object_colors = self.generate_color_map()

        # 커넥션 상태
        self.is_camera_connected = False

        # 현재 열린 다이얼로그 참조 저장 (검색 최적화)
        self.open_dialogs = []

        # 탐지 횟수 업데이트 타이머 - 업데이트 주기 설정
        self.detection_count_timer = QTimer(self)
        self.detection_count_timer.setInterval(2000)  # 2초마다 업데이트
        self.detection_count_timer.timeout.connect(self.update_detection_counts_in_dialogs)

        # 현재 탐지 횟수 캐싱
        self.current_detection_counts = {}

        # 시그널 연결
        self.connect_signals()

    def generate_color_map(self):
        """객체별 색상 맵 생성 - 다양한 색상으로 객체 구분"""
        colors = {
            "person": (0, 255, 0),  # 사람: 초록색 (BGR)
            "default": (0, 0, 255)  # 기본: 빨간색 (BGR)
        }

        # COCO 클래스 목록에서 각 클래스별 색상 생성
        try:
            with open("models/yolo/coco.names", "r") as f:
                coco_classes = f.read().strip().split("\n")

                # HSV 색상 공간에서 균등하게 분포된 색상 생성
                for i, cls in enumerate(coco_classes):
                    if cls != "person":  # 사람은 녹색으로 고정
                        # HSV 색상 공간에서 색상값만 균등하게 분포하여 생성
                        hue = i * 180 // len(coco_classes)
                        saturation = 255  # 채도 최대
                        value = 255  # 명도 최대

                        # HSV -> BGR 변환
                        color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
                        colors[cls] = (int(color[0]), int(color[1]), int(color[2]))
        except:
            pass  # 파일이 없으면 기본 색상만 사용

        return colors

    def connect_signals(self):
        """시그널 연결"""
        # 카메라 매니저 시그널
        self.camera_manager.frame_updated.connect(self.update_frame)
        self.camera_manager.fps_updated.connect(self.update_fps)
        self.camera_manager.system_status_signal.connect(self.update_system_status)

        # 탐지 매니저 시그널
        self.detector_manager.detection_result_signal.connect(self.update_detection_result)

        # 객체 탐지 횟수 업데이트 시그널 연결
        if hasattr(self.detector_manager, 'detection_counts_updated'):
            self.detector_manager.detection_counts_updated.connect(self.update_detection_counts)

        # 현재 탐지 수 업데이트 시그널 연결
        if hasattr(self.detector_manager, 'current_detection_updated'):
            try:
                self.detector_manager.current_detection_updated.connect(self.update_current_detections)
            except Exception as e:
                print(f"현재 탐지 업데이트 시그널 연결 실패: {str(e)}")

        # 메뉴/툴바 액션 연결
        self.actionExit.triggered.connect(self.close)
        self.actionConnectCamera.triggered.connect(self.show_camera_settings)
        self.actionDisconnectCamera.triggered.connect(self.on_disconnect_camera)
        self.actionEnableDetection.triggered.connect(self.toggle_detection)
        self.actionDetectionSettings.triggered.connect(self.show_detector_settings)
        self.actionObjectSelection.triggered.connect(self.show_object_selection)
        self.actionAbout.triggered.connect(self.show_about)

        # 카메라 연결 버튼 시그널 추가
        if hasattr(self, 'connect_button'):
            self.connect_button.clicked.connect(self.on_connect_camera)

        # 카메라 연결 해제 버튼 시그널 추가
        if hasattr(self, 'disconnect_button'):
            self.disconnect_button.clicked.connect(self.on_disconnect_camera)

    def update_current_detections(self, current_detections):
        """Update UI with currently detected object counts"""
        if not current_detections:
            return

        # Update each object's current detection count label
        for class_name, widgets in self.object_checkboxes.items():
            if 'count_label' in widgets:
                count_label = widgets['count_label']
                count = current_detections.get(class_name, 0)
                count_label.setText(f"탐지: {count}개")

                # Highlight labels for currently detected objects
                if count > 0:
                    count_label.setStyleSheet("color: #007BFF; font-weight: bold;")
                else:
                    count_label.setStyleSheet("")

    def update_frame(self, frame):
        """카메라 프레임 업데이트"""
        if frame is None:
            return

        # 이미 처리 중이면 큐 관리
        if self.processing_frame:
            # 큐가 가득 찼으면 이전 프레임 버림
            if len(self.frame_queue) >= self.max_queue_size:
                return
            self.frame_queue.append(frame.copy())
            return

        self.processing_frame = True

        try:
            # 마지막 프레임 저장 (깊은 복사 방지)
            self.last_frame = frame

            processed_frame = frame

            # 객체 탐지 실행 (탐지 매니저에 프레임 전달)
            if self.detector_manager.is_active() and self.actionEnableDetection.isChecked():
                # 객체 탐지 실행 시 오류 처리를 위한 try-except 블록
                try:
                    # 기본 방식으로 호출 시도
                    processed_frame, detections, _ = self.detector_manager.detect(frame)
                except TypeError as e:
                    print(f"탐지 함수 호출 오류: {e}")
                    # 대안 1: 매개변수 없이 호출
                    processed_frame, detections, _ = self.detector_manager.detect(frame)
                except Exception as e:
                    print(f"탐지 중 예상치 못한 오류: {e}")
                    # 오류 발생 시 원본 프레임 사용
                    processed_frame = frame
                    detections = []

            # OpenCV BGR 형식을 RGB로 변환
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape

            # Qt 이미지 변환
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 비디오 프레임 위젯의 실제 크기
            frame_width = self.videoFrame.width()
            frame_height = self.videoFrame.height()

            # 이미지 크기 조정 (비율 유지)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(frame_width, frame_height,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)

            self.videoFrame.setPixmap(scaled_pixmap)
            self.videoFrame.setAlignment(Qt.AlignCenter)

            # 큐에 프레임이 있으면 다음 프레임 처리
            if self.frame_queue:
                next_frame = self.frame_queue.pop(0)
                self.processing_frame = False
                self.update_frame(next_frame)
            else:
                self.processing_frame = False

        except Exception as e:
            print(f"프레임 업데이트 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            self.processing_frame = False

    def update_fps(self, fps):
        """FPS 업데이트 처리"""
        self.current_fps = fps

    def update_system_status(self, status):
        """시스템 상태 정보 업데이트 - 1초에 1회만 호출되도록 제한"""
        # 호출 빈도 제한
        current_time = getattr(self, 'last_status_update_time', 0)
        if time.time() - current_time < 0.5:  # 0.5초마다 업데이트로 변경
            return

        self.last_status_update_time = time.time()

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

        self.statusbar.showMessage(status_text)

    def update_detection_result(self, result_text):
        """객체 탐지 결과 업데이트"""
        self.resultsTextLabel.setText(result_text)

    def update_detection_counts(self, detection_counts):
        """객체별 탐지 횟수 업데이트"""
        # 현재 속성으로 저장 (나중에 다이얼로그에 전달하기 위함)
        self.current_detection_counts = detection_counts

    def update_detection_counts_in_dialogs(self):
        """열려 있는 다이얼로그의 탐지 횟수 업데이트"""
        # 탐지 횟수가 없거나 열린 다이얼로그가 없으면 건너뛰기
        if not hasattr(self, 'current_detection_counts') or not self.open_dialogs:
            return

        # 저장된 다이얼로그 참조만 업데이트
        for dialog in list(self.open_dialogs):  # 복사본으로 순회
            if dialog and hasattr(dialog, 'update_detection_counts'):
                try:
                    dialog.update_detection_counts(self.current_detection_counts)
                except:
                    # 다이얼로그가 더 이상 유효하지 않으면 목록에서 제거
                    if dialog in self.open_dialogs:
                        self.open_dialogs.remove(dialog)

    def show_camera_settings(self):
        """카메라 설정 다이얼로그 표시"""
        dialog = CameraSettingsDialog(self.camera_manager, self)
        self.open_dialogs.append(dialog)

        # 현재 연결된 카메라가 있으면 설정 값으로 초기화
        if self.is_camera_connected:
            current_camera = self.camera_manager.get_current_camera()
            if current_camera:
                index = dialog.cameraCombo.findText(current_camera)
                if index >= 0:
                    dialog.cameraCombo.setCurrentIndex(index)

        result = dialog.exec_()
        if dialog in self.open_dialogs:
            self.open_dialogs.remove(dialog)

        if result == QDialog.Accepted:
            settings = dialog.get_camera_settings()
            self.on_connect_camera(settings['camera'], settings['resolution'])

    def show_detector_settings(self):
        """탐지기 설정 다이얼로그 표시"""
        # 카메라가 연결되어 있는지 확인
        if not self.is_camera_connected:
            QMessageBox.warning(self, "경고", "카메라가 연결되어 있지 않습니다. 일부 설정이 적용되지 않을 수 있습니다.")

        # 탐지기 설정 다이얼로그 생성
        dialog = DetectorSettingsDialog(self.detector_manager, self)
        self.open_dialogs.append(dialog)

        # 다이얼로그 실행
        result = dialog.exec_()
        if dialog in self.open_dialogs:
            self.open_dialogs.remove(dialog)

        # 사용자가 '적용' 버튼을 클릭한 경우
        if result == QDialog.Accepted:
            # 설정 가져오기
            settings = dialog.get_detector_settings()

            # 현재 설정 저장
            self.saved_detection_mode = settings['detector_type']
            if settings['detector_type'] == 'YOLO':
                self.saved_yolo_version = settings['version']
                self.saved_yolo_model = settings['model']
                if 'confidence' in settings['params']:
                    self.saved_confidence = int(settings['params']['confidence'] * 100)

            # 객체 선택 목록이 있다면 파라미터에 추가
            if hasattr(self, 'selected_objects') and self.selected_objects:
                settings['params']['objects_to_detect'] = self.selected_objects

            # 탐지가 활성화된 경우에만 설정 적용
            if self.actionEnableDetection.isChecked():
                # 색상 정보도 전달
                settings['params']['colors'] = self.object_colors

                # 탐지기 설정 적용
                success, message = self.detector_manager.set_detector(
                    settings['detector_type'], settings['version'],
                    settings['model'], **settings['params'])

                # 결과 메시지 표시
                if success:
                    self.update_detector_info()
                    self.statusbar.showMessage(f"탐지기 설정 적용됨: {message}")
                else:
                    self.statusbar.showMessage(f"탐지기 설정 실패: {message}")
                    QMessageBox.warning(self, "설정 실패", f"탐지기 설정을 적용하지 못했습니다: {message}")

    def show_object_selection(self):
        """객체 선택 다이얼로그 표시"""
        try:
            # 필요한 모든 클래스가 있는지 확인
            if not hasattr(self, 'detector_manager'):
                QMessageBox.critical(self, "오류", "탐지 관리자가 초기화되지 않았습니다.")
                return

            # 현재 활성화된 탐지기가 YOLO인지 확인
            if self.detector_manager.is_active():
                detector_info = self.detector_manager.get_current_detector_info()
                if detector_info.get("type") != "YOLO":
                    QMessageBox.information(self, "알림",
                                            "객체 선택 기능은 YOLO 탐지기에서만 사용할 수 있습니다.\n"
                                            "현재 탐지기: " + detector_info.get("type", "알 수 없음"))
                    return

            # 객체 선택 다이얼로그 생성 (모달리스로 설정)
            try:
                from ui.object_selection_dialog import ObjectSelectionDialog
                dialog = ObjectSelectionDialog(self.detector_manager, self)
                dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                dialog.show()  # exec_() 대신 show() 사용하여 모달리스로 표시

                # 다이얼로그 참조 저장
                self.open_dialogs.append(dialog)

                # 현재 탐지 수 업데이트 연결
                if hasattr(self.detector_manager, 'current_detection_updated'):
                    self.detector_manager.current_detection_updated.connect(dialog.update_current_detections)

                # 다이얼로그 완료 시 처리를 위한 시그널 연결
                dialog.accepted.connect(lambda: self.on_object_selection_accepted(dialog))
                dialog.rejected.connect(lambda: self.on_object_selection_rejected(dialog))

            except Exception as e:
                QMessageBox.critical(self, "오류", f"객체 선택 다이얼로그 생성 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                return

        except Exception as e:
            QMessageBox.critical(self, "오류", f"객체 선택 다이얼로그에서 오류가 발생했습니다: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_object_selection_accepted(self, dialog):
        """객체 선택 다이얼로그 수락 시 처리"""
        try:
            # 선택된 객체 목록 가져오기
            self.selected_objects = dialog.get_selected_objects()

            # 카테고리별 색상 가져오기
            category_colors = dialog.get_category_colors()

            # 탐지기가 활성화되어 있다면 객체 선택 적용
            if self.detector_manager.is_active() and self.actionEnableDetection.isChecked():
                # 현재 탐지기 설정을 가져와서 객체 선택 정보만 업데이트
                detector_info = self.detector_manager.get_current_detector_info()
                if detector_info.get("status") == "설정됨":
                    detector_type = detector_info.get("type", "")
                    version = detector_info.get("version", "")
                    model = detector_info.get("model", "")

                    # 기존 매개변수 유지
                    existing_params = detector_info.get("params", {}).copy()

                    # 객체 선택 정보 및 색상 업데이트
                    existing_params['objects_to_detect'] = self.selected_objects
                    existing_params['category_colors'] = category_colors

                    # 탐지기 재설정
                    success, message = self.detector_manager.set_detector(
                        detector_type, version, model, **existing_params)

                    if success:
                        self.statusbar.showMessage("객체 선택 설정이 적용되었습니다.")
                    else:
                        self.statusbar.showMessage(f"객체 선택 설정 적용 실패: {message}")
        except Exception as e:
            print(f"객체 선택 적용 오류: {str(e)}")

        # 다이얼로그 참조 제거
        if dialog in self.open_dialogs:
            self.open_dialogs.remove(dialog)

    def on_object_selection_rejected(self, dialog):
        """객체 선택 다이얼로그 취소 시 처리"""
        # 다이얼로그 참조 제거
        if dialog in self.open_dialogs:
            self.open_dialogs.remove(dialog)

    def update_color_map(self):
        """선택된 객체에 따라 색상 맵 업데이트"""
        # 현재 색상 맵 유지
        updated_colors = self.object_colors.copy()

        return updated_colors

    def show_about(self):
        """프로그램 정보 다이얼로그 표시"""
        about_text = "객체 탐지 시스템\n버전 1.0\n\n객체 탐지 및 인식을 위한 프로그램입니다."
        QMessageBox.information(self, "프로그램 정보", about_text)

    def on_connect_camera(self, camera, resolution):
        """카메라 연결 처리"""
        # 연결 시도
        success, message = self.camera_manager.connect_camera(camera, resolution)

        if success:
            self.is_camera_connected = True
            self.actionConnectCamera.setEnabled(False)
            self.actionDisconnectCamera.setEnabled(True)

            # 탐지기 컨트롤 활성화
            self.actionEnableDetection.setEnabled(True)
            self.actionDetectionSettings.setEnabled(True)
            self.actionObjectSelection.setEnabled(True)

            # 타이머 시작
            self.detection_count_timer.start()

            # 이전에 탐지가 활성화되어 있었다면 설정 복원
            if hasattr(self, 'previous_detection_active') and self.previous_detection_active:
                self.actionEnableDetection.setChecked(True)
                self.toggle_detection(True)

            self.statusbar.showMessage(f"카메라 '{camera}' 연결됨, 해상도: {resolution}")
        else:
            self.statusbar.showMessage(f"카메라 연결 실패: {message}")

    def on_disconnect_camera(self):
        """카메라 연결 해제 처리"""
        # 현재 탐지 상태 저장
        self.previous_detection_active = self.actionEnableDetection.isChecked()

        # 진행 중인 프레임 처리 정리
        self.processing_frame = False
        self.frame_queue.clear()

        # 타이머 정지
        self.detection_count_timer.stop()

        self.camera_manager.disconnect_camera()
        self.is_camera_connected = False
        self.actionConnectCamera.setEnabled(True)
        self.actionDisconnectCamera.setEnabled(False)

        self.videoFrame.clear()
        self.videoFrame.setText("카메라가 연결되지 않았습니다.")
        self.resultsTextLabel.setText("탐지된 객체가 없습니다.")

        # 탐지기 비활성화 - 설정은 유지하되 컨트롤만 비활성화
        self.detector_manager.set_detector(None)
        self.actionEnableDetection.setChecked(False)
        self.actionEnableDetection.setEnabled(False)
        self.actionDetectionSettings.setEnabled(False)
        self.actionObjectSelection.setEnabled(False)

        # 상태 업데이트
        self.update_detector_info()
        self.statusbar.showMessage("카메라 연결 해제됨")

    def toggle_detection(self, state):
        """객체 탐지 활성화/비활성화"""
        if isinstance(state, bool):
            enabled = state
        else:
            enabled = self.actionEnableDetection.isChecked()

        # 활성화 상태에 따라 설정 적용
        if enabled:
            # 탐지기 설정 적용
            self.apply_detector_settings()
        else:
            # 탐지기 비활성화
            self.detector_manager.set_detector(None)
            self.update_detector_info()
            self.resultsTextLabel.setText("탐지된 객체가 없습니다.")

    def apply_detector_settings(self):
        """탐지기 설정 적용"""
        # 기본값으로 YOLO를 사용
        detector_type = "YOLO"
        version = "v4"  # 기본 버전
        model = "YOLOv4-tiny"  # 기본 모델
        conf_threshold = 0.4  # 기본 신뢰도 임계값

        # 사용자가 이전에 설정한 값이 있으면 사용
        if hasattr(self, 'saved_detection_mode'):
            detector_type = self.saved_detection_mode

        if detector_type == "YOLO" and hasattr(self, 'saved_yolo_version'):
            version = self.saved_yolo_version

        if hasattr(self, 'saved_yolo_model'):
            model = self.saved_yolo_model

        if hasattr(self, 'saved_confidence'):
            conf_threshold = self.saved_confidence / 100.0

        # 매개변수 설정
        params = {
            'conf_threshold': conf_threshold,
            'nms_threshold': 0.4,  # 기본값
            'objects_to_detect': self.selected_objects,  # 선택된 객체 목록 전달
            'colors': self.object_colors  # 색상 정보 전달
        }

        # 탐지기 설정
        if detector_type == "YOLO":
            success, message = self.detector_manager.set_detector(
                detector_type, version, model, **params)
        else:
            # HOG 등 다른 탐지기
            success, message = self.detector_manager.set_detector(detector_type)

        # 결과 표시
        if success:
            self.update_detector_info()
            self.statusbar.showMessage(f"탐지기 설정 적용됨: {message}")
        else:
            self.statusbar.showMessage(f"탐지기 설정 실패: {message}")
            # 실패 시 체크박스 해제
            self.actionEnableDetection.setChecked(False)

    def update_detector_info(self):
        """현재 설정된 탐지기 정보 업데이트"""
        info = self.detector_manager.get_current_detector_info()

        if info.get("status") == "미설정":
            self.detectorInfoLabel.setText("탐지기가 설정되지 않았습니다.")
            return

        # 정보 텍스트 구성
        detector_type = info.get("type", "")
        version = info.get("version", "")
        model = info.get("model", "")
        params = info.get("params", {})

        text = f"객체 탐지 모드: {detector_type}"
        if version:
            text += f" {version}"
        if model:
            text += f"\n{model}"

        # 매개변수 표시
        if params:
            conf = params.get('conf_threshold', 0)
            if conf:
                text += f" (신뢰도 임계값: {conf:.2f})"

        self.detectorInfoLabel.setText(text)

    def resizeEvent(self, event):
        """창 크기 변경 시 이벤트 처리"""
        super().resizeEvent(event)

        # 이전 프레임이 있으면 크기에 맞게 다시 그리기
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            self.update_frame(self.last_frame)

    def closeEvent(self, event):
        """프로그램 종료 시 정리 작업"""
        try:
            # 카메라 연결 해제
            if self.is_camera_connected:
                self.camera_manager.disconnect_camera()

            # 타이머 정지
            if hasattr(self, 'detection_count_timer') and self.detection_count_timer.isActive():
                self.detection_count_timer.stop()

            # 열려있는 다이얼로그 정리
            for dialog in list(self.open_dialogs):
                try:
                    if dialog and dialog.isVisible():
                        dialog.close()
                except:
                    pass
            self.open_dialogs.clear()

            # 프레임 처리 중지
            self.processing_frame = False
            self.frame_queue.clear()

            # 탐지기 해제
            if self.detector_manager:
                self.detector_manager.set_detector(None)

            print("프로그램이 안전하게 종료되었습니다.")
        except Exception as e:
            print(f"프로그램 종료 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()

        # 부모 클래스의 종료 이벤트 처리
        super().closeEvent(event)