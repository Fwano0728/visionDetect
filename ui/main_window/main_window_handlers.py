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
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage

from ui.error_handling import ErrorHandler

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

    def connect_signals(self):
        """이벤트 시그널 연결"""
        # 메뉴/툴바 액션 연결
        self.ui.action_Connect.triggered.connect(self.on_connect_camera)
        self.ui.action_Disconnect.triggered.connect(self.on_disconnect_camera)
        self.ui.action_StartDetection.triggered.connect(self.on_start_detection)
        self.ui.action_StopDetection.triggered.connect(self.on_stop_detection)
        self.ui.action_Screenshot.triggered.connect(self.on_take_screenshot)
        self.ui.action_Exit.triggered.connect(self.window.close)
        self.ui.action_About.triggered.connect(self.on_show_about)
        self.ui.action_ShowDashboard.triggered.connect(self.on_show_dashboard)
        self.ui.action_Fullscreen.triggered.connect(self.on_toggle_fullscreen)

        # 카메라 설정 탭
        self.ui.refreshCameraButton.clicked.connect(self.on_refresh_cameras)
        self.ui.connectButton.clicked.connect(self.on_connect_camera)
        self.ui.disconnectButton.clicked.connect(self.on_disconnect_camera)
        self.ui.cameraSettingsButton.clicked.connect(self.on_camera_settings)

        # 탐지 설정 탭
        self.ui.detectionCheckBox.stateChanged.connect(self.on_detection_toggled)
        self.ui.detectionModeComboBox.currentIndexChanged.connect(self.on_detection_mode_changed)
        self.ui.applyDetectorButton.clicked.connect(self.on_apply_detector)
        self.ui.yoloVersionComboBox.currentIndexChanged.connect(self.on_yolo_version_changed)
        self.ui.confidenceSlider.valueChanged.connect(self.on_confidence_changed)

        # 시스템 설정 탭
        self.ui.browseSavePathButton.clicked.connect(self.on_browse_save_path)
        self.ui.saveSettingsButton.clicked.connect(self.on_save_settings)

        # 결과 패널
        self.ui.saveResultsButton.clicked.connect(self.on_save_results)
        self.ui.clearResultsButton.clicked.connect(self.on_clear_results)
        self.ui.screenshotButton.clicked.connect(self.on_take_screenshot)

        # 비디오 프레임 컨텍스트 메뉴
        self.ui.videoFrame.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.videoFrame.customContextMenuRequested.connect(self.on_video_context_menu)

        logger.debug("이벤트 시그널 연결 완료")

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
                    self.ui.action_Connect.setEnabled(False)
                    self.ui.action_Disconnect.setEnabled(True)
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
                self.ui.action_Connect.setEnabled(True)
                self.ui.action_Disconnect.setEnabled(False)
                self.ui.detectionCheckBox.setEnabled(False)
                self.ui.action_StartDetection.setEnabled(False)
                self.ui.action_StopDetection.setEnabled(False)

                # 비디오 프레임 초기화
                self.ui.videoFrame.clear()
                self.ui.videoFrame.setText("카메라가 연결되지 않았습니다.")

                # 탐지 결과 초기화
                self.ui.resultsTextLabel.setText("탐지된 객체가 없습니다.")

                self.window.statusBar().showMessage("카메라 연결 해제됨")
                logger.info("카메라 연결 해제됨")

            except Exception as e:
                ErrorHandler.handle_exception(self.window, e, "카메라 연결 해제")

        def on_camera_settings(self):
            """카메라 고급 설정 다이얼로그 표시"""
            # 미구현 - 향후 카메라 속성 조정 다이얼로그 추가
            QMessageBox.information(self.window, "안내", "카메라 고급 설정 기능은 아직 구현되지 않았습니다.")

        # 탐지 관련 핸들러
        def on_detection_toggled(self, state):
            """객체 탐지 활성화/비활성화"""
            try:
                enabled = (state == Qt.Checked)

                # UI 요소 활성화/비활성화
                self.ui.detectionModeComboBox.setEnabled(enabled)
                self.ui.applyDetectorButton.setEnabled(enabled)

                # 액션 버튼 상태 업데이트
                self.ui.action_StartDetection.setEnabled(enabled)
                self.ui.action_StartDetection.setChecked(enabled)
                self.ui.action_StopDetection.setEnabled(enabled)

                # 스택 위젯 표시 여부
                current_mode = self.ui.detectionModeComboBox.currentText()
                if enabled:
                    if current_mode == "YOLO":
                        self.ui.detectionStackedWidget.setCurrentIndex(1)  # YOLO 페이지
                    else:
                        self.ui.detectionStackedWidget.setCurrentIndex(0)  # HOG 페이지
                else:
                    # 탐지 비활성화 시 탐지기 해제
                    self.window.detector_manager.set_detector(None)
                    self.update_current_detector_info()

                logger.info(f"객체 탐지 {'활성화' if enabled else '비활성화'}")

            except Exception as e:
                ErrorHandler.handle_exception(self.window, e, "탐지 설정 변경")

        def on_detection_mode_changed(self, index):
            """탐지 모드 변경 시 처리"""
            try:
                # 현재 선택된 모드
                mode = self.ui.detectionModeComboBox.currentText()

                # 스택 위젯 업데이트
                if mode == "YOLO":
                    self.ui.detectionStackedWidget.setCurrentIndex(1)  # YOLO 페이지

                    # YOLO 버전 목록 로드
                    self.load_yolo_versions()
                else:
                    self.ui.detectionStackedWidget.setCurrentIndex(0)  # HOG 페이지

                logger.debug(f"탐지 모드 변경: {mode}")

            except Exception as e:
                ErrorHandler.handle_exception(self.window, e, "탐지 모드 변경")

        def load_yolo_versions(self):
            """YOLO 버전 목록 로드"""
            try:
                self.ui.yoloVersionComboBox.clear()

                # 사용 가능한 YOLO 버전 가져오기
                versions = self.window.detector_manager.get_available_versions("YOLO")

                if versions:
                    self.ui.yoloVersionComboBox.addItems(versions)
                    # 첫 번째 버전 선택 및 모델 업데이트
                    self.on_yolo_version_changed(0)
                else:
                    logger.warning("사용 가능한 YOLO 버전이 없습니다.")

            except Exception as e:
                ErrorHandler.handle_exception(self.window, e, "YOLO 버전 로드")

        def on_yolo_version_changed(self, index):
            """YOLO 버전 변경 시 모델 목록 업데이트"""
            try:
                if index < 0:
                    return

                self.ui.yoloModelComboBox.clear()

                # 현재 선택된 버전
                version = self.ui.yoloVersionComboBox.currentText()

                # 해당 버전의 모델 목록 가져오기
                models = self.window.detector_manager.get_available_models("YOLO", version)

                if models:
                    self.ui.yoloModelComboBox.addItems(models)
                else:
                    logger.warning(f"YOLO {version}에 사용 가능한 모델이 없습니다.")

            except Exception as e:
                ErrorHandler.handle_exception(self.window, e, "YOLO 모델 로드")

        def on_confidence_changed(self, value):
            """신뢰도 슬라이더 값 변경 시 처리"""
            # 슬라이더 값을 0.0-1.0 범위로 변환
            confidence = value / 100.0
            self.ui.confidenceValueLabel.setText(f"{confidence:.2f}")

    def on_apply_detector(self):
        """탐지기 설정 적용"""
        try:
            # 현재 모드
            detector_type = self.ui.detectionModeComboBox.currentText()
            params = {}

            if detector_type == "YOLO":
                # YOLO 설정
                version = self.ui.yoloVersionComboBox.currentText()
                model = self.ui.yoloModelComboBox.currentText()
                conf_threshold = self.ui.confidenceSlider.value() / 100.0

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
                self.update_current_detector_info()

                # 액션 버튼 활성화
                self.ui.action_StartDetection.setEnabled(True)
                self.ui.action_StopDetection.setEnabled(False)

                logger.info(f"탐지기 설정 완료: {message}")
            else:
                ErrorHandler.show_error(self.window, "탐지기 설정 실패", message, "error")

        except Exception as e:
            ErrorHandler.handle_exception(self.window, e, "탐지기 설정")

    def update_current_detector_info(self):
        """현재 설정된 탐지기 정보 업데이트"""
        info = self.window.detector_manager.get_current_detector_info()

        if info.get("status") == "미설정":
            self.ui.currentDetectorInfoLabel.setText("탐지기가 설정되지 않았습니다.")
            return

        # 정보 텍스트 구성
        detector_type = info.get("type", "")
        version = info.get("version", "")
        model = info.get("model", "")
        params = info.get("params", {})

        text = f"탐지기: {detector_type}"
        if version and version != "Default":
            text += f" {version}"
        if model and model != "Default":
            text += f" ({model})"

        self.ui.currentDetectorInfoLabel.setText(text)

    def on_start_detection(self):
        """탐지 시작"""
        # UI 업데이트
        self.ui.action_StartDetection.setEnabled(False)
        self.ui.action_StartDetection.setChecked(True)
        self.ui.action_StopDetection.setEnabled(True)

        logger.info("객체 탐지 시작")

    def on_stop_detection(self):
        """탐지 중지"""
        # UI 업데이트
        self.ui.action_StartDetection.setEnabled(True)
        self.ui.action_StartDetection.setChecked(False)
        self.ui.action_StopDetection.setEnabled(False)

        logger.info("객체 탐지 중지")

        # 시스템 설정 핸들러
        def on_browse_save_path(self):
            """저장 경로 선택 다이얼로그"""
            try:
                current_path = self.ui.savePathEdit.text()

                # 폴더 선택 다이얼로그
                directory = QFileDialog.getExistingDirectory(
                    self.window,
                    "저장 폴더 선택",
                    current_path,
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
                )

                if directory:
                    self.ui.savePathEdit.setText(directory)

            except Exception as e:
                ErrorHandler.handle_exception(self.window, e, "폴더 선택")

        def on_save_settings(self):
            """시스템 설정 저장"""
            try:
                # 설정 저장
                self.window.init_manager.save_settings()

                # 자동 시작 등록/해제 (Windows의 경우)
                if os.name == 'nt':
                    self.update_windows_autostart()

                self.window.statusBar().showMessage("설정이 저장되었습니다.")
                logger.info("시스템 설정 저장됨")

            except Exception as e:
                ErrorHandler.handle_exception(self.window, e, "설정 저장")

        def update_windows_autostart(self):
            """Windows 시작 시 자동 실행 설정 (Windows 전용)"""
            try:
                import winreg

                auto_start = self.ui.autoStartCheckBox.isChecked()
                app_path = os.path.abspath(sys.argv[0])

                # 레지스트리 키
                key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
                app_name = "VisionDetect"

                # 레지스트리 열기
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    key_path,
                    0,
                    winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE
                )

                if auto_start:
                    # 자동 시작 등록
                    winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, f'"{app_path}"')
                    logger.info("Windows 시작 시 자동 실행 등록됨")
                else:
                    # 자동 시작 해제 (존재하는 경우)
                    try:
                        winreg.DeleteValue(key, app_name)
                        logger.info("Windows 시작 시 자동 실행 해제됨")
                    except:
                        # 키가 없는 경우 무시
                        pass

                winreg.CloseKey(key)

            except Exception as e:
                logger.error(f"자동 시작 설정 중 오류: {str(e)}")
                # 실패해도 다른 설정에는 영향 없음

        # 결과 관련 핸들러
        def on_save_results(self):
            """탐지 결과 저장"""
            try:
                # 저장할 내용이 없는 경우
                if self.ui.resultsTextLabel.text() == "탐지된 객체가 없습니다.":
                    ErrorHandler.show_error(self.window, "알림", "저장할 탐지 결과가 없습니다.", "info")
                    return

                # 저장 경로 가져오기
                save_path = self.ui.savePathEdit.text()
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
                save_path = self.ui.savePathEdit.text()
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
                    context_menu = QMenu(self.window)

                    # 항목 추가
                    save_action = context_menu.addAction("스크린샷 저장")
                    fullscreen_action = context_menu.addAction("전체 화면")
                    zoom_menu = QMenu("확대/축소", context_menu)
                    zoom_in_action = zoom_menu.addAction("확대 (+)")
                    zoom_reset_action = zoom_menu.addAction("원본 크기")
                    zoom_out_action = zoom_menu.addAction("축소 (-)")
                    context_menu.addMenu(zoom_menu)

                    # 카메라 연결 상태에 따른 활성화/비활성화
                    is_connected = self.window.camera_manager.is_connected()
                    save_action.setEnabled(is_connected)
                    fullscreen_action.setEnabled(is_connected)
                    zoom_menu.setEnabled(is_connected)

                    # 메뉴 표시 및 선택 처리
                    action = context_menu.exec_(self.ui.videoFrame.mapToGlobal(position))

                    if action == save_action:
                        self.on_take_screenshot()
                    elif action == fullscreen_action:
                        self.on_toggle_fullscreen()
                    elif action == zoom_in_action:
                        self.on_zoom_in()
                    elif action == zoom_reset_action:
                        self.on_zoom_reset()
                    elif action == zoom_out_action:
                        self.on_zoom_out()

                except Exception as e:
                    ErrorHandler.handle_exception(self.window, e, "컨텍스트 메뉴")

            def on_toggle_fullscreen(self):
                """전체 화면 전환"""
                if self.window.isFullScreen():
                    self.window.showNormal()
                    self.ui.action_Fullscreen.setChecked(False)
                else:
                    self.window.showFullScreen()
                    self.ui.action_Fullscreen.setChecked(True)

            def on_zoom_in(self):
                """화면 확대"""
                # 미구현 - 향후 구현 예정
                self.window.statusBar().showMessage("확대 기능은 아직 구현되지 않았습니다.")

            def on_zoom_reset(self):
                """화면 원본 크기로 복원"""
                # 미구현 - 향후 구현 예정
                self.window.statusBar().showMessage("원본 크기 복원 기능은 아직 구현되지 않았습니다.")

            def on_zoom_out(self):
                """화면 축소"""
                # 미구현 - 향후 구현 예정
                self.window.statusBar().showMessage("축소 기능은 아직 구현되지 않았습니다.")

            def on_show_dashboard(self):
                """대시보드 창 표시"""
                # 미구현 - 향후 구현 예정
                QMessageBox.information(self.window, "안내", "대시보드 기능은 아직 구현되지 않았습니다.")

            def on_show_about(self):
                """프로그램 정보 표시"""
                about_text = (
                    "객체 탐지 시스템\n"
                    "버전: 1.0.0\n\n"
                    "YOLO, OpenCV 등을 활용한 실시간 객체 탐지 시스템입니다.\n\n"
                    "© 2023 All Rights Reserved"
                )
                QMessageBox.about(self.window, "프로그램 정보", about_text)