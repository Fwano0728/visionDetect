#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
카메라 연결 및 관리 모듈
"""

import cv2
import time
import os
import logging
import subprocess
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import psutil

logger = logging.getLogger(__name__)


class CameraThread(QThread):
    """카메라 스트리밍을 위한 스레드 클래스"""
    frame_ready = pyqtSignal(object)
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, camera_id, resolution, fps=30, camera_model=""):
        super().__init__()
        self.camera_id = camera_id
        self.width, self.height = map(int, resolution.split('x'))
        self.fps = fps
        self.running = False
        self.cap = None
        self.camera_model = camera_model

        # 카메라 모델별 최적의 FPS 설정
        self.target_fps = self._get_optimal_fps(camera_model, fps)
        logger.debug(f"카메라 스레드 초기화: ID={camera_id}, 해상도={self.width}x{self.height}, FPS={self.target_fps}")

    def _get_optimal_fps(self, camera_model, requested_fps):
        """카메라 모델별 최적의 FPS 반환"""
        logger.debug(f"카메라 모델: {camera_model}, 요청 FPS: {requested_fps}")

        # 카메라 모델별 최대 FPS 제한
        max_fps = {
            "Logitech StreamCam": 60,
            "Logitech C920": 30,
            "Logitech C270": 30,
            "Microsoft LifeCam": 30,
            "Default": 30
        }

        # 해당 모델의 최대 FPS 가져오기
        for model_name, fps_limit in max_fps.items():
            if model_name in camera_model:
                return min(requested_fps, fps_limit)

        # 기본값
        return min(requested_fps, max_fps["Default"])

    def run(self):
        """스레드 실행 - 카메라에서 프레임을 지속적으로 가져옴"""
        try:
            # 여러 백엔드로 카메라 열기 시도
            success = False

            # 백엔드 목록 (시도할 순서대로)
            backends = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_FFMPEG, "FFMPEG"),
                (cv2.CAP_ANY, "기본")
            ]

            for backend, name in backends:
                try:
                    logger.info(f"{name} 백엔드로 카메라 열기 시도")
                    self.cap = cv2.VideoCapture(self.camera_id, backend)
                    if self.cap.isOpened():
                        logger.info(f"{name} 백엔드로 카메라 열기 성공")
                        success = True
                        break
                except Exception as be:
                    logger.warning(f"{name} 백엔드로 카메라 열기 실패: {str(be)}")

            if not success:
                # 마지막 시도로 아무 백엔드 지정 없이 열기
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    error_msg = f"모든 백엔드로 카메라 {self.camera_id} 열기 실패"
                    logger.error(error_msg)
                    self.error.emit(error_msg)
                    return

            # 여러 코덱 시도 (가장 효율적인 것부터)
            codecs = [
                ('MJPG', 'Motion JPEG'),
                ('YUYV', 'YUYV'),
                ('H264', 'H.264')
            ]

            for codec_code, codec_name in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_code)
                    self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    # 정상적으로 적용되었는지 확인
                    actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                    if actual_fourcc == fourcc:
                        logger.info(f"{codec_name} 코덱 설정 성공")
                        break
                except Exception as ce:
                    logger.warning(f"{codec_name} 코덱 설정 실패: {str(ce)}")

            # 프레임 속도 설정
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # 성능 향상을 위한 설정
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화

            # 현재 설정 확인
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()

            logger.info(
                f"카메라 설정: ID={self.camera_id}, 해상도={actual_width}x{actual_height}, FPS={actual_fps}, 코덱={fourcc_str}")

            # FPS 측정 변수
            frame_count = 0
            fps = 0
            fps_start_time = time.time()
            fps_update_interval = 0.5  # FPS 업데이트 간격 (초)

            # 재연결 관련 변수
            consecutive_failures = 0
            max_consecutive_failures = 5
            reconnect_attempts = 0
            max_reconnect_attempts = 3

            self.running = True

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"프레임 캡처 실패 ({consecutive_failures}/{max_consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        if reconnect_attempts < max_reconnect_attempts:
                            logger.warning(f"카메라 재연결 시도 ({reconnect_attempts + 1}/{max_reconnect_attempts})")

                            # 카메라 자원 해제 및 재연결
                            self.cap.release()
                            time.sleep(1.0)  # 잠시 대기

                            # 카메라 다시 열기
                            self.cap = cv2.VideoCapture(self.camera_id)
                            if self.cap.isOpened():
                                # 설정 다시 적용
                                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                                # 카운터 초기화
                                consecutive_failures = 0
                                reconnect_attempts += 1
                                logger.info("카메라 재연결 성공")
                            else:
                                reconnect_attempts += 1
                                logger.error("카메라 재연결 실패")
                                if reconnect_attempts >= max_reconnect_attempts:
                                    error_msg = "최대 재연결 시도 횟수 초과, 카메라 연결 종료"
                                    logger.error(error_msg)
                                    self.error.emit(error_msg)
                                    self.running = False
                                    break
                        else:
                            error_msg = "최대 재연결 시도 횟수 초과, 카메라 연결 종료"
                            logger.error(error_msg)
                            self.error.emit(error_msg)
                            self.running = False
                            break

                    time.sleep(0.1)  # 잠시 대기 후 재시도
                    continue
                else:
                    # 성공적으로 프레임을 가져왔으면 오류 카운터 초기화
                    consecutive_failures = 0

                # 프레임 카운트 증가
                frame_count += 1

                # FPS 계산 및 업데이트
                current_time = time.time()
                elapsed = current_time - fps_start_time

                if elapsed >= fps_update_interval:
                    fps = frame_count / elapsed
                    self.fps_updated.emit(fps)
                    fps_start_time = current_time
                    frame_count = 0

                # 프레임 준비 신호 발생
                self.frame_ready.emit(frame)

                # 과도한 CPU 사용 방지 (프레임 레이트에 맞춰 적절히 대기)
                sleep_time = max(1.0 / (self.target_fps * 1.5), 0.001)  # 최소 지연 1ms
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"카메라 스레드 오류: {str(e)}")
            self.error.emit(f"카메라 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            logger.info("카메라 스레드 종료")

    def stop(self):
        """스레드 정지"""
        self.running = False
        self.wait()
        logger.debug("카메라 스레드 중지 요청")


class CameraManager(QObject):
    """카메라 연결 및 관리 클래스"""

    # 시그널 정의
    frame_updated = pyqtSignal(object)
    fps_updated = pyqtSignal(float)
    system_status_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.current_camera = None
        self.available_cameras = self.scan_cameras()
        self.current_fps = 0
        self.load_update_time = 0
        self.system_load = {'cpu': 0, 'memory': 0}

        # 시스템 모니터링 타이머
        from PyQt5.QtCore import QTimer
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_status)
        self.system_timer.start(2000)  # 2초마다 시스템 상태 업데이트

        logger.info("카메라 매니저 초기화 완료")

    def scan_cameras(self):
        """시스템에 연결된 카메라 스캔"""
        available_cameras = {}

        # 터미널 명령어로 카메라 정보 확인
        try:
            if os.name == 'posix':  # Linux, macOS
                result = subprocess.run(['ls', '-l', '/dev/video*'], capture_output=True, text=True)
                logger.debug("시스템 카메라 장치:")
                logger.debug(result.stdout)
            elif os.name == 'nt':  # Windows
                logger.debug("Windows 시스템에서는 카메라 장치 확인 생략")
        except Exception as e:
            logger.error(f"카메라 장치 확인 중 오류: {str(e)}")

        # 가능한 모든 카메라 ID 확인
        for i in range(10):  # 0부터 9까지 카메라 ID 확인
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # 기본 이름
                        camera_name = f"Camera {i}"

                        # 카메라 이름 획득 시도
                        try:
                            if os.name == 'posix':  # Linux
                                device_path = f"/dev/video{i}"
                                cmd = f"v4l2-ctl -d {device_path} --all | grep 'Card\\|Driver'"
                                try:
                                    info = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                                    logger.debug(f"카메라 {i} 정보: {info.stdout}")

                                    # 카메라 모델명 추출
                                    if "StreamCam" in info.stdout:
                                        camera_name = "Logitech StreamCam"
                                    elif "C920" in info.stdout or "HD Pro Webcam" in info.stdout:
                                        camera_name = "Logitech C920"
                                    elif "C270" in info.stdout:
                                        camera_name = "Logitech C270"
                                    elif "LifeCam" in info.stdout:
                                        camera_name = "Microsoft LifeCam"
                                except:
                                    pass
                            elif os.name == 'nt':  # Windows
                                # Windows에서 카메라 이름 획득 로직 (미구현)
                                pass
                        except Exception as name_e:
                            logger.warning(f"카메라 이름 획득 중 오류: {str(name_e)}")

                        available_cameras[camera_name] = i
                        logger.info(f"카메라 발견: ID {i} - {camera_name}")
                    cap.release()
            except Exception as e:
                logger.warning(f"카메라 ID {i} 확인 중 오류: {str(e)}")

        # 카메라가 없으면 기본값 제공
        if not available_cameras:
            logger.warning("카메라를 찾지 못했습니다. 기본값을 사용합니다.")
            available_cameras = {"기본 카메라": 0}

        logger.info(f"사용 가능한 카메라: {available_cameras}")
        return available_cameras

    def connect_camera(self, camera_name, resolution):
        """카메라 연결"""
        if camera_name not in self.available_cameras:
            logger.error(f"카메라 '{camera_name}'을(를) 찾을 수 없습니다.")
            return False, f"카메라 '{camera_name}'을(를) 찾을 수 없습니다."

        # 이미 같은 카메라가 연결되어 있는지 확인
        if self.is_connected() and self.current_camera == camera_name:
            logger.info(f"카메라 '{camera_name}'은(는) 이미 연결되어 있습니다.")
            return True, f"카메라 '{camera_name}'은(는) 이미 연결되어 있습니다."

        # 다른 카메라가 연결되어 있는 경우에만 연결 해제
        if self.is_connected() and self.current_camera != camera_name:
            logger.info(f"다른 카메라가 연결되어 있습니다. 먼저 연결을 해제합니다.")
            self.disconnect_camera()

        # 카메라 ID 가져오기
        camera_id = self.available_cameras[camera_name]
        logger.info(f"카메라 '{camera_name}' 연결 시도 (ID: {camera_id}, 해상도: {resolution})")

        try:
            # 새 스레드 시작
            self.camera_thread = CameraThread(camera_id, resolution, camera_model=camera_name)

            # 시그널 연결
            self.camera_thread.frame_ready.connect(self.process_frame)
            self.camera_thread.fps_updated.connect(self.update_fps)
            self.camera_thread.error.connect(self.handle_error)

            # 스레드 시작
            self.camera_thread.start()
            self.current_camera = camera_name

            logger.info(f"카메라 '{camera_name}' 연결됨 (해상도: {resolution})")
            return True, f"카메라 '{camera_name}' 연결됨"

        except Exception as e:
            logger.error(f"카메라 연결 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"카메라 연결 오류: {str(e)}"

    def disconnect_camera(self):
        """카메라 연결 해제"""
        try:
            if self.camera_thread and self.camera_thread.isRunning():
                logger.info("카메라 연결 해제 중...")
                self.camera_thread.stop()
                self.camera_thread = None
                self.current_camera = None
                self.current_fps = 0
                logger.info("카메라 연결 해제 완료")
            else:
                logger.debug("연결된 카메라가 없습니다.")
        except Exception as e:
            logger.error(f"카메라 연결 해제 중 오류: {str(e)}")

    def process_frame(self, frame):
        """
        카메라 프레임 처리 및 신호 발생

        Args:
            frame (numpy.ndarray): 카메라 프레임
        """
        try:
            # 프레임 업데이트 신호 발생
            self.frame_updated.emit(frame)
        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {str(e)}")

    def update_fps(self, fps):
        """
        FPS 업데이트 처리

        Args:
            fps (float): FPS 값
        """
        self.current_fps = fps
        self.fps_updated.emit(fps)

    def handle_error(self, error_message):
        """
        카메라 오류 처리

        Args:
            error_message (str): 오류 메시지
        """
        logger.error(f"카메라 오류: {error_message}")
        self.error_signal.emit(error_message)

    def update_system_status(self):
        """시스템 상태 정보 업데이트 및 신호 발생"""
        try:
            # 시스템 부하 정보 가져오기
            system_load = self.get_system_load()

            # 시스템 상태 정보 시그널 발생
            self.system_status_signal.emit({
                'fps': self.current_fps,
                'cpu': system_load['cpu'],
                'memory': system_load['memory'],
                'gpu': system_load.get('gpu', 0)
            })
        except Exception as e:
            logger.error(f"시스템 상태 업데이트 중 오류: {str(e)}")

    def get_system_load(self):
        """
        시스템 부하 정보 반환

        Returns:
            dict: 시스템 부하 정보 (CPU, 메모리, GPU)
        """
        try:
            # CPU 및 메모리 정보
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent

            system_load = {
                'cpu': cpu_percent,
                'memory': memory_percent
            }

            # GPU 정보 추가 시도 (NVIDIA GPU가 있는 경우)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    gpu_load = float(result.stdout.strip())
                    system_load['gpu'] = gpu_load
            except:
                # GPU 정보를 가져올 수 없는 경우 무시
                pass

            return system_load

        except Exception as e:
            logger.error(f"시스템 부하 정보 가져오기 오류: {str(e)}")
            return {'cpu': 0, 'memory': 0}

    def is_connected(self):
        """
        카메라 연결 상태 확인

        Returns:
            bool: 카메라가 연결되어 있으면 True, 아니면 False
        """
        return self.camera_thread is not None and self.camera_thread.isRunning()

    def get_current_camera(self):
        """
        현재 연결된 카메라 이름 반환

        Returns:
            str: 카메라 이름 또는 None
        """
        return self.current_camera

    def get_camera_info(self):
        """
        현재 카메라 정보 반환

        Returns:
            dict: 카메라 정보
        """
        if not self.is_connected():
            return {"status": "disconnected"}

        return {
            "status": "connected",
            "name": self.current_camera,
            "fps": self.current_fps
        }