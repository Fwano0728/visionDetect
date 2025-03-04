#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import os
import psutil
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread


class CameraThread(QThread):
    """카메라 스트리밍을 위한 스레드 클래스"""
    frame_ready = pyqtSignal(object)
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, camera_id, resolution, camera_model=""):
        super().__init__()
        self.camera_id = camera_id
        self.width, self.height = map(int, resolution.split('x'))
        self.running = False
        self.cap = None
        self.camera_model = camera_model

        # 카메라 모델별 최적의 FPS 설정
        self.target_fps = self._get_optimal_fps(camera_model)

    def _get_optimal_fps(self, camera_model):
        """카메라 모델별 최적의 FPS 반환"""
        print(f"카메라 모델: {camera_model}")
        if "StreamCam" in camera_model:
            return 30  # StreamCam은 30fps로 제한 (USB 대역폭 이슈)
        elif "C920" in camera_model:
            return 30  # C920은 30fps
        else:
            return 30  # 기본값

        def run(self):
            """스레드 실행 - 카메라에서 프레임을 지속적으로 가져옴"""
            try:
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    self.error.emit(f"카메라 {self.camera_id} 열기 실패")
                    return

                # 포맷 설정 (MJPG 사용)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

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
                fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

                print(
                    f"카메라 설정: ID={self.camera_id}, 해상도={actual_width}x{actual_height}, FPS={actual_fps}, 코덱={fourcc_str}")

                # FPS 측정 변수
                frame_count = 0
                fps = 0
                fps_start_time = time.time()
                fps_update_interval = 0.5  # FPS 업데이트 간격 (초)

                self.running = True

                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.error.emit("프레임 캡처 실패")
                        break

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

                    # 과도한 CPU 사용 방지
                    sleep_time = max(1.0 / (self.target_fps * 1.5), 0.001)  # 최소 지연 1ms
                    time.sleep(sleep_time)

            except Exception as e:
                self.error.emit(f"카메라 오류: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                if self.cap:
                    self.cap.release()

        def stop(self):
            """스레드 정지"""
            self.running = False
            self.wait()

    class CameraManager(QObject):
        """카메라 연결 및 관리 클래스"""

        # 시그널 정의
        frame_updated = pyqtSignal(object)
        fps_updated = pyqtSignal(float)
        system_status_signal = pyqtSignal(dict)

        def __init__(self):
            super().__init__()
            self.camera_thread = None
            self.current_camera = None
            self.available_cameras = self.scan_cameras()
            self.current_fps = 0
            self.load_update_time = 0
            self.system_load = {'cpu': 0, 'memory': 0}

        def scan_cameras(self):
            """시스템에 연결된 카메라 스캔"""
            available_cameras = {}

            # 터미널 명령어로 카메라 정보 확인
            import subprocess
            try:
                result = subprocess.run(['ls', '-l', '/dev/video*'], capture_output=True, text=True)
                print("시스템 카메라 장치:")
                print(result.stdout)
            except Exception as e:
                print(f"카메라 장치 확인 중 오류: {str(e)}")

            # 가능한 모든 카메라 ID 확인
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # 기본 이름
                            camera_name = f"Camera {i}"

                            # v4l2-ctl 명령으로 카메라 모델명 가져오기 시도
                            try:
                                device_path = f"/dev/video{i}"
                                cmd = f"v4l2-ctl -d {device_path} --all | grep 'Card\\|Driver'"
                                info = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                                print(f"카메라 {i} 정보: {info.stdout}")

                                if "StreamCam" in info.stdout:
                                    camera_name = "Logitech StreamCam"
                                elif "C920" in info.stdout or "HD Pro Webcam" in info.stdout:
                                    camera_name = "Logitech C920"
                            except:
                                # 식별 실패 시 기본 매핑
                                if i == 0:
                                    camera_name = "Logitech C920"
                                elif i == 2:
                                    camera_name = "Logitech StreamCam"

                            available_cameras[camera_name] = i
                            print(f"카메라 발견: ID {i} - {camera_name}")
                        cap.release()
                except Exception as e:
                    print(f"카메라 ID {i} 확인 중 오류: {str(e)}")

            # 카메라가 없으면 기본값 제공
            if not available_cameras:
                print("카메라를 찾지 못했습니다. 기본값을 사용합니다.")
                available_cameras = {"기본 카메라": 0}

            print(f"사용 가능한 카메라: {available_cameras}")
            return available_cameras

        def connect_camera(self, camera_name, resolution):
            """카메라 연결"""
            if camera_name not in self.available_cameras:
                return False, f"카메라 '{camera_name}'을(를) 찾을 수 없습니다."

            camera_id = self.available_cameras[camera_name]
            print(f"카메라 '{camera_name}' 연결 시도 (ID: {camera_id})")

            # 이미 실행 중인 스레드가 있으면 중지
            if self.camera_thread and self.camera_thread.isRunning():
                self.disconnect_camera()

            # 새 스레드 시작
            self.camera_thread = CameraThread(camera_id, resolution, camera_model=camera_name)

            # 시그널 연결
            self.camera_thread.frame_ready.connect(self.process_frame)
            self.camera_thread.fps_updated.connect(self.update_fps)
            self.camera_thread.error.connect(self.handle_error)

            self.camera_thread.start()
            self.current_camera = camera_name

            return True, f"카메라 '{camera_name}' 연결됨"

        def disconnect_camera(self):
            """카메라 연결 해제"""
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop()
                self.camera_thread = None
                self.current_camera = None
                self.current_fps = 0

        def process_frame(self, frame):
            """카메라 프레임 처리 및 신호 발생"""
            # 시스템 부하 정보 가져오기
            system_load = self.get_system_load()

            # 시스템 상태 정보 시그널 발생
            self.system_status_signal.emit({
                'fps': self.current_fps,
                'cpu': system_load['cpu'],
                'memory': system_load['memory'],
                'gpu': system_load.get('gpu', 0)
            })

            # 프레임 업데이트 신호 발생
            self.frame_updated.emit(frame)

        def update_fps(self, fps):
            """FPS 업데이트 처리"""
            self.current_fps = fps
            self.fps_updated.emit(fps)

        def handle_error(self, error_message):
            """카메라 오류 처리"""
            print(f"카메라 오류: {error_message}")

        def get_system_load(self):
            """시스템 부하 정보 반환 (1초에 한 번만 업데이트)"""
            current_time = time.time()

            # 1초에 한 번만 업데이트
            if current_time - self.load_update_time > 1.0:
                self.load_update_time = current_time

                # CPU 및 메모리 정보
                self.system_load = {
                    'cpu': psutil.cpu_percent(interval=None),
                    'memory': psutil.virtual_memory().percent
                }

                # GPU 정보 추가 시도 (NVIDIA GPU가 있는 경우)
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True)
                    if result.returncode == 0:
                        gpu_load = float(result.stdout.strip())
                        self.system_load['gpu'] = gpu_load
                except:
                    pass

            return self.system_load

        def is_connected(self):
            """카메라 연결 상태 확인"""
            return self.camera_thread is not None and self.camera_thread.isRunning()

        def get_current_camera(self):
            """현재 연결된 카메라 이름 반환"""
            return self.current_camera