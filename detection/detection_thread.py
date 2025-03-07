#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# detection/detection_thread.py

from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import time  # time 모듈 명시적 import
import numpy as np
import logging
import traceback

logger = logging.getLogger(__name__)


class DetectionThread(QThread):
    """비동기 객체 탐지 스레드"""

    # 시그널 정의
    detection_complete = pyqtSignal(object, list, str, float)  # 결과 프레임, 탐지 결과, 메시지, FPS
    error = pyqtSignal(str)  # 오류 메시지

    def __init__(self, detector_manager):
        """스레드 초기화"""
        super().__init__()
        self.detector_manager = detector_manager
        self.frame = None
        self.running = False
        self.frame_ready = False
        self.detection_fps = 0.0
        self.last_detection_time = 0.0
        self.skip_frames = 0  # 프레임 건너뛰기 카운터

        logger.info("탐지 스레드 초기화 완료")

    def set_frame(self, frame):
        """새 프레임 설정

        Args:
            frame: 처리할 카메라 프레임
        """
        if frame is not None:
            self.frame = frame.copy()  # 안전하게 복사
            self.frame_ready = True

    def run(self):
        """스레드 메인 실행 루프"""
        logger.info("탐지 스레드 시작")
        self.running = True

        while self.running:
            # 프레임이 준비되지 않았거나 탐지기가 비활성화된 경우 짧게 대기
            if not self.frame_ready or not self.detector_manager.is_active():
                time.sleep(0.01)
                continue

            # 프레임 스킵 로직 (성능 최적화)
            self.skip_frames += 1
            if self.skip_frames < 2:  # 3번째 프레임마다 처리
                self.frame_ready = False
                continue

            self.skip_frames = 0

            try:
                # 프레임 복사 및 준비 상태 재설정
                current_frame = self.frame.copy()
                self.frame_ready = False

                # 프레임 크기 및 속도 최적화
                height, width = current_frame.shape[:2]
                scale_factor = 0.5  # 크기 감소

                # 큰 이미지만 리사이징 (작은 이미지는 그대로 처리)
                if width > 640:
                    process_frame = cv2.resize(current_frame, (0, 0), fx=scale_factor, fy=scale_factor)
                else:
                    process_frame = current_frame

                # 객체 탐지 수행
                try:
                    # 탐지 매니저의 detect 메서드 호출 (이제 4개 값 반환 예상)
                    result_frame, detections, message, detection_fps = self.detector_manager.detect(process_frame)

                    # 처리된 프레임을 원본 크기로 다시 확대 (필요한 경우)
                    if width > 640 and scale_factor < 1.0 and result_frame.shape != current_frame.shape:
                        result_frame = cv2.resize(result_frame, (width, height))

                    # 탐지 성능 저장
                    self.detection_fps = detection_fps

                    # 결과 시그널 발생
                    self.detection_complete.emit(result_frame, detections, message, detection_fps)

                except Exception as detect_error:
                    # 탐지 오류 발생 시 기본값 설정
                    error_msg = f"탐지 오류: {str(detect_error)}"
                    logger.error(error_msg)
                    self.error.emit(error_msg)

                    # 원본 프레임 반환
                    self.detection_complete.emit(current_frame, [], error_msg, 0.0)

            except Exception as e:
                # 전체 처리 과정에서 발생한 예외 처리
                error_msg = f"탐지 스레드 오류: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.error.emit(error_msg)
                time.sleep(0.1)  # 오류 발생 시 잠시 대기

        logger.info("탐지 스레드 종료")

    def stop(self):
        """스레드 중지"""
        logger.info("탐지 스레드 중지 요청")
        self.running = False
        self.wait()  # 스레드가 완전히 종료될 때까지 대기