#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import cv2
import numpy as np
from detection.yolo.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class YOLOv4Detector(YOLODetector):
    """YOLOv4 탐지기 구현"""

    def __init__(self):
        super().__init__()
        self._name = "YOLOv4"
        self._version = "v4"
        self._net = None
        self._output_layers = []
        self._supported_models = [
            {"name": "YOLOv4-tiny", "weights": "yolov4-tiny.weights", "config": "yolov4-tiny.cfg"},
            {"name": "YOLOv4", "weights": "yolov4.weights", "config": "yolov4.cfg"}
        ]

    def load_model(self, weights_path, config_path, classes_path="models/yolo/coco.names", **kwargs):
        """
        YOLOv4 모델 로드

        Args:
            weights_path (str): 가중치 파일 경로
            config_path (str): 설정 파일 경로
            classes_path (str): 클래스 이름 파일 경로
            **kwargs: 추가 매개변수

        Returns:
            tuple: (성공 여부, 메시지)
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(weights_path):
                logger.error(f"가중치 파일을 찾을 수 없음: {weights_path}")
                return False, f"가중치 파일을 찾을 수 없습니다: {weights_path}"

            if not os.path.exists(config_path):
                logger.error(f"설정 파일을 찾을 수 없음: {config_path}")
                return False, f"설정 파일을 찾을 수 없습니다: {config_path}"

            # 모델 로드
            logger.info(f"YOLOv4 모델 로드 중: {weights_path}, {config_path}")
            self._net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

            # GPU 사용 설정 (가능한 경우)
            try:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("CUDA 백엔드로 YOLOv4 설정 완료")
            except:
                logger.warning("CUDA 백엔드를 사용할 수 없어 CPU 모드로 실행합니다")

            # 출력 레이어 이름 가져오기
            layer_names = self._net.getLayerNames()
            try:
                self._output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
            except:
                self._output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

            # 클래스 이름 로드
            self._classes = self.load_classes(classes_path)
            if not self._classes:
                logger.warning(f"클래스 파일 로드 실패: {classes_path}, 기본 클래스 사용")
                self._classes = ["person"]  # 기본값

            self._initialized = True
            logger.info(f"YOLOv4 모델 로드 완료: {len(self._classes)}개 클래스")

            return True, f"YOLOv4 모델 로드 완료"

        except Exception as e:
            logger.error(f"YOLOv4 모델 로드 실패: {str(e)}")
            return False, f"YOLOv4 모델 로드 실패: {str(e)}"

    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.4, **kwargs):
        """
        YOLOv4 객체 탐지

        Args:
            frame (numpy.ndarray): 분석할 이미지 프레임
            conf_threshold (float): 신뢰도 임계값
            nms_threshold (float): 비최대 억제 임계값
            **kwargs: 추가 매개변수

        Returns:
            list: 탐지된 객체 목록
        """
        if not self._initialized or self._net is None:
            logger.warning("YOLOv4 모델이 초기화되지 않음")
            return []

        try:
            # 성능 최적화를 위한 옵션
            optimize = kwargs.get('optimize', True)
            target_size = kwargs.get('target_size', (416, 416))
            scale_factor = kwargs.get('scale_factor', None)

            # 원본 크기 저장
            height, width = frame.shape[:2]

            # 처리 속도를 위해 이미지 크기 조정 (선택 사항)
            if optimize and scale_factor is not None:
                # 이미지 축소 (처리 속도 향상)
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            else:
                small_frame = frame

            # YOLO 입력 형식으로 변환
            blob = cv2.dnn.blobFromImage(
                small_frame,
                1 / 255.0,
                target_size,
                swapRB=True,
                crop=False
            )

            # 입력 설정 및 추론 실행
            self._net.setInput(blob)
            outputs = self._net.forward(self._output_layers)

            # 원본 이미지 크기에 맞는 스케일 계산
            if optimize and scale_factor is not None:
                # 원본 크기 비율 계산
                scale_width = width / (small_frame.shape[1] * scale_factor)
                scale_height = height / (small_frame.shape[0] * scale_factor)
                size = (scale_width, scale_height)
            else:
                size = (width, height)

            # 결과 처리
            detections = self.process_detections(outputs, size, conf_threshold, nms_threshold)

            # 결과 로깅
            person_count = sum(1 for d in detections if d.get('class_name', '').lower() == 'person')
            logger.debug(f"YOLOv4 탐지 결과: 총 {len(detections)}개 객체, {person_count}명의 사람")

            return detections

        except Exception as e:
            logger.error(f"YOLOv4 탐지 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def release(self):
        """리소스 해제"""
        self._net = None
        self._output_layers = []
        self._initialized = False
        logger.info("YOLOv4 리소스 해제 완료")