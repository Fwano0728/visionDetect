#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DetectorBase(ABC):
    """
    객체 탐지기 기본 인터페이스
    모든 탐지기 구현체는 이 클래스를 상속해야 함
    """

    def __init__(self):
        self._name = "BaseDetector"
        self._version = "0.0"
        self._initialized = False
        self._supported_models = []
        logger.debug(f"탐지기 초기화: {self._name} {self._version}")

    @property
    def name(self):
        """탐지기 이름 반환"""
        return self._name

    @property
    def version(self):
        """탐지기 버전 반환"""
        return self._version

    @property
    def initialized(self):
        """초기화 상태 반환"""
        return self._initialized

    @property
    def available_models(self):
        """지원하는 모델 목록 반환"""
        return self._supported_models

    @abstractmethod
    def load_model(self, model_path, config_path=None, **kwargs):
        """
        모델 로드

        Args:
            model_path (str): 모델 파일 경로
            config_path (str, optional): 설정 파일 경로
            **kwargs: 추가 매개변수

        Returns:
            tuple: (성공 여부, 메시지)
        """
        pass

    @abstractmethod
    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.4, **kwargs):
        """
        이미지에서 객체 탐지 수행

        Args:
            frame (numpy.ndarray): 분석할 이미지 프레임
            conf_threshold (float): 신뢰도 임계값
            nms_threshold (float): 비최대 억제 임계값
            **kwargs: 추가 매개변수

        Returns:
            list: 탐지된 객체 목록 (각 객체는 딕셔너리 형태)
        """
        pass

    def preprocess(self, frame):
        """
        프레임 전처리 (크기 조정, 정규화 등)

        Args:
            frame (numpy.ndarray): 원본 이미지 프레임

        Returns:
            numpy.ndarray: 전처리된 프레임
        """
        # 기본 구현은 아무것도 하지 않음 (구현체에서 재정의 가능)
        return frame

    def postprocess(self, frame, detections):
        """
        탐지 결과 후처리 및 시각화

        Args:
            frame (numpy.ndarray): 원본 이미지 프레임
            detections (list): 탐지 결과 목록

        Returns:
            numpy.ndarray: 시각화된 프레임
        """
        # 기본 구현 - 각 탐지에 사각형 그리기
        result_frame = frame.copy()

        for detection in detections:
            box = detection.get('box', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0)
            class_name = detection.get('class_name', 'Unknown')

            x, y, w, h = box

            # 클래스에 따라 색상 결정
            if class_name.lower() == 'person':
                color = (0, 255, 0)  # 녹색 - 사람
            else:
                color = (255, 0, 0)  # 빨간색 - 기타 객체

            # 사각형 그리기
            import cv2
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)

            # 레이블 그리기
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(result_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_frame

    def release(self):
        """
        리소스 해제 (메모리 정리 등)
        """
        self._initialized = False
        logger.debug(f"탐지기 리소스 해제: {self._name} {self._version}")

    def __str__(self):
        """문자열 표현"""
        return f"{self._name} v{self._version} ({'초기화됨' if self._initialized else '미초기화'})"