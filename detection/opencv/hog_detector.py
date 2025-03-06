#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import cv2
import numpy as np
from detection.detector_base import DetectorBase

logger = logging.getLogger(__name__)


class HOGDetector(DetectorBase):
    """OpenCV HOG 사람 탐지기 구현"""

    def __init__(self):
        super().__init__()
        self._name = "OpenCV-HOG"
        self._version = "Default"
        self._hog = None
        self._supported_models = [
            {"name": "HOG-Default"}
        ]

    def load_model(self, model_path=None, config_path=None, **kwargs):
        """
        HOG 사람 탐지기 초기화

        Args:
            model_path (str, optional): 미사용 (호환성 유지용)
            config_path (str, optional): 미사용 (호환성 유지용)
            **kwargs: 추가 매개변수

        Returns:
            tuple: (성공 여부, 메시지)
        """
        try:
            # HOG 디스크립터 초기화
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # 클래스 설정 (HOG는 사람만 탐지)
            self._classes = ["person"]

            self._initialized = True
            logger.info("HOG 사람 탐지기 초기화 완료")

            return True, "HOG 사람 탐지기 초기화 완료"

        except Exception as e:
            logger.error(f"HOG 탐지기 초기화 실패: {str(e)}")
            return False, f"HOG 탐지기 초기화 실패: {str(e)}"

    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.4, **kwargs):
        """
        HOG를 사용하여 사람 탐지

        Args:
            frame (numpy.ndarray): 분석할 이미지 프레임
            conf_threshold (float): 신뢰도 임계값 (HOG에서는 제한적으로 사용)
            nms_threshold (float): 미사용 (호환성 유지용)
            **kwargs: 추가 매개변수

        Returns:
            list: 탐지된 객체 목록
        """
        if not self._initialized or self._hog is None:
            logger.warning("HOG 탐지기가 초기화되지 않음")
            return []

        try:
            # 성능을 위해 이미지 크기 조정
            height, width = frame.shape[:2]
            scale = kwargs.get('scale', 1.0)

            if width > 800 and scale == 1.0:
                scale = 800 / width

            if scale != 1.0:
                frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
            else:
                frame_resized = frame

            # HOG 매개변수
            win_stride = kwargs.get('win_stride', (8, 8))
            padding = kwargs.get('padding', (4, 4))
            scale_factor = kwargs.get('scale_factor', 1.05)

            # HOG 디스크립터를 사용하여 사람 탐지
            boxes, weights = self._hog.detectMultiScale(
                frame_resized,
                winStride=win_stride,
                padding=padding,
                scale=scale_factor
            )

            # 결과 처리
            detections = []

            for i, (x, y, w, h) in enumerate(boxes):
                # 원본 이미지 크기에 맞게 박스 크기 조정
                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)

                # 신뢰도
                confidence = float(weights[i]) if i < len(weights) else 0.5

                # 신뢰도 임계값 적용
                if confidence >= conf_threshold:
                    detections.append({
                        'box': [x, y, w, h],
                        'confidence': confidence,
                        'class_id': 0,
                        'class_name': 'person'
                    })

            # 결과 로깅
            logger.debug(f"HOG 탐지 결과: {len(detections)}명의 사람")

            return detections

        except Exception as e:
            logger.error(f"HOG 탐지 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def postprocess(self, frame, detections):
        """
        HOG 탐지 결과 시각화

        Args:
            frame (numpy.ndarray): 원본 이미지 프레임
            detections (list): 탐지 결과 목록

        Returns:
            numpy.ndarray: 시각화된 프레임
        """
        result_frame = frame.copy()

        for detection in detections:
            box = detection.get('box', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0)

            x, y, w, h = box

            # 사각형 그리기 (녹색)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 레이블 그리기
            label = f"Person {confidence:.2f}"
            cv2.putText(result_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_frame

    def release(self):
        """리소스 해제"""
        self._hog = None
        self._initialized = False
        logger.info("HOG 리소스 해제 완료")