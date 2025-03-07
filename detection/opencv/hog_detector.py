#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# detection/opencv/hog_detector.py

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

from detection.detector_base import DetectorBase


class HOGDetector(DetectorBase):
    """HOG 특성을 이용한 사람 탐지기"""

    def __init__(self):
        self._hog = None
        self._params = {}

    def initialize(self, **kwargs) -> Tuple[bool, str]:
        """탐지기 초기화"""
        try:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # 매개변수 저장
            self._params = {
                "winStride": kwargs.get("winStride", (8, 8)),
                "padding": kwargs.get("padding", (4, 4)),
                "scale": kwargs.get("scale", 1.05)
            }

            return True, "HOG 사람 탐지기 초기화 완료"
        except Exception as e:
            return False, f"HOG 탐지기 초기화 실패: {str(e)}"

    def detect(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
        """객체 탐지 수행"""
        if self._hog is None:
            return frame, [], "탐지기가 초기화되지 않았습니다."

        # 파라미터 업데이트 (런타임에 변경 가능)
        params = self._params.copy()
        params.update(kwargs)

        # 성능을 위해 이미지 크기 조정
        height, width = frame.shape[:2]
        if width > 800:
            scale_factor = 800 / width
            frame_resized = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        else:
            scale_factor = 1.0
            frame_resized = frame

        # HOG 디스크립터로 사람 탐지
        boxes, weights = self._hog.detectMultiScale(
            frame_resized,
            winStride=params["winStride"],
            padding=params["padding"],
            scale=params["scale"]
        )

        # 결과 이미지에 박스 그리기
        result_frame = frame.copy()
        detections = []

        for i, (x, y, w, h) in enumerate(boxes):
            # 원본 이미지 크기에 맞게 박스 크기 조정
            if scale_factor != 1.0:
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)

            # 박스 그리기
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            confidence = float(weights[i]) if i < len(weights) else 0.5
            label = f'Person {confidence:.2f}'
            cv2.putText(result_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 탐지 정보 저장
            detections.append({
                "class_id": 0,
                "class_name": "person",
                "confidence": confidence,
                "box": [x, y, w, h]
            })

        # 처리 시간 정보
        message = f"HOG 탐지 완료: {len(detections)}개 객체"

        return result_frame, detections, message

    @property
    def name(self) -> str:
        return "OpenCV-HOG"

    @property
    def version(self) -> str:
        return "Default"

    @property
    def available_models(self) -> List[Dict[str, str]]:
        return [{"name": "HOG-Default", "version": "Default"}]