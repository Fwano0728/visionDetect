#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from detection.detector_base import DetectorBase

logger = logging.getLogger(__name__)


class YOLODetector(DetectorBase):
    """
    YOLO 기본 탐지기 클래스
    모든 YOLO 버전 구현체의 기반 클래스
    """

    def __init__(self):
        super().__init__()
        self._name = "YOLO"
        self._version = "base"
        self._classes = []
        self._supported_models = []

    def load_classes(self, classes_file):
        """
        클래스 이름 파일 로드

        Args:
            classes_file (str): 클래스 파일 경로

        Returns:
            list: 클래스 이름 목록
        """
        try:
            if not os.path.exists(classes_file):
                logger.error(f"클래스 파일을 찾을 수 없음: {classes_file}")
                return []

            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]

            logger.info(f"{len(class_names)}개 클래스 로드 완료")
            return class_names
        except Exception as e:
            logger.error(f"클래스 파일 로드 중 오류: {str(e)}")
            return []

    def preprocess(self, frame, target_size=(416, 416)):
        """
        YOLO용 이미지 전처리

        Args:
            frame (numpy.ndarray): 원본 이미지
            target_size (tuple): 타겟 이미지 크기

        Returns:
            tuple: (전처리된 이미지, 원본 크기)
        """
        import cv2

        # 원본 크기 저장
        height, width = frame.shape[:2]

        # YOLO 입력 형식으로 변환
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            target_size,
            swapRB=True,
            crop=False
        )

        return blob, (width, height)

    def process_detections(self, outputs, size, conf_threshold, nms_threshold):
        """
        YOLO 탐지 결과 처리

        Args:
            outputs: 네트워크 출력
            size (tuple): 원본 이미지 크기 (width, height)
            conf_threshold (float): 신뢰도 임계값
            nms_threshold (float): 비최대 억제 임계값

        Returns:
            list: 처리된 탐지 결과
        """
        import cv2

        width, height = size
        boxes = []
        confidences = []
        class_ids = []

        # 출력 처리 (구현체에서 재정의 가능)
        # 기본 구현은 YOLOv4 형식
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    # 객체 위치 계산
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 좌표 계산
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 비최대 억제 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # 결과 가공
        detections = []

        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            elif isinstance(indices, np.ndarray) and len(indices.shape) > 1:
                indices = indices.flatten()

            for i in indices:
                box = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                class_name = self._classes[class_id] if class_id < len(self._classes) else f"Class {class_id}"

                detections.append({
                    'box': box,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

        return detections

    def load_model(self, model_path, config_path=None, **kwargs):
        """
        모델 로드 (자식 클래스에서 구현 필요)
        """
        raise NotImplementedError("자식 클래스에서 구현해야 합니다.")

    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.4, **kwargs):
        """
        객체 탐지 (자식 클래스에서 구현 필요)
        """
        raise NotImplementedError("자식 클래스에서 구현해야 합니다.")