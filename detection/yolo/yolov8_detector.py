#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from detection.yolo.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class YOLOv8Detector(YOLODetector):
    """YOLOv8 탐지기 구현 (Ultralytics 라이브러리 사용)"""

    def __init__(self):
        super().__init__()
        self._name = "YOLOv8"
        self._version = "v8"
        self._model = None
        self._supported_models = [
            {"name": "YOLOv8n", "weights": "yolov8n.pt"},
            {"name": "YOLOv8s", "weights": "yolov8s.pt"},
            {"name": "YOLOv8m", "weights": "yolov8m.pt"},
            {"name": "YOLOv8l", "weights": "yolov8l.pt"},
            {"name": "YOLOv8x", "weights": "yolov8x.pt"}
        ]

    def load_model(self, weights_path, config_path=None, **kwargs):
        """
        YOLOv8 모델 로드

        Args:
            weights_path (str): 모델 파일 경로
            config_path (str, optional): 미사용 (호환성 유지용)
            **kwargs: 추가 매개변수

        Returns:
            tuple: (성공 여부, 메시지)
        """
        try:
            # Ultralytics 패키지 가져오기 시도
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("ultralytics 패키지를 찾을 수 없음")
                return False, "YOLOv8을 사용하려면 'pip install ultralytics' 명령으로 패키지를 설치하세요."

            # 파일 존재 확인
            if not os.path.exists(weights_path):
                logger.error(f"모델 파일을 찾을 수 없음: {weights_path}")
                return False, f"모델 파일을 찾을 수 없습니다: {weights_path}"

            # 모델 로드
            logger.info(f"YOLOv8 모델 로드 중: {weights_path}")
            self._model = YOLO(weights_path)

            # 클래스 정보 가져오기
            self._classes = list(self._model.names.values())

            self._initialized = True
            logger.info(f"YOLOv8 모델 로드 완료: {len(self._classes)}개 클래스")

            return True, f"YOLOv8 모델 로드 완료"

        except Exception as e:
            logger.error(f"YOLOv8 모델 로드 실패: {str(e)}")
            return False, f"YOLOv8 모델 로드 실패: {str(e)}"

    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.45, **kwargs):
        """
        YOLOv8 객체 탐지

        Args:
            frame (numpy.ndarray): 분석할 이미지 프레임
            conf_threshold (float): 신뢰도 임계값
            nms_threshold (float): 비최대 억제 임계값 (iou_threshold로 사용)
            **kwargs: 추가 매개변수

        Returns:
            list: 탐지된 객체 목록
        """
        if not self._initialized or self._model is None:
            logger.warning("YOLOv8 모델이 초기화되지 않음")
            return []

        try:
            # 모델 예측 실행
            results = self._model(
                frame,
                conf=conf_threshold,
                iou=nms_threshold,
                verbose=False
            )

            # 첫 번째 결과 사용
            result = results[0]

            # 결과 변환
            detections = []

            # 박스 정보 추출
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]

                # 좌표를 YOLO 형식으로 변환 [x, y, width, height]
                detections.append({
                    'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

            # 결과 로깅
            person_count = sum(1 for d in detections if d.get('class_name', '').lower() == 'person')
            logger.debug(f"YOLOv8 탐지 결과: 총 {len(detections)}개 객체, {person_count}명의 사람")

            return detections

        except Exception as e:
            logger.error(f"YOLOv8 탐지 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def postprocess(self, frame, detections):
        """
        탐지 결과 시각화

        Args:
            frame (numpy.ndarray): 원본 이미지 프레임
            detections (list): 탐지 결과 목록

        Returns:
            numpy.ndarray: 시각화된 프레임
        """
        # 시각화 메서드 선택
        if hasattr(self._model, 'plot') and callable(self._model.plot):
            # YOLOv8의 내장 시각화 사용 (더 예쁜 시각화)
            try:
                # detections를 YOLOv8 Results 형식에 맞게 변환하는 대신
                # 마지막으로 실행된 결과를 사용하여 시각화 (내부적으로 결과 저장)
                result_frame = self._model.plot()
                return result_frame
            except:
                # 내장 시각화 실패 시 기본 구현 사용
                return super().postprocess(frame, detections)
        else:
            # 기본 구현 사용
            return super().postprocess(frame, detections)

    def release(self):
        """리소스 해제"""
        self._model = None
        self._initialized = False
        logger.info("YOLOv8 리소스 해제 완료")