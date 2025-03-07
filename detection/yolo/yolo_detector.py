# detection/yolo/yolo_detector.py

# detector_manager.py 파일 상단에 추가
import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import time  # time 모듈 추가
from typing import List, Dict, Tuple, Any

from detection.detector_base import DetectorBase


class YOLODetector(DetectorBase):
    """YOLO 탐지기 일반 인터페이스"""

    def __init__(self):
        self._detectors = {}
        self._current_detector = None
        self._model_name = "YOLO"
        self._version = None
        self._available_versions = ["v4"]

        # 지원되는 버전별 탐지기 초기화
        self._init_detectors()

    def _init_detectors(self):
        """사용 가능한 YOLO 버전별 탐지기 초기화"""
        try:
            # YOLOv4 탐지기 추가
            from detection.yolo.yolov4_detector import YOLOv4Detector
            self._detectors["v4"] = YOLOv4Detector()
            logger.info("YOLOv4 탐지기 등록 완료")

            # 다른 YOLO 버전 탐지기 추가 (필요시)
            # try:
            #     from detection.yolo.yolov8_detector import YOLOv8Detector
            #     self._detectors["v8"] = YOLOv8Detector()
            #     self._available_versions.append("v8")
            #     logger.info("YOLOv8 탐지기 등록 완료")
            # except ImportError:
            #     logger.warning("YOLOv8 탐지기를 불러올 수 없습니다. ultralytics 패키지가 필요합니다.")

        except Exception as e:
            logger.error(f"YOLO 탐지기 초기화 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def initialize(self, **kwargs) -> Tuple[bool, str]:
        """탐지기 초기화"""
        logger.info(f"YOLO 초기화 - 파라미터: {kwargs}")

        # 버전 선택
        version = kwargs.get("version", "v4")
        if version not in self._detectors:
            logger.error(f"지원되지 않는 YOLO 버전: {version}")
            logger.error(f"사용 가능한 버전: {list(self._detectors.keys())}")
            return False, f"지원되지 않는 YOLO 버전: {version}"

        # 선택한 버전의 탐지기 설정
        detector = self._detectors[version]
        self._current_detector = detector
        self._version = version

        # 선택한 탐지기 초기화
        logger.info(f"버전 {version} 탐지기 초기화 시도")
        success, message = detector.initialize(**kwargs)
        return success, message


    def detect(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Dict[str, Any]], str, float]:
        """객체 탐지 수행"""
        if self._current_detector is None:
            return frame, [], "YOLO 탐지기가 초기화되지 않았습니다.", 0.0

        # 현재 설정된 탐지기로 탐지 수행
        try:
            start_time = time.time()

            # 반환 값 형식 처리
            result = self._current_detector.detect(frame, **kwargs)

            if isinstance(result, tuple):
                if len(result) == 4:  # 4개 값 반환하는 경우
                    return result
                elif len(result) == 3:  # 3개 값만 반환하는 경우
                    result_frame, detections, message = result
                    detection_time = time.time() - start_time
                    detection_fps = 1.0 / detection_time if detection_time > 0 else 0
                    return result_frame, detections, message, detection_fps

            # 예상치 못한 형식
            return frame, [], "잘못된 탐지 결과 형식", 0.0
        except Exception as e:
            return frame, [], f"탐지 오류: {str(e)}", 0.0

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version or "Unknown"

    @property
    def available_versions(self) -> List[str]:
        """사용 가능한 YOLO 버전 목록"""
        versions = list(self._detectors.keys())
        logger.info(f"사용 가능한 YOLO 버전: {versions}")  # 디버깅용
        return versions if versions else ["v4"]  # 비어있으면 기본값 반환

    @property
    def available_models(self) -> List[Dict[str, str]]:
        """사용 가능한 모델 목록"""
        models = []
        for version, detector in self._detectors.items():
            version_models = detector.available_models
            models.extend(version_models)
        return models

    def get_models_for_version(self, version: str) -> List[str]:
        """특정 버전에 대한 모델 목록"""
        logger.info(f"버전 {version}에 대한 모델 목록 요청")
        if version in self._detectors:
            detector = self._detectors[version]
            if hasattr(detector, "available_models"):
                models = [model["name"] for model in detector.available_models]
                logger.info(f"버전 {version} 모델 목록: {models}")
                return models
        logger.warning(f"버전 {version}에 대한 모델 없음")
        return ["YOLOv4-tiny", "YOLOv4"]  # 기본 모델 목록 제공