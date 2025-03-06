#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class DetectorManager(QObject):
    """모든 객체 탐지 알고리즘을 관리하는 클래스"""

    # 시그널 정의
    detection_result_signal = pyqtSignal(str)
    detection_complete_signal = pyqtSignal(object, list, str)

    def __init__(self):
        super().__init__()
        self.detector_type = None
        self.detector_version = None
        self.detector_model = None
        self.detector_params = {}
        self.current_detector = None
        self.detectors = {}

        # 초기화
        self._init_detectors()
        logger.info("탐지기 매니저 초기화 완료")

    def _init_detectors(self):
        """사용 가능한 모든 탐지기 초기화"""
        # OpenCV HOG 탐지기
        from detection.opencv.hog_detector import HOGDetector
        self.detectors["OpenCV-HOG"] = {
            "versions": ["Default"],
            "models": {"Default": ["HOG-Default"]},
            "class": HOGDetector
        }

        # YOLO 탐지기
        try:
            from detection.yolo.yolov4_detector import YOLOv4Detector
            self.detectors["YOLO"] = {
                "versions": ["v4"],
                "models": {"v4": [model["name"] for model in YOLOv4Detector().available_models]},
                "class": {
                    "v4": YOLOv4Detector
                }
            }

            # YOLOv8 지원 확인
            try:
                from detection.yolo.yolov8_detector import YOLOv8Detector
                # 기존 YOLO 버전 목록에 v8 추가
                self.detectors["YOLO"]["versions"].append("v8")
                self.detectors["YOLO"]["models"]["v8"] = [model["name"] for model in YOLOv8Detector().available_models]
                self.detectors["YOLO"]["class"]["v8"] = YOLOv8Detector
                logger.info("YOLOv8 지원 활성화됨")
            except ImportError:
                logger.warning("YOLOv8 지원을 위해 ultralytics 패키지가 필요합니다.")
        except ImportError:
            logger.warning("YOLO 탐지기를 가져올 수 없습니다.")

        # 추가 탐지기 구현 시 여기에 추가

    def get_available_detectors(self):
        """사용 가능한 탐지기 유형 목록 반환"""
        return list(self.detectors.keys())

    def get_available_versions(self, detector_type):
        """지정된 탐지기 유형에 대한 사용 가능한 버전 목록 반환"""
        if detector_type in self.detectors:
            return self.detectors[detector_type]["versions"]
        return []

    def get_available_models(self, detector_type, version):
        """지정된 탐지기 유형 및 버전에 대한 사용 가능한 모델 목록 반환"""
        if detector_type in self.detectors:
            if version in self.detectors[detector_type]["models"]:
                return self.detectors[detector_type]["models"][version]
        return []

    def set_detector(self, detector_type, version=None, model=None, **params):
        """
        탐지기 설정

        Args:
            detector_type (str): 탐지기 유형 (YOLO, OpenCV-HOG 등)
            version (str, optional): 탐지기 버전
            model (str, optional): 모델 이름
            **params: 추가 매개변수

        Returns:
            tuple: (성공 여부, 메시지)
        """
        # 탐지기 비활성화
        if detector_type is None:
            # 기존 탐지기 해제
            if self.current_detector:
                self.current_detector.release()
                self.current_detector = None

            self.detector_type = None
            self.detector_version = None
            self.detector_model = None
            self.detector_params = {}
            logger.info("탐지기 비활성화됨")
            return True, "탐지기가 비활성화되었습니다."

        # 기존 설정과 같으면 아무것도 하지 않음
        if (detector_type == self.detector_type and
                version == self.detector_version and
                model == self.detector_model and
                params == self.detector_params):
            logger.debug("탐지기 설정이 동일하여 변경 없음")
            return True, "현재 설정과 동일합니다."

        # 기존 탐지기 해제
        if self.current_detector:
            self.current_detector.release()
            self.current_detector = None

        # 새 탐지기 설정
        if detector_type in self.detectors:
            # HOG 탐지기는 버전과 모델이 고정됨
            if detector_type == "OpenCV-HOG":
                detector_class = self.detectors[detector_type]["class"]
                detector_instance = detector_class()
                success, message = detector_instance.load_model(**params)

                if success:
                    self.current_detector = detector_instance
                    self.detector_type = detector_type
                    self.detector_version = "Default"
                    self.detector_model = "HOG-Default"
                    self.detector_params = params
                    logger.info(f"HOG 탐지기 설정 완료: {message}")
                    return True, message
                else:
                    logger.error(f"HOG 탐지기 설정 실패: {message}")
                    return False, message

            # YOLO 탐지기는 버전과 모델 지정 필요
            elif detector_type == "YOLO":
                if not version or version not in self.detectors[detector_type]["versions"]:
                    logger.error(f"잘못된 YOLO 버전: {version}")
                    return False, f"잘못된 YOLO 버전: {version}"

                if not model or model not in self.detectors[detector_type]["models"][version]:
                    logger.error(f"잘못된 YOLO 모델: {model}")
                    return False, f"잘못된 YOLO 모델: {model}"

                # 모델 파일 경로 설정
                models_dir = os.path.join("models", "yolo")
                os.makedirs(models_dir, exist_ok=True)

                # 탐지기 클래스 인스턴스 생성
                detector_class = self.detectors[detector_type]["class"][version]
                detector_instance = detector_class()

                # 모델 정보 가져오기
                model_info = None
                for m in detector_instance.available_models:
                    if m["name"] == model:
                        model_info = m
                        break

                if not model_info:
                    logger.error(f"모델 정보를 찾을 수 없음: {model}")
                    return False, f"모델 정보를 찾을 수 없습니다: {model}"

                # 모델 파일 경로
                weights_path = os.path.join(models_dir, model_info["weights"])
                config_path = os.path.join(models_dir, model_info.get("config", "")) if "config" in model_info else None

                # 파일 존재 여부 확인
                if not os.path.exists(weights_path):
                    logger.error(f"모델 파일을 찾을 수 없음: {weights_path}")
                    return False, f"모델 파일을 찾을 수 없습니다: {weights_path}"

                if config_path and not os.path.exists(config_path):
                    logger.error(f"설정 파일을 찾을 수 없음: {config_path}")
                    return False, f"설정 파일을 찾을 수 없습니다: {config_path}"

                # 모델 로드
                load_params = {**params}
                if config_path:
                    success, message = detector_instance.load_model(weights_path, config_path, **load_params)
                else:
                    success, message = detector_instance.load_model(weights_path, **load_params)

                if success:
                    self.current_detector = detector_instance
                    self.detector_type = detector_type
                    self.detector_version = version
                    self.detector_model = model
                    self.detector_params = params
                    logger.info(f"{detector_type} {version} ({model}) 설정 완료: {message}")
                    return True, message
                else:
                    logger.error(f"{detector_type} {version} ({model}) 설정 실패: {message}")
                    return False, message

        # 지원되지 않는 탐지기 유형
        logger.error(f"지원되지 않는 탐지기 유형: {detector_type}")
        return False, f"지원되지 않는 탐지기 유형: {detector_type}"

    def is_active(self):
        """탐지기 활성화 상태 확인"""
        return self.current_detector is not None

    def get_current_detector_info(self):
        """현재 설정된 탐지기 정보 반환"""
        if not self.current_detector:
            return {"status": "미설정"}

        return {
            "type": self.detector_type,
            "version": self.detector_version,
            "model": self.detector_model,
            "params": self.detector_params,
            "status": "설정됨"
        }

    def detect(self, frame):
        """
        객체 탐지 수행

        Args:
            frame (numpy.ndarray): 분석할 이미지 프레임

        Returns:
            tuple: (시각화된 프레임, 탐지 결과 목록, 결과 텍스트)
        """
        if not self.current_detector:
            return frame, [], "탐지기가 설정되지 않았습니다."

        try:
            # 탐지 시작 시간 측정
            start_time = time.time()

            # 탐지 매개변수
            conf_threshold = self.detector_params.get('conf_threshold', 0.5)
            nms_threshold = self.detector_params.get('nms_threshold', 0.4)

            # 객체 탐지 수행
            detections = self.current_detector.detect(frame, conf_threshold, nms_threshold)

            # 결과 시각화
            result_frame = self.current_detector.postprocess(frame, detections)

            # 탐지 시간 측정
            detection_time = time.time() - start_time

            # 결과 텍스트 생성
            person_count = sum(1 for d in detections if d.get('class_name', '').lower() == 'person')
            result_text = f"탐지된 사람: {person_count}명"

            if len(detections) > person_count:
                other_count = len(detections) - person_count
                result_text += f"\n기타 객체: {other_count}개"

            result_text += f"\n처리 시간: {detection_time:.3f}초"

            # 결과 시그널 발생
            self.detection_result_signal.emit(result_text)

            # 완료 시그널 발생
            self.detection_complete_signal.emit(result_frame, detections, result_text)

            return result_frame, detections, result_text

        except Exception as e:
            logger.error(f"탐지 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            return frame, [], f"탐지 중 오류 발생: {str(e)}"