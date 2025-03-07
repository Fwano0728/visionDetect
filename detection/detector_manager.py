# 탐지 모델 선택 코드
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# detection/detector_manager.py

# detector_manager.py 파일 상단에 추가
import logging
logger = logging.getLogger(__name__)
import os
import importlib
import inspect
import logging
import numpy as np
import time  # time 모듈 추가
from typing import Dict, List, Tuple, Any
from PyQt5.QtCore import QObject, pyqtSignal

from detection.detector_base import DetectorBase

logger = logging.getLogger(__name__)


class DetectorManager(QObject):
    """탐지기 관리 클래스"""

    # 시그널 정의
    detection_result_signal = pyqtSignal(str)
    detection_fps_updated = pyqtSignal(float)  # 탐지 FPS 업데이트를 위한 새 시그널


    def __init__(self):
        super().__init__()

        self.detectors = {}  # 등록된 탐지기
        self.active_detector = None  # 활성화된 탐지기
        self.active_detector_info = {}  # 활성화된 탐지기 정보
        self.detection_fps = 0.0  # 탐지 FPS 저장 변수

        # 탐지기 자동 등록
        self._register_detectors()

        # 비동기 처리를 위한 탐지 스레드
        self.detection_thread = None
        self.detection_fps = 0.0

    def _register_detectors(self):
        """탐지기 자동 등록"""
        # 기본 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"탐지기 기본 디렉토리: {base_dir}")

        # YOLO 탐지기 직접 등록
        try:
            from detection.yolo.yolo_detector import YOLODetector
            yolo_detector = YOLODetector()
            self.detectors["YOLO"] = {
                "instance": yolo_detector,
                "module": "detection.yolo.yolo_detector",
                "class": "YOLODetector"
            }
            logger.info("YOLO 탐지기 직접 등록 완료")
        except Exception as e:
            logger.error(f"YOLO 탐지기 등록 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        # 하위 디렉토리 탐색 (다른 탐지기가 있다면)
        for subdir in ['opencv', 'deepstream']:  # 'yolo'는 이미 직접 등록했으므로 제외
            detector_dir = os.path.join(base_dir, subdir)
            logger.debug(f"탐지기 하위 디렉토리 검색: {detector_dir}")

            if not os.path.isdir(detector_dir):
                logger.debug(f"디렉토리가 존재하지 않음: {detector_dir}")
                continue

            # 파이썬 파일 탐색
            for filename in os.listdir(detector_dir):
                if not filename.endswith('.py') or filename.startswith('__'):
                    continue

                # 모듈 로드
                module_name = f"detection.{subdir}.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)

                    # DetectorBase를 상속한 클래스 찾기
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                                issubclass(obj, DetectorBase) and
                                obj != DetectorBase):

                            try:
                                # 탐지기 인스턴스 생성
                                detector = obj()
                                detector_type = f"{detector.name}"

                                # 이미 등록된 탐지기 확인
                                if detector_type in self.detectors:
                                    logger.warning(f"중복된 탐지기 유형: {detector_type}")
                                    continue

                                # 탐지기 등록
                                self.detectors[detector_type] = {
                                    "instance": detector,
                                    "module": module_name,
                                    "class": name
                                }

                                logger.info(f"탐지기 등록: {detector_type} ({module_name}.{name})")
                            except Exception as e:
                                logger.error(f"탐지기 등록 실패: {module_name}.{name}, 오류: {str(e)}")

                except Exception as e:
                    logger.error(f"모듈 로드 실패: {module_name}, 오류: {str(e)}")

        logger.info(f"총 {len(self.detectors)}개 탐지기 등록 완료")
        logger.info(f"등록된 탐지기 목록: {list(self.detectors.keys())}")

    def get_available_detectors(self) -> List[str]:
        """사용 가능한 탐지기 목록 반환"""
        return list(self.detectors.keys())

    def get_available_versions(self, detector_type: str) -> List[str]:
        """사용 가능한 버전 목록 반환"""
        if detector_type in self.detectors:
            detector = self.detectors[detector_type]["instance"]
            if hasattr(detector, "available_versions"):
                return detector.available_versions
            return [detector.version]
        return []

    def get_available_models(self, detector_type: str, version: str = None) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        logger.info(f"모델 목록 요청: 탐지기={detector_type}, 버전={version}")

        if detector_type in self.detectors:
            detector = self.detectors[detector_type]["instance"]

            # YOLO 탐지기인 경우
            if detector_type == "YOLO" and hasattr(detector, "get_models_for_version"):
                models = detector.get_models_for_version(version)
                logger.info(f"YOLO {version} 버전 모델 목록: {models}")
                return models

            # 일반적인 경우
            if hasattr(detector, "available_models"):
                models = detector.available_models
                # 모델 이름만 추출
                model_names = [model["name"] for model in models if version is None or model.get("version") == version]
                logger.info(f"모델 목록: {model_names}")
                return model_names

        logger.warning(f"모델을 찾을 수 없음: 탐지기={detector_type}, 버전={version}")
        return []

    def set_detector(self, detector_type: str, version: str = None, model: str = None, **params) -> Tuple[bool, str]:
        """탐지기 설정"""
        try:
            # 탐지기 비활성화
            if detector_type is None:
                self.active_detector = None
                self.active_detector_info = {}

                # 실행 중인 경우 탐지 스레드 중지
                if hasattr(self, 'detection_thread') and self.detection_thread and self.detection_thread.isRunning():
                    self.detection_thread.stop()
                    self.detection_thread = None

                return True, "탐지기가 비활성화되었습니다."

            # 탐지기 존재 확인
            if detector_type not in self.detectors:
                return False, f"지원되지 않는 탐지기 유형: {detector_type}"

            detector = self.detectors[detector_type]["instance"]

            # 초기화 매개변수 준비
            init_params = {
                "version": version,
                "model": model,
                **params
            }

            # 탐지기 초기화
            try:
                success, message = detector.initialize(**init_params)
                if success:
                    self.active_detector = detector
                    self.active_detector_info = {
                        "type": detector_type,
                        "version": version,
                        "model": model,
                        "params": params,
                        "status": "설정됨"
                    }

                    # 탐지 스레드 초기화 및 시작
                    if hasattr(self,
                               'detection_thread') and self.detection_thread and self.detection_thread.isRunning():
                        self.detection_thread.stop()

                    # detection_thread 속성이 있는지 확인
                    if hasattr(self, 'detection_thread'):
                        from detection.detection_thread import DetectionThread
                        self.detection_thread = DetectionThread(self)
                        self.detection_thread.detection_complete.connect(self._handle_detection_result)
                        self.detection_thread.error.connect(self._handle_thread_error)
                        self.detection_thread.start()

                    return True, message
                else:
                    return False, message
            except Exception as e:
                logger.error(f"탐지기 초기화 중 오류: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False, f"탐지기 초기화 중 오류: {str(e)}"

        except Exception as e:
            logger.error(f"탐지기 설정 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"탐지기 설정 중 오류: {str(e)}"

    def _handle_thread_error(self, error_message):
        """스레드 오류 처리"""
        logger.error(f"탐지 스레드 오류: {error_message}")

    def _handle_detection_result(self, result_frame, detections, message, fps):
        """처리 스레드에서 결과 처리"""
        # 마지막 탐지 결과 저장
        self.active_detector_info["last_detections"] = detections
        self.active_detector_info["last_message"] = message
        self.active_detector_info["result_frame"] = result_frame

        # 탐지 FPS 업데이트
        self.detection_fps = fps

        # 결과 텍스트 생성
        result_text = self._generate_result_text(detections, message)

        # 시그널 발생
        self.detection_result_signal.emit(result_text)

    def process_frame(self, frame):
        """비동기 처리를 위한 프레임 전달"""
        if self.is_active() and self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.set_frame(frame)
            return True
        return False

    def get_current_detector_info(self) -> Dict[str, Any]:
        """현재 설정된 탐지기 정보 반환"""
        if not self.active_detector:
            return {"status": "미설정"}

        return self.active_detector_info

    def is_active(self) -> bool:
        """탐지기 활성화 상태 확인"""
        return self.active_detector is not None

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], str, float]:
        """객체 탐지 수행"""
        if not self.active_detector:
            return frame, [], "탐지기가 설정되지 않았습니다.", 0.0

        # 탐지기에 프레임 전달
        start_time = time.time()

        try:
            # 다양한 반환 값 패턴 처리
            result = self.active_detector.detect(
                frame, **self.active_detector_info.get("params", {})
            )

            # 결과 길이에 따라 처리
            if isinstance(result, tuple):
                if len(result) == 4:  # 새 형식 (4개 값)
                    result_frame, detections, message, detection_fps = result
                elif len(result) == 3:  # 이전 형식 (3개 값)
                    result_frame, detections, message = result
                    detection_time = time.time() - start_time
                    detection_fps = 1.0 / detection_time if detection_time > 0 else 0
                else:
                    # 예상치 못한 형식
                    raise ValueError(f"예상치 못한 반환 값 개수: {len(result)}")
            else:
                # 튜플이 아닌 형식 처리
                result_frame = result
                detections = []
                message = "탐지 결과 형식 오류"
                detection_time = time.time() - start_time
                detection_fps = 1.0 / detection_time if detection_time > 0 else 0

        except Exception as e:
            # 오류 발생 시 기본값 반환
            logger.error(f"탐지 오류: {str(e)}")
            import traceback
            traceback.print_exc()

            result_frame = frame.copy()
            detections = []
            message = f"탐지 중 오류 발생: {str(e)}"
            detection_fps = 0.0

        # 마지막 탐지 결과 저장
        self.active_detector_info["last_detections"] = detections
        self.active_detector_info["last_message"] = message
        self.detection_fps = detection_fps

        # FPS 업데이트 신호 발생
        self.detection_fps_updated.emit(detection_fps)

        # 결과 텍스트 생성
        result_text = self._generate_result_text(detections, message)

        # 시그널 발생
        self.detection_result_signal.emit(result_text)

        return result_frame, detections, result_text, detection_fps

    def _generate_result_text(self, detections: List[Dict[str, Any]], message: str) -> str:
        """탐지 결과 텍스트 생성"""
        if not detections:
            return message or "탐지된 객체가 없습니다."

        # 객체별 개수 계산
        object_counts = {}
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        # 결과 텍스트 생성
        result_text = "탐지 결과:\n"
        for class_name, count in object_counts.items():
            result_text += f"- {class_name}: {count}개\n"

        # 추가 메시지가 있으면 포함
        if message:
            result_text += f"\n{message}"

        return result_text

    def set_object_detection_settings(self, settings: Dict[str, Any]) -> None:
        """객체 탐지 설정 적용"""
        if self.active_detector and hasattr(self.active_detector, "set_detection_settings"):
            self.active_detector.set_detection_settings(settings)

    def get_detection_fps(self) -> float:
        """현재 탐지 FPS 반환"""
        return self.detection_fps