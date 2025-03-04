# models/yolo_manager.py
import os
import importlib
from .yolo.yolo_interface import YOLODetector


class YOLOManager:
    """YOLO 모델 버전 관리"""

    def __init__(self, models_dir="models/yolo"):
        self.models_dir = models_dir
        self._detectors = {}
        self._active_detector = None
        self._active_model = None

        # 모든 구현된 버전 로드
        self._load_detectors()

    def _load_detectors(self):
        """사용 가능한 모든 YOLO 탐지기 로드"""
        # 직접 구현체 추가
        from .yolo.yolov4 import YOLOv4Detector
        self._detectors["v4"] = YOLOv4Detector()

        try:
            # 다른 구현체 추가 (존재하는 경우)
            from .yolo.yolov8 import YOLOv8Detector
            self._detectors["v8"] = YOLOv8Detector()
        except ImportError:
            print("YOLOv8 지원을 위해 ultralytics 패키지가 필요합니다.")

    def get_available_versions(self):
        """사용 가능한 모든 YOLO 버전 반환"""
        return list(self._detectors.keys())

    def get_models_for_version(self, version):
        """특정 버전에서 사용 가능한 모델 목록 반환"""
        if version in self._detectors:
            return self._detectors[version].available_models
        return []

    def load_model(self, version, model_info):
        """특정 버전과 모델 로드"""
        if version not in self._detectors:
            return False, f"YOLO {version} 버전을 찾을 수 없습니다."

        detector = self._detectors[version]

        # 모델 파일 경로 준비
        weights_path = os.path.join(self.models_dir, model_info.get("weights", ""))
        config_path = os.path.join(self.models_dir, model_info.get("config", "")) if "config" in model_info else None

        # 모델 파일 존재 확인
        if not os.path.exists(weights_path):
            return False, f"모델 파일을 찾을 수 없습니다: {weights_path}"

        if config_path and not os.path.exists(config_path):
            return False, f"설정 파일을 찾을 수 없습니다: {config_path}"

        # 모델 로드
        success = detector.load_model(weights_path, config_path)
        if success:
            self._active_detector = detector
            self._active_model = model_info
            return True, f"{detector.name} {model_info['name']} 모델이 로드되었습니다."
        else:
            return False, "모델 로드에 실패했습니다."

    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.4):
        """현재 로드된 모델로 객체 탐지 수행"""
        if self._active_detector is None:
            return [], "모델이 로드되지 않았습니다."

        try:
            detections = self._active_detector.detect(frame, conf_threshold, nms_threshold)
            return detections, None
        except Exception as e:
            return [], f"탐지 중 오류 발생: {str(e)}"

    def get_active_model_info(self):
        """현재 활성화된 모델 정보 반환"""
        if self._active_detector and self._active_model:
            return {
                'version': self._active_detector.version,
                'name': self._active_detector.name,
                'model': self._active_model['name']
            }
        return None