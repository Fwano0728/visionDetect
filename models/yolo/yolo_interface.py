# models/yolo/yolo_interface.py
from abc import ABC, abstractmethod
import numpy as np


class YOLODetector(ABC):
    """모든 YOLO 버전이 구현해야 하는 인터페이스"""

    @abstractmethod
    def load_model(self, weights_path, config_path=None):
        """모델 로드"""
        pass

    @abstractmethod
    def detect(self, frame):
        """이미지에서 객체 탐지 수행"""
        pass

    @property
    @abstractmethod
    def name(self):
        """모델 이름 반환"""
        pass

    @property
    @abstractmethod
    def version(self):
        """모델 버전 반환"""
        pass

    @property
    def available_models(self):
        """사용 가능한 사전 훈련된 모델 목록"""
        return []