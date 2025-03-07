# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# detection/detector_base.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple, Any


class DetectorBase(ABC):
    """탐지기 기본 인터페이스"""

    @abstractmethod
    def initialize(self, **kwargs) -> Tuple[bool, str]:
        """탐지기 초기화"""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Dict[str, Any]], str, float]:
        """객체 탐지 수행 (프레임, 탐지결과, 메시지, FPS 반환)"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """탐지기 이름"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """탐지기 버전"""
        pass

    @property
    def available_models(self) -> List[Dict[str, str]]:
        """사용 가능한 모델 목록"""
        return []

    @property
    def parameters(self) -> Dict[str, Any]:
        """탐지기 매개변수"""
        return {}