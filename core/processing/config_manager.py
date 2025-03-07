#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# core/config_manager.py

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """전역 설정 관리 클래스"""

    _instance = None  # 싱글톤 패턴

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 기본 경로 설정
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.root_dir, "config")
        os.makedirs(self.config_dir, exist_ok=True)

        # 기본 설정 파일들
        self.settings = {
            "camera": {},
            "detector": {},
            "object_detection": {},
            "ui": {},
            "system": {}
        }

        # 설정 파일 로드
        self._load_all_settings()
        self._initialized = True

    def _get_config_path(self, config_name: str) -> str:
        """설정 파일 경로 반환"""
        return os.path.join(self.config_dir, f"{config_name}.json")

    def _load_all_settings(self) -> None:
        """모든 설정 파일 로드"""
        for config_name in self.settings.keys():
            self._load_settings(config_name)

    def _load_settings(self, config_name: str) -> Dict[str, Any]:
        """특정 설정 파일 로드"""
        config_path = self._get_config_path(config_name)

        if not os.path.exists(config_path):
            logger.info(f"설정 파일이 없습니다: {config_path}")
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.settings[config_name] = config
                return config
        except Exception as e:
            logger.error(f"설정 로드 중 오류 발생: {str(e)}")
            return {}

    def save_settings(self, config_name: str, data: Dict[str, Any]) -> bool:
        """설정 파일 저장"""
        config_path = self._get_config_path(config_name)

        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 설정 업데이트
            self.settings[config_name] = data
            logger.info(f"설정 저장 완료: {config_path}")
            return True
        except Exception as e:
            logger.error(f"설정 저장 중 오류 발생: {str(e)}")
            return False

    def get_setting(self, config_name: str, key: str = None, default=None):
        """설정값 가져오기"""
        config = self.settings.get(config_name, {})

        if key is None:
            return config

        return config.get(key, default)

    def update_setting(self, config_name: str, key: str, value: Any) -> bool:
        """설정값 업데이트"""
        if config_name not in self.settings:
            self.settings[config_name] = {}

        self.settings[config_name][key] = value
        return self.save_settings(config_name, self.settings[config_name])

    def get_ui_path(self, file_name: str) -> str:
        """UI 파일 경로 반환"""
        ui_dir = os.path.join(self.root_dir, "ui", "ui_files")
        return os.path.join(ui_dir, file_name)

    def get_asset_path(self, asset_type: str, file_name: str) -> str:
        """에셋 파일 경로 반환"""
        asset_dir = os.path.join(self.root_dir, "ui", "assets", asset_type)
        os.makedirs(asset_dir, exist_ok=True)
        return os.path.join(asset_dir, file_name)

    @staticmethod
    def get_instance():
        """ConfigManager 인스턴스 반환"""
        return ConfigManager()