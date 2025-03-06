#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import logging.handlers
from datetime import datetime


class Logger:
    """애플리케이션 로깅 시스템"""

    def __init__(self, log_dir='logs', log_level=logging.INFO):
        """
        로거 초기화

        Args:
            log_dir (str): 로그 파일 저장 디렉토리
            log_level (int): 로깅 레벨
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.logger = None

        # 로그 디렉토리 생성
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.setup_logger()

    def setup_logger(self):
        """로거 설정"""
        # 루트 로거 가져오기
        logger = logging.getLogger()
        logger.setLevel(self.log_level)

        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 현재 날짜로 로그 파일 이름 생성
        log_filename = os.path.join(
            self.log_dir,
            f"vision_detect_{datetime.now().strftime('%Y%m%d')}.log"
        )

        # 파일 핸들러 추가 (1MB 단위로 롤오버)
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=1024 * 1024,  # 1MB
            backupCount=10,
            encoding='utf-8'
        )

        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()

        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 핸들러 등록
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.logger = logger
        logger.info("로깅 시스템 초기화 완료")

        return logger

    def get_logger(self, name=None):
        """
        로거 객체 반환

        Args:
            name (str, optional): 로거 이름

        Returns:
            logging.Logger: 로거 객체
        """
        if name:
            return logging.getLogger(name)
        return self.logger


# 전역 로거 초기화 함수
def init_logging(log_dir='logs', log_level=logging.INFO):
    """
    전역 로깅 시스템 초기화

    Args:
        log_dir (str): 로그 파일 저장 디렉토리
        log_level (int): 로깅 레벨

    Returns:
        logging.Logger: 로거 객체
    """
    logger_instance = Logger(log_dir, log_level)
    return logger_instance.get_logger()