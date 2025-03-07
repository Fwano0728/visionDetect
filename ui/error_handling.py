#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import logging
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox


class ErrorHandler:
    """애플리케이션 에러 처리 클래스"""

    def __init__(self, parent=None):
        self.parent = parent
        self.setup_logger()

    def setup_logger(self):
        """로깅 설정"""
        # 로그 디렉토리 확인
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 로거 설정
        self.logger = logging.getLogger("ui.error_handling")
        self.logger.setLevel(logging.DEBUG)

        # 이미 핸들러가 설정되어 있는지 확인
        if not self.logger.handlers:
            # 파일 핸들러
            log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 포맷 설정
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 핸들러 추가
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log_error(self, message, show_dialog=False):
        """에러 로깅"""
        # 스택 트레이스 가져오기
        stack_trace = traceback.format_exc()

        # 로그 출력
        if "NoneType" in stack_trace:
            # 실제 예외가 발생하지 않은 경우
            self.logger.error(message)
        else:
            # 예외 발생
            self.logger.error(f"{message}\n{stack_trace}")

        # 대화상자 표시 (필요한 경우)
        if show_dialog and self.parent:
            QMessageBox.critical(self.parent, "오류 발생", message)

    def show_warning(self, message):
        """경고 메시지 표시"""
        self.logger.warning(message)
        if self.parent:
            QMessageBox.warning(self.parent, "경고", message)

    def show_info(self, message):
        """정보 메시지 표시"""
        self.logger.info(message)
        if self.parent:
            QMessageBox.information(self.parent, "알림", message)

    @classmethod
    def show_error(cls, parent, title, message, error_type="error", details=None):
        """
        오류 메시지 표시

        Args:
            parent: 부모 위젯
            title: 오류 제목
            message: 오류 메시지
            error_type: 오류 유형 (error, warning, info 등)
            details: 오류 상세 정보
        """
        from PyQt5.QtWidgets import QMessageBox

        # 로그에 오류 기록
        cls.log_error(message, details)

        # 메시지 박스 표시
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        if details:
            msg_box.setDetailedText(details)

        if error_type == "error":
            msg_box.setIcon(QMessageBox.Critical)
        elif error_type == "warning":
            msg_box.setIcon(QMessageBox.Warning)
        else:
            msg_box.setIcon(QMessageBox.Information)

        msg_box.exec_()