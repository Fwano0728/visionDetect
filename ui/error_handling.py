#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QMessageBox
import logging
import traceback

logger = logging.getLogger(__name__)


class ErrorHandler:
    """애플리케이션 오류 처리 클래스"""

    ERROR_LEVELS = {
        "info": QMessageBox.Information,
        "warning": QMessageBox.Warning,
        "error": QMessageBox.Critical,
        "question": QMessageBox.Question
    }

    LOG_LEVELS = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "question": logging.INFO
    }

    @staticmethod
    def show_error(parent, title, message, level="error", details=None):
        """
        오류 메시지 표시

        Args:
            parent: 부모 위젯
            title (str): 오류 제목
            message (str): 오류 메시지
            level (str): 오류 수준 (info, warning, error, question)
            details (str, optional): 상세 정보

        Returns:
            int: 대화상자 결과 코드
        """
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(ErrorHandler.ERROR_LEVELS.get(level, QMessageBox.Critical))

        if details:
            msg_box.setDetailedText(details)

        # 로깅
        log_level = ErrorHandler.LOG_LEVELS.get(level, logging.ERROR)
        logger.log(log_level, f"{title}: {message}" + (f" - {details}" if details else ""))

        return msg_box.exec_()

    @staticmethod
    def handle_exception(parent, e, operation_name):
        """
        예외 처리 및 메시지 표시

        Args:
            parent: 부모 위젯
            e (Exception): 발생한 예외
            operation_name (str): 작업 이름

        Returns:
            int: 대화상자 결과 코드
        """
        error_msg = f"{operation_name} 중 오류가 발생했습니다."
        details = f"{str(e)}\n\n{traceback.format_exc()}"

        # 로깅
        logger.error(f"{error_msg}\n{details}")

        return ErrorHandler.show_error(parent, "오류 발생", error_msg, "error", details)

    @staticmethod
    def confirm(parent, title, message, default_button=QMessageBox.No):
        """
        확인 메시지 표시

        Args:
            parent: 부모 위젯
            title (str): 제목
            message (str): 메시지
            default_button: 기본 버튼

        Returns:
            bool: 확인 시 True, 취소 시 False
        """
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(default_button)

        # 로깅
        logger.info(f"확인 메시지 표시: {title} - {message}")

        return msg_box.exec_() == QMessageBox.Yes