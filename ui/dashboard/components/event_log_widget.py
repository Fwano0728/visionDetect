#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
이벤트 로그 표시 위젯
"""

import logging
from datetime import datetime
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QLabel, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QColor, QFont, QTextCharFormat

logger = logging.getLogger(__name__)


class EventLogWidget(QGroupBox):
    """이벤트 로그를 표시하는 위젯"""

    def __init__(self, parent=None):
        """초기화"""
        super().__init__("이벤트 로그", parent)

        # 로그 항목 저장
        self.logs = []
        self.max_logs = 1000  # 최대 로그 개수

        # 필터 설정
        self.filters = {
            'level': 'all',  # 'info', 'warning', 'error', 'all'
            'keyword': '',
            'system_logs': True,
            'detection_logs': True
        }

        # UI 초기화
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 필터 영역
        filter_layout = QHBoxLayout()

        # 로그 레벨 필터
        filter_layout.addWidget(QLabel("레벨:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems(["모두", "정보", "경고", "오류"])
        self.level_combo.currentIndexChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.level_combo)

        # 유형 필터
        self.system_check = QCheckBox("시스템 로그")
        self.system_check.setChecked(True)
        self.system_check.stateChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.system_check)

        self.detection_check = QCheckBox("탐지 로그")
        self.detection_check.setChecked(True)
        self.detection_check.stateChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.detection_check)

        # 키워드 필터 (나중에 구현)
        filter_layout.addStretch(1)

        # 버튼
        self.clear_button = QPushButton("로그 지우기")
        self.clear_button.clicked.connect(self.clear_logs)
        filter_layout.addWidget(self.clear_button)

        layout.addLayout(filter_layout)

        # 로그 텍스트 영역
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)

        # 폰트 설정
        font = QFont("Courier New", 9)
        self.log_text.setFont(font)

        layout.addWidget(self.log_text, 1)  # 1은 stretch factor

    def add_log_entry(self, message, level="info", category="detection"):
        """
        로그 항목 추가

        Args:
            message (str): 로그 메시지
            level (str): 로그 레벨 ('info', 'warning', 'error')
            category (str): 로그 카테고리 ('system', 'detection')
        """
        try:
            # 로그 시간
            timestamp = datetime.now()

            # 로그 항목 생성
            log_entry = {
                'timestamp': timestamp,
                'message': message,
                'level': level,
                'category': category
            }

            # 로그 저장
            self.logs.append(log_entry)

            # 최대 개수 초과 시 오래된 로그 삭제
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]

            # UI 업데이트
            self.update_log_display()

        except Exception as e:
            logger.error(f"로그 항목 추가 오류: {str(e)}")

    def update_log_display(self):
        """로그 표시 업데이트"""
        try:
            # 현재 커서 위치 저장
            scrollbar = self.log_text.verticalScrollBar()
            was_at_bottom = scrollbar.value() == scrollbar.maximum()

            # 텍스트 지우기
            self.log_text.clear()

            # 필터링된 로그 표시
            for log in self.filter_logs():
                # 시간 형식
                time_str = log['timestamp'].strftime("%H:%M:%S")

                # 레벨에 따른 색상
                level_colors = {
                    'info': QColor(0, 0, 0),  # 검정
                    'warning': QColor(255, 140, 0),  # 주황
                    'error': QColor(255, 0, 0)  # 빨강
                }

                level_str = {
                    'info': 'INFO',
                    'warning': 'WARN',
                    'error': 'ERROR'
                }.get(log['level'], 'INFO')

                # 카테고리 표시
                category_str = {
                    'system': '[SYS]',
                    'detection': '[DET]'
                }.get(log['category'], '')

                # 전체 로그 라인
                log_line = f"{time_str} {level_str} {category_str} {log['message']}"

                # 포맷 설정
                format = QTextCharFormat()
                format.setForeground(level_colors.get(log['level'], QColor(0, 0, 0)))

                # 레벨에 따라 굵게 표시
                if log['level'] in ['warning', 'error']:
                    format.setFontWeight(QFont.Bold)

                # 텍스트 추가
                cursor = self.log_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(log_line + '\n', format)

            # 스크롤 위치 복원
            if was_at_bottom:
                scrollbar.setValue(scrollbar.maximum())

        except Exception as e:
            logger.error(f"로그 표시 업데이트 오류: {str(e)}")

    def filter_logs(self):
        """
        필터링된 로그 항목 반환

        Returns:
            list: 필터링된 로그 항목 목록
        """
        # 필터 적용
        filtered_logs = []

        for log in self.logs:
            # 레벨 필터
            if self.filters['level'] != 'all' and log['level'] != self.filters['level']:
                continue

            # 카테고리 필터
            if log['category'] == 'system' and not self.filters['system_logs']:
                continue

            if log['category'] == 'detection' and not self.filters['detection_logs']:
                continue

            # 키워드 필터 (나중에 구현)

            # 필터 통과
            filtered_logs.append(log)

        return filtered_logs

    def on_filter_changed(self):
        """필터 변경 처리"""
        try:
            # 레벨 필터
            level_map = {
                0: 'all',  # 모두
                1: 'info',  # 정보
                2: 'warning',  # 경고
                3: 'error'  # 오류
            }
            self.filters['level'] = level_map.get(self.level_combo.currentIndex(), 'all')

            # 카테고리 필터
            self.filters['system_logs'] = self.system_check.isChecked()
            self.filters['detection_logs'] = self.detection_check.isChecked()

            # 표시 업데이트
            self.update_log_display()

        except Exception as e:
            logger.error(f"필터 변경 처리 오류: {str(e)}")

    def clear_logs(self):
        """로그 지우기"""
        self.logs = []
        self.log_text.clear()

    def add_system_log(self, message, level="info"):
        """
        시스템 로그 추가

        Args:
            message (str): 로그 메시지
            level (str): 로그 레벨
        """
        self.add_log_entry(message, level, "system")

    def add_detection_log(self, message, level="info"):
        """
        탐지 로그 추가

        Args:
            message (str): 로그 메시지
            level (str): 로그 레벨
        """
        self.add_log_entry(message, level, "detection")