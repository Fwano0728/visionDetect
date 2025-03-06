#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
객체 탐지 결과를 그래프로 표시하는 컴포넌트
"""

import logging
from datetime import datetime, timedelta
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics
from PyQt5.QtWidgets import QWidget, QSizePolicy

logger = logging.getLogger(__name__)


class TimeAxisGraph(QWidget):
    """시간 축 그래프 위젯"""

    def __init__(self, parent=None):
        """초기화"""
        super().__init__(parent)
        self.setMinimumSize(600, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 데이터 저장 변수
        self.timestamps = []
        self.datasets = {}
        self.colors = {}
        self.display_range = 60  # 기본 표시 범위 (초)

        # 배경색 설정
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)

    def add_dataset(self, name, data=None, color=None):
        """
        데이터셋 추가

        Args:
            name (str): 데이터셋 이름
            data (list, optional): 초기 데이터
            color (QColor, optional): 그래프 색상
        """
        self.datasets[name] = data or []
        self.colors[name] = color or QColor(0, 0, 0)

    def set_data(self, timestamps, datasets):
        """
        모든 데이터 설정

        Args:
            timestamps (list): 타임스탬프 목록
            datasets (dict): 데이터셋 사전 {이름: 데이터목록}
        """
        self.timestamps = timestamps

        for name, data in datasets.items():
            if name in self.datasets:
                self.datasets[name] = data
            else:
                # 새 데이터셋 추가
                self.add_dataset(name, data)

        self.update()

    def set_display_range(self, seconds):
        """
        표시 범위 설정 (초)

        Args:
            seconds (int): 표시할 시간 범위 (초)
        """
        self.display_range = seconds
        self.update()

    def paintEvent(self, event):
        """페인트 이벤트 처리"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 그래프 영역 크기
        width = self.width()
        height = self.height()

        # 여백 설정
        left_margin = 50
        right_margin = 20
        top_margin = 20
        bottom_margin = 40

        # 그래프 영역
        graph_width = width - left_margin - right_margin
        graph_height = height - top_margin - bottom_margin

        # 배경 및 테두리 그리기
        painter.setPen(QPen(Qt.lightGray, 1))
        painter.setBrush(QBrush(QColor(240, 240, 240)))
        painter.drawRect(left_margin, top_margin, graph_width, graph_height)

        # 데이터가 없으면 종료
        if not self.timestamps or not self.datasets:
            # 안내 메시지 표시
            painter.setPen(QPen(Qt.black))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(left_margin, top_margin, graph_width, graph_height,
                             Qt.AlignCenter, "데이터가 없습니다.")
            return

        # Y축 범위 계산
        max_value = 0
        for dataset in self.datasets.values():
            if dataset:
                max_value = max(max_value, max(dataset))

        if max_value == 0:
            max_value = 10  # 기본값

        # Y축 눈금 그리기
        painter.setPen(QPen(Qt.gray, 1, Qt.DotLine))
        painter.setFont(QFont("Arial", 8))

        num_ticks = 5
        for i in range(num_ticks + 1):
            y_value = max_value * (num_ticks - i) / num_ticks
            y_pos = top_margin + (i * graph_height) / num_ticks

            # 수평선
            painter.drawLine(left_margin, y_pos, left_margin + graph_width, y_pos)

            # 레이블
            painter.setPen(Qt.black)
            painter.drawText(5, y_pos + 5, f"{y_value:.0f}")
            painter.setPen(QPen(Qt.gray, 1, Qt.DotLine))

        # 시간 범위 계산
        if len(self.timestamps) >= 2:
            time_range = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            if time_range < 1:
                time_range = self.display_range
        else:
            time_range = self.display_range

        # X축 눈금 그리기
        now = datetime.now()
        time_interval = time_range / 4  # 4개 구간

        painter.setPen(QPen(Qt.gray, 1, Qt.DotLine))
        for i in range(5):  # 5개 지점
            time_offset = time_range * (4 - i) / 4
            time_point = now - timedelta(seconds=time_offset)
            x_pos = left_margin + (i * graph_width) / 4

            # 수직선
            painter.drawLine(x_pos, top_margin, x_pos, top_margin + graph_height)

            # 레이블
            painter.setPen(Qt.black)
            time_label = time_point.strftime("%H:%M:%S")
            metrics = QFontMetrics(painter.font())
            text_width = metrics.width(time_label)
            painter.drawText(x_pos - text_width / 2, height - 10, time_label)
            painter.setPen(QPen(Qt.gray, 1, Qt.DotLine))

        # 각 데이터셋 그리기
        for name, data in self.datasets.items():
            if not data or len(data) < 2:
                continue

            color = self.colors.get(name, QColor(0, 0, 0))
            painter.setPen(QPen(color, 2))

            # 첫 번째 점
            prev_x = left_margin
            if max_value > 0:
                prev_y = top_margin + graph_height - (data[0] / max_value) * graph_height
            else:
                prev_y = top_margin + graph_height

            # 선 그리기
            for i in range(1, len(data)):
                # 시간에 따른 X 위치 계산
                if i < len(self.timestamps):
                    time_diff = (now - self.timestamps[i]).total_seconds()
                    x_pos = left_margin + graph_width - (time_diff / time_range) * graph_width
                else:
                    # 타임스탬프가 없는 경우 균등 분포
                    x_pos = left_margin + (i * graph_width) / (len(data) - 1)

                # Y 위치 계산
                if max_value > 0:
                    y_pos = top_margin + graph_height - (data[i] / max_value) * graph_height
                else:
                    y_pos = top_margin + graph_height

                # 선 그리기
                if 0 <= x_pos <= width and 0 <= y_pos <= height:
                    painter.drawLine(prev_x, prev_y, x_pos, y_pos)
                    prev_x, prev_y = x_pos, y_pos

        # 범례 그리기
        legend_x = left_margin + 10
        legend_y = top_margin + 10

        painter.setFont(QFont("Arial", 8))
        for name, color in self.colors.items():
            painter.setPen(Qt.black)
            painter.setBrush(QBrush(color))

            # 색상 표시
            painter.drawRect(legend_x, legend_y, 10, 10)

            # 이름 표시
            painter.drawText(legend_x + 15, legend_y + 10, name)

            # 다음 항목 위치
            legend_y += 20


class DetectionGraph(QGroupBox):
    """객체 탐지 그래프 위젯"""

    def __init__(self, parent=None):
        """초기화"""
        super().__init__("객체 탐지 추이", parent)

        # 그래프 데이터
        self.detection_data = {
            'timestamps': [],
            'person_count': [],
            'other_objects': []
        }

        # UI 초기화
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)

        # 컨트롤 영역
        control_layout = QHBoxLayout()

        # 시간 범위 선택기
        control_layout.addWidget(QLabel("시간 범위:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["1분", "5분", "10분", "30분", "1시간"])
        self.time_range_combo.setCurrentIndex(0)
        self.time_range_combo.currentIndexChanged.connect(self.on_time_range_changed)
        control_layout.addWidget(self.time_range_combo)

        # 우측 여백
        control_layout.addStretch(1)

        layout.addLayout(control_layout)

        # 그래프
        self.graph = TimeAxisGraph()
        layout.addWidget(self.graph)

        # 데이터셋 설정
        self.graph.add_dataset("사람 수", [], QColor(0, 120, 215))
        self.graph.add_dataset("다른 객체", [], QColor(0, 153, 0))

    def on_time_range_changed(self, index):
        """
        시간 범위 변경 처리

        Args:
            index (int): 콤보박스 인덱스
        """
        # 시간 범위 설정 (초)
        ranges = [60, 300, 600, 1800, 3600]
        if 0 <= index < len(ranges):
            self.graph.set_display_range(ranges[index])

    def update_graph(self, detection_stats):
        """
        그래프 데이터 업데이트

        Args:
            detection_stats (dict): 탐지 통계 데이터
        """
        try:
            # 데이터 업데이트
            datasets = {}
            if 'person_count' in detection_stats:
                datasets['사람 수'] = detection_stats['person_count']

            if 'other_objects' in detection_stats:
                datasets['다른 객체'] = detection_stats['other_objects']

            # 시간 정보가 없는 경우 현재 시간 사용
            timestamps = detection_stats.get('timestamps', [])
            if not timestamps and (datasets.get('사람 수') or datasets.get('다른 객체')):
                # 데이터 길이만큼 현재 시간으로 채움
                data_len = max(len(datasets.get('사람 수', [])), len(datasets.get('다른 객체', [])))
                now = datetime.now()
                timestamps = [now - timedelta(seconds=(data_len - i - 1)) for i in range(data_len)]

            # 그래프 데이터 설정
            self.graph.set_data(timestamps, datasets)

        except Exception as e:
            logger.error(f"탐지 그래프 업데이트 오류: {str(e)}")