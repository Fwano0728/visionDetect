#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
실시간 통계 패널 컴포넌트
"""

import logging
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QGridLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class StatsPanel(QGroupBox):
    """실시간 통계를 표시하는 패널"""

    def __init__(self, parent=None):
        """
        초기화

        Args:
            parent: 부모 위젯
        """
        super().__init__("실시간 통계", parent)

        # 통계 데이터
        self.stats = {
            'camera_connected': False,
            'detector_active': False,
            'fps': 0,
            'detector_type': "",
            'detector_model': "",
            'person_count': 0,
            'other_objects': 0,
            'total_detections': 0,
            'detection_rate': 0,
            'max_persons': 0,
            'avg_persons': 0
        }

        # UI 초기화
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        main_layout = QVBoxLayout(self)

        # 상태 및 통계 그리드
        grid = QGridLayout()

        # 상태 라벨
        status_font = QFont()
        status_font.setBold(True)

        self.status_labels = {}

        # 시스템 상태
        row = 0
        grid.addWidget(QLabel("시스템 상태:"), row, 0)

        row += 1
        grid.addWidget(QLabel("카메라 연결:"), row, 0)
        self.status_labels['camera_connected'] = QLabel("연결 안됨")
        self.status_labels['camera_connected'].setFont(status_font)
        grid.addWidget(self.status_labels['camera_connected'], row, 1)

        row += 1
        grid.addWidget(QLabel("탐지기 상태:"), row, 0)
        self.status_labels['detector_active'] = QLabel("비활성")
        self.status_labels['detector_active'].setFont(status_font)
        grid.addWidget(self.status_labels['detector_active'], row, 1)

        row += 1
        grid.addWidget(QLabel("현재 FPS:"), row, 0)
        self.status_labels['fps'] = QLabel("0")
        grid.addWidget(self.status_labels['fps'], row, 1)

        row += 1
        grid.addWidget(QLabel("탐지기 유형:"), row, 0)
        self.status_labels['detector_type'] = QLabel("없음")
        grid.addWidget(self.status_labels['detector_type'], row, 1)

        row += 1
        grid.addWidget(QLabel("모델:"), row, 0)
        self.status_labels['detector_model'] = QLabel("없음")
        grid.addWidget(self.status_labels['detector_model'], row, 1)

        # 구분선
        row += 1
        separator = QLabel("")
        separator.setFixedHeight(20)
        grid.addWidget(separator, row, 0, 1, 2)

        # 탐지 통계
        row += 1
        grid.addWidget(QLabel("탐지 통계:"), row, 0)

        row += 1
        grid.addWidget(QLabel("현재 인원:"), row, 0)
        self.status_labels['person_count'] = QLabel("0명")
        grid.addWidget(self.status_labels['person_count'], row, 1)

        row += 1
        grid.addWidget(QLabel("기타 객체:"), row, 0)
        self.status_labels['other_objects'] = QLabel("0개")
        grid.addWidget(self.status_labels['other_objects'], row, 1)

        row += 1
        grid.addWidget(QLabel("총 탐지 수:"), row, 0)
        self.status_labels['total_detections'] = QLabel("0")
        grid.addWidget(self.status_labels['total_detections'], row, 1)

        row += 1
        grid.addWidget(QLabel("탐지율:"), row, 0)
        self.status_labels['detection_rate'] = QLabel("0%")
        grid.addWidget(self.status_labels['detection_rate'], row, 1)

        row += 1
        grid.addWidget(QLabel("최대 인원:"), row, 0)
        self.status_labels['max_persons'] = QLabel("0명")
        grid.addWidget(self.status_labels['max_persons'], row, 1)

        row += 1
        grid.addWidget(QLabel("평균 인원:"), row, 0)
        self.status_labels['avg_persons'] = QLabel("0명")
        grid.addWidget(self.status_labels['avg_persons'], row, 1)

        main_layout.addLayout(grid)
        main_layout.addStretch(1)  # 아래쪽 여백

    def update_stats(self, stats):
        """
        통계 업데이트

        Args:
            stats (dict): 업데이트할 통계 데이터
        """
        # 통계 데이터 업데이트
        self.stats.update(stats)

        # UI 업데이트
        try:
            # 카메라 연결 상태
            if self.stats['camera_connected']:
                self.status_labels['camera_connected'].setText("연결됨")
                self.status_labels['camera_connected'].setStyleSheet("color: green")
            else:
                self.status_labels['camera_connected'].setText("연결 안됨")
                self.status_labels['camera_connected'].setStyleSheet("color: red")

            # 탐지기 상태
            if self.stats['detector_active']:
                self.status_labels['detector_active'].setText("활성")
                self.status_labels['detector_active'].setStyleSheet("color: green")
            else:
                self.status_labels['detector_active'].setText("비활성")
                self.status_labels['detector_active'].setStyleSheet("color: gray")

            # FPS
            self.status_labels['fps'].setText(f"{self.stats['fps']:.1f}")

            # 탐지기 정보
            detector_type = self.stats.get('detector_type', "")
            if detector_type:
                self.status_labels['detector_type'].setText(detector_type)
            else:
                self.status_labels['detector_type'].setText("없음")

            detector_model = self.stats.get('detector_model', "")
            if detector_model:
                self.status_labels['detector_model'].setText(detector_model)
            else:
                self.status_labels['detector_model'].setText("없음")

            # 탐지 통계
            if 'person_count' in stats:
                self.status_labels['person_count'].setText(f"{stats['person_count']}명")

            if 'other_objects' in stats:
                self.status_labels['other_objects'].setText(f"{stats['other_objects']}개")

            if 'total_detections' in stats:
                self.status_labels['total_detections'].setText(str(stats['total_detections']))

            if 'detection_rate' in stats:
                self.status_labels['detection_rate'].setText(f"{stats['detection_rate']:.1f}%")

            if 'max_persons' in stats:
                self.status_labels['max_persons'].setText(f"{stats['max_persons']}명")

            if 'avg_persons' in stats:
                self.status_labels['avg_persons'].setText(f"{stats['avg_persons']:.1f}명")

        except Exception as e:
            logger.error(f"통계 패널 업데이트 오류: {str(e)}")