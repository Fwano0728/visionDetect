#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
대시보드 위젯 모듈
"""

import os
import logging
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QTabWidget, QGroupBox, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# 대시보드 컴포넌트
from ui.dashboard.components.stats_panel import StatsPanel
from ui.dashboard.components.performance_monitor import PerformanceMonitor
from ui.dashboard.components.detection_graph import DetectionGraph
from ui.dashboard.components.event_log_widget import EventLogWidget

logger = logging.getLogger(__name__)


class DashboardWidget(QWidget):
    """객체 탐지 시스템 대시보드 위젯"""

    def __init__(self, camera_manager, detector_manager, parent=None):
        """
        초기화

        Args:
            camera_manager: 카메라 관리 객체
            detector_manager: 탐지기 관리 객체
            parent: 부모 위젯
        """
        super().__init__(parent)

        # 매니저 참조
        self.camera_manager = camera_manager
        self.detector_manager = detector_manager

        # 데이터 저장 변수
        self.detection_stats = {
            'person_count': [],
            'timestamps': [],
            'other_objects': []
        }

        self.performance_data = {
            'fps': [],
            'cpu': [],
            'memory': [],
            'gpu': [],
            'timestamps': []
        }

        # 타이머 설정
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_dashboard)

        # UI 초기화
        self.init_ui()

        # 컴포넌트 초기화
        self.init_components()

        # 타이머 시작
        self.update_timer.start(1000)  # 1초마다 업데이트

        logger.info("대시보드 위젯 초기화 완료")

    def init_ui(self):
        """UI 초기화"""
        # 메인 레이아웃
        main_layout = QVBoxLayout(self)

        # 타이틀 레이블
        title_label = QLabel("객체 탐지 대시보드")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)

        # 대시보드 탭 위젯
        self.tabs = QTabWidget()

        # 실시간 탭
        self.realtime_tab = QWidget()
        self.tabs.addTab(self.realtime_tab, "실시간")

        # 통계 탭
        self.stats_tab = QWidget()
        self.tabs.addTab(self.stats_tab, "통계")

        # 로그 탭
        self.log_tab = QWidget()
        self.tabs.addTab(self.log_tab, "이벤트 로그")

        # 레이아웃에 추가
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.tabs, 1)  # 탭이 남은 공간 차지

        # 창 크기 설정
        self.resize(1000, 700)
        self.setWindowTitle("객체 탐지 대시보드")

    def init_components(self):
        """대시보드 컴포넌트 초기화"""
        # 실시간 탭 설정
        self.setup_realtime_tab()

        # 통계 탭 설정
        self.setup_stats_tab()

        # 로그 탭 설정
        self.setup_log_tab()

    def setup_realtime_tab(self):
        """실시간 탭 설정"""
        layout = QGridLayout(self.realtime_tab)

        # 실시간 통계 패널
        self.stats_panel = StatsPanel()
        layout.addWidget(self.stats_panel, 0, 0)

        # 성능 모니터
        self.performance_monitor = PerformanceMonitor()
        layout.addWidget(self.performance_monitor, 0, 1)

        # 실시간 그래프
        self.detection_graph = DetectionGraph()
        layout.addWidget(self.detection_graph, 1, 0, 1, 2)

    def setup_stats_tab(self):
        """통계 탭 설정"""
        # 추후 구현
        layout = QVBoxLayout(self.stats_tab)
        label = QLabel("통계 기능은 아직 구현되지 않았습니다.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def setup_log_tab(self):
        """로그 탭 설정"""
        layout = QVBoxLayout(self.log_tab)
        self.event_log = EventLogWidget()
        layout.addWidget(self.event_log)

    def update_dashboard(self):
        """대시보드 데이터 업데이트"""
        try:
            # 현재 시간
            now = datetime.now()

            # 시스템 상태 정보 가져오기
            system_status = self.camera_manager.get_system_load()
            fps = getattr(self.camera_manager, 'current_fps', 0)

            # 탐지 정보 가져오기
            detector_info = self.detector_manager.get_current_detector_info()
            detector_active = detector_info.get("status") == "설정됨"

            # 성능 데이터 저장
            self.performance_data['timestamps'].append(now)
            self.performance_data['fps'].append(fps)
            self.performance_data['cpu'].append(system_status.get('cpu', 0))
            self.performance_data['memory'].append(system_status.get('memory', 0))
            self.performance_data['gpu'].append(system_status.get('gpu', 0))

            # 데이터 길이 제한 (최근 60초)
            max_len = 60
            if len(self.performance_data['timestamps']) > max_len:
                for key in self.performance_data:
                    self.performance_data[key] = self.performance_data[key][-max_len:]

            # UI 컴포넌트 업데이트
            if hasattr(self, 'stats_panel'):
                self.stats_panel.update_stats({
                    'camera_connected': self.camera_manager.is_connected(),
                    'detector_active': detector_active,
                    'fps': fps,
                    'detector_type': detector_info.get("type", ""),
                    'detector_model': detector_info.get("model", "")
                })

            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.update_data(self.performance_data)

            if hasattr(self, 'detection_graph'):
                self.detection_graph.update_graph(self.detection_stats)

        except Exception as e:
            logger.error(f"대시보드 업데이트 오류: {str(e)}")

    def add_detection_event(self, detections, timestamp=None):
        """
        탐지 이벤트 추가

        Args:
            detections (list): 탐지된 객체 목록
            timestamp (datetime, optional): 타임스탬프
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()

            # 사람 수 계산
            person_count = sum(1 for d in detections if d.get('class_name', '').lower() == 'person')
            other_count = len(detections) - person_count

            # 데이터 저장
            self.detection_stats['timestamps'].append(timestamp)
            self.detection_stats['person_count'].append(person_count)
            self.detection_stats['other_objects'].append(other_count)

            # 데이터 길이 제한 (최근 100개)
            max_len = 100
            if len(self.detection_stats['timestamps']) > max_len:
                for key in self.detection_stats:
                    self.detection_stats[key] = self.detection_stats[key][-max_len:]

            # 이벤트 로그에 추가
            if hasattr(self, 'event_log'):
                if person_count > 0:
                    self.event_log.add_log_entry(f"탐지: {person_count}명의 사람, {other_count}개의 다른 객체")

        except Exception as e:
            logger.error(f"탐지 이벤트 추가 오류: {str(e)}")

    def closeEvent(self, event):
        """창 닫기 이벤트"""
        # 타이머 중지
        self.update_timer.stop()
        event.accept()