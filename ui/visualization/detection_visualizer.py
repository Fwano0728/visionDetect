#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/visualization/detection_visualizer.py

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QBarSet, QBarSeries, QValueAxis, QBarCategoryAxis


class DetectionResultsWidget(QWidget):
    """객체 탐지 결과 시각화 위젯"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.detection_results = {}  # 클래스별 탐지 수
        self.object_colors = {}  # 클래스별 색상

    def initUI(self):
        """UI 초기화"""
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # 초기 레이블
        self.status_label = QLabel("탐지 결과 없음")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        # 차트 뷰
        self.pie_chart = QChart()
        self.pie_chart.setTitle("객체 탐지 결과")
        self.pie_chart.setAnimationOptions(QChart.SeriesAnimations)

        self.pie_chart_view = QChartView(self.pie_chart)
        self.pie_chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.pie_chart_view)

        # 시간별 탐지 추이 차트
        self.trend_chart = QChart()
        self.trend_chart.setTitle("시간별 탐지 추이")

        self.trend_chart_view = QChartView(self.trend_chart)
        self.trend_chart_view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.trend_chart_view)

        # 초기에는 추이 차트 숨김
        self.trend_chart_view.setVisible(False)

    def update_results(self, detections, message=None):
        """탐지 결과 업데이트"""
        if not detections:
            self.status_label.setText(message or "탐지된 객체가 없습니다.")
            self.status_label.setVisible(True)
            self.pie_chart_view.setVisible(False)
            self.trend_chart_view.setVisible(False)
            return

        # 클래스별 탐지 수 집계
        class_counts = {}
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # 결과 저장 및 업데이트
        self.detection_results = class_counts

        # 상태 레이블 업데이트
        status_text = f"총 {len(detections)}개 객체 탐지됨"
        if message:
            status_text += f"\n{message}"
        self.status_label.setText(status_text)

        # 파이 차트 업데이트
        self.update_pie_chart()

        # 모든 구성 요소 표시
        self.status_label.setVisible(True)
        self.pie_chart_view.setVisible(True)
        self.trend_chart_view.setVisible(len(self.history) > 1 if hasattr(self, 'history') else False)

    def update_pie_chart(self):
        """파이 차트 업데이트"""
        # 이전 시리즈 제거
        self.pie_chart.removeAllSeries()

        # 새 파이 시리즈 생성
        series = QPieSeries()

        for class_name, count in self.detection_results.items():
            # 슬라이스 추가
            slice = series.append(f"{class_name} ({count})", count)

            # 색상 설정
            if class_name in self.object_colors:
                slice.setBrush(QBrush(QColor(self.object_colors[class_name])))

            # 슬라이스 설정
            slice.setLabelVisible(True)

        # 시리즈를 차트에 추가
        self.pie_chart.addSeries(series)

        # 차트 새로고침
        self.pie_chart.update()

    def start_trend_tracking(self, max_history=30):
        """시간별 추이 추적 시작"""
        self.history = []
        self.max_history = max_history
        self.trend_chart_view.setVisible(True)

    def update_trend_chart(self):
        """추이 차트 업데이트"""
        if not hasattr(self, 'history') or len(self.history) < 2:
            return

        # 이전 시리즈 제거
        self.trend_chart.removeAllSeries()

        # 각 클래스에 대한 막대 세트 생성
        class_names = set()
        for result in self.history:
            class_names.update(result.keys())

        # X 축 카테고리 (시간점)
        categories = [str(i) for i in range(len(self.history))]

        # 각 클래스에 대해 막대 세트 생성
        bar_sets = []
        for class_name in class_names:
            bar_set = QBarSet(class_name)

            # 각 시간점에서의 값 설정
            for result in self.history:
                bar_set.append(result.get(class_name, 0))

            # 색상 설정
            if class_name in self.object_colors:
                bar_set.setColor(QColor(self.object_colors[class_name]))

            bar_sets.append(bar_set)

        # 막대 시리즈 생성 및 추가
        series = QBarSeries()
        for bar_set in bar_sets:
            series.append(bar_set)

        self.trend_chart.addSeries(series)

        # 축 설정
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        self.trend_chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        max_value = max([max(result.values()) if result else 0 for result in self.history]) + 1
        axis_y.setRange(0, max_value)
        self.trend_chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        # 범례 표시
        self.trend_chart.legend().setVisible(True)

        # 차트 새로고침
        self.trend_chart.update()

    def add_to_history(self):
        """현재 결과를 히스토리에 추가"""
        if hasattr(self, 'history'):
            self.history.append(self.detection_results.copy())

            # 최대 기록 개수 제한
            if len(self.history) > self.max_history:
                self.history.pop(0)

            # 추이 차트 업데이트
            self.update_trend_chart()

    def set_object_colors(self, color_map):
        """객체별 색상 설정"""
        self.object_colors = color_map
        self.update_pie_chart()