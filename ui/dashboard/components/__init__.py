# 대시보드 컴포넌트 패키지 초기화 파일
from ui.dashboard.components.stats_panel import StatsPanel
from ui.dashboard.components.performance_monitor import PerformanceMonitor
from ui.dashboard.components.detection_graph import DetectionGraph
from ui.dashboard.components.event_log_widget import EventLogWidget

__all__ = ['StatsPanel', 'PerformanceMonitor', 'DetectionGraph', 'EventLogWidget']