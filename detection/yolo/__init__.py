# detection/yolo/__init__.py

# 명시적으로 클래스 직접 내보내기
from detection.yolo.yolov4_detector import YOLOv4Detector

# 다른 YOLO 구현체 추가 가능
# from detection.yolo.yolov8_detector import YOLOv8Detector

__all__ = ['YOLOv4Detector']  # 내보낼 클래스 목록