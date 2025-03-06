# YOLO 패키지 초기화 파일
from detection.yolo.yolo_detector import YOLODetector
from detection.yolo.yolov4_detector import YOLOv4Detector
from detection.yolo.yolov8_detector import YOLOv8Detector

__all__ = ['YOLODetector', 'YOLOv4Detector', 'YOLOv8Detector']