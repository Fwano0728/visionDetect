# models/yolo/yolov8.py
from .yolo_interface import YOLODetector
import numpy as np
import cv2


class YOLOv8Detector(YOLODetector):
    """YOLOv8 구현체"""

    def __init__(self):
        self._model = None
        self._model_name = "YOLOv8"

    def load_model(self, weights_path, config_path=None):
        """Ultralytics YOLO 모델 로드"""
        try:
            # ultralytics 패키지 필요
            from ultralytics import YOLO
            self._model = YOLO(weights_path)
            return True
        except ImportError:
            print("ultralytics 패키지가 설치되지 않았습니다. 'pip install ultralytics' 명령으로 설치하세요.")
            return False
        except Exception as e:
            print(f"YOLOv8 모델 로드 실패: {str(e)}")
            return False

    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.45):
        """객체 탐지 수행"""
        if self._model is None:
            return []

        # ultralytics YOLO 모델 실행
        results = self._model(frame, conf=conf_threshold, iou=nms_threshold)
        result = results[0]  # 첫 번째 결과만 사용

        # 결과 변환
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]

            detections.append({
                'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })

        return detections

    @property
    def name(self):
        return self._model_name

    @property
    def version(self):
        return "v8"

    @property
    def available_models(self):
        return [
            {"name": "YOLOv8n", "weights": "yolov8n.pt"},
            {"name": "YOLOv8s", "weights": "yolov8s.pt"},
            {"name": "YOLOv8m", "weights": "yolov8m.pt"},
            {"name": "YOLOv8l", "weights": "yolov8l.pt"},
            {"name": "YOLOv8x", "weights": "yolov8x.pt"}
        ]