# models/yolo/yolov4.py
import cv2
import numpy as np
from .yolo_interface import YOLODetector


class YOLOv4Detector(YOLODetector):
    """YOLOv4 구현체"""

    def __init__(self):
        self._net = None
        self._classes = []
        self._output_layers = []
        self._model_name = "YOLOv4"

    def load_model(self, weights_path, config_path):
        """Darknet 모델 로드"""
        self._net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        # 클래스 이름 로드
        with open("models/yolo/coco.names", "r") as f:
            self._classes = f.read().strip().split("\n")

        # 출력 레이어
        layer_names = self._net.getLayerNames()
        try:
            self._output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
        except:
            self._output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

        return True

    def detect(self, frame, conf_threshold=0.5, nms_threshold=0.4):
        """객체 탐지 수행"""
        height, width = frame.shape[:2]

        # 블롭 변환
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)

        # 추론 실행
        outputs = self._net.forward(self._output_layers)

        # 결과 처리
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    # 객체 위치
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 비최대 억제 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        results = []
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            elif isinstance(indices, np.ndarray) and len(indices.shape) > 1:
                indices = indices.flatten()

            for i in indices:
                box = boxes[i]
                results.append({
                    'box': box,
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self._classes[class_ids[i]]
                })

        return results

    @property
    def name(self):
        return self._model_name

    @property
    def version(self):
        return "v4"

    @property
    def available_models(self):
        return [
            {"name": "YOLOv4-tiny", "weights": "yolov4-tiny.weights", "config": "yolov4-tiny.cfg"},
            {"name": "YOLOv4", "weights": "yolov4.weights", "config": "yolov4.cfg"}
        ]