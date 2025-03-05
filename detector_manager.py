#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import time
import inspect
from PyQt5.QtCore import QObject, pyqtSignal

# 카테고리별 색상 정의 (BGR 형식)
CATEGORY_COLORS = {
    "사람": (0, 255, 0),  # 초록색
    "탈것": (0, 0, 255),  # 빨간색
    "동물": (0, 165, 255),  # 주황색
    "생활용품": (128, 0, 128),  # 보라색
    "음식": (255, 0, 255),  # 분홍색
    "가구": (0, 128, 128),  # 청록색
    "전자기기": (255, 255, 0),  # 노란색
    "기타": (128, 128, 128)  # 회색
}


class DetectorManager(QObject):
    """객체 탐지 알고리즘 관리 클래스"""

    # 시그널 정의
    detection_result_signal = pyqtSignal(str)
    detection_counts_updated = pyqtSignal(dict)  # 객체별 탐지 횟수 시그널
    current_detection_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        # 카테고리 정의
        self.categories = {
            "사람": ["person"],
            "탈것": ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"],
            "동물": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "생활용품": ["backpack", "umbrella", "handbag", "tie", "suitcase", "bottle", "wine glass", "cup", "fork",
                     "knife", "spoon", "bowl"],
            "음식": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
            "가구": ["chair", "sofa", "pottedplant", "bed", "diningtable", "toilet"],
            "전자기기": ["tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                     "refrigerator", "sink"],
            "기타": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "frisbee", "skis",
                   "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                   "tennis racket", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        }

        # 현재 탐지 중인 객체 수 저장
        self.current_detections = {}

        # 시그널 추가
        self.current_detection_updated = pyqtSignal(dict)

        self.detector_type = None
        self.detector_version = None
        self.detector_model = None
        self.detector_params = {}
        self.detectors = {}

        # 초기화
        self._init_detectors()

        # 탐지 횟수 추적을 위한 변수 추가
        self.detection_counts = {}
        self.reset_detection_counts()

        # 마지막 탐지 횟수 업데이트 시간 초기화
        self.last_counts_update_time = 0

        # 현재 탐지 중인 객체 수 저장
        self.current_detections = {}

        # Initialize the current detections dictionary
        self.current_detections = {}

        # 탐지 수 업데이트 시그널 추가
        self.current_detection_updated = pyqtSignal(dict)

    def _init_detectors(self):
        """사용 가능한 탐지기 초기화"""
        # HOG 탐지기
        self.detectors["OpenCV-HOG"] = {
            "versions": ["Default"],
            "models": {"Default": ["HOG-Default"]},
            "init_func": self._init_hog_detector,
            "detect_func": self._detect_hog
        }

        # YOLO 탐지기
        self.detectors["YOLO"] = {
            "versions": ["v4"],  # 실제 모델 확인 후 필터링될 것임
            "models": {
                "v4": ["YOLOv4-tiny", "YOLOv4"],
            },
            "init_func": self._init_yolo_detector,
            "detect_func": self._detect_yolo
        }

        # 모델 파일 존재 여부 확인 및 모델 목록 필터링
        self._verify_model_files()

    def _verify_model_files(self):
        """각 모델의 파일 존재 여부 확인하고 실제 사용 가능한 모델 목록 업데이트"""
        models_dir = "models/yolo"

        # YOLO 모델 확인
        if "YOLO" in self.detectors:
            for version, models in list(self.detectors["YOLO"]["models"].items()):
                available_models = []
                for model_name in models:
                    # 모델별 파일 경로 확인
                    if version == "v4":
                        if model_name == "YOLOv4-tiny":
                            weights_path = os.path.join(models_dir, "yolov4-tiny.weights")
                            config_path = os.path.join(models_dir, "yolov4-tiny.cfg")
                        elif model_name == "YOLOv4":
                            weights_path = os.path.join(models_dir, "yolov4.weights")
                            config_path = os.path.join(models_dir, "yolov4.cfg")
                        else:
                            continue

                        # 파일 존재 여부 확인
                        if os.path.exists(weights_path) and os.path.exists(config_path):
                            available_models.append(model_name)
                            print(f"YOLO 모델 발견: {model_name}")

                    # YOLOv5 모델 (아직 구현되지 않음, 미래 확장성을 위한 자리 표시자)
                    elif version == "v5":
                        # v5는 파일이 없으므로 모델 목록에 추가하지 않음
                        continue

                    # YOLOv8 모델 (Ultralytics)
                    elif version == "v8":
                        try:
                            # Ultralytics 설치 여부 확인
                            import importlib.util
                            ultralytics_spec = importlib.util.find_spec("ultralytics")
                            if ultralytics_spec is not None:
                                # 패키지 설치됨, 이제 모델 파일 확인
                                model_file = None
                                if model_name == "YOLOv8n":
                                    model_file = "yolov8n.pt"
                                elif model_name == "YOLOv8s":
                                    model_file = "yolov8s.pt"
                                elif model_name == "YOLOv8m":
                                    model_file = "yolov8m.pt"
                                elif model_name == "YOLOv8l":
                                    model_file = "yolov8l.pt"
                                elif model_name == "YOLOv8x":
                                    model_file = "yolov8x.pt"

                                # 모델 파일 존재 여부 확인
                                if model_file and (
                                        os.path.exists(os.path.join(models_dir, model_file)) or
                                        os.path.exists(model_file)
                                ):
                                    available_models.append(model_name)
                                    print(f"YOLO 모델 발견: {model_name}")
                        except:
                            # Ultralytics 패키지가 설치되지 않음
                            pass

                # 사용 가능한 모델 목록 업데이트
                if available_models:
                    self.detectors["YOLO"]["models"][version] = available_models
                else:
                    # 사용 가능한 모델이 없는 버전은 목록에서 제거
                    if version in self.detectors["YOLO"]["versions"]:
                        self.detectors["YOLO"]["versions"].remove(version)
                    del self.detectors["YOLO"]["models"][version]

            # 가능한 버전이 없으면 빈 리스트로 설정
            if not self.detectors["YOLO"]["versions"]:
                print("사용 가능한 YOLO 모델이 없습니다.")
                # 기본값으로 v4를 유지하되 모델 없음을 표시
                self.detectors["YOLO"]["versions"] = ["v4"]
                self.detectors["YOLO"]["models"]["v4"] = []

    def _check_model_downloadable(self, model_file):
        """모델이 자동으로 다운로드 가능한지 확인"""
        # YOLOv8 모델은 자동 다운로드 지원
        if model_file.startswith("yolov8"):
            return True
        return False

    def get_available_detectors(self):
        """사용 가능한 탐지기 목록 반환"""
        return list(self.detectors.keys())

    def get_available_versions(self, detector_type):
        """사용 가능한 버전 목록 반환"""
        if detector_type in self.detectors:
            return self.detectors[detector_type]["versions"]
        return []

    def get_available_models(self, detector_type, version):
        """사용 가능한 모델 목록 반환"""
        if detector_type in self.detectors and version in self.detectors[detector_type]["models"]:
            return self.detectors[detector_type]["models"][version]
        return []

    def set_detector(self, detector_type, version=None, model=None, **params):
        """탐지기 설정"""
        # 탐지기 비활성화
        if detector_type is None:
            self.detector_type = None
            self.detector_version = None
            self.detector_model = None
            self.detector_params = {}
            return True, "탐지기가 비활성화되었습니다."

        # 기존 설정과 같으면 아무것도 하지 않음
        if (detector_type == self.detector_type and
                version == self.detector_version and
                model == self.detector_model and
                params == self.detector_params):
            return True, "현재 설정과 동일합니다."

        # 설정 저장
        self.detector_type = detector_type
        self.detector_version = version
        self.detector_model = model
        self.detector_params = params

        # 초기화 함수 호출
        if detector_type in self.detectors:
            init_func = self.detectors[detector_type]["init_func"]
            success, message = init_func(version, model, **params)
            return success, message

        return False, f"지원되지 않는 탐지기 유형: {detector_type}"

    def get_current_detector_info(self):
        """현재 설정된 탐지기 정보 반환"""
        if not self.detector_type:
            return {"status": "미설정"}

        return {
            "type": self.detector_type,
            "version": self.detector_version,
            "model": self.detector_model,
            "params": self.detector_params,
            "status": "설정됨"
        }

    def is_active(self):
        """탐지기 활성화 상태 확인"""
        return self.detector_type is not None

    def detect(self, frame, text_scale=1.0, colors=None):
        """객체 탐지 수행"""
        if not self.detector_type:
            return frame, [], "탐지기가 설정되지 않았습니다."

        if self.detector_type in self.detectors:
            detect_func = self.detectors[self.detector_type]["detect_func"]
            start_time = time.time()

            # 기본 색상 설정 (colors 매개변수가 전달되지 않은 경우)
            if colors is None:
                colors = {
                    "person": (0, 255, 0),  # 사람: 초록색 (BGR)
                    "default": (0, 0, 255)  # 기타 객체: 빨간색 (BGR)
                }

            # 안전하게 함수 호출
            try:
                result_frame, detections = detect_func(frame, text_scale=text_scale, colors=colors)
            except TypeError:
                # 인수가 맞지 않으면 기본 매개변수로 호출
                try:
                    result_frame, detections = detect_func(frame)
                except Exception as e:
                    print(f"탐지 함수 호출 중 오류 발생: {str(e)}")
                    result_frame = frame.copy()
                    detections = []
            except Exception as e:
                print(f"탐지 함수 호출 중 오류 발생: {str(e)}")
                result_frame = frame.copy()
                detections = []

            detection_time = time.time() - start_time

            # 결과 텍스트 생성
            person_count = sum(1 for d in detections if d.get("class_name", "").lower() == "person")
            other_count = len(detections) - person_count

            result_text = f"탐지된 사람: {person_count}명"
            if other_count > 0:
                result_text += f"\n기타 객체: {other_count}개"

            result_text += f"\n처리 시간: {detection_time:.3f}초"
            self.detection_result_signal.emit(result_text)

            # 탐지 횟수 업데이트 (필요한 경우)
            self.update_detection_counts(detections)

            return result_frame, detections, result_text

        return frame, [], "지원되지 않는 탐지기 유형입니다."

    # HOG 탐지기 관련 메서드
    def _init_hog_detector(self, version=None, model=None, **params):
        """HOG 사람 탐지기 초기화"""
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True, "HOG 사람 탐지기 초기화 완료"
        except Exception as e:
            print(f"HOG 탐지기 초기화 실패: {str(e)}")
            return False, f"HOG 탐지기 초기화 실패: {str(e)}"

    def _detect_hog(self, frame):
        """HOG를 사용하여 사람 탐지"""
        # 성능을 위해 이미지 크기 조정
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        else:
            scale = 1.0
            frame_resized = frame

        # HOG 디스크립터를 사용하여 사람 탐지
        boxes, weights = self.hog.detectMultiScale(
            frame_resized,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        # 결과 이미지에 박스 그리기
        result_frame = frame.copy()
        detections = []

        for i, (x, y, w, h) in enumerate(boxes):
            # 원본 이미지 크기에 맞게 박스 크기 조정
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)

            # 박스 그리기
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            confidence = float(weights[i]) if i < len(weights) else 0.5
            cv2.putText(result_frame, f'Person {confidence:.2f}',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 탐지 정보 저장
            detections.append({
                "class_id": 0,
                "class_name": "person",
                "confidence": confidence,
                "box": [x, y, w, h]
            })

        return result_frame, detections

    # YOLO 탐지기 관련 메서드
    def _init_yolo_detector(self, version="v4", model="YOLOv4-tiny", **params):
        """YOLO 객체 탐지기 초기화"""
        try:
            # 모델 파일 경로 설정
            models_dir = "models/yolo"
            os.makedirs(models_dir, exist_ok=True)

            # 버전과 모델에 따른 파일 경로
            if version == "v4":
                if model == "YOLOv4-tiny":
                    weights_path = os.path.join(models_dir, "yolov4-tiny.weights")
                    config_path = os.path.join(models_dir, "yolov4-tiny.cfg")
                elif model == "YOLOv4":
                    weights_path = os.path.join(models_dir, "yolov4.weights")
                    config_path = os.path.join(models_dir, "yolov4.cfg")
                else:
                    return False, f"지원되지 않는 YOLOv4 모델: {model}"

                # 파일 존재 확인
                if not os.path.exists(weights_path) or not os.path.exists(config_path):
                    return False, f"모델 파일을 찾을 수 없습니다: {weights_path} 또는 {config_path}"

                # YOLO 네트워크 로드
                self.yolo_net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

                # 레이어 이름 가져오기
                layer_names = self.yolo_net.getLayerNames()
                try:
                    self.yolo_output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
                except:
                    self.yolo_output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]

                # 클래스 이름 로드
                classes_path = os.path.join(models_dir, "coco.names")
                if os.path.exists(classes_path):
                    with open(classes_path, 'r') as f:
                        self.yolo_classes = f.read().strip().split('\n')
                else:
                    self.yolo_classes = ["person"]

                print(f"YOLO {version} ({model}) 모델 로드 완료")
                return True, f"YOLO {version} ({model}) 모델 로드 완료"

            # YOLOv8 (Ultralytics)
            elif version == "v8":
                try:
                    from ultralytics import YOLO

                    if model == "YOLOv8n":
                        model_path = "yolov8n.pt"
                    elif model == "YOLOv8s":
                        model_path = "yolov8s.pt"
                    elif model == "YOLOv8x":
                        model_path = "yolov8x.pt"
                    else:
                        return False, f"지원되지 않는 YOLOv8 모델: {model}"

                    self.yolo8_model = YOLO(model_path)
                    self.yolo_classes = self.yolo8_model.names

                    print(f"YOLOv8 ({model}) 모델 로드 완료")
                    return True, f"YOLOv8 ({model}) 모델 로드 완료"
                except ImportError:
                    return False, "YOLOv8을 사용하려면 'pip install ultralytics' 명령으로 패키지를 설치하세요."

            return False, f"지원되지 않는 YOLO 버전: {version}"
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"YOLO 모델 로드 실패: {str(e)}"

    def _detect_yolo(self, frame, text_scale=1.0, colors=None):
        """YOLO를 사용한 객체 탐지"""
        if self.detector_version == "v8" and hasattr(self, 'yolo8_model'):
            # YOLOv8 분기
            if colors is not None:
                return self._detect_yolov8(frame, text_scale, colors)
            else:
                return self._detect_yolov8(frame, text_scale)
        elif hasattr(self, 'yolo_net'):
            # YOLOv4 분기
            if colors is not None:
                return self._detect_yolov4(frame, text_scale, colors)
            else:
                return self._detect_yolov4(frame, text_scale)

        return frame, []

    def _detect_yolov4(self, frame, text_scale=1.0, colors=None):
        """YOLOv4를 사용한 객체 탐지"""
        # 매개변수
        conf_threshold = self.detector_params.get('conf_threshold', 0.4)
        nms_threshold = self.detector_params.get('nms_threshold', 0.4)

        # 객체 선택 목록 가져오기
        objects_to_detect = self.detector_params.get('objects_to_detect', {})

        # 카테고리별 색상 가져오기
        category_colors = self.detector_params.get('category_colors', {})

        # 기본 색상 설정 (카테고리별로 적용)
        default_colors = {}
        for category, color in CATEGORY_COLORS.items():
            default_colors[category] = color

        # 사용자 정의 색상이 있으면 적용
        if category_colors:
            for category, color in category_colors.items():
                default_colors[category] = color

        # 처리 속도를 위해 프레임 크기를 더 작게 감소
        scale_factor = 0.3  # 이미지 크기를 30%로 줄임
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        height, width = small_frame.shape[:2]

        # 이미지를 YOLO 입력 형식으로 변환
        blob = cv2.dnn.blobFromImage(small_frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)

        # 예측 실행
        outputs = self.yolo_net.forward(self.yolo_output_layers)

        # 결과 처리
        boxes = []
        confidences = []
        class_ids = []
        class_names = []
        categories = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # 클래스 이름 가져오기
                class_name = self.yolo_classes[class_id] if class_id < len(self.yolo_classes) else f"Class {class_id}"

                # 객체의 카테고리 결정
                category = self.get_category_for_class(class_name)

                # 신뢰도 임계값 이상이고 선택된 객체만 처리
                # 선택된 객체가 없으면 모든 객체 표시, 있으면 선택된 객체만 표시
                if confidence > conf_threshold and (not objects_to_detect or objects_to_detect.get(class_name, False)):
                    # 원본 이미지 크기에 맞게 좌표 계산
                    center_x = int((detection[0] * width) / scale_factor)
                    center_y = int((detection[1] * height) / scale_factor)
                    w = int((detection[2] * width) / scale_factor)
                    h = int((detection[3] * height) / scale_factor)

                    # 바운딩 박스 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # 화면 범위 내로 조정
                    x = max(0, x)
                    y = max(0, y)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    class_names.append(class_name)
                    categories.append(category)

        # 비최대 억제 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # 결과 이미지에 박스 그리기
        result_frame = frame.copy()
        detections = []

        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            elif isinstance(indices, np.ndarray) and len(indices.shape) > 1:
                indices = indices.flatten()

            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                class_id = class_ids[i]
                class_name = class_names[i]
                category = categories[i]
                confidence = confidences[i]

                # 카테고리에 따른 색상 선택
                color = default_colors.get(category, default_colors.get('기타', (128, 128, 128)))

                # 박스 그리기 - 경계선 굵기 증가
                line_thickness = 2
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, line_thickness)

                # 텍스트 배경 그리기 (가독성 개선)
                label = f'{class_name} {confidence:.2f}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = text_scale * 0.5  # 텍스트 크기 조정
                t_size = cv2.getTextSize(label, font, font_scale, 1)[0]

                # 텍스트 배경 박스
                cv2.rectangle(result_frame, (x, y - t_size[1] - 4), (x + t_size[0], y), color, -1)

                # 텍스트 그리기 (흰색)
                cv2.putText(result_frame, label, (x, y - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

                # 탐지 정보 저장 (카테고리 정보 추가)
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "category": category,
                    "confidence": confidence,
                    "box": [x, y, w, h]
                })

        # 현재 탐지 수 업데이트
        self.update_current_detections(detections)

        return result_frame, detections

    def _detect_yolov8(self, frame, text_scale=1.0, colors=None):
        """YOLOv8을 사용한 객체 탐지"""
        # 매개변수
        conf_threshold = self.detector_params.get('conf_threshold', 0.4)

        # 객체 선택 목록 가져오기
        objects_to_detect = self.detector_params.get('objects_to_detect', {})

        # 기본 색상 설정
        if colors is None:
            colors = {
                "person": (0, 255, 0),  # 사람: 초록색
                "default": (0, 0, 255)  # 기타 객체: 빨간색
            }

        # 모델 실행
        results = self.yolo8_model(frame, conf=conf_threshold)
        result = results[0]  # 첫 번째 결과

        # 탐지 정보 가져오기
        detections = []
        result_frame = frame.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]

            # 선택된 객체만 처리
            if not objects_to_detect or objects_to_detect.get(class_name, False):
                # 좌표 정수로 변환
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                # 객체 유형에 맞는 색상 선택
                if class_name in colors:
                    color = colors[class_name]
                else:
                    color = colors.get("default", (0, 255, 0))

                # 박스 그리기 - 경계선 굵기 증가
                line_thickness = 2
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, line_thickness)

                # 텍스트 배경 그리기 (가독성 개선)
                label = f'{class_name} {confidence:.2f}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = text_scale * 0.5  # 텍스트 크기 조정
                t_size = cv2.getTextSize(label, font, font_scale, 1)[0]

                # 텍스트 배경 박스
                cv2.rectangle(result_frame, (x, y - t_size[1] - 4), (x + t_size[0], y), color, -1)

                # 텍스트 그리기 (흰색)
                cv2.putText(result_frame, label, (x, y - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

                # 탐지 정보 저장
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [x, y, w, h]
                })

        return result_frame, detections

    def get_objects_to_detect(self):
        """탐지할 객체 목록 반환"""
        # 객체 선택 상태가 없으면 기본값 반환 (모든 객체 선택)
        if not hasattr(self, 'objects_to_detect'):
            # COCO 클래스 목록 가져오기
            coco_classes = []
            try:
                with open("models/yolo/coco.names", "r") as f:
                    coco_classes = f.read().strip().split("\n")
            except:
                # 기본 COCO 클래스 목록 (80개 클래스)
                coco_classes = ["person", "bicycle", "car", ...]

            # 기본값: 모든 객체 선택
            self.objects_to_detect = {cls: True for cls in coco_classes}

        return self.objects_to_detect

    def set_objects_to_detect(self, objects_dict):
        """탐지할 객체 목록 설정"""
        # 객체 선택 상태 저장
        self.objects_to_detect = objects_dict

        # 현재 활성화된 탐지기가 있으면 설정 적용
        if self.is_active():
            # 현재 매개변수 가져오기
            params = self.detector_params.copy() if hasattr(self, 'detector_params') else {}
            params['objects_to_detect'] = objects_dict

            # 탐지기 재설정 (같은 설정으로)
            return self.set_detector(
                self.detector_type,
                self.detector_version,
                self.detector_model,
                **params
            )

        return True, "탐지 객체 설정이 저장되었습니다."

    def reset_detection_counts(self):
        """탐지 횟수 초기화"""
        self.detection_counts = {}

        # COCO 클래스 목록 가져오기
        try:
            with open("models/yolo/coco.names", "r") as f:
                coco_classes = f.read().strip().split("\n")
                self.detection_counts = {cls: 0 for cls in coco_classes}
        except:
            # 기본적으로는 빈 딕셔너리로 시작
            pass

        # 시그널 발생
        self.detection_counts_updated.emit(self.detection_counts)

    def update_detection_counts(self, detections):
        """객체별 탐지 횟수 업데이트"""
        # 이 메서드가 구현되지 않은 경우를 위한 안전 확인
        if not hasattr(self, 'detection_counts'):
            self.detection_counts = {}

        # 이번 프레임에서 탐지된 객체 카운트
        current_counts = {}
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            current_counts[class_name] = current_counts.get(class_name, 0) + 1

        # 전체 카운트 업데이트
        for class_name, count in current_counts.items():
            self.detection_counts[class_name] = self.detection_counts.get(class_name, 0) + count

        # 시그널 발생 (있는 경우)
        if hasattr(self, 'detection_counts_updated'):
            self.detection_counts_updated.emit(self.detection_counts)

    def get_detection_counts(self):
        """현재 객체별 탐지 횟수 반환"""
        if not hasattr(self, 'detection_counts'):
            self.detection_counts = {}
        return self.detection_counts

    def update_current_detections(self, detections):
        """현재 프레임에서 탐지된 객체 수 업데이트"""
        # 현재 프레임의 탐지 수 계산
        current_counts = {}
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            current_counts[class_name] = current_counts.get(class_name, 0) + 1

        # 변경 사항이 있을 때만 시그널 발생
        if current_counts != self.current_detections:
            self.current_detections = current_counts
            self.current_detection_updated.emit(self.current_detections)

        return current_counts

    def get_current_detections(self):
        """Return current object detection counts"""
        return self.current_detections

    def get_category_for_class(self, class_name):
        """클래스 이름에 해당하는 카테고리 반환"""
        for category, classes in self.categories.items():
            if class_name in classes:
                return category
        return "기타"  # 기본 카테고리

# 카테고리별 색상 정의 (BGR 형식)
CATEGORY_COLORS = {
    'person': (0, 255, 0),     # 초록색
    'animal': (0, 165, 255),   # 주황색
    'vehicle': (0, 0, 255),    # 빨간색
    'household': (255, 0, 0),  # 파란색
    'electronic': (255, 255, 0), # 청록색
    'food': (255, 0, 255),     # 분홍색
    'outdoor': (128, 0, 128),  # 보라색
    'other': (128, 128, 128)   # 회색
}

# 객체 카테고리 정의
OBJECT_CATEGORIES = {
    'person': '사람',
    'animal': '동물',
    'vehicle': '교통수단',
    'household': '가정용품',
    'electronic': '전자제품',
    'food': '음식',
    'outdoor': '야외용품',
    'other': '기타'
}

# 카테고리별 클래스 매핑
CATEGORY_MAPPING = {
    # 사람
    'person': 'person',

    # 동물
    'bird': 'animal', 'cat': 'animal', 'dog': 'animal', 'horse': 'animal',
    'sheep': 'animal', 'cow': 'animal', 'elephant': 'animal', 'bear': 'animal',
    'zebra': 'animal', 'giraffe': 'animal',

    # 교통수단
    'bicycle': 'vehicle', 'car': 'vehicle', 'motorbike': 'vehicle', 'aeroplane': 'vehicle',
    'bus': 'vehicle', 'train': 'vehicle', 'truck': 'vehicle', 'boat': 'vehicle',

    # 가정용품
    'chair': 'household', 'sofa': 'household', 'pottedplant': 'household', 'bed': 'household',
    'diningtable': 'household', 'toilet': 'household', 'tvmonitor': 'household',

    # 전자제품
    'laptop': 'electronic', 'mouse': 'electronic', 'remote': 'electronic', 'keyboard': 'electronic',
    'cell phone': 'electronic', 'microwave': 'electronic', 'oven': 'electronic',
    'toaster': 'electronic', 'refrigerator': 'electronic',

    # 음식
    'banana': 'food', 'apple': 'food', 'sandwich': 'food', 'orange': 'food', 'broccoli': 'food',
    'carrot': 'food', 'hot dog': 'food', 'pizza': 'food', 'donut': 'food', 'cake': 'food',

    # 야외용품
    'backpack': 'outdoor', 'umbrella': 'outdoor', 'handbag': 'outdoor', 'tie': 'outdoor',
    'suitcase': 'outdoor', 'frisbee': 'outdoor', 'skis': 'outdoor', 'snowboard': 'outdoor',
    'sports ball': 'outdoor', 'kite': 'outdoor'
}

# 카테고리별 색상 정의 (BGR 형식)
CATEGORY_COLORS = {
    'person': (0, 255, 0),  # 초록색
    'animal': (0, 165, 255),  # 주황색
    'vehicle': (0, 0, 255),  # 빨간색
    'household': (255, 0, 0),  # 파란색
    'electronic': (255, 255, 0),  # 청록색
    'food': (255, 0, 255),  # 분홍색
    'outdoor': (128, 0, 128),  # 보라색
    'other': (128, 128, 128)  # 회색
}




