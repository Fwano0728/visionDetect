# detection/yolo/yolov4_detector.py

import cv2
import os
import numpy as np
from typing import List, Dict, Tuple, Any
import platform
import subprocess
import time

from detection.detector_base import DetectorBase


class YOLOv4Detector(DetectorBase):
    """YOLOv4 탐지기 구현"""

    def __init__(self):
        self._net = None
        self._classes = []
        self._output_layers = []
        self._model_name = "YOLOv4"
        self._version = "v4"
        self._model = None
        self._weights_path = None
        self._config_path = None
        self._input_size = (416, 416)  # 기본 입력 크기
        self._use_cuda = False
        self._cuda_initialized = False

    def _check_cuda_support(self):
        """CUDA 지원 여부를 확인하고 자세한 정보를 출력합니다."""
        print("===== CUDA 지원 진단 시작 =====")

        # OpenCV CUDA 지원 확인
        opencv_cuda_support = hasattr(cv2, 'cuda')
        print(f"OpenCV CUDA 모듈 존재: {opencv_cuda_support}")

        # OpenCV 버전 확인
        print(f"OpenCV 버전: {cv2.__version__}")

        # CUDA 장치 확인
        cuda_devices = 0
        if opencv_cuda_support:
            try:
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                print(f"감지된 CUDA 장치 수: {cuda_devices}")

                if cuda_devices > 0:
                    # CUDA 장치 정보 출력
                    device_name = cv2.cuda.getDevice()
                    print(f"현재 CUDA 장치: {device_name}")

                    # 시스템에서 GPU 정보 확인 (Linux)
                    if platform.system() == "Linux":
                        try:
                            nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
                            print("NVIDIA-SMI 출력:")
                            print(nvidia_smi)
                        except:
                            print("nvidia-smi 명령을 실행할 수 없습니다.")
                else:
                    print("CUDA 장치를 찾을 수 없습니다.")
            except Exception as e:
                print(f"CUDA 장치 확인 중 오류 발생: {str(e)}")

        # OpenCV DNN 백엔드 확인
        backends = []
        if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'):
            backends.append("CUDA")
        if hasattr(cv2.dnn, 'DNN_BACKEND_OPENCV'):
            backends.append("OpenCV")
        if hasattr(cv2.dnn, 'DNN_BACKEND_DEFAULT'):
            backends.append("Default")
        print(f"사용 가능한 DNN 백엔드: {', '.join(backends)}")

        # OpenCV DNN 타겟 확인
        targets = []
        if hasattr(cv2.dnn, 'DNN_TARGET_CUDA'):
            targets.append("CUDA")
        if hasattr(cv2.dnn, 'DNN_TARGET_CUDA_FP16'):
            targets.append("CUDA_FP16")
        if hasattr(cv2.dnn, 'DNN_TARGET_CPU'):
            targets.append("CPU")
        print(f"사용 가능한 DNN 타겟: {', '.join(targets)}")

        print("===== CUDA 지원 진단 완료 =====")

        # CUDA 메모리 사용 최적화 시도
        if opencv_cuda_support and cuda_devices > 0:
            try:
                # CUDA 장치 설정
                cv2.cuda.setDevice(0)
                # 비동기 작업을 위한 스트림 생성
                self._cuda_stream = cv2.cuda.Stream()
                print("CUDA 스트림 생성 완료")
            except Exception as e:
                print(f"CUDA 스트림 생성 실패: {str(e)}")

        # CUDA 사용 가능 여부 반환
        return opencv_cuda_support and cuda_devices > 0


    def initialize(self, **kwargs) -> Tuple[bool, str]:
        """탐지기 초기화"""
        # 로그 출력
        print(f"YOLOv4 초기화 - 파라미터: {kwargs}")

        # CUDA 진단 실행
        cuda_available = self._check_cuda_support()
        print(f"CUDA 사용 가능: {cuda_available}")

        # 강제 CUDA 사용 옵션 확인
        force_cuda = kwargs.get("force_cuda", False)
        if force_cuda:
            print("CUDA 사용이 강제로 설정되었습니다.")

        # 모델 선택
        model = kwargs.get("model", "YOLOv4-tiny")

        # 모델 디렉토리 설정
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "models", "yolo")

        # 디렉토리 존재 확인
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            print(f"모델 디렉토리 생성: {models_dir}")

        # 디렉토리 내용 확인
        print(f"모델 디렉토리 경로: {models_dir}")
        if os.path.exists(models_dir):
            print(f"모델 디렉토리 파일 목록: {os.listdir(models_dir)}")

        # 모델에 따른 파일 경로 설정
        if model == "YOLOv4-tiny":
            self._weights_path = os.path.join(models_dir, "yolov4-tiny.weights")
            self._config_path = os.path.join(models_dir, "yolov4-tiny.cfg")
        elif model == "YOLOv4":
            self._weights_path = os.path.join(models_dir, "yolov4.weights")
            self._config_path = os.path.join(models_dir, "yolov4.cfg")
        else:
            return False, f"지원하지 않는 모델: {model}"

        # 입력 크기 설정 - 콤보박스에서 선택한 값 사용
        input_size = kwargs.get("input_size", "416x416")  # 기본값은 416x416
        print(f"입력 크기 설정: {input_size}")  # 디버깅용

        if isinstance(input_size, str) and "x" in input_size:
            width, height = map(int, input_size.split("x"))
            self._input_size = (width, height)
            print(f"YOLO 입력 크기 설정됨: {self._input_size}")
        else:
            # 기본값 설정
            self._input_size = (416, 416)
            print(f"기본 YOLO 입력 크기 사용: {self._input_size}")

        # 파일 존재 확인
        print(f"가중치 파일 경로: {self._weights_path}")
        print(f"설정 파일 경로: {self._config_path}")

        if not os.path.exists(self._weights_path):
            # 파일 다운로드 시도
            if self.download_model_file(model):
                print(f"모델 파일 다운로드 완료: {model}")
            else:
                return False, f"가중치 파일을 찾을 수 없습니다: {self._weights_path}"

        if not os.path.exists(self._config_path):
            # 설정 파일 다운로드 시도
            if self.download_config_file(model):
                print(f"설정 파일 다운로드 완료: {model}")
            else:
                return False, f"설정 파일을 찾을 수 없습니다: {self._config_path}"

        # 클래스 파일 경로
        classes_path = os.path.join(models_dir, "coco.names")
        if not os.path.exists(classes_path):
            # 클래스 파일 다운로드 시도
            if self.download_class_file():
                print("클래스 파일 다운로드 완료")
            else:
                return False, f"클래스 파일을 찾을 수 없습니다: {classes_path}"

        try:
            # 네트워크 로드
            self._net = cv2.dnn.readNetFromDarknet(self._config_path, self._weights_path)

            # CUDA 구성 (가용성 확인 후)
            if cuda_available or force_cuda:
                try:
                    # CUDA 설정이 있는지 확인
                    if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA') and hasattr(cv2.dnn, 'DNN_TARGET_CUDA'):
                        # 먼저 네트워크를 생성한 후 CUDA 설정
                        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        print("CUDA 백엔드 및 타겟 설정 완료")

                        # 테스트로 간단한 추론 실행하여 CUDA가 작동하는지 확인
                        dummy_input = np.zeros((self._input_size[1], self._input_size[0], 3), dtype=np.uint8)
                        blob = cv2.dnn.blobFromImage(dummy_input, 1 / 255.0, self._input_size, swapRB=True, crop=False)
                        self._net.setInput(blob)

                        try:
                            layer_names = self._net.getLayerNames()
                            output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
                            _ = self._net.forward(output_layers)
                            print("CUDA 테스트 추론 성공 - GPU 가속이 활성화되었습니다")
                            self._use_cuda = True
                        except Exception as e:
                            print(f"CUDA 테스트 추론 실패: {str(e)}")
                            print("CPU 모드로 전환합니다")
                            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                            self._use_cuda = False
                    else:
                        print("OpenCV에 CUDA 백엔드 또는 타겟이 없습니다 - CPU 모드 사용")
                        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                        self._use_cuda = False
                except Exception as e:
                    print(f"CUDA 설정 중 오류 발생: {str(e)}")
                    print("CPU 모드로 전환합니다")
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self._use_cuda = False
            else:
                print("CUDA를 사용할 수 없습니다 - CPU 모드 사용")
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self._use_cuda = False

            # 클래스 이름 로드
            with open(classes_path, 'r') as f:
                self._classes = f.read().strip().split('\n')

            # 출력 레이어 설정
            layer_names = self._net.getLayerNames()
            try:
                self._output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
            except:
                self._output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

            # 모델 정보 저장
            self._model = model
            self._cuda_initialized = True

            acceleration_status = "GPU 가속" if self._use_cuda else "CPU 모드"
            print(f"YOLOv4 초기화 완료 - 모델: {model}, 입력 크기: {self._input_size}, 가속: {acceleration_status}")
            return True, f"YOLOv4 {model} 초기화 완료 ({acceleration_status})"
        except Exception as e:
            print(f"초기화 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"초기화 중 오류 발생: {str(e)}"

    def download_model_file(self, model):
        """모델 파일 다운로드"""
        import urllib.request

        # 모델별 URL
        urls = {
            "YOLOv4-tiny": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
            "YOLOv4": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"
        }

        if model not in urls:
            print(f"알 수 없는 모델: {model}")
            return False

        try:
            print(f"모델 파일 다운로드 중: {model}")
            urllib.request.urlretrieve(urls[model], self._weights_path)
            print(f"모델 파일 다운로드 완료: {self._weights_path}")
            return True
        except Exception as e:
            print(f"모델 파일 다운로드 실패: {str(e)}")
            return False

    def download_config_file(self, model):
        """설정 파일 다운로드"""
        import urllib.request

        # 설정 파일 URL
        urls = {
            "YOLOv4-tiny": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            "YOLOv4": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
        }

        if model not in urls:
            print(f"알 수 없는 모델: {model}")
            return False

        try:
            print(f"설정 파일 다운로드 중: {model}")
            urllib.request.urlretrieve(urls[model], self._config_path)
            print(f"설정 파일 다운로드 완료: {self._config_path}")

            # CUDA를 위한 추가 설정
            self._modify_config_for_cuda(self._config_path)

            return True
        except Exception as e:
            print(f"설정 파일 다운로드 실패: {str(e)}")
            return False

    def _modify_config_for_cuda(self, config_path):
        """CUDA 성능 최적화를 위해 설정 파일 수정"""
        try:
            # 설정 파일 읽기
            with open(config_path, 'r') as f:
                config_lines = f.readlines()

            # 설정 파일에 CUDA 최적화 설정 추가
            modified = False
            for i, line in enumerate(config_lines):
                # 배치 크기 최적화 (CUDA는 배치 처리에서 성능이 향상)
                if line.strip().startswith('batch='):
                    config_lines[i] = 'batch=1\n'  # 추론용 배치 크기
                    modified = True

                # 세분화 최적화
                if line.strip().startswith('subdivisions='):
                    config_lines[i] = 'subdivisions=1\n'  # 추론용 세분화
                    modified = True

            if modified:
                # 수정된 설정 파일 저장
                with open(config_path, 'w') as f:
                    f.writelines(config_lines)
                print(f"CUDA 최적화를 위해 설정 파일이 수정되었습니다: {config_path}")
        except Exception as e:
            print(f"설정 파일 수정 중 오류 발생: {str(e)}")

    def download_class_file(self):
        """클래스 파일 다운로드"""
        import urllib.request

        # 클래스 파일 URL
        url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

        try:
            print("클래스 파일 다운로드 중")
            classes_path = os.path.join(os.path.dirname(self._weights_path), "coco.names")
            urllib.request.urlretrieve(url, classes_path)
            print(f"클래스 파일 다운로드 완료: {classes_path}")
            return True
        except Exception as e:
            print(f"클래스 파일 다운로드 실패: {str(e)}")
            return False

    def detect(self, frame: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Dict[str, Any]], str, float]:
        """객체 탐지 수행"""
        if self._net is None:
            return frame, [], "탐지기가 초기화되지 않았습니다.", 0.0

        # 매개변수 설정
        conf_threshold = kwargs.get('conf_threshold', 0.4)
        nms_threshold = kwargs.get('nms_threshold', 0.4)

        # 성능 측정 시작
        start_time = time.time()  # time 모듈을 사용

        # 프레임 크기
        height, width = frame.shape[:2]

        # 입력 크기 가져오기
        input_width, input_height = self._input_size

        # 이미지를 블롭으로 변환
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_width, input_height), swapRB=True, crop=False)
        self._net.setInput(blob)

        # 이미 초기화된 경우 CUDA 설정을 다시 시도하지 않음
        if not self._cuda_initialized:
            print("경고: 탐지기가 제대로 초기화되지 않았습니다.")

        # 추론 실행
        outputs = self._net.forward(self._output_layers)

        # 성능 측정 (추론 시간)
        inference_time = time.time() - start_time
        detection_fps = 1.0 / inference_time if inference_time > 0 else 0

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
                    # 객체 위치 계산
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 좌표 계산
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 비최대 억제 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # 결과 생성
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
                confidence = confidences[i]
                class_name = self._classes[class_id] if class_id < len(self._classes) else f"Class {class_id}"

                # 박스 그리기
                color = (0, 255, 0)  # 초록색 (RGB)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)

                # 레이블 그리기
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(result_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 탐지 정보 저장
                detections.append({
                    'box': box,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

            # 성능 정보 표시
            acceleration_type = "GPU (CUDA)" if self._use_cuda else "CPU"
            cv2.putText(result_frame, f"FPS: {detection_fps:.1f} | {acceleration_type}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 탐지 결과 메시지
            message = f"총 {len(detections)}개 객체 탐지됨 (임계값: {conf_threshold:.2f}, FPS: {detection_fps:.1f}, 모드: {acceleration_type})"

            return result_frame, detections, message, detection_fps

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version

    @property
    def available_models(self) -> List[Dict[str, str]]:
        return [
            {"name": "YOLOv4-tiny", "version": "v4"},
            {"name": "YOLOv4", "version": "v4"}
        ]

    @property
    def is_using_cuda(self) -> bool:
        """현재 CUDA를 사용 중인지 여부 반환"""
        return self._use_cuda

    def get_backend_info(self) -> Dict[str, Any]:
        """현재 백엔드 정보 반환"""
        if self._net is None:
            return {
                "initialized": False,
                "backend": "None",
                "target": "None",
                "cuda": False
            }

        backend_id = self._net.getPreferableBackend()
        target_id = self._net.getPreferableTarget()

        backend_name = "Unknown"
        if backend_id == cv2.dnn.DNN_BACKEND_CUDA:
            backend_name = "CUDA"
        elif backend_id == cv2.dnn.DNN_BACKEND_DEFAULT:
            backend_name = "Default"
        elif hasattr(cv2.dnn, 'DNN_BACKEND_OPENCV') and backend_id == cv2.dnn.DNN_BACKEND_OPENCV:
            backend_name = "OpenCV"

        target_name = "Unknown"
        if target_id == cv2.dnn.DNN_TARGET_CUDA:
            target_name = "CUDA"
        elif target_id == cv2.dnn.DNN_TARGET_CPU:
            target_name = "CPU"
        elif hasattr(cv2.dnn, 'DNN_TARGET_CUDA_FP16') and target_id == cv2.dnn.DNN_TARGET_CUDA_FP16:
            target_name = "CUDA_FP16"

        return {
            "initialized": self._cuda_initialized,
            "backend": backend_name,
            "target": target_name,
            "cuda": self._use_cuda
        }