# detector_test.py 파일 생성
import cv2
import time
import os
import numpy as np
import json
from datetime import datetime


class DetectorTester:
    """객체 탐지 모델 테스트 및 결과 기록 클래스"""

    def __init__(self, output_dir='test_results'):
        """초기화"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.results = []
        self.current_test = None

    def start_test(self, model_name, model_config=None):
        """새 테스트 시작"""
        self.current_test = {
            'model_name': model_name,
            'model_config': model_config or {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'frames': [],
            'metrics': {
                'avg_fps': 0,
                'avg_detection_time': 0,
                'avg_accuracy': 0
            }
        }
        print(f"테스트 시작: {model_name}")

    def record_frame(self, frame_data):
        """프레임 결과 기록"""
        if self.current_test is None:
            print("테스트가 시작되지 않았습니다.")
            return

        self.current_test['frames'].append(frame_data)

    def end_test(self):
        """테스트 종료 및 결과 저장"""
        if self.current_test is None:
            print("테스트가 시작되지 않았습니다.")
            return

        # 평균 지표 계산
        frames = self.current_test['frames']
        if frames:
            self.current_test['metrics']['avg_fps'] = sum(f.get('fps', 0) for f in frames) / len(frames)
            self.current_test['metrics']['avg_detection_time'] = sum(f.get('detection_time', 0) for f in frames) / len(
                frames)

        # 결과 저장
        self.results.append(self.current_test)

        # 파일로 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.output_dir}/{self.current_test['model_name']}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_test, f, ensure_ascii=False, indent=2)

        print(f"테스트 결과 저장됨: {filename}")
        self.current_test = None

    def save_test_image(self, frame, detections, filename=None):
        """테스트 이미지 저장"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/test_image_{timestamp}.jpg"

        cv2.imwrite(filename, frame)
        print(f"테스트 이미지 저장됨: {filename}")

        return filename