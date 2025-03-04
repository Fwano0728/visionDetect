# test_yolo.py 파일 생성
import cv2
import time
import numpy as np
import os
from detector_test import DetectorTester

# YOLO 모델 설정
yolo_configs = [
    {
        'name': 'YOLOv4-tiny',
        'weights': 'models/yolo/yolov4-tiny.weights',
        'config': 'models/yolo/yolov4-tiny.cfg',
        'confidence': 0.4,
        'nms_threshold': 0.3
    },
    # 다른 YOLO 모델 추가 가능
]


def test_yolo_detector(config, test_video='test.mp4', frame_limit=100):
    """YOLO 모델 테스트"""
    # 모델 로드
    net = cv2.dnn.readNetFromDarknet(config['config'], config['weights'])

    # 출력 레이어 이름
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 비디오 로드
    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {test_video}")
        return

    # 테스터 초기화
    tester = DetectorTester()
    tester.start_test(config['name'], config)

    frame_count = 0
    while cap.isOpened() and frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 객체 탐지 시작
        start_time = time.time()

        # 이미지를 YOLO 입력 형식으로 변환
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # 예측 실행
        outputs = net.forward(output_layers)

        # 결과 처리
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > config['confidence']:
                    # 객체 위치 계산
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 바운딩 박스 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 비최대 억제 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, config['confidence'], config['nms_threshold'])

        # 결과 이미지에 박스 그리기
        result_frame = frame.copy()
        people_count = 0

        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            elif isinstance(indices, np.ndarray) and len(indices.shape) > 1:
                indices = indices.flatten()

            for i in indices:
                box = boxes[i]
                x, y, w, h = box

                if class_ids[i] == 0:  # 사람 클래스
                    people_count += 1
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f'Person {confidences[i]:.2f}'
                    cv2.putText(result_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        detection_time = time.time() - start_time
        fps = 1.0 / detection_time if detection_time > 0 else 0

        # 결과 기록
        tester.record_frame({
            'frame_id': frame_count,
            'people_count': people_count,
            'detection_time': detection_time,
            'fps': fps,
            'detections': [{
                'box': boxes[i],
                'confidence': confidences[i]
            } for i in indices] if len(indices) > 0 else []
        })

        # 5번째 프레임마다 이미지 저장
        if frame_count % 5 == 0:
            tester.save_test_image(result_frame, people_count,
                                   f"test_results/{config['name']}_frame_{frame_count}.jpg")

        print(f"프레임 {frame_count}: 인식된 사람 {people_count}명, FPS: {fps:.1f}")

    cap.release()
    tester.end_test()
    print(f"{config['name']} 테스트 완료!")


if __name__ == "__main__":
    for config in yolo_configs:
        test_yolo_detector(config)