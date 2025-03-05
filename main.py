#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import sys
import os

# PyQt5와 OpenCV 충돌 방지를 위한 환경 변수 설정
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(sys.executable), "lib", "python3.10",
                                                         "site-packages", "PyQt5", "Qt5", "plugins")

# 필요한 경우 환경 변수 설정
os.environ["QT_DEBUG_PLUGINS"] = "0"  # 플러그인 디버그 메시지 비활성화

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def setup_environment():
    """환경 설정 및 필요한 디렉토리 확인"""
    # 필요한 디렉토리 확인
    required_dirs = ['models', 'models/yolo', 'logs', 'configs', 'test_results']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"디렉토리 생성: {directory}")


if __name__ == "__main__":
    # 환경 설정
    setup_environment()

    # Qt 애플리케이션 시작
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # 애플리케이션 실행
    sys.exit(app.exec_())