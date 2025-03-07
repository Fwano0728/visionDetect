#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
"""
객체 탐지 시스템 메인 스크립트
"""
import cv2
print(f"OpenCV 버전: {cv2.__version__}")

import sys
import os
import logging
import argparse
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

# 로그 설정을 위해 먼저 기본 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정 - 프로젝트 루트를 시스템 경로에 추가
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    # 로그 설정
    from core.logger import init_logging

    # 매니저 클래스
    from core.camera.camera_manager import CameraManager
    from detection.detector_manager import DetectorManager

    # UI 클래스
    from ui.main_window.main_window import MainWindow
    from ui.error_handling import ErrorHandler

    # 버전 정보
    __version__ = "1.0.0"


    def setup_environment():
        """환경 설정 및 필요한 디렉토리 확인"""
        # 필요한 디렉토리 확인
        required_dirs = [
            'models',
            'models/yolo',
            'logs',
            'config',
            'ui/ui_files',
            'ui/generated',
            'test_results'
        ]

        for directory in required_dirs:
            full_path = os.path.join(PROJECT_ROOT, directory)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                logger.info(f"디렉토리 생성: {full_path}")


    def parse_arguments():
        """명령줄 인수 파싱"""
        parser = argparse.ArgumentParser(description="객체 탐지 시스템")

        # 로그 레벨
        parser.add_argument('--log-level', '-l',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            default='INFO',
                            help='로그 레벨 (기본값: INFO)')

        # UI 파일 컴파일
        parser.add_argument('--compile-ui', '-c', action='store_true',
                            help='UI 파일 컴파일 수행')

        # 설정 파일 경로
        parser.add_argument('--config', '-f',
                            default='config/settings.yaml',
                            help='설정 파일 경로 (기본값: config/settings.yaml)')

        # 자동 연결
        parser.add_argument('--auto-connect', '-a', action='store_true',
                            help='시작 시 마지막 카메라에 자동 연결')

        # 최소화 시작
        parser.add_argument('--minimized', '-m', action='store_true',
                            help='최소화 상태로 시작')

        # 버전 표시
        parser.add_argument('--version', '-v', action='version',
                            version=f'객체 탐지 시스템 v{__version__}')

        return parser.parse_args()


    def compile_ui_files():
        """UI 파일 컴파일"""
        try:
            from scripts.ui_compiler import compile_all_ui_files

            logger.info("UI 파일 컴파일 중...")
            ui_dir = os.path.join(PROJECT_ROOT, 'ui', 'ui_files')
            output_dir = os.path.join(PROJECT_ROOT, 'ui', 'generated')

            success, fail = compile_all_ui_files(ui_dir, output_dir)

            if fail > 0:
                logger.warning(f"UI 파일 컴파일 결과: 성공 {success}개, 실패 {fail}개")
            else:
                logger.info(f"UI 파일 컴파일 완료: {success}개 파일")

            return success > 0  # 하나 이상 성공했으면 True 반환

        except ImportError as e:
            logger.error(f"UI 컴파일러를 가져올 수 없습니다: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"UI 파일 컴파일 중 오류 발생: {str(e)}")
            return False


    def show_splash_screen(app):
        """스플래시 화면 표시"""
        # 스플래시 이미지 경로
        splash_image_path = os.path.join(PROJECT_ROOT, 'resources', 'images', 'splash.png')

        # 스플래시 이미지가 있으면 표시
        if os.path.exists(splash_image_path):
            splash_pixmap = QPixmap(splash_image_path)
            splash = QSplashScreen(splash_pixmap)
            splash.show()

            # 스플래시 화면에 메시지 표시
            splash.showMessage("객체 탐지 시스템 초기화 중...",
                               Qt.AlignBottom | Qt.AlignCenter,
                               Qt.white)

            # 이벤트 처리
            app.processEvents()

            return splash

        # 이미지가 없으면 None 반환
        return None


    def main():
        """메인 함수"""
        # 인수 파싱
        args = parse_arguments()

        # 로그 설정
        log_level = getattr(logging, args.log_level)
        logger = init_logging(log_level=log_level)
        logger.info(f"객체 탐지 시스템 v{__version__} 시작")

        # 환경 설정
        setup_environment()

        # UI 컴파일
        if args.compile_ui:
            if not compile_ui_files():
                logger.error("UI 파일 컴파일 실패, 프로그램을 계속합니다.")

        # Qt 애플리케이션 생성
        app = QApplication(sys.argv)
        app.setApplicationName("ObjectDetection")
        app.setOrganizationName("VisionDetect")

        # 스플래시 화면 표시
        splash = show_splash_screen(app)

        try:
            # 매니저 초기화
            if splash:
                splash.showMessage("카메라 매니저 초기화 중...",
                                   Qt.AlignBottom | Qt.AlignCenter,
                                   Qt.white)
                app.processEvents()

            camera_manager = CameraManager()

            if splash:
                splash.showMessage("탐지기 매니저 초기화 중...",
                                   Qt.AlignBottom | Qt.AlignCenter,
                                   Qt.white)
                app.processEvents()

            detector_manager = DetectorManager()

            # 메인 윈도우 생성
            if splash:
                splash.showMessage("사용자 인터페이스 로드 중...",
                                   Qt.AlignBottom | Qt.AlignCenter,
                                   Qt.white)
                app.processEvents()

            main_window = MainWindow(camera_manager, detector_manager)

            # 시작 상태 설정
            if args.minimized:
                if splash:
                    splash.showMessage("최소화 모드로 시작 중...",
                                       Qt.AlignBottom | Qt.AlignCenter,
                                       Qt.white)
                    app.processEvents()

                main_window.showMinimized()
            else:
                if splash:
                    splash.showMessage("프로그램 시작 중...",
                                       Qt.AlignBottom | Qt.AlignCenter,
                                       Qt.white)
                    app.processEvents()

                main_window.show()

            # 자동 연결 설정
            if args.auto_connect:
                # 지연 시작 (UI가 완전히 로드된 후)
                QTimer.singleShot(1000, main_window.connect_last_camera)

            # 스플래시 화면 닫기
            if splash:
                splash.finish(main_window)

            # 애플리케이션 실행
            return app.exec_()

        except Exception as e:
            logger.critical(f"프로그램 초기화 중 심각한 오류 발생: {str(e)}")
            import traceback
            logger.critical(traceback.format_exc())

            # 스플래시 화면이 있는 경우 닫기
            if splash:
                splash.close()

            # 오류 대화상자 표시
            error_app = QApplication.instance() or QApplication(sys.argv)
            ErrorHandler.show_error(None, "초기화 오류",
                                    "프로그램을 초기화하는 중 심각한 오류가 발생했습니다.",
                                    "error", str(e))

            return 1

except Exception as e:
    print(f"초기 모듈 로드 중 오류 발생: {str(e)}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())