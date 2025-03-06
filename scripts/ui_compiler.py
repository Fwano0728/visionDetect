#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI 컴파일러 - .ui 파일을 .py 파일로 변환하는 유틸리티
"""

import os
import sys
import glob
import subprocess
import logging
import argparse
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('ui_compiler')


def find_pyuic5():
    """pyuic5 실행 파일 찾기"""
    try:
        # 환경 변수에서 실행 시도
        result = subprocess.run(['pyuic5', '--version'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode == 0:
            return 'pyuic5'
    except:
        pass

    # 직접 경로 지정
    pyuic5_paths = [
        os.path.join(sys.prefix, 'bin', 'pyuic5'),
        os.path.join(sys.prefix, 'Scripts', 'pyuic5.exe'),
        os.path.join(sys.prefix, 'Scripts', 'pyuic5-script.py'),
    ]

    for path in pyuic5_paths:
        if os.path.exists(path):
            return path

    # 설치 요청
    logger.error("pyuic5를 찾을 수 없습니다. PyQt5-tools를 설치하세요.")
    logger.error("pip install PyQt5-tools")
    return None


def compile_ui_file(ui_file, output_dir, overwrite=False):
    """
    단일 UI 파일 컴파일

    Args:
        ui_file (str): .ui 파일 경로
        output_dir (str): 출력 디렉토리
        overwrite (bool): 기존 파일 덮어쓰기 여부

    Returns:
        bool: 성공 여부
    """
    # 출력 파일 이름 생성
    basename = os.path.basename(ui_file)
    py_file = os.path.join(output_dir, 'ui_' + os.path.splitext(basename)[0] + '.py')

    # 기존 파일 확인
    if os.path.exists(py_file) and not overwrite:
        ui_time = os.path.getmtime(ui_file)
        py_time = os.path.getmtime(py_file)

        # UI 파일이 더 최근에 수정되지 않았다면 건너뛰기
        if ui_time <= py_time:
            logger.info(f"최신 상태입니다: {py_file}")
            return True

    # pyuic5 찾기
    pyuic5 = find_pyuic5()
    if not pyuic5:
        return False

    # 명령 구성
    cmd = [pyuic5, '-x', ui_file, '-o', py_file]

    # 컴파일 실행
    try:
        logger.info(f"컴파일 중: {ui_file} -> {py_file}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            logger.error(f"컴파일 실패: {ui_file}")
            logger.error(f"오류: {result.stderr}")
            return False

        # 파일 수정 - 불필요한 부분 제거 및 헤더 추가
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 헤더 추가
        header = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
자동 생성된 UI 파일 - 수정하지 마세요!
소스: {basename}
날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
\"\"\"

"""
        # 실행 코드 제거
        if content.endswith(
                'if __name__ == "__main__":\n    import sys\n    app = QtWidgets.QApplication(sys.argv)\n    MainWindow = QtWidgets.QMainWindow()\n    ui = Ui_MainWindow()\n    ui.setupUi(MainWindow)\n    MainWindow.show()\n    sys.exit(app.exec_())'):
            content = content.rsplit('if __name__ == "__main__":', 1)[0]

        # 헤더 추가 및 저장
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(header + content)

        logger.info(f"컴파일 완료: {py_file}")
        return True

    except Exception as e:
        logger.error(f"컴파일 중 오류 발생: {str(e)}")
        return False


def compile_all_ui_files(ui_dir, output_dir, overwrite=False):
    """
    모든 UI 파일 컴파일

    Args:
        ui_dir (str): UI 파일 디렉토리
        output_dir (str): 출력 디렉토리
        overwrite (bool): 기존 파일 덮어쓰기 여부

    Returns:
        tuple: (성공 개수, 실패 개수)
    """
    # 디렉토리 확인 및 생성
    if not os.path.exists(ui_dir):
        logger.error(f"UI 디렉토리를 찾을 수 없음: {ui_dir}")
        return 0, 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"출력 디렉토리 생성됨: {output_dir}")

    # UI 파일 찾기
    ui_files = glob.glob(os.path.join(ui_dir, '*.ui'))

    if not ui_files:
        logger.warning(f"UI 파일을 찾을 수 없음: {ui_dir}")
        return 0, 0

    # 컴파일 실행
    success_count = 0
    fail_count = 0

    for ui_file in ui_files:
        if compile_ui_file(ui_file, output_dir, overwrite):
            success_count += 1
        else:
            fail_count += 1

    # 요약 출력
    logger.info(f"컴파일 완료: 성공 {success_count}개, 실패 {fail_count}개")
    return success_count, fail_count


def main():
    """명령줄 인터페이스"""
    parser = argparse.ArgumentParser(description='UI 파일 컴파일러')
    parser.add_argument('--ui-dir', '-u', default='ui/ui_files',
                        help='UI 파일이 있는 디렉토리 (기본값: ui/ui_files)')
    parser.add_argument('--output-dir', '-o', default='ui/generated',
                        help='컴파일된 Python 파일 출력 디렉토리 (기본값: ui/generated)')
    parser.add_argument('--overwrite', '-w', action='store_true',
                        help='기존 파일 강제 덮어쓰기')

    args = parser.parse_args()

    # 컴파일 실행
    success, fail = compile_all_ui_files(args.ui_dir, args.output_dir, args.overwrite)

    # 종료 코드 설정
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())