#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import uic

# UI 파일 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "ui", "ui_files")
TOOLS_DIALOG_UI = os.path.join(UI_DIR, "tools_dialog.ui")


class ToolsDialog(QDialog):
    """도구 및 기타 기능 다이얼로그"""

    # 시그널 정의
    object_selection_requested = pyqtSignal()  # 객체 선택 요청

    def __init__(self, parent=None):
        # 부모를 None으로 설정하여 독립적인 창으로 작동
        super().__init__(None)

        # 윈도우 플래그 설정
        self.setWindowFlags(Qt.Window)
        self.setWindowModality(Qt.NonModal)

        # UI 파일 로드
        if os.path.exists(TOOLS_DIALOG_UI):
            uic.loadUi(TOOLS_DIALOG_UI, self)
        else:
            raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {TOOLS_DIALOG_UI}")

        self.parent_window = parent  # 부모 창 참조 저장

        # 시그널 연결
        self.objectSelectionButton.clicked.connect(self.on_object_selection_clicked)
        self.closeButton.clicked.connect(self.close)

        # 부모 윈도우 핸들러와 시그널 연결
        if self.parent_window:
            self.object_selection_requested.connect(self.parent_window.handlers.show_object_selection_dialog)

    def on_object_selection_clicked(self):
        """객체 선택 버튼 클릭 처리"""
        self.object_selection_requested.emit()
        self.close()