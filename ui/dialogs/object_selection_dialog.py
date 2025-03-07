#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ui/dialogs/object_selection_dialog.py

import os
import json
from PyQt5.QtWidgets import (QDialog, QCheckBox, QColorDialog, QMessageBox,
                             QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
                             QWidget, QPushButton, QLabel, QGroupBox)
from PyQt5.QtGui import QColor, QBrush, QFont
from PyQt5.QtCore import Qt, pyqtSignal

# 설정 파일 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))), "config")
OBJECT_CONFIG_FILE = os.path.join(CONFIG_DIR, "object_detection_config.json")

# COCO 객체 카테고리 정의
OBJECT_CATEGORIES = {
    "person": ["person"],
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket"],
    "furniture": ["chair", "sofa", "bed", "dining table", "toilet"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "electronics": ["tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                    "refrigerator"],
    "others": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "backpack",
               "umbrella", "handbag", "tie", "suitcase", "potted plant", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush", "book", "sink"]
}

# 카테고리별 기본 색상
DEFAULT_COLORS = {
    "person": "#FF0000",  # 빨강
    "animals": "#00FF00",  # 초록
    "vehicles": "#0000FF",  # 파랑
    "sports": "#FFFF00",  # 노랑
    "furniture": "#FF00FF",  # 마젠타
    "kitchen": "#00FFFF",  # 시안
    "electronics": "#FF8000",  # 주황
    "others": "#8000FF"  # 보라
}

# 카테고리 이름 한글 변환
CATEGORY_NAMES = {
    "person": "사람",
    "animals": "동물",
    "vehicles": "탈것",
    "sports": "스포츠",
    "furniture": "가구",
    "kitchen": "주방용품",
    "electronics": "전자제품",
    "others": "기타"
}


class ObjectSelectionDialog(QDialog):
    """객체 선택 및 색상 설정 다이얼로그"""

    # 설정 변경 시그널
    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        # 부모를 유지하면서 독립 창으로 설정
        super().__init__(parent)

        # 독립적인 창으로 동작하도록 설정
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowTitle("객체 선택 및 색상 설정")
        self.resize(700, 550)

        # 설정 디렉토리 확인
        os.makedirs(CONFIG_DIR, exist_ok=True)

        # 체크박스와 색상 버튼 저장용 딕셔너리
        self.category_items = {}  # 카테고리 트리 아이템
        self.object_items = {}  # 객체 트리 아이템

        # UI 초기화
        self.init_ui()

        # 시그널 연결
        self.connect_signals()

        # 설정 로드
        self.load_settings()

    def init_ui(self):
        """UI 초기화"""
        # 메인 레이아웃
        main_layout = QVBoxLayout(self)

        # 그룹 박스
        selection_group = QGroupBox("객체 선택")
        selection_layout = QVBoxLayout(selection_group)

        # 글로벌 컨트롤 레이아웃
        global_controls = QHBoxLayout()
        self.select_all_button = QPushButton("모두 선택")
        self.deselect_all_button = QPushButton("모두 해제")
        self.info_label = QLabel("선택된 객체: 0/80")

        global_controls.addWidget(self.select_all_button)
        global_controls.addWidget(self.deselect_all_button)
        global_controls.addStretch()
        global_controls.addWidget(self.info_label)

        selection_layout.addLayout(global_controls)

        # 트리 위젯
        self.object_tree = QTreeWidget()
        self.object_tree.setHeaderLabels(["객체", "색상"])
        self.object_tree.setColumnWidth(0, 500)  # 첫 번째 열 너비 설정
        self.object_tree.setColumnWidth(1, 150)  # 두 번째 열 너비 설정
        self.object_tree.setAlternatingRowColors(True)
        self.object_tree.setAnimated(True)

        # 큰 글꼴 설정
        font = QFont()
        font.setPointSize(12)  # 글꼴 크기 증가
        self.object_tree.setFont(font)

        # 아이템 높이 설정
        self.object_tree.setStyleSheet("""
            QTreeWidget::item {
                height: 30px;  /* 아이템 높이 증가 */
            }
        """)

        # 각 카테고리를 트리 아이템으로 추가
        for category, objects in OBJECT_CATEGORIES.items():
            # 카테고리 아이템 생성
            category_item = QTreeWidgetItem(self.object_tree)
            category_item.setText(0, CATEGORY_NAMES.get(category, category))
            category_item.setFlags(category_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            category_item.setCheckState(0, Qt.Checked)  # 기본적으로 체크됨

            # 글꼴 설정 (카테고리는 더 굵게)
            category_font = QFont()
            category_font.setPointSize(12)
            category_font.setBold(True)
            category_item.setFont(0, category_font)

            # 색상 표시 및 변경 버튼 텍스트
            color_text = "변경"
            category_item.setText(1, color_text)
            category_item.setBackground(1, QBrush(QColor(DEFAULT_COLORS[category])))

            # 카테고리 아이템 저장
            self.category_items[category] = category_item

            # 객체 아이템 추가
            self.object_items[category] = {}
            for obj in objects:
                object_item = QTreeWidgetItem(category_item)
                object_item.setText(0, obj)
                object_item.setFlags(object_item.flags() | Qt.ItemIsUserCheckable)
                object_item.setCheckState(0, Qt.Checked)  # 기본적으로 체크됨

                # 객체 아이템 글꼴 설정
                object_item.setFont(0, font)

                # 색상 표시 (카테고리와 같은 색상)
                object_item.setBackground(1, QBrush(QColor(DEFAULT_COLORS[category])))

                # 객체 아이템 저장
                self.object_items[category][obj] = object_item

        # 모든 카테고리 확장
        self.object_tree.expandAll()

        selection_layout.addWidget(self.object_tree)
        main_layout.addWidget(selection_group)

        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("저장 및 적용")
        self.reset_button = QPushButton("초기화")
        self.close_button = QPushButton("닫기")

        # 버튼 글꼴도 크게 설정
        button_font = QFont()
        button_font.setPointSize(12)
        self.save_button.setFont(button_font)
        self.reset_button.setFont(button_font)
        self.close_button.setFont(button_font)

        self.save_button.setMinimumHeight(40)
        self.reset_button.setMinimumHeight(40)
        self.close_button.setMinimumHeight(40)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def connect_signals(self):
        """시그널 연결"""
        # 전역 컨트롤 버튼
        self.select_all_button.clicked.connect(lambda: self.select_all(True))
        self.deselect_all_button.clicked.connect(lambda: self.select_all(False))

        # 저장 및 닫기 버튼
        self.save_button.clicked.connect(self.save_settings)
        self.reset_button.clicked.connect(self.reset_settings)
        self.close_button.clicked.connect(self.close)

        # 트리 아이템 변경 시 선택된 객체 수 업데이트
        self.object_tree.itemChanged.connect(self.update_selected_count)

        # 트리 아이템 클릭 시 처리
        self.object_tree.itemClicked.connect(self.on_item_clicked)

    def on_item_clicked(self, item, column):
        """아이템 클릭 처리"""
        # 색상 열만 처리
        if column != 1:
            return

        # 카테고리 아이템인 경우에만 색상 변경
        category = None
        for cat, cat_item in self.category_items.items():
            if item == cat_item:
                category = cat
                break

        if category:
            self.change_category_color(category)

    def select_all(self, checked):
        """모든 카테고리의 모든 객체 선택/해제"""
        state = Qt.Checked if checked else Qt.Unchecked

        # 트리 항목 변경 시그널 블록
        self.object_tree.blockSignals(True)

        for category, item in self.category_items.items():
            item.setCheckState(0, state)

        # 트리 항목 변경 시그널 해제
        self.object_tree.blockSignals(False)

        # 선택된 객체 수 업데이트
        self.update_selected_count()

    def select_category(self, category, checked):
        """특정 카테고리의 모든 객체 선택/해제"""
        if category in self.category_items:
            state = Qt.Checked if checked else Qt.Unchecked

            # 트리 항목 변경 시그널 블록
            self.object_tree.blockSignals(True)

            self.category_items[category].setCheckState(0, state)

            # 트리 항목 변경 시그널 해제
            self.object_tree.blockSignals(False)

            # 선택된 객체 수 업데이트
            self.update_selected_count()

    def change_category_color(self, category):
        """카테고리 색상 변경"""
        if category in self.category_items:
            # 현재 색상 가져오기
            background = self.category_items[category].background(1)
            current_color = QColor(DEFAULT_COLORS[category])
            if background:
                current_color = background.color()

            # 색상 선택 대화상자 표시
            color = QColorDialog.getColor(current_color, self, f"{CATEGORY_NAMES.get(category, category)} 카테고리 색상 선택")

            # 유효한 색상이 선택된 경우
            if color.isValid():
                # 트리 아이템 배경 색상 변경
                self.category_items[category].setBackground(1, QBrush(color))

                # 자식 객체들의 색상도 변경
                for obj_item in self.object_items[category].values():
                    obj_item.setBackground(1, QBrush(color))

    def update_selected_count(self):
        """선택된 객체 수 업데이트"""
        total_objects = 0
        selected_objects = 0

        for category, objects in self.object_items.items():
            for obj, item in objects.items():
                total_objects += 1
                if item.checkState(0) == Qt.Checked:
                    selected_objects += 1

        self.info_label.setText(f"선택된 객체: {selected_objects}/{total_objects}")

    def get_current_settings(self):
        """현재 설정 가져오기"""
        settings = {
            "objects": {},
            "colors": {}
        }

        # 객체 선택 상태
        for category, objects in self.object_items.items():
            for obj, item in objects.items():
                settings["objects"][obj] = (item.checkState(0) == Qt.Checked)

        # 카테고리 색상
        for category, item in self.category_items.items():
            background = item.background(1)
            if background:
                color = background.color().name()
                settings["colors"][category] = color
            else:
                settings["colors"][category] = DEFAULT_COLORS[category]

        return settings

    def load_settings(self):
        """저장된 설정 로드"""
        # 기본 설정
        settings = {
            "objects": {},
            "colors": DEFAULT_COLORS.copy()
        }

        # 모든 객체를 기본적으로 선택됨으로 설정
        for category, objects in OBJECT_CATEGORIES.items():
            for obj in objects:
                settings["objects"][obj] = True

        # 설정 파일이 있으면 로드
        if os.path.exists(OBJECT_CONFIG_FILE):
            try:
                with open(OBJECT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)

                # 객체 선택 상태 적용
                if "objects" in saved_settings:
                    settings["objects"].update(saved_settings["objects"])

                # 색상 적용
                if "colors" in saved_settings:
                    settings["colors"].update(saved_settings["colors"])
            except Exception as e:
                print(f"설정 로드 중 오류 발생: {str(e)}")

        # UI에 설정 적용
        self.apply_settings(settings)

    def apply_settings(self, settings):
        """설정을 UI에 적용"""
        # 트리 항목 변경 시그널 블록
        self.object_tree.blockSignals(True)

        # 객체 선택 상태 적용
        if "objects" in settings:
            for category, objects in self.object_items.items():
                for obj, item in objects.items():
                    if obj in settings["objects"]:
                        item.setCheckState(0, Qt.Checked if settings["objects"][obj] else Qt.Unchecked)

        # 색상 적용
        if "colors" in settings:
            for category, color in settings["colors"].items():
                if category in self.category_items:
                    self.category_items[category].setBackground(1, QBrush(QColor(color)))

                    # 자식 객체들의 색상도 변경
                    for obj_item in self.object_items[category].values():
                        obj_item.setBackground(1, QBrush(QColor(color)))

        # 트리 항목 변경 시그널 해제
        self.object_tree.blockSignals(False)

        # 선택된 객체 수 업데이트
        self.update_selected_count()

    def save_settings(self):
        """설정 저장"""
        settings = self.get_current_settings()

        try:
            with open(OBJECT_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)

            # 설정 변경 시그널 발생
            self.settings_changed.emit(settings)

            QMessageBox.information(self, "저장 완료", "객체 탐지 설정이 저장되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"설정 저장 중 오류가 발생했습니다: {str(e)}")

    def reset_settings(self):
        """설정 초기화"""
        # 확인 대화상자
        reply = QMessageBox.question(self, "설정 초기화",
                                     "모든 설정을 기본값으로 초기화하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 기본 설정
            settings = {
                "objects": {},
                "colors": DEFAULT_COLORS.copy()
            }

            # 모든 객체를 선택됨으로 설정
            for category, objects in OBJECT_CATEGORIES.items():
                for obj in objects:
                    settings["objects"][obj] = True

            # UI에 설정 적용
            self.apply_settings(settings)

            # 설정 파일 삭제 (있는 경우)
            if os.path.exists(OBJECT_CONFIG_FILE):
                try:
                    os.remove(OBJECT_CONFIG_FILE)
                except Exception as e:
                    print(f"설정 파일 삭제 중 오류 발생: {str(e)}")

            QMessageBox.information(self, "초기화 완료", "모든 설정이 기본값으로 초기화되었습니다.")

    @staticmethod
    def load_object_settings():
        """객체 탐지 설정 로드 (정적 메서드)"""
        # 기본 설정
        settings = {
            "objects": {},
            "colors": DEFAULT_COLORS.copy()
        }

        # 모든 객체를 기본적으로 선택됨으로 설정
        for category, objects in OBJECT_CATEGORIES.items():
            for obj in objects:
                settings["objects"][obj] = True

        # 설정 파일이 있으면 로드
        if os.path.exists(OBJECT_CONFIG_FILE):
            try:
                with open(OBJECT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)

                # 객체 선택 상태 적용
                if "objects" in saved_settings:
                    settings["objects"].update(saved_settings["objects"])

                # 색상 적용
                if "colors" in saved_settings:
                    settings["colors"].update(saved_settings["colors"])
            except Exception as e:
                print(f"설정 로드 중 오류 발생: {str(e)}")

        return settings

    def closeEvent(self, event):
        """다이얼로그 닫기 이벤트 처리"""
        # 설정이 변경되었지만 저장되지 않은 경우 확인
        current_settings = self.get_current_settings()

        try:
            # 저장된 설정 로드
            with open(OBJECT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved_settings = json.load(f)

            # 설정이 다른 경우
            if current_settings != saved_settings:
                reply = QMessageBox.question(self, "설정 저장",
                                             "변경된 설정을 저장하시겠습니까?",
                                             QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                             QMessageBox.Yes)

                if reply == QMessageBox.Yes:
                    self.save_settings()
                elif reply == QMessageBox.Cancel:
                    event.ignore()
                    return
        except:
            # 설정 파일이 없는 경우 기본 설정과 비교
            default_settings = {
                "objects": {obj: True for category, objects in OBJECT_CATEGORIES.items() for obj in objects},
                "colors": DEFAULT_COLORS.copy()
            }

            if current_settings != default_settings:
                reply = QMessageBox.question(self, "설정 저장",
                                             "변경된 설정을 저장하시겠습니까?",
                                             QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                             QMessageBox.Yes)

                if reply == QMessageBox.Yes:
                    self.save_settings()
                elif reply == QMessageBox.Cancel:
                    event.ignore()
                    return

        # 기본 닫기 이벤트 처리
        super().closeEvent(event)