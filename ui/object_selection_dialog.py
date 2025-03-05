# ui/object_selection_dialog.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QListWidget, QListWidgetItem, QLineEdit, QCheckBox,
                             QAbstractItemView, QMessageBox, QScrollArea, QWidget,
                             QFrame, QColorDialog, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSettings
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter, QBrush
import os
import time
import json


class ColorButton(QPushButton):
    """색상 선택 버튼 위젯"""

    def __init__(self, color=Qt.green, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(25, 25)
        self.update_color()

    def update_color(self):
        """버튼 배경색 업데이트"""
        self.setStyleSheet(f"background-color: {self.color.name()}; border: 1px solid #888888;")

    def set_color(self, color):
        """색상 설정"""
        self.color = color
        self.update_color()

    def get_color(self):
        """현재 색상 반환"""
        return self.color


class ObjectSelectionDialog(QDialog):
    """객체 선택 대화상자"""

    def __init__(self, detector_manager, parent=None):
        super().__init__(parent)

        # 모달리스 대화상자로 설정
        self.setWindowModality(Qt.NonModal)

        # 기본 속성 설정
        self.detector_manager = detector_manager
        self.parent = parent

        # 객체 선택 상태 초기화
        self.selected_objects = {}
        self.object_checkboxes = {}
        self.category_colors = {}

        # 객체 카테고리 정의
        self.define_categories()

        # 설정 로드
        self.load_settings()

        # UI 초기화
        self.init_ui()

        # 사용 가능한 객체 목록 로드
        self.load_objects()

        # 시그널 연결
        self.connect_signals()

    def define_categories(self):
        """객체 카테고리 정의"""
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

        # 기본 색상 설정
        self.default_colors = {
            "사람": QColor(0, 255, 0),  # 초록색
            "탈것": QColor(0, 0, 255),  # 빨간색
            "동물": QColor(255, 165, 0),  # 주황색
            "생활용품": QColor(128, 0, 128),  # 보라색
            "음식": QColor(255, 192, 203),  # 분홍색
            "가구": QColor(0, 128, 128),  # 청록색
            "전자기기": QColor(255, 255, 0),  # 노란색
            "기타": QColor(192, 192, 192)  # 회색
        }

    def load_settings(self):
        """설정 로드"""
        try:
            # 설정 파일 경로
            settings_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         'config', 'object_colors.json')

            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    saved_colors = json.load(f)

                # 저장된 색상 로드
                for category, color_str in saved_colors.items():
                    self.category_colors[category] = QColor(color_str)
            else:
                # 기본 색상 사용
                self.category_colors = self.default_colors.copy()
        except Exception as e:
            print(f"설정 로드 오류: {str(e)}")
            # 오류 발생 시 기본 색상 사용
            self.category_colors = self.default_colors.copy()

    def save_settings(self):
        """설정 저장"""
        try:
            # 설정 디렉토리 경로
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
            os.makedirs(config_dir, exist_ok=True)

            # 설정 파일 경로
            settings_file = os.path.join(config_dir, 'object_colors.json')

            # 색상을 문자열로 변환하여 저장
            color_dict = {category: color.name() for category, color in self.category_colors.items()}

            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(color_dict, f, ensure_ascii=False, indent=2)

            print("객체 색상 설정이 저장되었습니다.")
        except Exception as e:
            print(f"설정 저장 오류: {str(e)}")

    def init_ui(self):
        """UI 초기화"""
        # 창 제목 및 크기 설정
        self.setWindowTitle("객체 선택")
        self.setMinimumSize(600, 500)

        # 메인 레이아웃
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # 검색 영역
        search_layout = QHBoxLayout()
        search_label = QLabel("검색:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("객체 이름 검색...")
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        main_layout.addLayout(search_layout)

        # 객체 목록 영역 (스크롤 가능)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.object_list_layout = QVBoxLayout(scroll_content)
        self.object_list_layout.setSpacing(10)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)  # 확장 가능하도록 1의 stretch 값 지정

        # 모두 선택/해제 버튼
        select_buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("모두 선택")
        self.deselect_all_button = QPushButton("모두 해제")
        select_buttons_layout.addWidget(self.select_all_button)
        select_buttons_layout.addWidget(self.deselect_all_button)
        select_buttons_layout.addStretch()
        main_layout.addLayout(select_buttons_layout)

        # 적용/취소 버튼
        buttons_layout = QHBoxLayout()
        self.apply_button = QPushButton("적용")
        self.cancel_button = QPushButton("취소")
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.apply_button)
        buttons_layout.addWidget(self.cancel_button)
        main_layout.addLayout(buttons_layout)

        # 카테고리별 박스 및 색상 버튼 저장
        self.category_color_buttons = {}

    def load_objects(self):
        """사용 가능한 객체 목록 로드"""
        try:
            # 먼저 COCO 클래스 목록 불러오기
            coco_classes = []
            try:
                coco_names_path = "models/yolo/coco.names"
                with open(coco_names_path, "r") as f:
                    coco_classes = f.read().strip().split("\n")
            except Exception as e:
                print(f"COCO 클래스 목록 로드 오류: {str(e)}")
                # 오류 시 기본 클래스 목록 사용
                coco_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train"]

            # 부모 윈도우에서 기존 선택된 객체 목록 가져오기
            if self.parent and hasattr(self.parent, 'selected_objects'):
                self.selected_objects = self.parent.selected_objects.copy()
            else:
                # 기본값으로 person만 선택
                self.selected_objects = {cls: (cls == "person") for cls in coco_classes}

            # 카테고리별로 클래스 정렬
            categorized_classes = {}
            for category in self.categories:
                categorized_classes[category] = []

            # 각 클래스를 해당 카테고리에 할당
            for class_name in coco_classes:
                assigned = False
                for category, items in self.categories.items():
                    if class_name in items:
                        categorized_classes[category].append(class_name)
                        assigned = True
                        break

                # 할당되지 않은 클래스는 기타 카테고리에 추가
                if not assigned:
                    categorized_classes["기타"].append(class_name)

            # 카테고리별로 UI 생성
            self.object_checkboxes = {}

            # 각 카테고리별 그룹 생성
            for category, classes in categorized_classes.items():
                if not classes:  # 비어있는 카테고리는 건너뛰기
                    continue

                # 카테고리 그룹 박스 생성
                category_group = QGroupBox(f"{category} 카테고리")
                category_layout = QVBoxLayout(category_group)

                # 카테고리 헤더 (선택/해제 버튼, 색상 선택 버튼)
                header_layout = QHBoxLayout()

                # 카테고리 선택/해제 버튼
                select_button = QPushButton("모두 선택")
                select_button.setFixedWidth(80)
                select_button.clicked.connect(lambda checked, cat=category: self.select_category(cat))

                deselect_button = QPushButton("모두 해제")
                deselect_button.setFixedWidth(80)
                deselect_button.clicked.connect(lambda checked, cat=category: self.deselect_category(cat))

                # 색상 선택 버튼
                color_button = ColorButton(
                    self.category_colors.get(category, self.default_colors.get(category, Qt.green)))
                color_button.clicked.connect(lambda checked, cat=category, btn=None: self.select_color(cat, btn))
                self.category_color_buttons[category] = color_button

                color_label = QLabel("색상:")

                # 헤더 레이아웃에 추가
                header_layout.addWidget(select_button)
                header_layout.addWidget(deselect_button)
                header_layout.addStretch()
                header_layout.addWidget(color_label)
                header_layout.addWidget(color_button)

                category_layout.addLayout(header_layout)

                # 객체 목록 추가
                for class_name in classes:
                    item_layout = QHBoxLayout()

                    # 체크박스
                    checkbox = QCheckBox(class_name)
                    checkbox.setChecked(self.selected_objects.get(class_name, False))

                    # 현재 탐지 개수 레이블
                    count_label = QLabel("탐지: 0개")
                    count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    # 레이아웃에 추가
                    item_layout.addWidget(checkbox)
                    item_layout.addStretch()
                    item_layout.addWidget(count_label)

                    category_layout.addLayout(item_layout)

                    # 참조 저장
                    self.object_checkboxes[class_name] = {
                        'checkbox': checkbox,
                        'count_label': count_label,
                        'category': category
                    }

                # 메인 레이아웃에 카테고리 그룹 추가
                self.object_list_layout.addWidget(category_group)

            # 마지막에 스트레치 추가하여 위쪽 정렬
            self.object_list_layout.addStretch()

        except Exception as e:
            print(f"객체 목록 로드 오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def connect_signals(self):
        """시그널 연결"""
        try:
            # 버튼 시그널 연결
            self.apply_button.clicked.connect(self.accept)
            self.cancel_button.clicked.connect(self.reject)

            # 검색 시그널 연결
            self.search_box.textChanged.connect(self.filter_objects)

            # 모두 선택/해제 시그널 연결
            self.select_all_button.clicked.connect(self.select_all_objects)
            self.deselect_all_button.clicked.connect(self.deselect_all_objects)
        except Exception as e:
            print(f"시그널 연결 오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def select_all_objects(self):
        """모든 객체 선택"""
        for class_name, widgets in self.object_checkboxes.items():
            checkbox = widgets['checkbox']
            if checkbox.parent().isVisible():  # 필터링된 항목만 처리
                checkbox.setChecked(True)

    def deselect_all_objects(self):
        """모든 객체 선택 해제"""
        for class_name, widgets in self.object_checkboxes.items():
            checkbox = widgets['checkbox']
            if checkbox.parent().isVisible():  # 필터링된 항목만 처리
                checkbox.setChecked(False)

    def filter_objects(self, text):
        """검색어로 객체 필터링"""
        search_text = text.lower()

        # 모든 객체 순회하며 필터링
        for class_name, widgets in self.object_checkboxes.items():
            parent_widget = widgets['checkbox'].parent()
            while parent_widget and not isinstance(parent_widget, QGroupBox):
                parent_widget = parent_widget.parent()

            # 검색어가 빈 문자열이거나 클래스 이름에 검색어가 포함되면 표시
            if not search_text or search_text in class_name.lower():
                widgets['checkbox'].parent().setVisible(True)
                if parent_widget:
                    parent_widget.setVisible(True)
            else:
                widgets['checkbox'].parent().setVisible(False)

        # 카테고리별로 모든 항목이 숨겨졌는지 확인하고 카테고리 그룹 표시 여부 결정
        for category, _ in self.categories.items():
            category_visible = False

            for class_name, widgets in self.object_checkboxes.items():
                if widgets.get('category') == category and widgets['checkbox'].parent().isVisible():
                    category_visible = True
                    break

            # 카테고리의 첫 번째 객체의 부모 QGroupBox 찾기
            for class_name, widgets in self.object_checkboxes.items():
                if widgets.get('category') == category:
                    parent_widget = widgets['checkbox'].parent()
                    while parent_widget and not isinstance(parent_widget, QGroupBox):
                        parent_widget = parent_widget.parent()

                    if parent_widget:
                        parent_widget.setVisible(category_visible)
                    break

    def select_category(self, category):
        """특정 카테고리의 모든 객체 선택"""
        for class_name, widgets in self.object_checkboxes.items():
            if widgets.get('category') == category:
                checkbox = widgets['checkbox']
                if checkbox.isVisible():  # 필터링된 항목만 처리
                    checkbox.setChecked(True)

    def deselect_category(self, category):
        """특정 카테고리의 모든 객체 선택 해제"""
        for class_name, widgets in self.object_checkboxes.items():
            if widgets.get('category') == category:
                checkbox = widgets['checkbox']
                if checkbox.isVisible():  # 필터링된 항목만 처리
                    checkbox.setChecked(False)

    def select_color(self, category, button=None):
        """카테고리 색상 선택"""
        # 현재 색상 가져오기
        current_color = self.category_colors.get(category, self.default_colors.get(category, Qt.green))

        # 색상 대화상자 표시
        color = QColorDialog.getColor(current_color, self, f"{category} 카테고리 색상 선택")

        # 유효한 색상이 선택되었으면 저장
        if color.isValid():
            self.category_colors[category] = color

            # 색상 버튼 업데이트
            if category in self.category_color_buttons:
                self.category_color_buttons[category].set_color(color)

            # 색상 설정 저장
            self.save_settings()

    def update_current_detections(self, current_detections):
        """현재 탐지 중인 객체 수 업데이트"""
        if not current_detections:
            return

        # 각 객체의 현재 탐지 수 레이블 업데이트
        for class_name, widgets in self.object_checkboxes.items():
            count_label = widgets['count_label']
            count = current_detections.get(class_name, 0)
            count_label.setText(f"탐지: {count}개")

            # 현재 탐지 중인 객체에 대해 시각적 강조 표시
            if count > 0:
                count_label.setStyleSheet("color: #007BFF; font-weight: bold;")
            else:
                count_label.setStyleSheet("")

    def get_selected_objects(self):
        """선택된 객체 목록 반환"""
        selected = {}
        for class_name, widgets in self.object_checkboxes.items():
            checkbox = widgets['checkbox']
            selected[class_name] = checkbox.isChecked()
        return selected

    def get_category_colors(self):
        """카테고리별 색상 반환"""
        # QColor 객체를 BGR 튜플로 변환 (OpenCV에서 사용하기 위함)
        bgr_colors = {}
        for category, color in self.category_colors.items():
            # QColor의 RGB를 OpenCV의 BGR로 변환
            bgr_colors[category] = (color.blue(), color.green(), color.red())
        return bgr_colors

    def accept(self):
        """확인 버튼 클릭 처리"""
        # 선택된 객체가 있는지 확인
        selected = self.get_selected_objects()
        if not any(selected.values()):
            QMessageBox.warning(self, "경고", "최소한 하나의 객체를 선택해야 합니다.")
            return

        # 색상 설정 저장
        self.save_settings()

        # 대화상자 수락
        super().accept()

    def reject(self):
        """취소 버튼 클릭 처리"""
        # 대화상자 거부
        super().reject()

    def closeEvent(self, event):
        """대화상자 닫기 이벤트 처리"""
        # 부모 클래스에서 닫기 이벤트 처리
        super().closeEvent(event)