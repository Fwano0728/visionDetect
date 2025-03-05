from PyQt5.QtWidgets import (QDialog, QFrame, QHBoxLayout, QVBoxLayout, QCheckBox,
                            QLabel, QPushButton, QScrollArea, QWidget, QProgressBar,
                            QMessageBox)  # QMessageBox 추가
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QFont
from PyQt5 import uic
import os
import time
import json  # json 모듈 추가

class ObjectSelectionDialog(QDialog):
    def __init__(self, detector_manager, parent=None):
        super().__init__(parent)

        # Store references
        self.detector_manager = detector_manager
        self.parent = parent

        # Get camera manager from parent (safely)
        if parent and hasattr(parent, 'camera_manager'):
            self.camera_manager = parent.camera_manager
        else:
            self.camera_manager = None

        # Initialize UI FIRST - this will create all UI elements including apply_button
        self.init_ui()

        # Load available objects
        self.load_objects()

        # THEN connect signals - now all UI elements exist
        self.connect_signals()

    def init_ui(self):
        """UI 요소 초기화"""
        # Set window title and size
        self.setWindowTitle("객체 선택")
        self.setMinimumSize(600, 400)

        # Main layout
        layout = QVBoxLayout(self)

        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("검색:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("객체 이름 검색...")
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)

        # Object list with checkboxes
        self.object_list = QListWidget()
        self.object_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.object_list)

        # Select/Deselect All buttons
        select_buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("모두 선택")
        self.deselect_all_button = QPushButton("모두 해제")
        select_buttons_layout.addWidget(self.select_all_button)
        select_buttons_layout.addWidget(self.deselect_all_button)
        layout.addLayout(select_buttons_layout)

        # Apply/Cancel buttons
        buttons_layout = QHBoxLayout()
        self.apply_button = QPushButton("적용")
        self.cancel_button = QPushButton("취소")
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.apply_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

    # Other methods remain the same...

    def load_coco_classes(self):
        """COCO 클래스 목록과 한글 이름 로드"""
        # COCO 클래스 목록
        try:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            coco_path = os.path.join(project_dir, 'models', 'yolo', 'coco.names')
            with open(coco_path, "r") as f:
                self.coco_classes = f.read().strip().split("\n")
        except:
            # 기본 COCO 클래스 목록 (80개)
            self.coco_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
                                 "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                                 "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                                 "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                                 "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                                 "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                                 "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                                 "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                                 "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                 "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
                                 "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                                 "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                 "scissors", "teddy bear", "hair drier", "toothbrush"]

        # 한글 이름 사전
        self.korean_names = {
            "person": "사람",
            "bicycle": "자전거",
            "car": "자동차",
            "motorbike": "오토바이",
            "bus": "버스",
            "train": "기차",
            "truck": "트럭",
            "cat": "고양이",
            "dog": "개"
            # 필요한 경우 더 많은 한글 이름 추가
        }

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

        # 모든 클래스가 카테고리에 포함되었는지 확인
        categorized_classes = [cls for category in self.categories.values() for cls in category]
        for cls in self.coco_classes:
            if cls not in categorized_classes:
                self.categories["기타"].append(cls)

    def add_category_header(self, category_name, category_items):
        """카테고리 헤더 추가"""
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 5, 0, 5)

        # 카테고리 체크박스
        category_checkbox = QCheckBox(category_name)
        category_checkbox.setFont(QFont("", 10, QFont.Bold))
        category_checkbox.setObjectName(f"category_{category_name}")

        # 현재 카테고리의 모든 항목이 선택되어 있는지 확인
        all_selected = all(self.selected_objects.get(item, False) for item in category_items)
        some_selected = any(self.selected_objects.get(item, False) for item in category_items)

        if all_selected:
            category_checkbox.setCheckState(Qt.Checked)
        elif some_selected:
            category_checkbox.setCheckState(Qt.PartiallyChecked)
        else:
            category_checkbox.setCheckState(Qt.Unchecked)

        # 카테고리 체크박스 상태 변경 시 모든 항목에 적용
        category_checkbox.stateChanged.connect(
            lambda state, items=category_items, cat=category_name: self.toggle_category(items, state, cat)
        )

        header_layout.addWidget(category_checkbox)

        # 카테고리 항목 수 표시
        count_label = QLabel(f"({len(category_items)})")
        header_layout.addWidget(count_label)

        header_layout.addStretch()

        # 카테고리 탐지 횟수 레이블 추가
        category_count_label = QLabel("총 탐지 수: 0")
        category_count_label.setObjectName(f"category_count_{category_name}")
        header_layout.addWidget(category_count_label)

        # 접기/펼치기 버튼
        toggle_button = QPushButton("▼" if self.category_expanded.get(category_name, True) else "▶")
        toggle_button.setFixedWidth(30)
        toggle_button.setObjectName(f"toggle_{category_name}")
        toggle_button.clicked.connect(
            lambda _, cat=category_name: self.toggle_category_expansion(cat)
        )

        header_layout.addWidget(toggle_button)

        return header_frame

    def create_checkbox_for_class(self, cls):
        """클래스별 체크박스 생성"""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 2, 0, 2)

        # 체크박스 생성
        checkbox = QCheckBox(f"{cls} ({self.korean_names.get(cls, '')})")
        checkbox.setObjectName(f"checkbox_{cls}")
        # 이전 선택 상태로 초기화
        checkbox.setChecked(self.selected_objects.get(cls, False))
        checkbox.stateChanged.connect(lambda state, c=cls: self.on_checkbox_changed(c, state))
        layout.addWidget(checkbox)

        # 카운트 레이블
        count_label = QLabel("탐지 수: 0")
        count_label.setObjectName(f"count_{cls}")

        # 진행 표시기 추가 (새로 추가)
        progress_bar = QProgressBar()
        progress_bar.setObjectName(f"progress_{cls}")
        progress_bar.setRange(0, 100)  # 최대 100으로 설정
        progress_bar.setValue(0)
        progress_bar.setFixedWidth(50)
        progress_bar.setFixedHeight(10)
        progress_bar.setTextVisible(False)

        layout.addWidget(progress_bar)
        layout.addWidget(count_label)

        return frame

    def connect_signals(self):
        """시그널 연결"""
        # 버튼 시그널 연결
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # 검색 시그널 연결
        self.search_box.textChanged.connect(self.filter_objects)

        # 모두 선택/해제 시그널 연결
        self.select_all_button.clicked.connect(self.select_all_objects)
        self.deselect_all_button.clicked.connect(self.deselect_all_objects)

        # No camera manager signals - removed as suggested earlier

    def setup_debounced_search(self):
        """검색어 입력 지연 처리 설정"""
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.perform_search)

        # 검색 입력란에 이벤트 연결
        self.searchEdit.textChanged.connect(self.start_search_timer)

    def start_search_timer(self):
        """검색 타이머 시작"""
        self.search_timer.stop()
        self.search_timer.start(300)  # 300ms 후에 검색 실행

    def perform_search(self):
        """실제 검색 수행"""
        search_text = self.searchEdit.text().lower()
        self.filter_objects(search_text)

    def filter_objects(self, search_text):
        """검색어로 객체 필터링"""
        # 검색어가 없으면 모든 항목 표시
        if not search_text:
            for category_name in self.categories.keys():
                container = self.findChild(QWidget, f"container_{category_name}")
                header = self.findChild(QFrame, f"category_{category_name}").parent()
                if container and header:
                    container.setVisible(self.category_expanded.get(category_name, True))
                    header.setVisible(True)
            return

        # 검색어로 필터링
        for category_name, category_items in self.categories.items():
            category_visible = False
            container = self.findChild(QWidget, f"container_{category_name}")
            header = self.findChild(QCheckBox, f"category_{category_name}").parent()

            if not container or not header:
                continue

            # 카테고리 이름이 검색어를 포함하면 전체 표시
            if search_text in category_name.lower():
                container.setVisible(True)
                header.setVisible(True)
                continue

            # 카테고리 내 항목 필터링
            for cls in category_items:
                checkbox = self.findChild(QCheckBox, f"checkbox_{cls}")
                if not checkbox:
                    continue

                # 항목이 검색어를 포함하는지 확인
                item_visible = search_text in cls.lower() or search_text in self.korean_names.get(cls, '').lower()

                # 체크박스가 있는 프레임 찾기
                frame = checkbox.parent()
                if frame:
                    frame.setVisible(item_visible)

                if item_visible:
                    category_visible = True

            # 카테고리 표시 여부 설정
            container.setVisible(category_visible)
            header.setVisible(category_visible)

    def add_preset_controls(self):
        """프리셋 저장/불러오기 컨트롤 추가"""
        # 프리셋 컨테이너
        preset_frame = QFrame()
        preset_layout = QHBoxLayout(preset_frame)

        # 프리셋 이름 입력
        self.presetNameEdit = QLineEdit()
        self.presetNameEdit.setPlaceholderText("프리셋 이름 입력...")

        # 저장 버튼
        save_button = QPushButton("저장")
        save_button.clicked.connect(self.save_current_preset)

        # 프리셋 선택 콤보박스
        self.presetCombo = QComboBox()
        self.update_preset_list()

        # 불러오기 버튼
        load_button = QPushButton("불러오기")
        load_button.clicked.connect(self.load_selected_preset)

        # 삭제 버튼
        delete_button = QPushButton("삭제")
        delete_button.clicked.connect(self.delete_selected_preset)

        # 레이아웃에 추가
        preset_layout.addWidget(QLabel("프리셋:"))
        preset_layout.addWidget(self.presetNameEdit)
        preset_layout.addWidget(save_button)
        preset_layout.addWidget(self.presetCombo)
        preset_layout.addWidget(load_button)
        preset_layout.addWidget(delete_button)

        # 메인 레이아웃에 추가
        self.verticalLayout.insertWidget(1, preset_frame)  # 상단에 추가

    def update_preset_list(self):
        """저장된 프리셋 목록 업데이트"""
        if not hasattr(self, 'presetCombo'):
            return

        self.presetCombo.clear()

        # 프리셋 디렉토리 확인
        presets_dir = self.get_presets_directory()

        if not os.path.exists(presets_dir):
            return

        # 모든 프리셋 파일 찾기
        preset_files = [f[:-5] for f in os.listdir(presets_dir) if f.endswith('.json')]

        if preset_files:
            self.presetCombo.addItems(preset_files)

    def get_presets_directory(self):
        """프리셋 디렉토리 경로 반환"""
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        presets_dir = os.path.join(project_dir, 'config', 'presets')
        os.makedirs(presets_dir, exist_ok=True)
        return presets_dir

    def save_current_preset(self):
        """현재 선택 상태를 프리셋으로 저장"""
        import json

        preset_name = self.presetNameEdit.text().strip()
        if not preset_name:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "프리셋 이름을 입력해주세요.")
            return

        # 프리셋 파일 경로
        preset_file = os.path.join(self.get_presets_directory(), f"{preset_name}.json")

        # 선택 상태 저장
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(self.selected_objects, f, ensure_ascii=False, indent=2)

        # 프리셋 목록 업데이트
        self.update_preset_list()

        # 저장 성공 메시지
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "알림", f"프리셋 '{preset_name}'이(가) 저장되었습니다.")

    def load_selected_preset(self):
        """선택된 프리셋 불러오기"""
        import json

        preset_name = self.presetCombo.currentText()
        if not preset_name:
            return

        # 프리셋 파일 경로
        preset_file = os.path.join(self.get_presets_directory(), f"{preset_name}.json")

        if not os.path.exists(preset_file):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", f"프리셋 '{preset_name}'을(를) 찾을 수 없습니다.")
            return

        try:
            # 프리셋 불러오기
            with open(preset_file, 'r', encoding='utf-8') as f:
                self.selected_objects = json.load(f)

            # UI 업데이트
            self.update_checkboxes()
            self.update_selection_count()

            # 성공 메시지
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "알림", f"프리셋 '{preset_name}'이(가) 로드되었습니다.")
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"프리셋을 불러오는 중 오류가 발생했습니다: {str(e)}")

    def delete_selected_preset(self):
        """선택된 프리셋 삭제"""
        preset_name = self.presetCombo.currentText()
        if not preset_name:
            return

        # 확인 대화상자
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, "확인", f"프리셋 '{preset_name}'을(를) 삭제하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply != QMessageBox.Yes:
            return

        # 프리셋 파일 경로
        preset_file = os.path.join(self.get_presets_directory(), f"{preset_name}.json")

        try:
            if os.path.exists(preset_file):
                os.remove(preset_file)

            # 프리셋 목록 업데이트
            self.update_preset_list()

            # 성공 메시지
            QMessageBox.information(self, "알림", f"프리셋 '{preset_name}'이(가) 삭제되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"프리셋을 삭제하는 중 오류가 발생했습니다: {str(e)}")

    def toggle_category(self, items, state, category_name):
        """카테고리 내 모든 항목 선택/해제"""
        # Qt.PartiallyChecked 상태일 때 처리
        if state == Qt.PartiallyChecked:
            return

        checked = (state == Qt.Checked)

        # 해당 카테고리의 모든 항목 업데이트
        for cls in items:
            self.selected_objects[cls] = checked

            # 체크박스 UI 업데이트
            checkbox = self.findChild(QCheckBox, f"checkbox_{cls}")
            if checkbox and checkbox.isChecked() != checked:
                checkbox.setChecked(checked)

        # 선택 개수 업데이트
        self.update_selection_count()

    def toggle_category_expansion(self, category_name):
        """카테고리 접기/펼치기 토글"""
        # 현재 상태 반전
        self.category_expanded[category_name] = not self.category_expanded.get(category_name, True)
        is_expanded = self.category_expanded[category_name]

        # 컨테이너 표시 상태 변경
        container = self.findChild(QWidget, f"container_{category_name}")
        if container:
            container.setVisible(is_expanded)

        # 토글 버튼 텍스트 변경
        toggle_button = self.findChild(QPushButton, f"toggle_{category_name}")
        if toggle_button:
            toggle_button.setText("▼" if is_expanded else "▶")

    def update_checkboxes(self):
        """모든 체크박스 상태 업데이트"""
        # 개별 클래스 체크박스 업데이트
        for cls in self.coco_classes:
            checkbox = self.findChild(QCheckBox, f"checkbox_{cls}")
            if checkbox:
                is_checked = self.selected_objects.get(cls, False)
                if checkbox.isChecked() != is_checked:
                    checkbox.setChecked(is_checked)

        # 카테고리 체크박스 업데이트
        for category_name, category_items in self.categories.items():
            category_checkbox = self.findChild(QCheckBox, f"category_{category_name}")
            if category_checkbox:
                all_selected = all(self.selected_objects.get(item, False) for item in category_items)
                some_selected = any(self.selected_objects.get(item, False) for item in category_items)

                if all_selected:
                    category_checkbox.setCheckState(Qt.Checked)
                elif some_selected:
                    category_checkbox.setCheckState(Qt.PartiallyChecked)
                else:
                    category_checkbox.setCheckState(Qt.Unchecked)

    def select_all(self):
        """모든 객체 선택"""
        for cls in self.coco_classes:
            self.selected_objects[cls] = True

        self.update_checkboxes()
        self.update_selection_count()

    def deselect_all(self):
        """모든 객체 선택 해제"""
        for cls in self.coco_classes:
            self.selected_objects[cls] = False

        self.update_checkboxes()
        self.update_selection_count()

    def reset_to_default(self):
        """기본 설정으로 초기화 (person만 선택)"""
        for cls in self.coco_classes:
            self.selected_objects[cls] = (cls == "person")

        self.update_checkboxes()
        self.update_selection_count()

    def on_checkbox_changed(self, class_name, state):
        """체크박스 상태가 변경될 때 호출"""
        checked = (state == Qt.Checked)
        self.selected_objects[class_name] = checked

        # 관련 카테고리 체크박스 상태 업데이트
        for category_name, category_items in self.categories.items():
            if class_name in category_items:
                self.update_category_checkbox(category_name, category_items)

        # 선택 개수 업데이트
        self.update_selection_count()

    def update_category_checkbox(self, category_name, category_items):
        """카테고리 체크박스 상태 업데이트"""
        category_checkbox = self.findChild(QCheckBox, f"category_{category_name}")
        if not category_checkbox:
            return

        all_selected = all(self.selected_objects.get(item, False) for item in category_items)
        some_selected = any(self.selected_objects.get(item, False) for item in category_items)

        if all_selected:
            category_checkbox.setCheckState(Qt.Checked)
        elif some_selected:
            category_checkbox.setCheckState(Qt.PartiallyChecked)
        else:
            category_checkbox.setCheckState(Qt.Unchecked)

    def update_selection_count(self):
        """선택된 객체 수 업데이트"""
        selected_count = sum(1 for val in self.selected_objects.values() if val)
        total_count = len(self.coco_classes)

        # 선택된 객체 수 표시 레이블 업데이트
        if hasattr(self, 'selectionCountLabel'):
            self.selectionCountLabel.setText(f"선택됨: {selected_count}/{total_count}")

    def get_selected_objects(self):
        """선택된 객체 딕셔너리 반환"""
        return self.selected_objects

    def update_detection_counts(self, detection_counts):
        """각 객체별 탐지 횟수 업데이트"""
        # 최소 업데이트 간격 설정 (너무 자주 업데이트하면 UI가 느려질 수 있음)
        current_time = time.time()
        if hasattr(self, 'last_count_update') and current_time - self.last_count_update < 0.5:
            return
        self.last_count_update = current_time

        # Person 카운트 업데이트
        person_count = self.findChild(QLabel, "count_person")
        if person_count and "person" in detection_counts:
            person_count.setText(f"탐지 수: {detection_counts['person']}")

        # 다른 객체들 카운트 업데이트
        for cls in self.coco_classes:
            if cls == 'person':
                continue

            count_label = self.findChild(QLabel, f"count_{cls}")
            if count_label:
                count = detection_counts.get(cls, 0)
                count_label.setText(f"탐지 수: {count}")

        # 카테고리별 통계 업데이트 (추가 기능)
        for category_name, category_items in self.categories.items():
            category_count = sum(detection_counts.get(cls, 0) for cls in category_items)
            category_label = self.findChild(QLabel, f"category_count_{category_name}")
            if category_label:
                category_label.setText(f"총 탐지 수: {category_count}")

        # 다른 객체들 카운트 업데이트
        for cls in self.coco_classes:
            if cls == 'person':
                continue

            count_label = self.findChild(QLabel, f"count_{cls}")
            progress_bar = self.findChild(QProgressBar, f"progress_{cls}")

            if count_label:
                count = detection_counts.get(cls, 0)
                count_label.setText(f"탐지 수: {count}")

                # 진행 표시기 업데이트
                if progress_bar:
                    # 최대값 찾기 (모든 객체 중 최대 탐지 수)
                    max_count = max(detection_counts.values()) if detection_counts else 1
                    value = min(100, int((count / max_count) * 100)) if max_count > 0 else 0
                    progress_bar.setValue(value)