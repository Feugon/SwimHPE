import sys
import os
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QFileDialog, QLabel, QSlider,
                             QCheckBox, QProgressBar, QMessageBox, QApplication,
                             QTabWidget)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from model_inference import ModelInference


# ---------------------------------------------------------------------------
# PlayerPanel — one self-contained video player + inference panel
# ---------------------------------------------------------------------------

class PlayerPanel(QWidget):
    """A fully independent video player with its own model, timer, and cache."""

    model_loaded = pyqtSignal(str)   # emits model filename when a model loads

    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path

        # Video state
        self.video_capture = None
        self.original_frame = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30
        self.is_playing = False

        # Inference state
        self.model_inference = ModelInference()
        self.show_keypoints = False
        self.conf_threshold = 0.5
        self.use_tta = False
        self.keypoints_cache = {}
        self.inference_complete = False
        self.processing_frames = False
        self._infer_times = deque(maxlen=30)  # rolling window of inference durations (seconds)

        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()
        self.init_model()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Video display
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumHeight(400)
        layout.addWidget(self.video_label, stretch=1)

        # Seek slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        layout.addWidget(self.progress_slider)

        # Frame counter + inference FPS
        info_row = QHBoxLayout()
        self.frame_info_label = QLabel("Frame: 0 / 0")
        self.frame_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.fps_label = QLabel("Inference: -- FPS")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        info_row.addWidget(self.frame_info_label)
        info_row.addStretch()
        info_row.addWidget(self.fps_label)
        layout.addLayout(info_row)

        # Playback controls
        controls_row = QHBoxLayout()
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setEnabled(False)
        self.prev_button = QPushButton("< Prev")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setEnabled(False)
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        for btn in (self.load_button, self.play_pause_button, self.prev_button, self.next_button):
            controls_row.addWidget(btn)
        layout.addLayout(controls_row)

        # Inference controls
        inference_row = QHBoxLayout()
        self.process_all_button = QPushButton("Process All Frames")
        self.process_all_button.clicked.connect(self.process_all_frames)
        self.process_all_button.setEnabled(False)
        self.keypoints_checkbox = QCheckBox("Show Keypoints")
        self.keypoints_checkbox.setEnabled(False)
        self.keypoints_checkbox.stateChanged.connect(self.toggle_keypoints)
        self.tta_checkbox = QCheckBox("TTA")
        self.tta_checkbox.setToolTip("Test-Time Augmentation: slower but more accurate")
        self.tta_checkbox.stateChanged.connect(self._on_tta_changed)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        for w in (self.process_all_button, self.keypoints_checkbox, self.tta_checkbox, self.progress_bar):
            inference_row.addWidget(w)
        layout.addLayout(inference_row)

        # Confidence threshold slider
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Conf threshold:"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(50)
        self.thresh_slider.setFixedWidth(200)
        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        self.thresh_value_label = QLabel("0.50")
        self.thresh_value_label.setFixedWidth(35)
        thresh_row.addWidget(self.thresh_slider)
        thresh_row.addWidget(self.thresh_value_label)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        # Model row: status label + browse button
        model_row = QHBoxLayout()
        self.model_status_label = QLabel("Model: Not loaded")
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.browse_model_button = QPushButton("Browse Model...")
        self.browse_model_button.setFixedWidth(140)
        self.browse_model_button.clicked.connect(self.browse_model)
        model_row.addWidget(self.model_status_label, stretch=1)
        model_row.addWidget(self.browse_model_button)
        layout.addLayout(model_row)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def init_model(self):
        primary_path = self.model_path or str(Path(__file__).parent.parent / "models" / "yolo26n-pose.pt")
        success = self.model_inference.load_model(model_path=primary_path)
        if success:
            name = os.path.basename(primary_path)
            self.model_status_label.setText(f"Model: {name}")
            self.model_status_label.setStyleSheet("color: green;")
            self.process_all_button.setEnabled(True)
            self.model_loaded.emit(name)
        else:
            success = self.model_inference.load_model()
            if success:
                self.model_status_label.setText("Model: Default (fallback)")
                self.model_status_label.setStyleSheet("color: orange;")
                self.process_all_button.setEnabled(True)
                self.model_loaded.emit("Default")
            else:
                self.model_status_label.setText("Model: Failed to load")
                self.model_status_label.setStyleSheet("color: red;")

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "Model Files (*.pt *.pth *.mlpackage);;All Files (*)"
        )
        if not path:
            return
        path = path.rstrip("/")  # normalize .mlpackage bundle paths
        was_playing = self.is_playing
        self.pause_video()
        success = self.model_inference.load_model(model_path=path)
        if success:
            name = os.path.basename(path)
            self.model_status_label.setText(f"Model: {name}")
            self.model_status_label.setStyleSheet("color: green;")
            self.process_all_button.setEnabled(True)
            # Invalidate cache — new model means old keypoints are stale
            self.keypoints_cache = {}
            self.inference_complete = False
            self._infer_times.clear()
            self.fps_label.setText("Inference: -- FPS")
            self.keypoints_checkbox.setChecked(False)
            self.keypoints_checkbox.setEnabled(False)
            self.tta_checkbox.setEnabled(True)
            self.model_loaded.emit(name)
        else:
            self.model_status_label.setText(f"Model: Failed to load {os.path.basename(path)}")
            self.model_status_label.setStyleSheet("color: red;")
        if was_playing:
            self.play_video()

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if not path:
            return

        if self.video_capture:
            self.video_capture.release()

        self.video_capture = cv2.VideoCapture(path)
        if not self.video_capture.isOpened():
            QMessageBox.warning(self, "Error", f"Cannot open video: {path}")
            return

        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_number = 0
        self.keypoints_cache = {}
        self.inference_complete = False
        self._infer_times.clear()
        self.fps_label.setText("Inference: -- FPS")
        self.keypoints_checkbox.setChecked(False)
        self.keypoints_checkbox.setEnabled(False)
        self.tta_checkbox.setEnabled(True)
        self.progress_slider.setRange(0, max(0, self.total_frames - 1))
        self.progress_slider.setValue(0)
        self.timer.setInterval(int(1000 / self.fps))

        for btn in (self.play_pause_button, self.prev_button, self.next_button):
            btn.setEnabled(True)
        if self.model_inference.model_loaded:
            self.process_all_button.setEnabled(True)

        ret, frame = self.video_capture.read()
        if ret:
            self.original_frame = frame
            self.process_and_display_frame(frame)
        self.update_frame_info()

    def release(self):
        """Release resources (call before removing the panel)."""
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_all_frames(self):
        if not self.video_capture or not self.video_capture.isOpened():
            QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        if not self.model_inference.model_loaded:
            QMessageBox.warning(self, "No Model", "Model is not loaded.")
            return

        was_playing = self.is_playing
        self.pause_video()
        self.processing_frames = True
        self.process_all_button.setEnabled(False)
        self.tta_checkbox.setEnabled(False)
        self.progress_bar.setRange(0, self.total_frames)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.total_frames):
            ret, frame = self.video_capture.read()
            if not ret:
                break
            t0 = time.perf_counter()
            self.keypoints_cache[i] = self.model_inference.predict(frame, use_tta=self.use_tta)
            self._infer_times.append(time.perf_counter() - t0)
            self._update_fps_label()
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.processing_frames = False
        self.inference_complete = True
        self.progress_bar.setVisible(False)
        self.keypoints_checkbox.setEnabled(True)

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        if self.original_frame is not None:
            self.process_and_display_frame(self.original_frame)
        if was_playing:
            self.play_video()

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        self.is_playing = True
        self.play_pause_button.setText("Pause")
        self.timer.start()

    def pause_video(self):
        self.is_playing = False
        self.play_pause_button.setText("Play")
        self.timer.stop()

    def next_frame(self):
        if not self.video_capture:
            return
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame_number = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.original_frame = frame
            self.process_and_display_frame(frame)
            self.progress_slider.setValue(self.current_frame_number)
            self.update_frame_info()
        else:
            self.pause_video()

    def previous_frame(self):
        if not self.video_capture:
            return
        target = max(0, self.current_frame_number - 1)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame_number = target
            self.original_frame = frame
            self.process_and_display_frame(frame)
            self.progress_slider.setValue(self.current_frame_number)
            self.update_frame_info()

    def slider_pressed(self):
        if self.is_playing:
            self.timer.stop()

    def slider_released(self):
        if not self.video_capture:
            return
        target = self.progress_slider.value()
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame_number = target
            self.original_frame = frame
            self.process_and_display_frame(frame)
            self.update_frame_info()
        if self.is_playing:
            self.timer.start()

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _on_thresh_changed(self, value: int):
        self.conf_threshold = value / 100.0
        self.thresh_value_label.setText(f"{self.conf_threshold:.2f}")
        if self.original_frame is not None:
            self.process_and_display_frame(self.original_frame)

    def _on_tta_changed(self, state: int):
        self.use_tta = bool(state)

    def toggle_keypoints(self, checked):
        self.show_keypoints = bool(checked)
        if self.original_frame is not None:
            self.process_and_display_frame(self.original_frame)

    def process_and_display_frame(self, frame):
        display = frame.copy()
        if self.show_keypoints:
            persons = self.keypoints_cache.get(self.current_frame_number)
            if persons is None and self.model_inference.model_loaded:
                t0 = time.perf_counter()
                persons = self.model_inference.predict(frame, use_tta=self.use_tta)
                self._infer_times.append(time.perf_counter() - t0)
                self._update_fps_label()
            if persons:
                self.draw_keypoints(display, persons)
        self.display_frame(display)

    def draw_keypoints(self, frame, all_persons):
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            (5, 11), (6, 12),
        ]
        for keypoints in all_persons:
            pts = []
            for kp in keypoints:
                x, y, conf = kp
                pts.append((int(x), int(y), float(conf)))
                if conf >= self.conf_threshold:
                    color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)

            for a, b in skeleton:
                if a < len(pts) and b < len(pts):
                    xa, ya, ca = pts[a]
                    xb, yb, cb = pts[b]
                    if ca >= self.conf_threshold and cb >= self.conf_threshold:
                        cv2.line(frame, (xa, ya), (xb, yb), (255, 0, 255), 2)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def update_frame_info(self):
        self.frame_info_label.setText(f"Frame: {self.current_frame_number} / {self.total_frames}")

    def _update_fps_label(self):
        if not self._infer_times:
            return
        avg_s = sum(self._infer_times) / len(self._infer_times)
        fps = 1.0 / avg_s if avg_s > 0 else 0
        self.fps_label.setText(f"Inference: {fps:.1f} FPS")


# ---------------------------------------------------------------------------
# VideoPlayer — main window that hosts tabs of PlayerPanels
# ---------------------------------------------------------------------------

class VideoPlayer(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path
        self._tab_counter = 0

        self.setWindowTitle("SwimHPE Video Player")
        self.setGeometry(100, 100, 1200, 850)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        layout.addWidget(self.tab_widget, stretch=1)

        new_tab_btn = QPushButton("+ New Tab")
        new_tab_btn.setFixedHeight(28)
        new_tab_btn.clicked.connect(self.add_tab)
        layout.addWidget(new_tab_btn)

        self.add_tab()   # open one tab on start

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    def add_tab(self):
        self._tab_counter += 1
        panel = PlayerPanel(model_path=self.model_path)
        title = f"Tab {self._tab_counter}"
        idx = self.tab_widget.addTab(panel, title)
        self.tab_widget.setCurrentIndex(idx)

        # Update tab title whenever the panel loads a model
        panel.model_loaded.connect(lambda name, i=idx: self._rename_tab(i, name))

        self._update_close_buttons()

    def _close_tab(self, index: int):
        if self.tab_widget.count() <= 1:
            return   # keep at least one tab
        panel = self.tab_widget.widget(index)
        if isinstance(panel, PlayerPanel):
            panel.release()
        self.tab_widget.removeTab(index)
        self._update_close_buttons()

    def _rename_tab(self, index: int, model_name: str):
        # The index captured in the lambda may shift if tabs were closed;
        # find the panel by iterating instead.
        panel = self.tab_widget.widget(index)
        if panel is not None:
            # Shorten to fit the tab bar (max ~20 chars)
            short = model_name if len(model_name) <= 20 else model_name[:17] + "…"
            self.tab_widget.setTabText(index, short)

    def _update_close_buttons(self):
        # Hide the close button on the last remaining tab
        only_one = self.tab_widget.count() == 1
        bar = self.tab_widget.tabBar()
        for i in range(self.tab_widget.count()):
            btn = bar.tabButton(i, bar.ButtonPosition.RightSide)
            if btn:
                btn.setVisible(not only_one)
