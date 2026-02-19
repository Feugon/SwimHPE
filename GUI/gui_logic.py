import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QFileDialog, QLabel, QSlider, QCheckBox, QProgressBar,
                             QSplitter, QTabWidget, QFrame, QMessageBox, QDoubleSpinBox)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from model_inference import ModelInference
from stroke_rate import StrokeRateAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tempfile
import math
from smoothing import compute_blended_keypoints
# Ensure we can import from the cycle/ directory
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cycle'))
except Exception:
    pass

# Ensure robust imports for decoder utilities
try:
    from cycle.pose_decode import (
        reconstruct_pose_from_angles as decode_reconstruct,
        compute_angles_from_kp as decode_compute_angles,
        estimate_lengths_from_pts as decode_estimate_lengths,
        rotate_points as decode_rotate,
        mirror_points_x as decode_mirror_x,
    )
except Exception:
    try:
        # Add project root so 'cycle' package is discoverable
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from cycle.pose_decode import (
            reconstruct_pose_from_angles as decode_reconstruct,
            compute_angles_from_kp as decode_compute_angles,
            estimate_lengths_from_pts as decode_estimate_lengths,
            rotate_points as decode_rotate,
            mirror_points_x as decode_mirror_x,
        )
    except Exception:
        # Fallback: add cycle dir and import module directly
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cycle'))
        from pose_decode import (
            reconstruct_pose_from_angles as decode_reconstruct,
            compute_angles_from_kp as decode_compute_angles,
            estimate_lengths_from_pts as decode_estimate_lengths,
            rotate_points as decode_rotate,
            mirror_points_x as decode_mirror_x,
        )

# Import nearest anchor search with similar robustness
try:
    from cycle.nearest_search import find_nearest_anchors
except Exception:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from cycle.nearest_search import find_nearest_anchors  # type: ignore
    except Exception:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cycle'))
        from nearest_search import find_nearest_anchors  # type: ignore

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SwimHPE Video Player")
        self.setGeometry(100, 100, 1200, 800)
        
        # Video variables
        self.video_capture = None
        self.current_frame = None
        self.original_frame = None  # Store original frame without annotations
        self.is_playing = False
        self.total_frames = 0
        self.current_frame_number = 0
        self.fps = 30
        
        # Model inference
        self.model_inference = ModelInference()
        self.show_keypoints = False
        self.keypoints_cache = {}  # Cache keypoints for each frame
        self.inference_complete = False
        self.processing_frames = False
        
        # Stroke rate analysis
        self.stroke_analyzer = StrokeRateAnalyzer()
        
        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Anchor resources: directory containing multiple angles JSONs
        self.anchors_json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cycle", "angles_json")
        # Default blend parameter (kept for internal helpers)
        self.blend_p = 0.5
        
        self.init_ui()
        
    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create main splitter for video and analysis panels
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Add stroke analysis toggle button to the bottom
        stroke_toggle_layout = QHBoxLayout()
        self.toggle_analysis_button = QPushButton("Show Stroke Analysis")
        self.toggle_analysis_button.clicked.connect(self.toggle_analysis_panel)
        self.toggle_analysis_button.setEnabled(False)  # Disabled until inference is complete
        stroke_toggle_layout.addWidget(self.toggle_analysis_button)
        stroke_toggle_layout.addStretch()
        main_layout.addLayout(stroke_toggle_layout)
        
        # Left panel - Video player
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Video display area
        self.video_label = QLabel("No video loaded")
        self.video_label.setMinimumSize(600, 450)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: #f0f0f0;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)
        
        # Video progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        video_layout.addWidget(self.progress_slider)
        
        # Frame info label
        self.frame_info_label = QLabel("Frame: 0 / 0")
        self.frame_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.frame_info_label)
        
        self.main_splitter.addWidget(video_widget)
        
        # Right panel - Analysis panel (collapsible)
        self.analysis_widget = QWidget()
        self.analysis_widget.setMinimumWidth(400)
        analysis_layout = QVBoxLayout(self.analysis_widget)
        
        # Analysis tabs
        self.analysis_tabs = QTabWidget()

        # Stroke Analysis tab (existing content)
        self.analysis_content = QFrame()
        content_layout = QVBoxLayout(self.analysis_content)
        
        # Matplotlib figure for stroke analysis
        self.stroke_figure = Figure(figsize=(8, 6))
        self.stroke_canvas = FigureCanvas(self.stroke_figure)
        content_layout.addWidget(self.stroke_canvas)
        
        # Stroke analysis controls
        stroke_controls_layout = QHBoxLayout()
        
        self.analyze_strokes_button = QPushButton("Analyze Strokes")
        self.analyze_strokes_button.clicked.connect(self.analyze_strokes)
        self.analyze_strokes_button.setEnabled(False)
        stroke_controls_layout.addWidget(self.analyze_strokes_button)
        
        self.stroke_info_label = QLabel("No stroke analysis")
        stroke_controls_layout.addWidget(self.stroke_info_label)
        
        content_layout.addLayout(stroke_controls_layout)

        self.analysis_tabs.addTab(self.analysis_content, "Stroke Analysis")

        # NOTE: Removed Anchor Match tab and image display

        analysis_layout.addWidget(self.analysis_tabs)
        
        self.main_splitter.addWidget(self.analysis_widget)
        
        # Initially hide the analysis panel completely
        self.analysis_widget.setVisible(False)
        self.analysis_panel_open = False
        
        # Model controls layout
        model_layout = QHBoxLayout()
        
        # Process all frames button
        self.process_all_button = QPushButton("Process All Frames")
        self.process_all_button.clicked.connect(self.process_all_frames)
        self.process_all_button.setEnabled(False)
        model_layout.addWidget(self.process_all_button)
        
        # Show keypoints checkbox
        self.keypoints_checkbox = QCheckBox("Show Keypoints")
        self.keypoints_checkbox.toggled.connect(self.toggle_keypoints)
        self.keypoints_checkbox.setEnabled(False)
        model_layout.addWidget(self.keypoints_checkbox)

        # Show smoothed pose checkbox
        self.smooth_checkbox = QCheckBox("Show Smoothed Pose")
        self.smooth_checkbox.toggled.connect(lambda _: (self.original_frame is not None) and self.process_and_display_frame(self.original_frame))
        self.smooth_checkbox.setEnabled(False)
        model_layout.addWidget(self.smooth_checkbox)

        # p-value control for smoothing
        self.p_label = QLabel("p:")
        model_layout.addWidget(self.p_label)
        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(0.0, 1.0)
        self.p_spin.setSingleStep(0.05)
        self.p_spin.setDecimals(2)
        self.p_spin.setValue(self.blend_p)
        self.p_spin.setEnabled(False)
        self.p_spin.valueChanged.connect(self.set_blend_p)
        model_layout.addWidget(self.p_spin)
        
        # Find closest anchor button (console only)
        self.find_anchor_button = QPushButton("Find Closest Anchor")
        self.find_anchor_button.setObjectName("find_anchor")
        self.find_anchor_button.clicked.connect(self.find_anchor)
        self.find_anchor_button.setEnabled(True)
        model_layout.addWidget(self.find_anchor_button)
        
        # Processing status label
        self.processing_status_label = QLabel("")
        model_layout.addWidget(self.processing_status_label)
        
        # Progress bar for inference
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        model_layout.addWidget(self.progress_bar)
        
        # Model status label
        self.model_status_label = QLabel("Model: Not loaded")
        model_layout.addWidget(self.model_status_label)
        
        main_layout.addLayout(model_layout)
        
        # Control buttons layout
        controls_layout = QHBoxLayout()
        
        # Load video button
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        controls_layout.addWidget(self.load_button)
        
        # Play/Pause button
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setEnabled(False)
        controls_layout.addWidget(self.play_pause_button)
        
        # Previous frame button
        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setEnabled(False)
        controls_layout.addWidget(self.prev_button)
        
        # Next frame button
        self.next_button = QPushButton("Next Frame")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        controls_layout.addWidget(self.next_button)
        
        main_layout.addLayout(controls_layout)
        
        # Initialize model
        self.init_model()

    def show_error_message(self, title, message, details=None):
        """Show error message dialog to user"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if details:
            msg_box.setDetailedText(str(details))
        msg_box.exec()
    
    def show_warning_message(self, title, message):
        """Show warning message dialog to user"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()
    
    def reset_processing_state(self):
        """Reset processing state after error or completion"""
        self.processing_frames = False
        self.process_all_button.setText("Process All Frames")
        self.process_all_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.processing_status_label.setText("")
    def toggle_analysis_panel(self):
        """Toggle the analysis panel visibility"""
        try:
            if self.analysis_panel_open:
                # Hide analysis panel
                self.analysis_widget.setVisible(False)
                self.toggle_analysis_button.setText("Show Stroke Analysis")
                self.analysis_panel_open = False
            else:
                # Show analysis panel
                self.analysis_widget.setVisible(True)
                self.toggle_analysis_button.setText("Hide Stroke Analysis")
                self.analysis_panel_open = True
                # Set splitter proportions (70% video, 30% analysis)
                self.main_splitter.setSizes([700, 300])
        except Exception as e:
            self.show_error_message("Panel Error", "Failed to toggle analysis panel", e)
        
    def init_model(self):
        """Initialize the pose estimation model"""
        try:
            # Use the finetuned model
            finetuned_model_path = "/Users/artemkiryukhin/Desktop/Code/Projects/SwimHPE/yolo26n-pose.pt"
            success = self.model_inference.load_model(model_path=finetuned_model_path)
            if success:
                self.model_status_label.setText("Model: Finetuned Model Loaded")
                self.model_status_label.setStyleSheet("color: green")
                self.process_all_button.setEnabled(True)
            else:
                # Fall back to default model if finetuned model fails
                print("Finetuned model failed to load, trying default model...")
                success = self.model_inference.load_model()
                if success:
                    self.model_status_label.setText("Model: Default Model Loaded")
                    self.model_status_label.setStyleSheet("color: orange")
                    self.process_all_button.setEnabled(True)
                    self.show_warning_message("Model Warning", "Finetuned model failed to load. Using default model instead.")
                else:
                    self.model_status_label.setText("Model: Failed to load")
                    self.model_status_label.setStyleSheet("color: red")
                    self.process_all_button.setEnabled(False)
                    self.show_warning_message("Model Warning", "Both finetuned and default models failed to load. Please check your installation.")
        except Exception as e:
            self.model_status_label.setText(f"Model: Error")
            self.model_status_label.setStyleSheet("color: red")
            self.process_all_button.setEnabled(False)
            self.show_error_message("Model Error", "Failed to initialize pose estimation model", e)
    
    def process_all_frames(self):
        """Process all frames in the video with pose estimation"""
        try:
            if not self.video_capture or not self.video_capture.isOpened():
                self.show_warning_message("Video Error", "No video loaded or video file is not accessible.")
                return
                
            if not self.model_inference.model_loaded:
                self.show_warning_message("Model Error", "Pose estimation model is not loaded.")
                return
            
            if self.processing_frames:
                return  # Already processing
            
            self.processing_frames = True
            self.process_all_button.setText("Processing...")
            self.process_all_button.setEnabled(False)
            
            # Show and setup progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(self.total_frames)
            self.progress_bar.setValue(0)
            
            # Pause video if playing
            was_playing = self.is_playing
            if self.is_playing:
                self.pause_video()
            
            # Store current position
            original_frame_number = self.current_frame_number
            
            print(f"Starting inference on {self.total_frames} frames...")
            
            # Process each frame with error handling
            processed_frames = 0
            for frame_idx in range(self.total_frames):
                try:
                    # Update progress bar and status
                    self.progress_bar.setValue(frame_idx)
                    self.processing_status_label.setText(f"Processing frame {frame_idx + 1}/{self.total_frames}")
                    
                    # Process events to update GUI and check for interruptions
                    QApplication.processEvents()
                    
                    # Check if processing was cancelled
                    if not self.processing_frames:
                        break
                    
                    # Seek to frame
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = self.video_capture.read()
                    
                    if ret:
                        try:
                            # Run inference on this frame
                            keypoints = self.model_inference.predict(frame)
                            self.keypoints_cache[frame_idx] = keypoints
                            processed_frames += 1
                        except Exception as e:
                            print(f"Error processing frame {frame_idx}: {e}")
                            self.keypoints_cache[frame_idx] = None
                    else:
                        print(f"Could not read frame {frame_idx}")
                        self.keypoints_cache[frame_idx] = None
                        
                except Exception as e:
                    print(f"Critical error at frame {frame_idx}: {e}")
                    # Continue processing other frames instead of stopping
                    self.keypoints_cache[frame_idx] = None
            
            # Complete the progress bar
            self.progress_bar.setValue(self.total_frames)
            
            # Processing complete
            self.inference_complete = True
            self.processing_frames = False
            self.process_all_button.setText("Process All Frames")
            self.process_all_button.setEnabled(True)
            self.keypoints_checkbox.setEnabled(True)
            self.analyze_strokes_button.setEnabled(True)
            self.toggle_analysis_button.setEnabled(True)
            if hasattr(self, 'smooth_checkbox'):
                self.smooth_checkbox.setEnabled(True)
            if hasattr(self, 'p_spin'):
                self.p_spin.setEnabled(True)
            
            if processed_frames > 0:
                self.processing_status_label.setText(f"Inference complete! Processed {processed_frames}/{self.total_frames} frames")
                # Set FPS for stroke analyzer
                self.stroke_analyzer.set_fps(self.fps)
                print(f"Inference complete! Processed {processed_frames} frames.")
            else:
                self.processing_status_label.setText("Processing failed - no frames processed")
                self.show_error_message("Processing Error", "Failed to process any frames. Please check the video file and model.")
            
            # Restore original position
            try:
                self.current_frame_number = original_frame_number
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, original_frame_number)
                ret, frame = self.video_capture.read()
                if ret:
                    self.original_frame = frame.copy()
                    self.process_and_display_frame(frame)
                    self.update_frame_info()
            except Exception as e:
                print(f"Error restoring video position: {e}")
            
            # Resume playback if it was playing
            if was_playing:
                self.play_video()
                
        except Exception as e:
            self.reset_processing_state()
            self.show_error_message("Processing Error", "An unexpected error occurred during frame processing", e)
            print(f"Critical error in process_all_frames: {e}")
    
    def analyze_strokes(self):
        """Analyze stroke rate from cached keypoints"""
        try:
            if not self.inference_complete or not self.keypoints_cache:
                self.show_warning_message("Analysis Error", "Please process all frames first before analyzing strokes.")
                return
            
            # Clear previous data
            self.stroke_analyzer.clear_data()
            
            # Add all frame data to analyzer
            valid_frames = 0
            for frame_number, keypoints in self.keypoints_cache.items():
                if keypoints is not None:
                    try:
                        self.stroke_analyzer.add_frame_data(frame_number, keypoints)
                        valid_frames += 1
                    except Exception as e:
                        print(f"Error adding frame {frame_number} to analyzer: {e}")
            
            if valid_frames == 0:
                self.show_warning_message("Analysis Error", "No valid keypoint data found for stroke analysis.")
                return
            
            # Detect stroke events
            try:
                stroke_events = self.stroke_analyzer.detect_stroke_events()
                print(f"Detected {len(stroke_events)} stroke cycles")
            except Exception as e:
                self.show_error_message("Stroke Detection Error", "Failed to detect stroke events", e)
                return
            
            # Calculate stroke rate
            try:
                stroke_rate, intervals = self.stroke_analyzer.calculate_stroke_rate()
            except Exception as e:
                self.show_error_message("Calculation Error", "Failed to calculate stroke rate", e)
                return
            
            # Get summary
            try:
                summary = self.stroke_analyzer.get_stroke_summary()
            except Exception as e:
                self.show_error_message("Summary Error", "Failed to generate stroke summary", e)
                return
            
            # Update info label
            try:
                if summary['total_strokes'] > 0:
                    self.stroke_info_label.setText(
                        f"Strokes: {summary['total_strokes']} | "
                        f"Rate: {summary['stroke_rate_per_minute']:.1f} SPM | "
                        f"Duration: {summary['analysis_duration']:.1f}s"
                    )
                else:
                    self.stroke_info_label.setText("No strokes detected")
            except Exception as e:
                self.stroke_info_label.setText("Error in analysis")
                print(f"Error updating info label: {e}")
            
            # Plot the analysis
            try:
                self.plot_stroke_analysis()
            except Exception as e:
                self.show_error_message("Plot Error", "Failed to generate stroke analysis plot", e)
                
        except Exception as e:
            self.show_error_message("Analysis Error", "An unexpected error occurred during stroke analysis", e)
    
    def plot_stroke_analysis(self):
        """Plot hand displacement and stroke events"""
        try:
            self.stroke_figure.clear()
            
            # Get plot data
            left_frames, left_displacements, right_frames, right_displacements = self.stroke_analyzer.get_plot_data()
            
            if not left_frames and not right_frames:
                ax = self.stroke_figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
                self.stroke_canvas.draw()
                return
            
            # Create subplot
            ax = self.stroke_figure.add_subplot(111)
            
            # Convert frames to time
            time_left = [f / self.fps for f in left_frames] if left_frames else []
            time_right = [f / self.fps for f in right_frames] if right_frames else []
            
            # Plot hand displacements
            if left_frames:
                ax.plot(time_left, left_displacements, 'b-', label='Left Hand', linewidth=2, alpha=0.7)
            if right_frames:
                ax.plot(time_right, right_displacements, 'r-', label='Right Hand', linewidth=2, alpha=0.7)
            
            # Plot zero line (torso reference)
            if time_left or time_right:
                all_times = time_left + time_right
                if all_times:
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Torso Line')
            
            # Mark stroke cycles
            try:
                stroke_cycles = self.stroke_analyzer.stroke_cycles
                for cycle in stroke_cycles:
                    start_time = cycle['start_frame'] / self.fps
                    end_time = cycle['end_frame'] / self.fps
                    ax.axvspan(start_time, end_time, alpha=0.2, color='green')
            except Exception as e:
                print(f"Error marking stroke cycles: {e}")
            
            # Formatting
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Hand Position Relative to Torso (normalized)')
            ax.set_title('Stroke Analysis - Hand Movement Relative to Torso')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text annotation for stroke rate
            try:
                summary = self.stroke_analyzer.get_stroke_summary()
                if summary['total_strokes'] > 0:
                    ax.text(0.02, 0.98, 
                           f"Stroke Rate: {summary['stroke_rate_per_minute']:.1f} SPM\n"
                           f"Total Cycles: {summary['total_cycles']}\n"
                           f"Total Strokes: {summary['total_strokes']}",
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except Exception as e:
                print(f"Error adding summary text: {e}")
            
            self.stroke_figure.tight_layout()
            self.stroke_canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_stroke_analysis: {e}")
            # Show a simple error plot
            try:
                self.stroke_figure.clear()
                ax = self.stroke_figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Error generating plot:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                self.stroke_canvas.draw()
            except:
                pass  # If even the error plot fails, just continue
    
    def toggle_keypoints(self, checked):
        """Toggle keypoint display"""
        self.show_keypoints = checked
        # Redraw current frame with or without keypoints
        if self.original_frame is not None:
            self.process_and_display_frame(self.original_frame)
    
    def load_video(self):
        """Load a video file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Video File", 
                "", 
                "Video Files (*.mp4 *.avi *.mov *.webm *.mkv);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Release previous video if any
            if self.video_capture is not None:
                self.video_capture.release()
            
            # Load new video
            self.video_capture = cv2.VideoCapture(file_path)
            
            if not self.video_capture.isOpened():
                self.show_error_message("Video Error", f"Could not open video file:\n{file_path}")
                return
            
            try:
                # Get video properties
                self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if self.fps <= 0 or self.total_frames <= 0:
                    raise ValueError("Invalid video properties")
                
                # Update slider range
                self.progress_slider.setMaximum(self.total_frames - 1)
                
                # Enable controls
                self.play_pause_button.setEnabled(True)
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                
                # Reset cache and inference status
                self.keypoints_cache = {}
                self.inference_complete = False
                self.keypoints_checkbox.setEnabled(False)
                self.processing_status_label.setText("")
                self.progress_bar.setVisible(False)
                self.analyze_strokes_button.setEnabled(False)
                self.stroke_analyzer.clear_data()
                self.stroke_info_label.setText("No stroke analysis")
                if hasattr(self, 'smooth_checkbox'):
                    self.smooth_checkbox.setChecked(False)
                    self.smooth_checkbox.setEnabled(False)
                if hasattr(self, 'p_spin'):
                    self.blend_p = 0.5
                    self.p_spin.setValue(self.blend_p)
                    self.p_spin.setEnabled(False)
                
                # Load and display first frame
                self.current_frame_number = 0
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                if ret:
                    self.original_frame = frame.copy()
                    self.process_and_display_frame(frame)
                    self.update_frame_info()
                else:
                    raise ValueError("Could not read first frame")
                
                # Update timer interval based on FPS
                self.timer.setInterval(int(1000 / self.fps))
                
                print(f"Loaded video: {os.path.basename(file_path)}")
                print(f"FPS: {self.fps}, Total frames: {self.total_frames}")
                
            except Exception as e:
                self.show_error_message("Video Properties Error", "Failed to read video properties", e)
                if self.video_capture is not None:
                    self.video_capture.release()
                    self.video_capture = None
                    
        except Exception as e:
            self.show_error_message("Load Error", "An unexpected error occurred while loading the video", e)
    
    def process_and_display_frame(self, frame):
        """Process frame with model and display it"""
        try:
            if frame is not None:
                display_frame = frame.copy()
                base_keypoints = None

                need_keypoints = self.model_inference.model_loaded and (
                    self.show_keypoints or (hasattr(self, 'smooth_checkbox') and self.smooth_checkbox.isChecked())
                )

                if need_keypoints:
                    # Use cached keypoints if available, otherwise run inference
                    if self.inference_complete and self.current_frame_number in self.keypoints_cache:
                        base_keypoints = self.keypoints_cache[self.current_frame_number]
                    else:
                        try:
                            base_keypoints = self.model_inference.predict(frame)
                        except Exception as e:
                            print(f"Error during model inference: {e}")

                # Optionally draw base keypoints
                if self.show_keypoints and base_keypoints is not None:
                    try:
                        display_frame = self.draw_keypoints(display_frame, base_keypoints)
                    except Exception as e:
                        print(f"Error drawing keypoints: {e}")

                # Optionally overlay smoothed/blended pose (only after inference complete)
                if (self.inference_complete and hasattr(self, 'smooth_checkbox') and self.smooth_checkbox.isChecked()) and base_keypoints is not None:
                    try:
                        blended = compute_blended_keypoints(base_keypoints, frame.shape, self.anchors_json_dir, self.blend_p)
                        if blended is not None:
                            display_frame = self.draw_keypoints_overlay(display_frame, blended, color=(0, 200, 255))
                    except Exception as e:
                        print(f"Error computing/drawing smoothed pose: {e}")
                
                self.current_frame = display_frame
                self.display_frame(display_frame)
        except Exception as e:
            print(f"Error in process_and_display_frame: {e}")
            # Continue with original frame if processing fails
            if frame is not None:
                self.current_frame = frame
                self.display_frame(frame)
    
    def display_frame(self, frame):
        """Convert CV2 frame to QPixmap and display it in the video label"""
        if frame is None:
            return
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error displaying frame: {e}")

    def update_frame_info(self):
        """Update frame counter label"""
        try:
            self.frame_info_label.setText(f"Frame: {self.current_frame_number} / {self.total_frames}")
        except Exception:
            pass

    def draw_keypoints(self, frame, keypoints):
        """Draw keypoints and skeleton on the frame"""
        if keypoints is None or len(keypoints) == 0:
            return frame
        
        # Define skeleton connections (COCO format)
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Legs
            (5, 11), (6, 12)  # Torso
        ]
        
        # Draw keypoints
        for i, (x, y, confidence) in enumerate(keypoints):
            if confidence > 0.5:  # Only draw confident keypoints
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)  # Green for high confidence, Yellow for medium
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                cv2.putText(frame, str(i), (int(x) + 5, int(y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw skeleton connections
        for connection in skeleton_connections:
            if len(keypoints) > max(connection):
                pt1_idx, pt2_idx = connection
                if (keypoints[pt1_idx][2] > 0.5 and keypoints[pt2_idx][2] > 0.5):
                    pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                    pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
        
        return frame
    
    def draw_keypoints_overlay(self, frame, keypoints, color=(0, 200, 255)):
        """Draw keypoints in a single color overlay (circles+lines)."""
        if keypoints is None or len(keypoints) == 0:
            return frame
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            (5, 11), (6, 12)
        ]
        for i, (x, y, confidence) in enumerate(keypoints):
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        for a, b in skeleton_connections:
            if a < len(keypoints) and b < len(keypoints):
                pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
                pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(frame, pt1, pt2, color, 1)
        return frame

    def _latest_angles_json(self) -> str | None:
        """Pick the most recently modified .json from self.anchors_json_dir."""
        try:
            if not os.path.isdir(self.anchors_json_dir):
                return None
            files = [
                os.path.join(self.anchors_json_dir, f)
                for f in os.listdir(self.anchors_json_dir)
                if f.lower().endswith('.json')
            ]
            if not files:
                return None
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files[0]
        except Exception:
            return None

    def _list_angles_jsons(self):
        """Return a list of all .json files under self.anchors_json_dir."""
        try:
            if not os.path.isdir(self.anchors_json_dir):
                return []
            return [
                os.path.join(self.anchors_json_dir, f)
                for f in os.listdir(self.anchors_json_dir)
                if f.lower().endswith('.json')
            ]
        except Exception:
            return []

    def find_anchor(self):
        """Find the closest anchor pose across all angles JSONs and print results."""
        try:
            if find_nearest_anchors is None:
                self.show_error_message("Import Error", "Nearest search module not available.")
                return
            if self.original_frame is None:
                self.show_warning_message("Frame Error", "No frame available. Load a video first.")
                return
            if not self.model_inference.model_loaded:
                self.show_warning_message("Model Error", "Pose estimation model is not loaded.")
                return

            # Collect all angle JSON files
            json_paths = self._list_angles_jsons()
            if not json_paths:
                self.show_warning_message("Anchors JSON", "No angles JSON files found in configured directory.")
                return

            # Get keypoints for current frame (use cache if available)
            keypoints = None
            if self.inference_complete and self.current_frame_number in self.keypoints_cache:
                keypoints = self.keypoints_cache[self.current_frame_number]
            if keypoints is None:
                keypoints = self.model_inference.predict(self.original_frame)
            if keypoints is None or len(keypoints) < 17:
                self.show_warning_message("Inference Error", "No keypoints detected in the current frame.")
                return

            # Build YOLO-style label content (normalized coordinates)
            frame = self.original_frame
            h, w = frame.shape[:2]
            xs = [kp[0] for kp in keypoints if kp[2] > 0]
            ys = [kp[1] for kp in keypoints if kp[2] > 0]
            if not xs or not ys:
                self.show_warning_message("Keypoints Error", "No confident keypoints to form a label.")
                return
            x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
            y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
            # Small padding
            pad = 0.05
            bw = max(1.0, (x_max - x_min) * (1 + pad))
            bh = max(1.0, (y_max - y_min) * (1 + pad))
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            # Normalize
            x_center = cx / w
            y_center = cy / h
            width = min(1.0, bw / w)
            height = min(1.0, bh / h)

            # Assemble keypoints normalized with visibility mapped to {0,1,2}
            def vis_from_conf(c):
                return 2 if c is not None and c > 0.5 else 1
            parts = [
                0, x_center, y_center, width, height
            ]
            for i in range(17):
                x, y, conf = keypoints[i]
                parts.extend([x / w, y / h, vis_from_conf(conf)])

            label_line = " ".join(str(v) for v in parts)

            # Write temp label file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write(label_line + "\n")
                temp_label_path = tmp.name

            all_candidates = []
            try:
                for jp in json_paths:
                    try:
                        res = find_nearest_anchors(temp_label_path, jp, None, topk=3)
                    except Exception as e:
                        print(f"[anchor] Error searching {os.path.basename(jp)}: {e}")
                        continue
                    if not res or not res[0]['nearest']:
                        continue
                    for item in res[0]['nearest']:
                        cand = dict(item)
                        cand['source_json'] = os.path.basename(jp)
                        all_candidates.append(cand)
            finally:
                try:
                    os.remove(temp_label_path)
                except Exception:
                    pass

            if not all_candidates:
                print("[anchor] No matching anchor found across all JSONs.")
                return

            # Sort globally and take top-3
            all_candidates.sort(key=lambda x: x.get('distance', float('inf')))
            top = all_candidates[:3]
            print("[anchor] Nearest anchors across all JSONs (top-3):")
            for rank, item in enumerate(top, start=1):
                print(f"  {rank}. frame={item['frame']} file={item['frame_file']} idx={item['anchor_pose_index']} dist={item['distance']:.4f} src={item['source_json']}")

        except Exception as e:
            self.show_error_message("Anchor Search Error", "Failed to find closest anchor", e)

    def _wrapped_angle_diff(self, a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _rotate_vec(self, vx: float, vy: float, delta: float) -> tuple[float, float]:
        c, s = math.cos(delta), math.sin(delta)
        return vx * c - vy * s, vx * s + vy * c

    def _compute_blended_keypoints(self, base_keypoints, frame_shape):
        """Blend model angles with nearest anchor angles and return new keypoints in pixel coords.
        Reconstruct via decoder using blended angles and align to image orientation.
        """
        if find_nearest_anchors is None:
            return None
        h, w = frame_shape[:2]
        # Build a temporary YOLO label for nearest search
        xs = [kp[0] for kp in base_keypoints if kp[2] > 0]
        ys = [kp[1] for kp in base_keypoints if kp[2] > 0]
        if not xs or not ys:
            return None
        x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
        y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
        pad = 0.05
        bw = max(1.0, (x_max - x_min) * (1 + pad))
        bh = max(1.0, (y_max - y_min) * (1 + pad))
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        x_center = cx / w
        y_center = cy / h
        width = min(1.0, bw / w)
        height = min(1.0, bh / h)
        def vis_from_conf(c):
            return 2 if c is not None and c > 0.5 else 1
        parts = [0, x_center, y_center, width, height]
        for i in range(17):
            x, y, conf = base_keypoints[i]
            parts.extend([x / w, y / h, vis_from_conf(conf)])
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(" ".join(str(v) for v in parts) + "\n")
            temp_label_path = tmp.name

        # Find best anchor across all JSONs
        json_paths = self._list_angles_jsons()
        best_item = None
        try:
            for jp in json_paths:
                try:
                    res = find_nearest_anchors(temp_label_path, jp, None, topk=1)
                except Exception:
                    continue
                if not res or not res[0]['nearest']:
                    continue
                cand = res[0]['nearest'][0]
                if best_item is None or cand['distance'] < best_item['distance']:
                    best_item = cand
        finally:
            try:
                os.remove(temp_label_path)
            except Exception:
                pass
        if best_item is None:
            return None
        anchor_angles = best_item.get('angles', {}) or {}

        # Compute prediction angles with anchorClasses Pose
        try:
            from cycle.initialization.anchorClasses import Pose, Keypoint
        except Exception:
            try:
                sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cycle', 'initialization'))
                from anchorClasses import Pose, Keypoint  # type: ignore
            except Exception:
                Pose = None
        pred_angles = {}
        canon_pts = None
        if Pose is not None:
            pose = Pose()
            coco_names = [
                "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
                "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
            ]
            for i, name in enumerate(coco_names):
                x, y, conf = base_keypoints[i]
                pose.keypoints[name] = Keypoint(x=x, y=y, v=2 if conf and conf > 0.5 else 1)
            pose.centralize(); pose.rotate()
            pred_angles = pose.calculate_angles() or {}
            # Collect canonicalized coordinates for length estimation
            canon_pts = []
            for name in coco_names:
                k = pose.keypoints[name]
                canon_pts.append((k.x, k.y))
        # Estimate lengths from canonicalized current frame if possible
        lengths = decode_estimate_lengths(canon_pts) if canon_pts is not None else None

        # Blend angles (wrap-aware), include torso_tilt and limb angles
        blended = {}
        p = float(self.blend_p)
        keys_to_blend = [
            'torso_tilt',
            'L_shoulder_to_elbow','R_shoulder_to_elbow',
            'L_elbow_flexion','R_elbow_flexion',
            'L_hip_flexion','R_hip_flexion',
            'L_knee_flexion','R_knee_flexion'
        ]
        for k in keys_to_blend:
            pv = pred_angles.get(k)
            av = anchor_angles.get(k)
            if pv is None and av is None:
                continue
            if pv is None:
                blended[k] = float(av)
                continue
            if av is None:
                blended[k] = float(pv)
                continue
            diff = self._wrapped_angle_diff(float(av), float(pv))
            blended[k] = float(pv) + (1.0 - p) * diff

        # Reconstruct in canonical space
        pts = decode_reconstruct(blended, lengths=lengths)

        # Align to image orientation using torso_tilt delta and reflect x-axis
        comp_pix = decode_compute_angles(base_keypoints)
        tt_pix = comp_pix.get('torso_tilt', 0.0)
        tt_blend = float(blended.get('torso_tilt', 0.0))
        phi = self._wrapped_angle_diff(float(tt_pix), tt_blend)
        pts = decode_rotate(pts, phi)
        pts = decode_mirror_x(pts)

        # Translate to pixel hip center
        hip_cx = (base_keypoints[11][0] + base_keypoints[12][0]) / 2.0
        hip_cy = (base_keypoints[11][1] + base_keypoints[12][1]) / 2.0
        pts = [(x + hip_cx, y + hip_cy) for (x,y) in pts]

        # Convert to keypoints with confidence
        out = []
        for x, y in pts:
            out.append((float(x), float(y), 1.0))
        return out

    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        """Start video playback"""
        if self.video_capture and self.video_capture.isOpened():
            self.is_playing = True
            self.play_pause_button.setText("Pause")
            self.timer.start()

    def pause_video(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_pause_button.setText("Play")
        self.timer.stop()

    def next_frame(self):
        """Advance to next frame"""
        try:
            if self.video_capture and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret:
                    self.original_frame = frame.copy()
                    self.current_frame_number += 1
                    self.process_and_display_frame(frame)
                    self.update_frame_info()
                    self.progress_slider.setValue(self.current_frame_number)
                else:
                    # End of video reached
                    self.pause_video()
        except Exception as e:
            print(f"Error advancing to next frame: {e}")
            self.pause_video()

    def previous_frame(self):
        """Go to previous frame"""
        try:
            if self.video_capture and self.video_capture.isOpened():
                # Decrease frame number, ensuring it doesn't go below 0
                self.current_frame_number = max(0, self.current_frame_number - 1)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                ret, frame = self.video_capture.read()
                if ret:
                    self.original_frame = frame.copy()
                    self.process_and_display_frame(frame)
                    self.update_frame_info()
                    self.progress_slider.setValue(self.current_frame_number)
        except Exception as e:
            print(f"Error going to previous frame: {e}")
            self.pause_video()

    def slider_pressed(self):
        """Handle slider press - pause video if playing"""
        try:
            if self.is_playing:
                self.timer.stop()
        except Exception as e:
            print(f"Error pausing for slider: {e}")

    def slider_released(self):
        """Handle slider release - seek to new position"""
        try:
            if self.video_capture and self.video_capture.isOpened():
                new_frame = self.progress_slider.value()
                self.current_frame_number = new_frame
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                ret, frame = self.video_capture.read()
                if ret:
                    self.original_frame = frame.copy()
                    self.process_and_display_frame(frame)
                    self.update_frame_info()
                    # Reset to current frame for next read
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                
                # Resume playback if it was playing
                if self.is_playing:
                    self.timer.start()
        except Exception as e:
            print(f"Error seeking to frame: {e}")

    def set_blend_p(self, value: float):
        """Update smoothing blend parameter p and refresh overlay if active."""
        try:
            self.blend_p = float(value)
            if hasattr(self, 'smooth_checkbox') and self.smooth_checkbox.isChecked() and self.original_frame is not None and self.inference_complete:
                self.process_and_display_frame(self.original_frame)
        except Exception as e:
            print(f"Error updating blend p: {e}")