import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import deque
from enum import Enum

class StrokePhase(Enum):
    LEFT_SIDE = 1
    CROSSING = 2
    RIGHT_SIDE = 3

class StrokeRateAnalyzer:
    def __init__(self):
        # Data storage
        self.frame_data = []  # Store (frame, keypoints) for each frame
        self.smoothed_positions = {}  # Smoothed keypoint positions
        self.torso_vectors = []  # Dynamic torso reference vectors
        self.wrist_relative_positions = {'left': [], 'right': []}  # Relative to torso line
        self.stroke_cycles = []  # Detected complete stroke cycles
        self.fps = 30
        
        # 12-body-keypoint indices (face KPs removed; body re-indexed 0–11)
        self.LEFT_WRIST = 4
        self.RIGHT_WRIST = 5
        self.LEFT_SHOULDER = 0
        self.RIGHT_SHOULDER = 1
        self.LEFT_HIP = 6
        self.RIGHT_HIP = 7
        
        # Stroke detection parameters
        self.confidence_threshold = 0.5
        self.smoothing_window = 7  # For Savitzky-Golay filter
        self.crossing_threshold = 0.02  # Minimum relative distance to count as crossing
        self.min_cycle_frames = 30  # Minimum frames between stroke cycles
        
        # State machine for stroke detection
        self.current_phase = StrokePhase.CROSSING
        self.last_cycle_frame = -1
        self.phase_start_frame = 0
        
    def set_fps(self, fps):
        """Set the video FPS for time calculations"""
        self.fps = fps
    
    def clear_data(self):
        """Clear all stored data"""
        self.frame_data = []
        self.smoothed_positions = {}
        self.torso_vectors = []
        self.wrist_relative_positions = {'left': [], 'right': []}
        self.stroke_cycles = []
        self.current_phase = StrokePhase.CROSSING
        self.last_cycle_frame = -1
        self.phase_start_frame = 0
    
    def add_frame_data(self, frame_number, keypoints):
        """Add keypoint data for a frame"""
        if keypoints is None or len(keypoints) < 12:
            return
        
        # Store raw frame data
        self.frame_data.append((frame_number, keypoints))
    
    def _extract_keypoint(self, keypoints, index):
        """Extract keypoint with confidence check"""
        if keypoints[index][2] > self.confidence_threshold:
            return np.array([keypoints[index][0], keypoints[index][1]])
        return None
    
    def _calculate_torso_vector(self, keypoints):
        """Calculate dynamic torso reference vector from shoulders or hips"""
        # Try shoulders first (better for swimming)
        left_shoulder = self._extract_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._extract_keypoint(keypoints, self.RIGHT_SHOULDER)
        
        if left_shoulder is not None and right_shoulder is not None:
            # Use shoulder line as torso reference
            shoulder_center = (left_shoulder + right_shoulder) / 2
            shoulder_vector = right_shoulder - left_shoulder
            # Perpendicular vector (torso direction)
            torso_vector = np.array([-shoulder_vector[1], shoulder_vector[0]])
            torso_vector = torso_vector / np.linalg.norm(torso_vector)
            return shoulder_center, torso_vector
        
        # Fallback to hips
        left_hip = self._extract_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._extract_keypoint(keypoints, self.RIGHT_HIP)
        
        if left_hip is not None and right_hip is not None:
            hip_center = (left_hip + right_hip) / 2
            hip_vector = right_hip - left_hip
            torso_vector = np.array([-hip_vector[1], hip_vector[0]])
            torso_vector = torso_vector / np.linalg.norm(torso_vector)
            return hip_center, torso_vector
        
        return None, None
    
    def _smooth_positions(self):
        """Apply smoothing to keypoint positions to reduce jitter"""
        if len(self.frame_data) < self.smoothing_window:
            return
        
        # Extract time series data for each keypoint
        frames = [data[0] for data in self.frame_data]
        
        # Initialize storage
        keypoint_series = {
            'left_wrist': [],
            'right_wrist': [],
            'torso_centers': [],
            'torso_vectors': []
        }
        
        # Extract positions for each frame
        for frame, keypoints in self.frame_data:
            left_wrist = self._extract_keypoint(keypoints, self.LEFT_WRIST)
            right_wrist = self._extract_keypoint(keypoints, self.RIGHT_WRIST)
            torso_center, torso_vector = self._calculate_torso_vector(keypoints)
            
            keypoint_series['left_wrist'].append(left_wrist)
            keypoint_series['right_wrist'].append(right_wrist)
            keypoint_series['torso_centers'].append(torso_center)
            keypoint_series['torso_vectors'].append(torso_vector)
        
        # Apply smoothing where data is available
        self.smoothed_positions = {'frames': frames}
        
        for key in ['left_wrist', 'right_wrist', 'torso_centers', 'torso_vectors']:
            valid_indices = [i for i, pos in enumerate(keypoint_series[key]) if pos is not None]
            
            if len(valid_indices) > self.smoothing_window:
                valid_positions = np.array([keypoint_series[key][i] for i in valid_indices])
                
                # Apply Savitzky-Golay filter
                if valid_positions.ndim == 2:
                    smoothed_x = savgol_filter(valid_positions[:, 0], self.smoothing_window, 3)
                    smoothed_y = savgol_filter(valid_positions[:, 1], self.smoothing_window, 3)
                    smoothed_positions = np.column_stack((smoothed_x, smoothed_y))
                else:
                    smoothed_positions = valid_positions
                
                # Map back to full timeline
                full_smoothed = [None] * len(frames)
                for idx, valid_idx in enumerate(valid_indices):
                    full_smoothed[valid_idx] = smoothed_positions[idx]
                
                self.smoothed_positions[key] = full_smoothed
            else:
                self.smoothed_positions[key] = keypoint_series[key]
    
    def _calculate_relative_positions(self):
        """Calculate wrist positions relative to torso line"""
        if not self.smoothed_positions:
            return
        
        frames = self.smoothed_positions['frames']
        
        for hand in ['left', 'right']:
            wrist_key = f'{hand}_wrist'
            self.wrist_relative_positions[hand] = []
            
            for i, frame in enumerate(frames):
                wrist_pos = self.smoothed_positions[wrist_key][i]
                torso_center = self.smoothed_positions['torso_centers'][i]
                torso_vector = self.smoothed_positions['torso_vectors'][i]
                
                if wrist_pos is not None and torso_center is not None and torso_vector is not None:
                    # Vector from torso center to wrist
                    wrist_vector = wrist_pos - torso_center
                    
                    # Project onto torso direction (positive = towards head, negative = towards feet)
                    relative_position = np.dot(wrist_vector, torso_vector)
                    
                    # Also calculate lateral distance (for crossing detection)
                    lateral_vector = np.array([torso_vector[1], -torso_vector[0]])
                    lateral_position = np.dot(wrist_vector, lateral_vector)
                    
                    self.wrist_relative_positions[hand].append({
                        'frame': frame,
                        'longitudinal': relative_position,
                        'lateral': lateral_position,
                        'visible': True
                    })
                else:
                    self.wrist_relative_positions[hand].append({
                        'frame': frame,
                        'longitudinal': 0,
                        'lateral': 0,
                        'visible': False
                    })
    
    def _detect_stroke_cycles(self):
        """Detect complete stroke cycles using state machine"""
        self.stroke_cycles = []
        
        # Get combined data for both wrists
        left_data = self.wrist_relative_positions['left']
        right_data = self.wrist_relative_positions['right']
        
        if not left_data or not right_data:
            return
        
        # State machine variables
        self.current_phase = StrokePhase.CROSSING
        self.last_cycle_frame = -1
        cycle_start_frame = 0
        
        for i in range(len(left_data)):
            frame = left_data[i]['frame']
            
            # Skip if too close to last cycle
            if frame - self.last_cycle_frame < self.min_cycle_frames:
                continue
            
            left_lateral = left_data[i]['lateral']
            right_lateral = right_data[i]['lateral']
            left_visible = left_data[i]['visible']
            right_visible = right_data[i]['visible']
            
            # Skip if key points are not visible
            if not (left_visible and right_visible):
                continue
            
            # State machine logic
            if self.current_phase == StrokePhase.CROSSING:
                # Look for clear separation to either side
                if left_lateral < -self.crossing_threshold and right_lateral > self.crossing_threshold:
                    self.current_phase = StrokePhase.LEFT_SIDE
                    self.phase_start_frame = frame
                elif right_lateral < -self.crossing_threshold and left_lateral > self.crossing_threshold:
                    self.current_phase = StrokePhase.RIGHT_SIDE
                    self.phase_start_frame = frame
            
            elif self.current_phase == StrokePhase.LEFT_SIDE:
                # Look for crossing back to right side
                if right_lateral < -self.crossing_threshold and left_lateral > self.crossing_threshold:
                    # Complete stroke cycle detected
                    cycle_duration = frame - cycle_start_frame
                    if cycle_duration > self.min_cycle_frames:
                        self.stroke_cycles.append({
                            'start_frame': cycle_start_frame,
                            'end_frame': frame,
                            'duration_frames': cycle_duration,
                            'duration_seconds': cycle_duration / self.fps
                        })
                        self.last_cycle_frame = frame
                    
                    self.current_phase = StrokePhase.RIGHT_SIDE
                    self.phase_start_frame = frame
                    cycle_start_frame = frame
            
            elif self.current_phase == StrokePhase.RIGHT_SIDE:
                # Look for crossing back to left side
                if left_lateral < -self.crossing_threshold and right_lateral > self.crossing_threshold:
                    # Complete stroke cycle detected
                    cycle_duration = frame - cycle_start_frame
                    if cycle_duration > self.min_cycle_frames:
                        self.stroke_cycles.append({
                            'start_frame': cycle_start_frame,
                            'end_frame': frame,
                            'duration_frames': cycle_duration,
                            'duration_seconds': cycle_duration / self.fps
                        })
                        self.last_cycle_frame = frame
                    
                    self.current_phase = StrokePhase.LEFT_SIDE
                    self.phase_start_frame = frame
                    cycle_start_frame = frame
    
    def detect_stroke_events(self, min_distance_frames=15):
        """Main method to process all frames and detect stroke cycles"""
        if len(self.frame_data) < 10:
            return []
        
        # Step 1: Apply smoothing to reduce jitter
        self._smooth_positions()
        
        # Step 2: Calculate wrist positions relative to torso line
        self._calculate_relative_positions()
        
        # Step 3: Detect complete stroke cycles using state machine
        self._detect_stroke_cycles()
        
        return self.stroke_cycles
    
    def calculate_stroke_rate(self):
        """Calculate stroke rate in strokes per minute based on detected cycles"""
        if len(self.stroke_cycles) < 2:
            return 0, []
        
        # Calculate cycle durations
        cycle_durations = [cycle['duration_seconds'] for cycle in self.stroke_cycles]
        
        if not cycle_durations:
            return 0, []
        
        # Calculate average cycle duration
        avg_cycle_duration = np.mean(cycle_durations)
        
        # Stroke rate = cycles per minute (each cycle = 2 strokes in freestyle)
        cycles_per_minute = 60 / avg_cycle_duration
        strokes_per_minute = cycles_per_minute * 2  # Each cycle has 2 strokes
        
        return strokes_per_minute, cycle_durations
    
    def get_plot_data(self):
        """Get data for plotting wrist positions relative to torso"""
        if not self.wrist_relative_positions['left'] and not self.wrist_relative_positions['right']:
            return [], [], [], []
        
        # Extract lateral positions (crossing data)
        left_frames = []
        left_lateral = []
        right_frames = []
        right_lateral = []
        
        for data in self.wrist_relative_positions['left']:
            if data['visible']:
                left_frames.append(data['frame'])
                left_lateral.append(data['lateral'])
        
        for data in self.wrist_relative_positions['right']:
            if data['visible']:
                right_frames.append(data['frame'])
                right_lateral.append(data['lateral'])
        
        return left_frames, left_lateral, right_frames, right_lateral
    
    def get_stroke_summary(self):
        """Get a summary of stroke analysis"""
        stroke_rate, cycle_durations = self.calculate_stroke_rate()
        
        total_time = 0
        if self.frame_data:
            start_frame = self.frame_data[0][0]
            end_frame = self.frame_data[-1][0]
            total_time = (end_frame - start_frame) / self.fps
        
        summary = {
            'total_cycles': len(self.stroke_cycles),
            'total_strokes': len(self.stroke_cycles) * 2,  # 2 strokes per cycle
            'stroke_rate_per_minute': stroke_rate,
            'avg_cycle_duration': np.mean(cycle_durations) if cycle_durations else 0,
            'stroke_cycles': self.stroke_cycles,
            'analysis_duration': total_time
        }
        
        return summary