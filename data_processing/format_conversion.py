import os

def calculate_bounding_box(keypoint_coords, padding=80):
    """
    Calculate bounding box from visible keypoint coordinates only.
    
    Args:
        keypoint_coords (dict): Dictionary with normalized coordinates and visibility
                               {'keypoint_name': {'x': float, 'y': float, 'v': int}}
        padding (float): Padding to add around the bounding box in normalized coordinates (default: 5)
    """
    if not keypoint_coords:
        return 
    
    x_coords = []
    y_coords = []
    
    for _, coords in keypoint_coords.items():
        if 'x' in coords and 'y' in coords and 'v' in coords:
            # Only use visible keypoints (v=2.0) for bounding box calculation
            if coords['v'] == 2.0:
                x_coords.append(coords['x'])
                y_coords.append(coords['y'])
    
    if not x_coords or not y_coords:
        return 
    
    # Convert padding from pixels to normalized coordinates (assuming 1920x1080 as reference)
    padding_norm = padding / 1920  # Normalize padding
    
    x_min = max(0, min(x_coords) - padding_norm)
    x_max = min(1, max(x_coords) + padding_norm)
    y_min = max(0, min(y_coords) - padding_norm)
    y_max = min(1, max(y_coords) + padding_norm)

    width = x_max - x_min
    height = y_max - y_min
    
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)


def convert_to_yolo(coco_annotation_file, img_width=1920, img_height=1080):
    """
    Convert COCO format annotation file to YOLO pose format with normalized coordinates.
    
    Args:
        coco_annotation_file (str): Path to the COCO annotation file
        img_width (int): Width of the image for normalization (default: 1920)
        img_height (int): Height of the image for normalization (default: 1080)
    """
    coco_to_yolo_mapping = {
        'Nose': 0,
        'LEye': 1, 'Eye_L': 1, 
        'REye': 2, 'Eye_R': 2,
        'LEar': 3, 'Ear_L': 3,
        'REar': 4, 'Ear_R': 4,
        'LShoulder': 5, 'L Clavicle': 5,  
        'RShoulder': 6, 'R Clavicle': 6,
        'LElbow': 7, 'L Forearm': 7,  
        'RElbow': 8, 'R Forearm': 8,
        'LWrist': 9, 'L Hand': 9,  
        'RWrist': 10, 'R Hand': 10,
        'LHip': 11, 'MidHip': 11,  
        'RHip': 12,
        'LKnee': 13, 'L Calf': 13,  
        'RKnee': 14, 'R Calf': 14,
        'LAnkle': 15, 'L Foot': 15,  
        'RAnkle': 16, 'R Foot': 16
    }
    
    with open(coco_annotation_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print(f"Warning: File {coco_annotation_file} has insufficient data")
        print(lines)
        return
    
    header = lines[0].strip()
    keypoint_names = header.split(';')
    
    keypoint_names = [name for name in keypoint_names if name.strip() != '' and not name.endswith('.z')]
    
    col_to_keypoint = {}
    for col_idx, name in enumerate(keypoint_names):
        if '.x' in name:
            keypoint_name = name.replace('.x', '')
            col_to_keypoint[col_idx] = {'name': keypoint_name, 'coord': 'x'}
        elif '.y' in name:
            keypoint_name = name.replace('.y', '')
            col_to_keypoint[col_idx] = {'name': keypoint_name, 'coord': 'y'}

    converted_lines = []
    
    for line_idx, line in enumerate(lines[1:], 1):
        try:
            line = line.strip().replace(',', '.')
            original_values = line.split(';')
            
            filtered_values = []
            original_header_names = header.split(';')
            
            for value, header_name in zip(original_values, original_header_names):
                if header_name.strip() != '' and not header_name.endswith('.z'):
                    filtered_values.append(value)
            
            values = filtered_values
            
            if len(values) != len(keypoint_names):
                print(f"Warning: Line {line_idx} has {len(values)} values but expected {len(keypoint_names)}")
                continue
            
            yolo_keypoints = ['0.0'] * 51  # 17 keypoints * 3 coordinates (x, y, v) = 51
            
            # Track if any keypoints are visible
            has_visible_keypoints = False
            
            keypoint_coords = {}
            for col_idx, value in enumerate(values):
                if col_idx in col_to_keypoint:
                    kp_info = col_to_keypoint[col_idx]
                    kp_name = kp_info['name']
                    coord_type = kp_info['coord']
                    
                    if kp_name not in keypoint_coords:
                        keypoint_coords[kp_name] = {}
                    
                    keypoint_coords[kp_name][coord_type] = float(value)
            
            # First pass: normalize keypoints and add visibility
            normalized_keypoint_coords = {}
            for kp_name, coords in keypoint_coords.items():
                yolo_idx = None
                for coco_name, yolo_index in coco_to_yolo_mapping.items():
                    if kp_name == coco_name:
                        yolo_idx = yolo_index
                        break
                
                if yolo_idx is not None and 'x' in coords and 'y' in coords:
                    # Flip y-coordinate: y becomes height - y_val
                    flipped_y = img_height - coords['y']
                    
                    # Normalize coordinates to [0,1] range and round to 6 decimal places
                    normalized_x = round(coords['x'] / img_width, 6)
                    normalized_y = round(flipped_y / img_height, 6)
                    
                    # Check visibility: v=2.0 if inside image bounds, v=1.0 if outside
                    if (0 <= coords['x'] <= img_width) and (0 <= flipped_y <= img_height):
                        visibility = 2.0
                        has_visible_keypoints = True  # Set flag when we find a visible keypoint
                    else:
                        visibility = 1.0
                    
                    # Store normalized coordinates with visibility for bounding box calculation
                    normalized_keypoint_coords[kp_name] = {
                        'x': normalized_x,
                        'y': normalized_y,
                        'v': visibility
                    }
                    
                    # YOLO format: x, y, v for each keypoint
                    yolo_keypoints[yolo_idx * 3] = str(normalized_x)
                    yolo_keypoints[yolo_idx * 3 + 1] = str(normalized_y)
                    yolo_keypoints[yolo_idx * 3 + 2] = f"{visibility:.6f}"
            
            # Only create annotation if there are visible keypoints
            if has_visible_keypoints:
                bbox = calculate_bounding_box(normalized_keypoint_coords)
                if bbox:
                    x_center, y_center, width, height = bbox
                    # Format: class_id x_center y_center width height keypoints...
                    bbox_line = f"0 {x_center} {y_center} {width} {height} {' '.join(yolo_keypoints)}"
                    converted_lines.append(bbox_line)
                else:
                    # Should not happen if has_visible_keypoints is True, but safety fallback
                    converted_lines.append("")
            else:
                # All joints are invisible, output empty string
                converted_lines.append("")
            
        except Exception as e:
            print(f"Error processing line {line_idx}: {e}")
            continue
    
    return converted_lines 