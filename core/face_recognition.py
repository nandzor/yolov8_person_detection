# face_recognition.py
import cv2
import os
import glob
import re
from utils.config import FACES_DIR

def is_face_already_exists(frame, bbox, model_name='Facenet', distance_threshold=0.6):
    from deepface import DeepFace
    import shutil
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    person_crop = frame[y1:y2, x1:x2].copy()
    if person_crop.size == 0:
        return None, None
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None, None
    face = max(faces, key=lambda rect: rect[2]*rect[3])
    fx, fy, fw, fh = face

    # Add padding for better visualization
    padding = 20
    fx_pad = fx - padding
    fy_pad = fy - padding
    fw_pad = fw + (2 * padding)
    fh_pad = fh + (2 * padding)

    # Create padded absolute bbox
    face_abs_bbox_padded = (
        max(0, x1 + fx_pad),
        max(0, y1 + fy_pad),
        min(w, x1 + fx_pad + fw_pad),
        min(h, y1 + fy_pad + fh_pad)
    )

    # Enforce a 1:1 aspect ratio for consistency
    x1_p, y1_p, x2_p, y2_p = face_abs_bbox_padded
    w_p = x2_p - x1_p
    h_p = y2_p - y1_p
    cx = x1_p + w_p // 2
    cy = y1_p + h_p // 2
    size = max(w_p, h_p)
    
    face_abs_bbox = (
        max(0, cx - size // 2),
        max(0, cy - size // 2),
        min(w, cx + size // 2),
        min(h, cy + size // 2)
    )

    face_crop = person_crop[fy:fy+fh, fx:fx+fw].copy()
    # Ensure faces_temp directory exists
    faces_temp_dir = os.path.join(FACES_DIR, '..', 'faces_temp')
    os.makedirs(faces_temp_dir, exist_ok=True)
    temp_face_path = os.path.join(faces_temp_dir, 'temp_face_check.png')
    # Only create temp_face_check.png if neither temp_face_check.png nor any temp_face_check_*.png exists
    existing_temp_files = glob.glob(os.path.join(faces_temp_dir, 'temp_face_check_*.png'))
    if not os.path.exists(temp_face_path) and not existing_temp_files:
        cv2.imwrite(temp_face_path, face_crop)
    # Use the available temp file for comparison
    temp_files_to_compare = [temp_face_path] if os.path.exists(temp_face_path) else existing_temp_files
    image_files = glob.glob(os.path.join(FACES_DIR, 'face_*.png'))
    if not image_files:
        for f in temp_files_to_compare:
            if os.path.exists(f):
                os.remove(f)
        return None, face_abs_bbox
    for file in image_files:
        try:
            for temp_file in temp_files_to_compare:
                result = DeepFace.verify(img1_path=temp_file, img2_path=file, model_name=model_name, enforce_detection=False)
                if result['verified'] and result['distance'] < distance_threshold:
                    match = re.match(r'face_(.+)\.png', os.path.basename(file))
                    if match:
                        name = match.group(1).replace('_', ' ').title()
                        # Rename temp_face_check.png to temp_face_check_{name}.png if not already exists
                        temp_face_named = os.path.join(faces_temp_dir, f'temp_face_check_{match.group(1)}.png')
                        if os.path.exists(temp_face_path) and not os.path.exists(temp_face_named):
                            shutil.move(temp_face_path, temp_face_named)
                        # Clean up other temp files except the named one
                        for f in temp_files_to_compare:
                            if f != temp_face_named and os.path.exists(f):
                                os.remove(f)
                    else:
                        name = os.path.splitext(os.path.basename(file))[0]
                    return name, face_abs_bbox
        except Exception as e:
            continue
    # Clean up temp files if no match
    for f in temp_files_to_compare:
        if os.path.exists(f):
            os.remove(f)
    return None, face_abs_bbox
