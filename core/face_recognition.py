# face_recognition.py
import cv2
import os
import glob
import re
from utils.config import FACES_DIR

def is_face_already_exists(frame, bbox, model_name='Facenet', distance_threshold=0.6):
    from deepface import DeepFace
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    person_crop = frame[y1:y2, x1:x2].copy()
    if person_crop.size == 0:
        return None
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda rect: rect[2]*rect[3])
    fx, fy, fw, fh = face
    face_crop = person_crop[fy:fy+fh, fx:fx+fw].copy()
    temp_face_path = os.path.join(FACES_DIR, 'temp_face_check.png')
    cv2.imwrite(temp_face_path, face_crop)
    image_files = glob.glob(os.path.join(FACES_DIR, 'face_*.png'))
    if not image_files:
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)
        return None
    for file in image_files:
        try:
            result = DeepFace.verify(img1_path=temp_face_path, img2_path=file, model_name=model_name, enforce_detection=False)
            if result['verified'] and result['distance'] < distance_threshold:
                match = re.match(r'face_(.+)\.png', os.path.basename(file))
                if match:
                    name = match.group(1).replace('_', ' ').title()
                else:
                    name = os.path.splitext(os.path.basename(file))[0]
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                return name
        except Exception as e:
            continue
    if os.path.exists(temp_face_path):
        os.remove(temp_face_path)
    return None
