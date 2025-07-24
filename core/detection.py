# detection.py
# Modul utama untuk deteksi, tracking, dan face recognition

from utils.config import *
from utils.api import send_api_alert
from utils.drawing import draw_person_box
from core.face_recognition import is_face_already_exists
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from core.tracking import update_tracker, tracked_objects, pending_candidates, persons_alerted

def run_detection_and_alerting():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return

    print("Webcam berhasil dibuka. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame.")
            break

        results = model(frame, stream=True, classes=[0], verbose=False)
        person_detections = []
        for r in results:
            for box in r.boxes:
                if box.conf[0] > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_detections.append((x1, y1, x2, y2))
        update_tracker(person_detections, frame)

        # Visualisasi dan eksekusi untuk objek yang sudah terkonfirmasi (tracked_objects)
        # (Logika visualisasi dan face recognition dapat dipanggil dari utils.drawing)
        draw_person_box(frame, tracked_objects, pending_candidates)

        cv2.imshow('Sistem Deteksi Orang Baru (Tekan Q untuk Keluar)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam dilepaskan dan jendela ditutup.")
