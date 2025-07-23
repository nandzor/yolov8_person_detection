# -*- coding: utf-8 -*-
"""
Deskripsi:
Script ini mengimplementasikan alur kerja untuk mendeteksi orang baru yang memasuki
jangkauan kamera, mengirimkan peringatan melalui API, dan menyimpan gambar wajah
orang tersebut.
Versi ini telah dioptimasi untuk meningkatkan akurasi pelacakan dan mengurangi
kesalahan deteksi orang baru yang salah.

Alur Kerja yang Diimplementasikan:
1.  Inisialisasi Sistem: Memuat model YOLO, tracker, dan variabel status.
2.  Akuisisi Frame Video: Menangkap video dari webcam.
3.  Proses Deteksi Orang: Menggunakan YOLOv8 untuk mendeteksi orang.
4.  Proses Pelacakan dan Asosiasi ID: Memberikan ID unik untuk setiap orang
    dan melacaknya antar frame.
5.  Eksekusi untuk Orang Baru (dengan Konfirmasi): Memicu logika saat ID baru
    terdeteksi dan terkonfirmasi secara konsisten selama beberapa frame.
6.  Generasi Peringatan via API & Simpan Wajah: Memformat data, mengirimkannya
    ke endpoint API, dan menyimpan potongan gambar wajah.
7.  Pengulangan (Loop): Memproses video secara terus-menerus.
8.  Manajemen Status: Menghapus ID yang sudah tidak terdeteksi.

Dependensi:
- ultralytics
- opencv-python
- numpy
- requests
- scipy
"""

from ultralytics import YOLO
import cv2
import numpy as np
import requests
import json
from datetime import datetime
from scipy.spatial import distance as dist # Import eksplisit untuk kejelasan
import os # Ditambahkan untuk operasi direktori
import threading
import queue

# Inisialisasi Haar Cascade untuk deteksi wajah
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# --- LANGKAH 1: INISIALISASI SISTEM ---

# URL Endpoint API untuk mengirim peringatan
# Menggunakan httpbin.org untuk tujuan demonstrasi. Ganti dengan URL API Anda.
API_ENDPOINT = "https://httpbin.org/post"
FACES_DIR = "faces" # Nama folder untuk menyimpan wajah

# --- Konfigurasi Pelacak (Tracker) yang Dioptimasi ---
# Tips Tuning:
# - MAX_DISAPPEARED_FRAMES: Naikkan jika orang sering hilang sementara (oklusi).
#   Turunkan jika ingin ID cepat dihapus saat orang keluar.
# - MAX_DISTANCE: Naikkan jika orang bergerak cepat. Turunkan jika orang
#   cenderung bergerak lambat dan berdekatan untuk menghindari salah asosiasi.
# - CONFIRMATION_FRAMES_THRESHOLD: Naikkan untuk keyakinan lebih tinggi sebelum
#   mengirim alert, mengurangi false positive. Turunkan untuk alert yang lebih cepat.

MAX_DISAPPEARED_FRAMES = 50
MAX_DISTANCE = 60
CONFIRMATION_FRAMES_THRESHOLD = 10 # Orang harus terdeteksi konsisten selama 10 frame



# Variabel Global untuk Manajemen Status
next_person_id = 1
tracked_objects = {}
pending_candidates = {}
next_temp_id = 1
persons_alerted = {}

# Queue dan thread untuk face recognition & crop agar tidak blocking
face_task_queue = queue.Queue()
face_result_queue = queue.Queue()
stop_face_worker = threading.Event()

def face_worker():
    while not stop_face_worker.is_set():
        try:
            task = face_task_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if task['type'] == 'recognition':
            frame, bbox, temp_id = task['frame'], task['bbox'], task['temp_id']
            matched_id = is_face_already_exists(frame, bbox)
            face_result_queue.put({'temp_id': temp_id, 'matched_id': matched_id})
        elif task['type'] == 'save_crop':
            frame, bbox, person_id = task['frame'], task['bbox'], task['person_id']
            save_face_crop(frame, bbox, person_id)
        face_task_queue.task_done()

# Start face worker thread
face_thread = threading.Thread(target=face_worker, daemon=True)
face_thread.start()


def is_face_already_exists(frame, bbox, model_name='Facenet', distance_threshold=0.3):
    """Bandingkan crop wajah dengan semua wajah di folder faces menggunakan DeepFace. Return True jika match."""
    import glob
    from deepface import DeepFace
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    person_crop = frame[y1:y2, x1:x2].copy()
    if person_crop.size == 0:
        return False
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return False
    face = max(faces, key=lambda rect: rect[2]*rect[3])
    fx, fy, fw, fh = face
    face_crop = person_crop[fy:fy+fh, fx:fx+fw].copy()
    # Simpan crop wajah sementara
    temp_face_path = os.path.join(FACES_DIR, 'temp_face_check.png')
    cv2.imwrite(temp_face_path, face_crop)
    # Cari semua file gambar di folder faces (one-to-many)
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_files.extend(glob.glob(os.path.join(FACES_DIR, ext)))
    # Hapus file temp dari list jika ada
    image_files = [f for f in image_files if not f.endswith('temp_face_check.png')]
    if not image_files:
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)
        return False
    # One-to-many face recognition dengan DeepFace
    for file in image_files:
        try:
            result = DeepFace.verify(img1_path=temp_face_path, img2_path=file, model_name=model_name, enforce_detection=False)
            if result['verified'] and result['distance'] < distance_threshold:
                # Ambil ID dari nama file (img_person<ID>.png)
                import re
                match = re.search(r'img_person(\d+)', os.path.basename(file))
                matched_id = int(match.group(1)) if match else None
                print(f"[FACE RECOG] Wajah match dengan {file} (distance={result['distance']:.3f}), gunakan ID: {matched_id}")
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                return matched_id
        except Exception as e:
            print(f"[FACE RECOG] Error membandingkan dengan {file}: {e}")
            continue
    if os.path.exists(temp_face_path):
        os.remove(temp_face_path)
    return None

def register_object(centroid, bbox, frame=None, matched_id=None):
    """Mendaftarkan objek baru yang sudah terkonfirmasi ke tracked_objects dengan ID unik, setelah face recognition."""
    global next_person_id
    # Face recognition: jika wajah sudah ada, update tracked_objects[ID] dengan data baru
    if matched_id:
        tracked_objects[matched_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'disappeared': 0,
            'confirmed_frames': CONFIRMATION_FRAMES_THRESHOLD
        }
        print(f"[INFO] Wajah sudah ada di database, update ID: {matched_id}, batal generate ID baru dan alert.")
        return False  # False = bukan ID baru
    tracked_objects[next_person_id] = {
        'centroid': centroid,
        'bbox': bbox,
        'disappeared': 0,
        'confirmed_frames': CONFIRMATION_FRAMES_THRESHOLD # Sudah pasti lolos threshold
    }
    print(f"[INFO] Objek terkonfirmasi, didaftarkan sebagai ID: {next_person_id}")
    next_person_id += 1
    return True  # True = ID baru

def deregister_object(object_id):
    """Menghapus objek yang sudah lama tidak terdeteksi."""
    print(f"[INFO] ID {object_id} dihapus dari pelacakan karena tidak terdeteksi.")
    del tracked_objects[object_id]
    # Juga hapus dari daftar yang sudah diberi peringatan
    if object_id in persons_alerted:
        del persons_alerted[object_id]


def update_tracker(detections, frame=None):
    """
    Memperbarui status pelacak berdasarkan deteksi baru.
    Sekarang, ID hanya diberikan saat sudah terkonfirmasi (CONFIRMATION_FRAMES_THRESHOLD).
    """
    global next_temp_id
    # Step 1: Update tracked_objects (yang sudah punya ID)
    if len(detections) == 0:
        for object_id in list(tracked_objects.keys()):
            tracked_objects[object_id]['disappeared'] += 1
            if tracked_objects[object_id]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                deregister_object(object_id)
        for temp_id in list(pending_candidates.keys()):
            pending_candidates[temp_id]['disappeared'] += 1
            if pending_candidates[temp_id]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                del pending_candidates[temp_id]
        return

    input_centroids = np.zeros((len(detections), 2), dtype="int")
    input_bboxes = [None] * len(detections)
    for i, (x1, y1, x2, y2) in enumerate(detections):
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        input_centroids[i] = (cX, cY)
        input_bboxes[i] = (x1, y1, x2, y2)

    # Gabungkan semua objek yang sedang dipantau (ID dan calon)
    all_objects = list(tracked_objects.items()) + list(pending_candidates.items())
    all_ids = [k for k, _ in all_objects]
    all_centroids = np.array([data['centroid'] for _, data in all_objects]) if all_objects else np.zeros((0,2))

    if len(all_objects) == 0:
        # Semua deteksi baru, masukkan ke pending_candidates
        for i in range(len(input_centroids)):
            pending_candidates[next_temp_id] = {
                'centroid': input_centroids[i],
                'bbox': input_bboxes[i],
                'disappeared': 0,
                'confirmed_frames': 1
            }
            next_temp_id += 1
        return

    # Asosiasikan deteksi ke objek yang sudah ada (baik ID maupun calon)
    D = dist.cdist(all_centroids, input_centroids)
    rows = D.min(axis=1).argsort()
    cols = D.argmin(axis=1)[rows]

    used_rows = set()
    used_cols = set()

    for row, col in zip(rows, cols):
        if row in used_rows or col in used_cols:
            continue
        if D[row, col] > MAX_DISTANCE:
            continue

        obj_id = all_ids[row]
        if obj_id in tracked_objects:
            tracked_objects[obj_id]['centroid'] = input_centroids[col]
            tracked_objects[obj_id]['bbox'] = input_bboxes[col]
            tracked_objects[obj_id]['disappeared'] = 0
            tracked_objects[obj_id]['confirmed_frames'] += 1
        elif obj_id in pending_candidates:
            pending_candidates[obj_id]['centroid'] = input_centroids[col]
            pending_candidates[obj_id]['bbox'] = input_bboxes[col]
            pending_candidates[obj_id]['disappeared'] = 0
            pending_candidates[obj_id]['confirmed_frames'] += 1
            # Jika sudah lolos threshold, masukkan ke queue untuk face recognition
            if pending_candidates[obj_id]['confirmed_frames'] >= CONFIRMATION_FRAMES_THRESHOLD:
                if frame is not None:
                    face_task_queue.put({'type': 'recognition', 'frame': frame.copy(), 'bbox': pending_candidates[obj_id]['bbox'], 'temp_id': obj_id})
        used_rows.add(row)
        used_cols.add(col)

    # Objek yang tidak terdeteksi lagi
    unused_rows = set(range(D.shape[0])).difference(used_rows)
    for row in unused_rows:
        obj_id = all_ids[row]
        if obj_id in tracked_objects:
            tracked_objects[obj_id]['disappeared'] += 1
            if tracked_objects[obj_id]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                deregister_object(obj_id)
        elif obj_id in pending_candidates:
            pending_candidates[obj_id]['disappeared'] += 1
            if pending_candidates[obj_id]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                del pending_candidates[obj_id]

    # Deteksi baru yang tidak terasosiasi ke objek manapun
    unused_cols = set(range(D.shape[1])).difference(used_cols)
    for col in unused_cols:
        pending_candidates[next_temp_id] = {
            'centroid': input_centroids[col],
            'bbox': input_bboxes[col],
            'disappeared': 0,
            'confirmed_frames': 1
        }
        next_temp_id += 1

def send_api_alert(person_id, timestamp, bounding_box):
    """LANGKAH 6a: Mengirimkan data peringatan ke API."""
    payload = {
        "eventType": "new_person_detected",
        "personId": f"Person {person_id}",
        "timestamp": timestamp.isoformat(),
        "location": "Ruang Monitoring Utama",
        "cameraId": "CAM-01",
        "boundingBox": [bounding_box[0], bounding_box[1], bounding_box[2]-bounding_box[0], bounding_box[3]-bounding_box[1]] # format [x, y, w, h]
    }
    
    headers = {'Content-Type': 'application/json'}

    print(f"[API] Mengirim data untuk Person {person_id}...")
    try:
        response = requests.post(API_ENDPOINT, data=json.dumps(payload), headers=headers, timeout=5)
        response.raise_for_status()
        print(f"[API] Sukses! Respons server (status {response.status_code}):")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"[API] Gagal mengirim data: {e}")


def save_face_crop(frame, bbox, person_id):
    """Hanya menyimpan crop wajah jika benar-benar terdeteksi wajah di dalam bounding box orang."""
    os.makedirs(FACES_DIR, exist_ok=True)
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    person_crop = frame[y1:y2, x1:x2].copy()
    if person_crop.size == 0:
        print(f"[WARNING] Gagal memotong area orang untuk Person {person_id}, bounding box mungkin di luar frame.")
        return False

    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        face = max(faces, key=lambda rect: rect[2]*rect[3])
        fx, fy, fw, fh = face
        face_area = fw * fh
        person_area = (x2 - x1) * (y2 - y1)
        if person_area > 0 and (face_area / person_area) <= 0.3:
            face_crop = person_crop[fy:fy+fh, fx:fx+fw].copy()
            filename = os.path.join(FACES_DIR, f"img_person{person_id}.png")
            try:
                cv2.imwrite(filename, face_crop)
                print(f"[INFO] Wajah untuk Person {person_id} disimpan ke {filename} (area wajah <= 70%)")
                return True
            except Exception as e:
                print(f"[ERROR] Gagal menyimpan file wajah untuk Person {person_id}: {e}")
                return False
        else:
            print(f"[INFO] Area wajah kurang dari 70% bounding box orang untuk Person {person_id}, tidak menyimpan crop.")
            return False
    else:
        print(f"[INFO] Tidak ditemukan wajah pada area orang untuk Person {person_id}, tidak menyimpan crop.")
        return False


def run_detection_and_alerting():
    """Fungsi utama untuk menjalankan seluruh alur kerja."""
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

        update_tracker(person_detections, frame=frame)

        # --- Ambil hasil face recognition dari worker dan update tracked_objects jika perlu ---
        while not face_result_queue.empty():
            result = face_result_queue.get()
            temp_id = result['temp_id']
            matched_id = result['matched_id']
            if temp_id in pending_candidates:
                centroid = pending_candidates[temp_id]['centroid']
                bbox = pending_candidates[temp_id]['bbox']
                if matched_id:
                    # Jika wajah sudah ada, update tracked_objects[matched_id] agar bbox dan centroid tetap up-to-date
                    tracked_objects[matched_id] = {
                        'centroid': centroid,
                        'bbox': bbox,
                        'disappeared': 0,
                        'confirmed_frames': CONFIRMATION_FRAMES_THRESHOLD
                    }
                    print(f"[INFO] Wajah sudah ada (ID: {matched_id}), update bbox & centroid, pending_candidates {temp_id} dihapus, tidak buat ID baru.")
                    del pending_candidates[temp_id]
                else:
                    # Jika wajah belum ada, buat ID baru
                    if register_object(centroid, bbox, matched_id=None):
                        del pending_candidates[temp_id]
            face_result_queue.task_done()

        # --- LANGKAH 5: EKSEKUSI ORANG BARU (DENGAN LOGIKA KONFIRMASI & FACE RECOGNITION) ---

        # Visualisasi dan eksekusi untuk objek yang sudah terkonfirmasi (tracked_objects)
        for object_id, data in list(tracked_objects.items()):
            # Cek apakah objek sudah dikonfirmasi dan belum pernah dikirimi alert
            if data['confirmed_frames'] >= CONFIRMATION_FRAMES_THRESHOLD and object_id not in persons_alerted:
                # Hanya lakukan crop dan alert jika ID ini benar-benar baru (bukan hasil match)
                # Cek apakah file img_person{object_id}.png sudah ada, jika sudah, skip crop & alert
                filename = os.path.join(FACES_DIR, f"img_person{object_id}.png")
                if not os.path.exists(filename):
                    face_task_queue.put({'type': 'save_crop', 'frame': frame.copy(), 'bbox': data['bbox'], 'person_id': object_id})
                    timestamp = datetime.now()
                    persons_alerted[object_id] = timestamp
                    print(f"[ALERT] Person {object_id} terkonfirmasi sebagai orang baru (wajah terdeteksi).")
                    send_api_alert(object_id, timestamp, data['bbox'])
                else:
                    # Sudah pernah dicrop, jangan lakukan crop/alert lagi
                    persons_alerted[object_id] = datetime.now()

            # Gambar visualisasi pada frame
            centroid = tuple(map(int, data['centroid']))
            bbox = tuple(map(int, data['bbox']))
            if object_id in persons_alerted:
                label = f"Person {object_id}"
                color = (0, 255, 0)
            else:
                label = f"ID {object_id} (Confirming...)"
                color = (0, 255, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, centroid, 4, color, -1)

        # Visualisasi untuk calon objek (pending_candidates)
        for temp_id, data in pending_candidates.items():
            bbox = tuple(map(int, data['bbox']))
            centroid = tuple(map(int, data['centroid']))
            label = "Face Recognition"
            color = (255, 0, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, centroid, 4, color, -1)

        cv2.imshow('Sistem Deteksi Orang Baru (Tekan Q untuk Keluar)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_face_worker.set()
    face_thread.join(timeout=1)
    print("Webcam dilepaskan dan jendela ditutup.")

if __name__ == "__main__":
    run_detection_and_alerting()
