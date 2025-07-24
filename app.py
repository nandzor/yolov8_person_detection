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
import glob
import re

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
# tracked_objects hanya berisi objek yang sudah terkonfirmasi (sudah punya ID)
tracked_objects = {}
# Buffer calon objek yang belum terkonfirmasi: {temp_id: {'centroid': (x,y), 'bbox': (x1,y1,x2,y2), 'disappeared': 0, 'confirmed_frames': 1}}
pending_candidates = {}
next_temp_id = 1
# Menyimpan orang yang sudah diberi peringatan untuk mencegah duplikasi alert
# {objectID: timestamp}
persons_alerted = {}


def is_face_already_exists(frame, bbox, model_name='Facenet', distance_threshold=0.6):
    """Bandingkan crop wajah dengan semua wajah di folder faces menggunakan DeepFace. Return nama jika match, None jika tidak."""
    import glob
    from deepface import DeepFace
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    person_crop = frame[y1:y2, x1:x2].copy()
    if person_crop.size == 0:
        return None
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
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
                import re
                match = re.match(r'face_(.+)\.png', os.path.basename(file))
                if match:
                    name = match.group(1).replace('_', ' ').title()
                else:
                    name = os.path.splitext(os.path.basename(file))[0]
                print(f"[FACE RECOG] Wajah match dengan {file} (distance={result['distance']:.3f}), gunakan nama: {name}")
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                return name
        except Exception as e:
            print(f"[FACE RECOG] Error membandingkan dengan {file}: {e}")
            continue
    if os.path.exists(temp_face_path):
        os.remove(temp_face_path)
    return None

def register_object(centroid, bbox, frame=None):
    """Mendaftarkan objek baru yang sudah terkonfirmasi ke tracked_objects dengan ID unik, setelah face recognition."""
    global next_person_id
    # Face recognition: jika wajah sudah ada, update tracked_objects[ID] dengan data baru
    matched_name = None
    if frame is not None:
        matched_name = is_face_already_exists(frame, bbox)
    if matched_name:
        # Update tracked_objects dengan nama jika sudah ada, jika belum tambahkan
        tracked_objects[matched_name] = {
            'centroid': centroid,
            'bbox': bbox,
            'disappeared': 0,
            'confirmed_frames': CONFIRMATION_FRAMES_THRESHOLD
        }
        print(f"[INFO] Wajah sudah ada di database, update nama: {matched_name}, batal generate ID baru dan alert.")
        return False
    tracked_objects[next_person_id] = {
        'centroid': centroid,
        'bbox': bbox,
        'disappeared': 0,
        'confirmed_frames': CONFIRMATION_FRAMES_THRESHOLD # Sudah pasti lolos threshold
    }
    print(f"[INFO] Objek terkonfirmasi, didaftarkan sebagai ID: {next_person_id}")
    next_person_id += 1
    return True

def deregister_object(object_id):
    """Menghapus objek yang sudah lama tidak terdeteksi."""
    print(f"[INFO] ID {object_id} dihapus dari pelacakan karena tidak terdeteksi.")
    del tracked_objects[object_id]
    # Juga hapus dari daftar yang sudah diberi peringatan
    if object_id in persons_alerted:
        del persons_alerted[object_id]


def update_tracker(detections):
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
        # Update pending_candidates juga
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
        # Update data
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
            # Jika sudah lolos threshold, daftarkan ke tracked_objects
            if pending_candidates[obj_id]['confirmed_frames'] >= CONFIRMATION_FRAMES_THRESHOLD:
                # Kirim frame ke register_object untuk face recognition
                frame_ref = None
                try:
                    import inspect
                    frame_ref = inspect.currentframe().f_back.f_locals.get('frame', None)
                except Exception:
                    frame_ref = None
                # Jika wajah sudah ada, batal register dan hapus dari pending
                if register_object(pending_candidates[obj_id]['centroid'], pending_candidates[obj_id]['bbox'], frame=frame_ref):
                    del pending_candidates[obj_id]
                else:
                    del pending_candidates[obj_id]
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
        
        update_tracker(person_detections)


        # --- LANGKAH 5: EKSEKUSI ORANG BARU (DENGAN LOGIKA KONFIRMASI & FACE RECOGNITION) ---

        # Visualisasi dan eksekusi untuk objek yang sudah terkonfirmasi (tracked_objects)
        # --- One-to-one face recognition assignment ---
        from deepface import DeepFace
        face_files = glob.glob(os.path.join(FACES_DIR, 'face_*.png'))
        face_names = []
        for file in face_files:
            match = re.match(r'face_(.+)\.png', os.path.basename(file))
            if match:
                face_names.append(match.group(1).replace('_', ' ').title())
            else:
                face_names.append(os.path.splitext(os.path.basename(file))[0])
        recog_results = []
        for object_id, data in list(tracked_objects.items()):
            bbox = data['bbox']
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            person_crop = frame[y1:y2, x1:x2].copy()
            if person_crop.size == 0 or not face_files:
                recog_results.append({'object_id': object_id, 'name': None, 'distance': 1e9, 'centroid': data['centroid'], 'bbox': bbox})
                continue
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) == 0:
                recog_results.append({'object_id': object_id, 'name': None, 'distance': 1e9, 'centroid': data['centroid'], 'bbox': bbox})
                continue
            face = max(faces, key=lambda rect: rect[2]*rect[3])
            fx, fy, fw, fh = face
            face_crop = person_crop[fy:fy+fh, fx:fx+fw].copy()
            temp_face_path = os.path.join(FACES_DIR, f'temp_face_check_{object_id}.png')
            cv2.imwrite(temp_face_path, face_crop)
            best_name = None
            best_distance = 1e9
            for file, name in zip(face_files, face_names):
                try:
                    result = DeepFace.verify(img1_path=temp_face_path, img2_path=file, model_name='Facenet', enforce_detection=False)
                    if result['verified'] and result['distance'] < best_distance:
                        best_distance = result['distance']
                        best_name = name
                except Exception as e:
                    continue
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)
            recog_results.append({'object_id': object_id, 'name': best_name, 'distance': best_distance, 'centroid': data['centroid'], 'bbox': bbox})
        # Assignment one-to-one: pilih pasangan (object, name) dengan distance terkecil, satu nama hanya untuk satu object
        assigned_names = set()
        for r in sorted(recog_results, key=lambda x: x['distance']):
            if r['name'] and r['name'] not in assigned_names and r['distance'] < 0.6:
                label = r['name']
                color = (0, 255, 0)
                assigned_names.add(r['name'])
            else:
                label = "Unknown Person"
                color = (0, 165, 255)
            bbox = r['bbox']
            centroid = r['centroid']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, tuple(centroid), 4, color, -1)

        # Visualisasi dan face recognition untuk calon objek (pending_candidates)
        for temp_id, data in pending_candidates.items():
            bbox = data['bbox']
            centroid = data['centroid']
            # Lakukan face recognition 1-to-many sebelum generate ID baru
            from deepface import DeepFace
            import glob
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            person_crop = frame[y1:y2, x1:x2].copy()
            label = "Face Recognition"
            color = (255, 0, 0)  # Biru untuk proses face recognition
            exist_face = False
            matched_id = None
            if person_crop.size != 0:
                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(faces) > 0:
                    face = max(faces, key=lambda rect: rect[2]*rect[3])
                    fx, fy, fw, fh = face
                    face_crop = person_crop[fy:fy+fh, fx:fx+fw].copy()
                    temp_face_path = os.path.join(FACES_DIR, 'temp_face_check.png')
                    cv2.imwrite(temp_face_path, face_crop)
                    image_files = []
                    for ext in ('*.png', '*.jpg', '*.jpeg'):
                        image_files.extend(glob.glob(os.path.join(FACES_DIR, ext)))
                    image_files = [f for f in image_files if not f.endswith('temp_face_check.png')]
                    for file in image_files:
                        try:
                            result = DeepFace.verify(img1_path=temp_face_path, img2_path=file, model_name='Facenet', enforce_detection=False)
                            if result['verified'] and result['distance'] < 0.6:
                                import re
                                match = re.search(r'img_person(\d+)', os.path.basename(file))
                                matched_id = int(match.group(1)) if match else None
                                exist_face = True
                                break
                        except Exception as e:
                            continue
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
            if exist_face:
                # Cari nama label dari file face jika ada
                label_name = None
                for file in image_files:
                    import re, os
                    match = re.search(r'img_person(\d+)', os.path.basename(file))
                    if matched_id and match and int(match.group(1)) == matched_id:
                        # Cek jika ada file dengan format face_<nama>.png
                        alt_name = os.path.basename(file).replace('img_person', 'face_')
                        alt_path = os.path.join(FACES_DIR, alt_name)
                        if os.path.exists(alt_path):
                            label_name = os.path.splitext(alt_name)[0].replace('face_', '')
                        else:
                            label_name = f"ID {matched_id}"
                        break
                if label_name:
                    label = f"{label_name}"
                else:
                    label = f"Exist Face (ID {matched_id})"
                color = (0, 0, 255)  # Merah jika wajah sudah ada
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, tuple(centroid), 4, color, -1)
            # Jika wajah sudah ada, jangan buat person baru, pending_candidates tetap dipertahankan
            # (tidak perlu hapus, biarkan tracker mengelola lifecycle-nya)

        cv2.imshow('Sistem Deteksi Orang Baru (Tekan Q untuk Keluar)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam dilepaskan dan jendela ditutup.")

if __name__ == "__main__":
    run_detection_and_alerting()
