# -*- coding: utf-8 -*-
"""
Deskripsi:
Script ini mengimplementasikan alur kerja untuk mendeteksi orang baru yang memasuki
jangkauan kamera, mengirimkan peringatan melalui API, dan menyimpan gambar wajah
orang tersebut.
Versi ini telah diperbaiki untuk:
1. Mencegah duplikasi person yang sama dengan face recognition yang lebih akurat
2. Bounding box fokus pada area kepala/wajah
3. Mencegah penyimpanan person yang sudah ada

Alur Kerja yang Diimplementasikan:
1.  Inisialisasi Sistem: Memuat model YOLO, tracker, dan variabel status.
2.  Akuisisi Frame Video: Menangkap video dari webcam.
3.  Proses Deteksi Orang: Menggunakan YOLOv8 untuk mendeteksi orang.
4.  Deteksi Wajah: Menggunakan Haar Cascade untuk deteksi wajah dari area orang
5.  Proses Pelacakan dan Asosiasi ID: Memberikan ID unik untuk setiap orang
    dan melacaknya antar frame.
6.  Face Recognition: Membandingkan wajah dengan database existing
7.  Eksekusi untuk Orang Baru (dengan Konfirmasi): Memicu logika saat ID baru
    terdeteksi dan terkonfirmasi secara konsisten selama beberapa frame.
8.  Generasi Peringatan via API & Simpan Wajah: Memformat data, mengirimkannya
    ke endpoint API, dan menyimpan potongan gambar wajah.
9.  Pengulangan (Loop): Memproses video secara terus-menerus.
10. Manajemen Status: Menghapus ID yang sudah tidak terdeteksi.

Dependensi:
- ultralytics
- opencv-python
- numpy
- requests
- scipy
- deepface
"""

from ultralytics import YOLO
import cv2
import numpy as np
import requests
import json
from datetime import datetime
from scipy.spatial import distance as dist
import os
import threading
import queue
import glob
import re

# Inisialisasi Haar Cascade untuk deteksi wajah
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# --- LANGKAH 1: INISIALISASI SISTEM ---

# URL Endpoint API untuk mengirim peringatan
API_ENDPOINT = "https://httpbin.org/post"
FACES_DIR = "faces"

# --- Konfigurasi Pelacak (Tracker) yang Dioptimasi ---
MAX_DISAPPEARED_FRAMES = 50
MAX_DISTANCE = 60
CONFIRMATION_FRAMES_THRESHOLD = 10

# Face Recognition Configuration
FACE_RECOGNITION_THRESHOLD = 0.4  # Lowered threshold for better accuracy
FACE_MODEL = 'Facenet'  # Options: VGG-Face, Facenet, OpenFace, DeepFace

# Variabel Global untuk Manajemen Status
next_person_id = 1
tracked_objects = {}
pending_candidates = {}
next_temp_id = 1
persons_alerted = {}
face_database = {}  # Cache for face embeddings

# Queue dan thread untuk face recognition & crop agar tidak blocking
face_task_queue = queue.Queue()
face_result_queue = queue.Queue()
stop_face_worker = threading.Event()

def extract_head_region(bbox, expansion_factor=0.3):
    """
    Mengekstrak area kepala dari bounding box orang.
    expansion_factor: faktor ekspansi untuk area kepala (0.3 = 30% lebih besar)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Area kepala biasanya 25-30% bagian atas dari tubuh
    head_height = int(height * 0.3)
    head_y1 = y1
    head_y2 = y1 + head_height
    
    # Expand sedikit ke samping untuk kepala
    head_expansion = int(width * expansion_factor)
    head_x1 = max(0, x1 - head_expansion)
    head_x2 = x2 + head_expansion
    
    return (head_x1, head_y1, head_x2, head_y2)

def face_worker():
    """Worker thread untuk face recognition yang tidak blocking"""
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

def load_face_database():
    """Load semua wajah yang sudah ada ke dalam database untuk perbandingan"""
    global face_database
    face_database = {}
    
    os.makedirs(FACES_DIR, exist_ok=True)
    
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_files.extend(glob.glob(os.path.join(FACES_DIR, ext)))
    
    for file_path in image_files:
        try:
            # Extract person ID from filename
            match = re.search(r'img_person(\d+)', os.path.basename(file_path))
            if match:
                person_id = int(match.group(1))
                face_database[person_id] = file_path
                print(f"[FACE DB] Loaded Person {person_id}: {file_path}")
        except Exception as e:
            print(f"[FACE DB] Error loading {file_path}: {e}")

def is_face_already_exists(frame, bbox):
    """
    Membandingkan wajah dari bounding box dengan database wajah yang ada.
    Return person_id jika match, None jika tidak match.
    """
    try:
        from deepface import DeepFace
    except ImportError:
        print("[ERROR] DeepFace tidak terinstall. Install dengan: pip install deepface")
        return None
    
    # Extract head region instead of full body
    head_bbox = extract_head_region(bbox)
    x1, y1, x2, y2 = head_bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    head_crop = frame[y1:y2, x1:x2].copy()
    if head_crop.size == 0:
        return None
    
    # Detect face in head region
    gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    if len(faces) == 0:
        print("[FACE RECOG] Tidak ada wajah terdeteksi di area kepala")
        return None
    
    # Use the largest face
    face = max(faces, key=lambda rect: rect[2]*rect[3])
    fx, fy, fw, fh = face
    face_crop = head_crop[fy:fy+fh, fx:fx+fw].copy()
    
    # Save temporary face for comparison
    temp_face_path = os.path.join(FACES_DIR, 'temp_face_check.png')
    cv2.imwrite(temp_face_path, face_crop)
    
    try:
        # Compare with all faces in database
        for person_id, db_face_path in face_database.items():
            try:
                result = DeepFace.verify(
                    img1_path=temp_face_path, 
                    img2_path=db_face_path, 
                    model_name=FACE_MODEL, 
                    enforce_detection=False
                )
                
                distance = result['distance']
                verified = result['verified']
                
                print(f"[FACE RECOG] Comparing with Person {person_id}: distance={distance:.3f}, verified={verified}")
                
                if verified and distance < FACE_RECOGNITION_THRESHOLD:
                    print(f"[FACE RECOG] MATCH! Wajah cocok dengan Person {person_id} (distance={distance:.3f})")
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
                    return person_id
                    
            except Exception as e:
                print(f"[FACE RECOG] Error membandingkan dengan Person {person_id}: {e}")
                continue
        
        print("[FACE RECOG] Tidak ada match, ini adalah wajah baru")
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)
        return None
        
    except Exception as e:
        print(f"[FACE RECOG] Error dalam face recognition: {e}")
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)
        return None

def register_object(centroid, bbox, frame=None, matched_id=None):
    """Mendaftarkan objek baru yang sudah terkonfirmasi ke tracked_objects dengan ID unik, setelah face recognition."""
    global next_person_id
    
    # Face recognition: jika wajah sudah ada, update tracked_objects[ID] dengan data baru
    if matched_id:
        # Update existing person's tracking data
        tracked_objects[matched_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'disappeared': 0,
            'confirmed_frames': CONFIRMATION_FRAMES_THRESHOLD,
            'head_bbox': extract_head_region(bbox)
        }
        print(f"[INFO] Wajah sudah ada di database (Person {matched_id}), update tracking data, tidak buat ID baru.")
        return False, matched_id  # False = bukan ID baru, tapi return matched_id
    
    # Create new person ID
    new_person_id = next_person_id
    tracked_objects[new_person_id] = {
        'centroid': centroid,
        'bbox': bbox,
        'disappeared': 0,
        'confirmed_frames': CONFIRMATION_FRAMES_THRESHOLD,
        'head_bbox': extract_head_region(bbox)
    }
    print(f"[INFO] Objek terkonfirmasi sebagai orang baru, didaftarkan sebagai Person {new_person_id}")
    next_person_id += 1
    return True, new_person_id  # True = ID baru

def deregister_object(object_id):
    """Menghapus objek yang sudah lama tidak terdeteksi."""
    print(f"[INFO] Person {object_id} dihapus dari pelacakan karena tidak terdeteksi.")
    if object_id in tracked_objects:
        del tracked_objects[object_id]
    # Juga hapus dari daftar yang sudah diberi peringatan
    if object_id in persons_alerted:
        del persons_alerted[object_id]

def update_tracker(detections, frame=None):
    """
    Memperbarui status pelacak berdasarkan deteksi baru.
    Sekarang menggunakan head region untuk tracking yang lebih akurat.
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

    # Calculate centroids using head region for better tracking
    input_centroids = np.zeros((len(detections), 2), dtype="int")
    input_bboxes = [None] * len(detections)
    input_head_bboxes = [None] * len(detections)
    
    for i, (x1, y1, x2, y2) in enumerate(detections):
        head_bbox = extract_head_region((x1, y1, x2, y2))
        # Use head center for tracking instead of body center
        hx1, hy1, hx2, hy2 = head_bbox
        cX = int((hx1 + hx2) / 2.0)
        cY = int((hy1 + hy2) / 2.0)
        input_centroids[i] = (cX, cY)
        input_bboxes[i] = (x1, y1, x2, y2)
        input_head_bboxes[i] = head_bbox

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
                'head_bbox': input_head_bboxes[i],
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
            tracked_objects[obj_id]['head_bbox'] = input_head_bboxes[col]
            tracked_objects[obj_id]['disappeared'] = 0
            tracked_objects[obj_id]['confirmed_frames'] += 1
        elif obj_id in pending_candidates:
            pending_candidates[obj_id]['centroid'] = input_centroids[col]
            pending_candidates[obj_id]['bbox'] = input_bboxes[col]
            pending_candidates[obj_id]['head_bbox'] = input_head_bboxes[col]
            pending_candidates[obj_id]['disappeared'] = 0
            pending_candidates[obj_id]['confirmed_frames'] += 1
            
            # Jika sudah lolos threshold, masukkan ke queue untuk face recognition
            if pending_candidates[obj_id]['confirmed_frames'] >= CONFIRMATION_FRAMES_THRESHOLD:
                if frame is not None:
                    face_task_queue.put({
                        'type': 'recognition', 
                        'frame': frame.copy(), 
                        'bbox': pending_candidates[obj_id]['bbox'], 
                        'temp_id': obj_id
                    })
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
            'head_bbox': input_head_bboxes[col],
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
        "boundingBox": [bounding_box[0], bounding_box[1], bounding_box[2]-bounding_box[0], bounding_box[3]-bounding_box[1]]
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
    """Menyimpan crop wajah dari area kepala yang terdeteksi."""
    os.makedirs(FACES_DIR, exist_ok=True)
    
    # Use head region for face cropping
    head_bbox = extract_head_region(bbox)
    x1, y1, x2, y2 = head_bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    head_crop = frame[y1:y2, x1:x2].copy()
    if head_crop.size == 0:
        print(f"[WARNING] Gagal memotong area kepala untuk Person {person_id}")
        return False

    gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        # Use largest face
        face = max(faces, key=lambda rect: rect[2]*rect[3])
        fx, fy, fw, fh = face
        face_crop = head_crop[fy:fy+fh, fx:fx+fw].copy()
        
        filename = os.path.join(FACES_DIR, f"img_person{person_id}.png")
        try:
            cv2.imwrite(filename, face_crop)
            # Update face database
            face_database[person_id] = filename
            print(f"[INFO] Wajah Person {person_id} disimpan ke {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan file wajah untuk Person {person_id}: {e}")
            return False
    else:
        print(f"[INFO] Tidak ditemukan wajah pada area kepala untuk Person {person_id}")
        return False

def run_detection_and_alerting():
    """Fungsi utama untuk menjalankan seluruh alur kerja."""
    # Load existing face database
    load_face_database()
    
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

        # YOLO Detection
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
                
                is_new, final_person_id = register_object(centroid, bbox, matched_id=matched_id)
                
                if matched_id:
                    print(f"[INFO] Face match found: menggunakan existing Person {matched_id}")
                else:
                    print(f"[INFO] Face baru: membuat Person {final_person_id}")
                
                del pending_candidates[temp_id]
            face_result_queue.task_done()

        # --- VISUALISASI DAN EKSEKUSI UNTUK ORANG BARU ---

        # Visualisasi dan eksekusi untuk objek yang sudah terkonfirmasi (tracked_objects)
        for object_id, data in list(tracked_objects.items()):
            if data['confirmed_frames'] >= CONFIRMATION_FRAMES_THRESHOLD and object_id not in persons_alerted:
                # Cek apakah file sudah ada (untuk menghindari duplikasi)
                filename = os.path.join(FACES_DIR, f"img_person{object_id}.png")
                if not os.path.exists(filename):
                    # Hanya simpan dan alert jika benar-benar baru
                    face_task_queue.put({
                        'type': 'save_crop', 
                        'frame': frame.copy(), 
                        'bbox': data['bbox'], 
                        'person_id': object_id
                    })
                    timestamp = datetime.now()
                    persons_alerted[object_id] = timestamp
                    print(f"[ALERT] Person {object_id} terkonfirmasi sebagai orang baru!")
                    send_api_alert(object_id, timestamp, data['head_bbox'])
                else:
                    # Sudah ada, jangan alert lagi
                    persons_alerted[object_id] = datetime.now()
                    print(f"[INFO] Person {object_id} sudah pernah diproses sebelumnya")

            # Gambar visualisasi pada frame menggunakan HEAD BBOX
            centroid = tuple(map(int, data['centroid']))
            head_bbox = tuple(map(int, data['head_bbox']))
            
            if object_id in persons_alerted:
                label = f"Person {object_id}"
                color = (0, 255, 0)  # Green for confirmed
            else:
                label = f"Person {object_id} (Confirming...)"
                color = (0, 255, 255)  # Yellow for confirming
                
            # Draw HEAD bounding box instead of full body
            cv2.rectangle(frame, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), color, 2)
            cv2.putText(frame, label, (head_bbox[0], head_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, centroid, 4, color, -1)

        # Visualisasi untuk calon objek (pending_candidates)
        for temp_id, data in pending_candidates.items():
            head_bbox = tuple(map(int, data['head_bbox']))
            centroid = tuple(map(int, data['centroid']))
            progress = min(data['confirmed_frames'], CONFIRMATION_FRAMES_THRESHOLD)
            label = f"Detecting ({progress}/{CONFIRMATION_FRAMES_THRESHOLD})"
            color = (255, 0, 0)  # Red for pending
            
            # Draw HEAD bounding box for pending candidates too
            cv2.rectangle(frame, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), color, 2)
            cv2.putText(frame, label, (head_bbox[0], head_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, centroid, 4, color, -1)

        # Display info
        info_text = f"Tracked: {len(tracked_objects)} | Pending: {len(pending_candidates)} | Database: {len(face_database)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Sistem Deteksi Orang Baru - Head Focus (Tekan Q untuk Keluar)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_face_worker.set()
    face_thread.join(timeout=1)
    print("Webcam dilepaskan dan jendela ditutup.")

if __name__ == "__main__":
    run_detection_and_alerting()