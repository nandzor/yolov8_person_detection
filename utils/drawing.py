# drawing.py
import cv2
from core.face_recognition import is_face_already_exists

def draw_person_box(frame, tracked_objects, pending_candidates):
    assigned_names = set()
    recog_results = []
    for object_id, data in list(tracked_objects.items()):
        bbox = data['bbox']
        centroid = data['centroid']
        print(f"[DRAW] Deteksi bbox: {bbox} untuk object_id: {object_id}")
        name = is_face_already_exists(frame, bbox)
        recog_results.append({'object_id': object_id, 'name': name, 'centroid': centroid, 'bbox': bbox})
    for r in recog_results:
        if r['name'] and r['name'] not in assigned_names:
            label = r['name']
            color = (0, 255, 0)
            assigned_names.add(r['name'])
        else:
            label = f"Unknown Person (ID {r['object_id']})"
            color = (0, 165, 255)
        bbox = r['bbox']
        centroid = r['centroid']
        print(f"[DRAW] Gambar bbox: {bbox} label: {label}")
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, tuple(centroid), 4, color, -1)
