# drawing.py
import cv2
from core.face_recognition import is_face_already_exists
import numpy as np

def draw_person_box(frame, tracked_objects, pending_candidates):
    assigned_names = set()
    recog_results = []
    for object_id, data in list(tracked_objects.items()):
        bbox = data['bbox']
        centroid = data['centroid']

        # Face recognition and bbox logic
        if 'face_name' not in data or data['frames'] == 1 or data['frames'] % 30 == 0:
            name, face_bbox_new = is_face_already_exists(frame, bbox)
            tracked_objects[object_id]['face_name'] = name
            
            # Intelligent Fallback: Only update face_bbox if a face was actually found
            if face_bbox_new is not None:
                tracked_objects[object_id]['last_seen_face_bbox'] = face_bbox_new
        
        name = tracked_objects[object_id].get('face_name')
        # Use last known face bbox; if none, fallback to body bbox
        face_bbox = tracked_objects[object_id].get('last_seen_face_bbox', bbox)

        # Apply smoothing
        alpha_face = 0.4 # Adjusted for stability
        smoothed_bbox = tuple((alpha_face * np.array(face_bbox) + (1 - alpha_face) * np.array(data.get('smoothed_bbox', face_bbox))).astype(int))
        tracked_objects[object_id]['smoothed_bbox'] = smoothed_bbox
        
        recog_results.append({'object_id': object_id, 'name': name, 'centroid': centroid, 'bbox': smoothed_bbox})
    
    for r in recog_results:
        object_id = r['object_id']
        if r['name'] and r['name'] not in assigned_names:
            label = r['name']
            color = (0, 255, 0)
            assigned_names.add(r['name'])
        else:
            label = f"Unknown Person (ID {r['object_id']})"
            color = (0, 165, 255)
        
        bbox = r['bbox']
        
        # Only print every 30 frames
        if tracked_objects[object_id]['frames'] % 30 == 0:
            print(f"[DRAW] Gambar bbox: {bbox} label: {label}")
        
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, tuple(centroid), 4, color, -1)
