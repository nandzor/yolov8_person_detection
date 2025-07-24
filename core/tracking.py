# tracking.py
import numpy as np
from utils.config import MAX_DISAPPEARED_FRAMES, MAX_DISTANCE, CONFIRMATION_FRAMES_THRESHOLD

tracked_objects = {}
pending_candidates = {}
persons_alerted = {}

def update_tracker(detections, frame=None):
    global tracked_objects, pending_candidates
    if len(detections) == 0:
        for object_id in list(tracked_objects.keys()):
            tracked_objects[object_id]['disappeared'] += 1
            if tracked_objects[object_id]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                del tracked_objects[object_id]
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
    # Simple centroid-based tracking logic
    next_object_id = max(list(tracked_objects.keys()) + [0]) + 1 if tracked_objects else 0
    used_rows = set()
    used_cols = set()
    # If there are no tracked objects, register all detections as new
    if len(tracked_objects) == 0:
        for i, bbox in enumerate(input_bboxes):
            tracked_objects[next_object_id] = {
                'bbox': bbox,
                'centroid': tuple(input_centroids[i]),
                'disappeared': 0,
                'frames': 1
            }
            next_object_id += 1
    else:
        object_ids = list(tracked_objects.keys())
        object_centroids = [tracked_objects[oid]['centroid'] for oid in object_ids]
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        assigned = set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > MAX_DISTANCE:
                continue
            object_id = object_ids[row]
            tracked_objects[object_id]['bbox'] = input_bboxes[col]
            tracked_objects[object_id]['centroid'] = tuple(input_centroids[col])
            tracked_objects[object_id]['disappeared'] = 0
            tracked_objects[object_id]['frames'] += 1
            used_rows.add(row)
            used_cols.add(col)
            assigned.add(col)
        # Mark disappeared for unassigned tracked objects
        for i, object_id in enumerate(object_ids):
            if i not in used_rows:
                tracked_objects[object_id]['disappeared'] += 1
                if tracked_objects[object_id]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                    del tracked_objects[object_id]
        # Register new detections
        for i in range(len(input_bboxes)):
            if i not in assigned:
                tracked_objects[next_object_id] = {
                    'bbox': input_bboxes[i],
                    'centroid': tuple(input_centroids[i]),
                    'disappeared': 0,
                    'frames': 1
                }
                next_object_id += 1
