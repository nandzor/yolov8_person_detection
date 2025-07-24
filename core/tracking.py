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
    # ... (lanjutkan logika tracker sesuai kebutuhan)
