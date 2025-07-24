# api.py
import requests
import json
from utils.config import API_ENDPOINT

def send_api_alert(person_id, timestamp, bounding_box):
    payload = {
        "eventType": "new_person_detected",
        "personId": str(person_id),
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
