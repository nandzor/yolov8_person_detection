from core.detection import run_detection_and_alerting
# main.py
import cv2
import time



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("KAMERA GAK BISA!")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # SIMULASI DATA
    tracked_objects = {
        1: {
            'bbox': [100, 100, 200, 200],
            'centroid': [150, 150],
            'frames': 100,
            'face_name': 'TEST'  # INI YANG BUAT HIJAU
        }
    }
    
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # UPDATE SIMULASI
        tracked_objects[1]['frames'] += 1
        
        # DRAW YANG AMAN
        try:
            draw_person_box(frame, tracked_objects, [])
        except:
            pass  # TETAP JALAN WALAUPUN ERROR
            
        # FPS
        current_time = time.time()
        fps = 1 / (current_time - fps_time) if (current_time - fps_time) > 0 else 0
        fps_time = current_time
        cv2.putText(frame, f"FPS:{int(fps)}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('TEST AJA', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("SELESAI - GAK ADA FORCE CLOSE!")

if __name__ == "__main__":
    run_detection_and_alerting()
