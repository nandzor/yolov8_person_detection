# drawing.py
import cv2
import numpy as np

def draw_person_box(frame, tracked_objects, pending_candidates):
    """Versi super aman - tidak akan pernah force close"""
    
    # JANGAN PAKE THREAD, JANGAN PAKE IMPORT RIBET
    try:
        for object_id, data in tracked_objects.items():
            try:
                # AMBIL DATA DENGAN AMAN
                bbox = data.get('bbox')
                centroid = data.get('centroid')
                frames = data.get('frames', 0)
                
                # VALIDASI DATA
                if not bbox or not centroid:
                    continue
                if len(bbox) != 4:
                    continue
                    
                # GAK USAH RECOGNITION LAGI, PAKE YANG SUDAH ADA
                name = data.get('face_name', '')
                
                # PILIH WARNA YANG AMAN
                try:
                    if name and str(name).lower() not in ['', 'unknown', 'none', 'null']:
                        color = (0, 255, 0)  # HIJAU
                        label = str(name)[:20]  # BATAS PANJANG LABEL
                    else:
                        color = (0, 165, 255)  # ORANGE
                        label = f"ID{object_id}"
                except:
                    color = (0, 165, 255)
                    label = f"ID{object_id}"
                
                # CONVERT COORDINATE DENGAN AMAN
                try:
                    x1 = int(float(bbox[0]))
                    y1 = int(float(bbox[1]))
                    x2 = int(float(bbox[2]))
                    y2 = int(float(bbox[3]))
                    cx = int(float(centroid[0]))
                    cy = int(float(centroid[1]))
                except:
                    continue  # SKIP KALAU ERROR
                
                # VALIDASI NILAI
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue
                if x1 > frame.shape[1] or x2 > frame.shape[1]:
                    continue
                if y1 > frame.shape[0] or y2 > frame.shape[0]:
                    continue
                    
                # GAMBAR DENGAN AMAN
                try:
                    # RECTANGLE
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    
                    # TEXT (cek dulu biar gak keluar frame)
                    text_y = y1 - 5 if y1 > 15 else y1 + 15
                    cv2.putText(frame, label, (x1, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # CENTROID
                    cv2.circle(frame, (cx, cy), 2, color, -1)
                    
                except Exception as draw_error:
                    # DIEM AJA KALAU ERROR DRAWING
                    pass
                    
            except Exception as object_error:
                # DIEM AJA KALAU ERROR OBJECT
                continue
                
    except Exception as main_error:
        # DIEM AJA KALAU ERROR UTAMA
        pass

def draw_simple_fps(frame, fps):
    """FPS yang super aman"""
    try:
        if fps and fps > 0:
            cv2.putText(frame, f"FPS:{int(fps)}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except:
        pass  # DIEM AJA