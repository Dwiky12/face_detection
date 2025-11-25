#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face Recognition Pipeline = Viola–Jones (Haar) + CNN Embedding (Recognition) + Farneback Optical Flow (Tracking)

- DETECTION:    Viola–Jones via OpenCV Haar Cascade (bukan CNN detector)
- RECOGNITION:  face_recognition (dlib ResNet) -> embedding CNN -> nearest neighbor by distance
- TRACKING:     Dense Optical Flow Farneback untuk mem-propagasi bbox di frame antara re-detection

Tekan 'q' untuk keluar.
"""

import os
import sys
import cv2
import numpy as np

# --- 0. IMPORT & DEPENDENCY CHECK ------------------------------------------------
try:
    import face_recognition  # dlib ResNet di balik layar untuk face encodings (CNN embedding)
except Exception as e:
    print("Error: Tidak bisa import 'face_recognition'. Pastikan sudah terpasang:\n"
          "  pip install face_recognition\n"
          "Catatan: butuh 'cmake' dan 'dlib' terpasang dengan benar.")
    print(f"Detail: {e}")
    sys.exit(1)

# Loader gambar (untuk EXIF & HEIC support opsional)
from PIL import Image, ImageOps
try:
    import pillow_heif  # opsional: untuk HEIC/HEIF
    pillow_heif.register_heif_opener()
except Exception:
    pass

# --- 1. CONFIG -------------------------------------------------------------------
# Arahkan ke FOLDER PAYUNG tertinggi (fungsi di bawah akan REKURSIF menelusuri subfolder)
DATASET_DIR = "dataset"

RE_DETECT_INTERVAL = 10      # jalankan deteksi+recognition setiap N frame
DIST_THRESHOLD = 0.60        # ambang jarak (semakin kecil = semakin ketat)
SCALE_FACTOR = 1.1           # Haar detectMultiScale
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}

# --- 2. LOAD DETECTOR (VIOLA–JONES / HAAR) --------------------------------------
def load_haar_cascade():
    # Pakai path bawaan OpenCV agar robust di berbagai lingkungan
    haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        raise IOError("Gagal memuat Haar Cascade. Cek instalasi OpenCV.")
    return face_cascade

try:
    face_cascade = load_haar_cascade()
    print("OK: Viola–Jones (Haar) detector loaded.")
except Exception as e:
    print(f"Error load Haar: {e}")
    sys.exit(1)

# --- 3. LOADER GAMBAR & BUILD ENCODINGS (REKURSIF) -------------------------------
def load_image_rgb(path, max_side=1600):
    """
    Load gambar ke RGB np.uint8 (H, W, 3).
    - Perbaiki orientasi EXIF (umum di foto iPhone).
    - Resize jika sisi terpanjang > max_side (stabilkan & percepat deteksi).
    - HEIC/HEIF didukung jika pillow-heif terpasang.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
    return np.array(img)

def build_known_faces(dataset_dir, upsample=1, use_cnn_fallback=True):
    """
    REKURSIF menelusuri semua subfolder.
    Label = nama folder tempat file berada (basename dari folder yang berisi gambar).
    Deteksi wajah: HOG dulu; kalau 0 → fallback ke CNN (lebih lambat, lebih akurat).
    """
    known_encodings, known_names = [], []

    if not os.path.isdir(dataset_dir):
        print(f"Peringatan: folder dataset '{dataset_dir}' tidak ditemukan. Recognition akan 'Unknown'.")
        return known_encodings, known_names

    print(f"Memuat dataset (rekursif) dari: {os.path.abspath(dataset_dir)}")
    per_person_counts = {}
    files_total = 0

    for root, _, files in os.walk(dataset_dir):
        if not files:
            continue
        person_name = os.path.basename(root).strip()
        if not person_name:
            continue

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                continue

            files_total += 1
            img_path = os.path.join(root, fname)
            try:
                img = load_image_rgb(img_path)
            except Exception as e:
                print(f"[SKIP] {img_path}: {e}")
                continue

            # Deteksi lokasi wajah → lebih stabil jika dikirim ke face_encodings
            boxes = face_recognition.face_locations(img, number_of_times_to_upsample=upsample, model="hog")
            if not boxes and use_cnn_fallback:
                boxes = face_recognition.face_locations(img, number_of_times_to_upsample=upsample, model="cnn")

            if not boxes:
                print(f"[NO FACE] {img_path}")
                continue

            encs = face_recognition.face_encodings(img, known_face_locations=boxes)
            if not encs:
                print(f"[NO ENC]  {img_path}")
                continue

            # Ambil wajah terbesar jika >1
            if len(encs) > 1:
                areas = [abs((b[2]-b[0])*(b[1]-b[3])) for b in boxes]
                enc = encs[int(np.argmax(areas))]
            else:
                enc = encs[0]

            known_encodings.append(enc)
            known_names.append(person_name)
            per_person_counts[person_name] = per_person_counts.get(person_name, 0) + 1
            print(f"[OK] {img_path}")

    # Ringkasan
    print("\n== Ringkasan Dataset ==")
    for p, c in sorted(per_person_counts.items()):
        print(f"- {p}: {c} encoding")
    print(f"Total orang: {len(per_person_counts)} | File diproses: {files_total} | Total encoding: {len(known_encodings)}\n")
    return known_encodings, known_names

known_encodings, known_names = build_known_faces(DATASET_DIR, upsample=1, use_cnn_fallback=True)

# --- 4. VIDEO CAPTURE SETUP ------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam (index 0). Coba ganti index (1/2) atau cek izin kamera.")
    sys.exit(1)

# --- 5. UTILITAS -----------------------------------------------------------------
def clamp_bbox(x, y, w, h, W, H):
    """Pastikan bbox tetap di dalam frame."""
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def recognize_face_by_distance(rgb_frame, box_fr, known_encodings, known_names, threshold=DIST_THRESHOLD):
    """
    Recognition via CNN embedding:
    - Ambil embedding (face_recognition.face_encodings)
    - Hitung jarak (face_distance) ke semua known_encodings
    - Pilih jarak minimum; jika <= threshold → ambil labelnya, else "Unknown"
    """
    if not known_encodings:
        return "Unknown"

    encs = face_recognition.face_encodings(rgb_frame, [box_fr])
    if not encs:
        return "Unknown"

    face_encoding = encs[0]
    dists = face_recognition.face_distance(known_encodings, face_encoding)
    if len(dists) == 0:
        return "Unknown"

    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    return known_names[best_idx] if best_dist <= threshold else "Unknown"

# --- 6. MAIN LOOP ----------------------------------------------------------------
tracked_faces = []   # list of dict: {'box': (x,y,w,h), 'name': str}
prev_gray = None
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Peringatan: gagal membaca frame dari kamera. Menghentikan.")
            break

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Buffer untuk pembaruan daftar tracked_faces di frame ini saat re-detect
        new_tracked_faces = []

        # --- PART 1: DETECT + RECOGNIZE setiap N frame ---
        if frame_count % RE_DETECT_INTERVAL == 0:
            faces_vj = face_cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=MIN_SIZE
            )

            for (x, y, w, h) in faces_vj:
                # Ubah (x,y,w,h) ke (top, right, bottom, left) untuk face_recognition
                box_fr = (y, x + w, y + h, x)

                # CNN embedding untuk recognition (BUKAN deteksi) + threshold by distance
                name = recognize_face_by_distance(rgb_frame, box_fr, known_encodings, known_names, threshold=DIST_THRESHOLD)

                # Simpan & gambar (deteksi baru → kotak hijau)
                x, y, w, h = clamp_bbox(int(x), int(y), int(w), int(h), W, H)
                new_tracked_faces.append({'box': (x, y, w, h), 'name': name})

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} (Detected)", (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Perbarui master list setelah re-detection
            tracked_faces = new_tracked_faces

        # --- PART 2: TRACK dengan Farneback di frame antara re-detection ---
        else:
            if prev_gray is not None and tracked_faces:
                # Optical Flow Farneback (dense)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                for face in tracked_faces:
                    (x, y, w, h) = face['box']
                    name = face['name']

                    # Pastikan crop berada di dalam frame flow
                    x, y, w, h = clamp_bbox(int(x), int(y), int(w), int(h), W, H)

                    if y + h <= flow.shape[0] and x + w <= flow.shape[1] and w > 0 and h > 0:
                        region = flow[y:y + h, x:x + w]
                        if region.size > 0:
                            avg_dx = float(np.nanmean(region[:, :, 0]))
                            avg_dy = float(np.nanmean(region[:, :, 1]))
                            if np.isnan(avg_dx): avg_dx = 0.0
                            if np.isnan(avg_dy): avg_dy = 0.0

                            new_x = int(round(x + avg_dx))
                            new_y = int(round(y + avg_dy))
                            new_x, new_y, w, h = clamp_bbox(new_x, new_y, w, h, W, H)

                            face['box'] = (new_x, new_y, w, h)

                            # Gambar (tracking → kotak kuning)
                            cv2.rectangle(frame, (new_x, new_y), (new_x + w, new_y + h), (0, 255, 255), 2)
                            cv2.putText(frame, f"{name} (Tracking)", (new_x, max(0, new_y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- FINALIZE FRAME ---
        prev_gray = gray  # simpan frame abu untuk perhitungan flow berikutnya
        frame_count += 1

        cv2.imshow("FaceID: Haar + CNN Embedding + Farneback", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Pastikan resource dirilis meskipun terjadi error
    cap.release()
    cv2.destroyAllWindows()
