#!/usr/bin/env python3
"""
Complete Real-Time Facial Recognition System
Author: (Assistant)
Requirements: pip install opencv-python mtcnn deepface numpy scikit-learn pillow tensorflow==2.15.0
"""

import os

# Suppress TensorFlow warnings and oneDNN messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import sqlite3
import csv
import json
import time
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# ==================== CONFIGURATION ====================
CONFIG = {
    "threshold": 0.45,  # cosine distance threshold
    "camera_id": 0,
    "frame_skip": 3,  # process every 3rd frame
    "anti_spoofing": True,
    "attendance_cooldown": 60,  # seconds between same person attendance
    "db_path": "face_db.sqlite",
    "attendance_csv": "attendance.csv",
    "unknown_log": "unknown_log.csv",
    "embedding_model": "Facenet512"
}


# ==================== DATABASE MANAGER ====================
class FaceDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cache = {}  # name -> embedding (numpy array)
        self._init_db()
        self._load_cache()

    def _init_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                embedding BLOB,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def _load_cache(self):
        cur = self.conn.execute('SELECT name, embedding FROM faces')
        for name, blob in cur.fetchall():
            emb = np.frombuffer(blob, dtype=np.float64)
            self.cache[name] = emb

    def add_face(self, name, embedding):
        self.cache[name] = embedding
        blob = embedding.tobytes()
        self.conn.execute('INSERT OR REPLACE INTO faces (name, embedding) VALUES (?, ?)', (name, blob))
        self.conn.commit()

    def delete_face(self, name):
        if name in self.cache:
            del self.cache[name]
        self.conn.execute('DELETE FROM faces WHERE name = ?', (name,))
        self.conn.commit()

    def get_all_names(self):
        return list(self.cache.keys())

    def get_all_embeddings(self):
        return np.array(list(self.cache.values())) if self.cache else np.empty((0, 512))

    def close(self):
        self.conn.close()


# ==================== ATTENDANCE LOGGER ====================
class AttendanceLogger:
    def __init__(self, csv_path, cooldown_sec):
        self.csv_path = csv_path
        self.cooldown = cooldown_sec
        self.last_marked = {}
        self._ensure_csv()

    def _ensure_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Timestamp"])

    def mark(self, name):
        now = datetime.now()
        if name in self.last_marked:
            diff = (now - self.last_marked[name]).total_seconds()
            if diff < self.cooldown:
                return False
        self.last_marked[name] = now
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")])
        return True

    def get_stats(self):
        if not os.path.exists(self.csv_path):
            return {}
        stats = {}
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 1:
                    name = row[0]
                    stats[name] = stats.get(name, 0) + 1
        return stats


# ==================== UNKNOWN LOGGER ====================
class UnknownLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp"])

    def log(self):
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


# ==================== ANTI-SPOOFING ====================
class LivenessDetector:
    def __init__(self, use_deepface=True):
        self.use_deepface = use_deepface

    def is_live(self, face_img_rgb):
        """
        Simple check using DeepFace's anti_spoofing.
        Returns True if real face, False if spoof/photo.
        """
        if not self.use_deepface:
            return True
        try:
            # DeepFace.extract_faces has anti_spoofing parameter
            result = DeepFace.extract_faces(
                img_path=face_img_rgb,
                detector_backend='skip',  # already have face crop
                anti_spoofing=True
            )
            # If it raises or returns spoof flag, handle
            if result and isinstance(result, list) and len(result) > 0:
                # Depending on version, it may have 'is_real' key
                return result[0].get('is_real', True)
            return True
        except Exception as e:
            # If anti_spoofing fails, assume real to avoid blocking
            print(f"Liveness check error: {e}")
            return True


# ==================== FACE RECOGNITION PIPELINE ====================
class FaceRecognitionPipeline:
    def __init__(self, config):
        self.config = config

        # Initialize MTCNN with error handling for different versions
        try:
            # Try with parameters (newer versions)
            self.detector = MTCNN(min_face_size=30, thresholds=[0.6, 0.7, 0.7])
        except TypeError:
            try:
                # Try without min_face_size
                self.detector = MTCNN(thresholds=[0.6, 0.7, 0.7])
            except TypeError:
                # Fallback to default
                self.detector = MTCNN()

        self.db = FaceDatabase(config["db_path"])
        self.attendance = AttendanceLogger(config["attendance_csv"], config["attendance_cooldown"])
        self.unknown_logger = UnknownLogger(config["unknown_log"])
        self.liveness = LivenessDetector(use_deepface=config["anti_spoofing"])
        self.threshold = config["threshold"]
        self.frame_skip = config["frame_skip"]
        self.frame_counter = 0

    def get_embedding(self, face_img_bgr):
        """Extract 512-d embedding from face crop (BGR)."""
        # Convert BGR to RGB for DeepFace
        face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        try:
            embedding = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.config["embedding_model"],
                detector_backend='mtcnn',  # Use MTCNN to avoid RetinaFace issues
                enforce_detection=False
            )[0]['embedding']
            return np.array(embedding)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def match(self, emb):
        """Return (name, distance) or (None, inf) if unknown."""
        if len(self.db.cache) == 0:
            return None, float('inf')
        known_embs = self.db.get_all_embeddings()
        if len(known_embs) == 0:
            return None, float('inf')

        similarities = cosine_similarity([emb], known_embs)[0]
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        dist = 1 - best_sim
        if dist < self.threshold:
            name = self.db.get_all_names()[best_idx]
            return name, dist
        return None, dist

    def process_frame(self, frame_bgr):
        """
        Detect faces, compute embeddings, match, draw results.
        Returns annotated frame.
        """
        self.frame_counter += 1
        process = (self.frame_counter % self.frame_skip == 0)

        # Always detect faces (fast)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            detections = self.detector.detect_faces(rgb)
        except Exception as e:
            print(f"Detection error: {e}")
            return frame_bgr

        for det in detections:
            x, y, w, h = det['box']
            # Ensure coordinates are within frame
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame_bgr.shape[1] - x), min(h, frame_bgr.shape[0] - y)
            if w <= 0 or h <= 0:
                continue
            face_crop = frame_bgr[y:y + h, x:x + w]
            if face_crop.size == 0:
                continue

            # Default label & color
            label = "Unknown"
            color = (0, 255, 255)  # yellow
            is_known = False

            # Anti-spoofing (only if processing this frame)
            spoof = False
            if process and self.config["anti_spoofing"]:
                try:
                    if not self.liveness.is_live(face_crop):
                        spoof = True
                        label = "SPOOF"
                        color = (0, 0, 255)  # red
                except Exception as e:
                    print(f"Liveness check error: {e}")

            if not spoof and process:
                emb = self.get_embedding(face_crop)
                if emb is not None:
                    name, dist = self.match(emb)
                    if name is not None:
                        label = f"{name} ({1 - dist:.2f})"
                        color = (0, 255, 0)  # green
                        is_known = True
                        # Mark attendance
                        try:
                            self.attendance.mark(name)
                        except Exception as e:
                            print(f"Attendance error: {e}")
                    else:
                        # Log unknown
                        try:
                            self.unknown_logger.log()
                        except Exception as e:
                            print(f"Unknown logger error: {e}")

            # Draw bounding box and label
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_bgr, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame_bgr

    def register_user(self, name, num_samples=10):
        """Capture num_samples from webcam, average embedding, store."""
        cap = cv2.VideoCapture(self.config["camera_id"])
        if not cap.isOpened():
            raise Exception("Cannot open camera for registration")

        embeddings = []
        sample_count = 0

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            # Show instruction on frame
            cv2.putText(frame, f"Look at camera. Samples: {sample_count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("Register - Press SPACE to capture, ESC to cancel", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # space
                # Detect face
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    faces = self.detector.detect_faces(rgb)
                except Exception as e:
                    print(f"Detection error during registration: {e}")
                    continue

                if len(faces) == 0:
                    cv2.putText(frame, "No face detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Register", frame)
                    cv2.waitKey(500)
                    continue

                # Use first face
                x, y, w, h = faces[0]['box']
                x, y = max(0, x), max(0, y)
                face_crop = frame[y:y + h, x:x + w]
                emb = self.get_embedding(face_crop)
                if emb is not None:
                    embeddings.append(emb)
                    sample_count += 1
                    print(f"Captured sample {sample_count}/{num_samples}")
                else:
                    cv2.putText(frame, "Embedding failed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Register", frame)
                    cv2.waitKey(500)
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                raise Exception("Registration cancelled")

        cap.release()
        cv2.destroyAllWindows()

        if len(embeddings) == 0:
            raise Exception("No embeddings captured")

        avg_embedding = np.mean(embeddings, axis=0)
        # Normalize (optional but helps)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        self.db.add_face(name, avg_embedding)
        return True

    def get_stats(self):
        return self.attendance.get_stats()

    def get_known_names(self):
        return self.db.get_all_names()

    def delete_user(self, name):
        self.db.delete_face(name)

    def close(self):
        self.db.close()


# ==================== TKINTER GUI ====================
class FaceRecognitionGUI:
    def __init__(self, root, pipeline):
        self.root = root
        self.pipeline = pipeline
        self.root.title("Real-Time Face Recognition System")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Video capture
        self.cap = None
        self.running = False
        self.after_id = None

        # UI Components
        self.create_widgets()

        # Start video automatically
        self.start_video()

    def create_widgets(self):
        # Left frame: video feed
        left_frame = Frame(self.root, width=800, height=600)
        left_frame.pack(side=LEFT, padx=10, pady=10, fill=BOTH, expand=True)
        self.video_label = Label(left_frame, bg="black")
        self.video_label.pack(fill=BOTH, expand=True)

        # Right frame: controls and logs
        right_frame = Frame(self.root, width=350)
        right_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)

        # Title
        Label(right_frame, text="Face Recognition Control", font=("Arial", 16)).pack(pady=5)

        # Threshold slider
        threshold_frame = Frame(right_frame)
        threshold_frame.pack(pady=5, fill=X)
        Label(threshold_frame, text="Matching Threshold:").pack(side=LEFT)
        self.threshold_var = DoubleVar(value=self.pipeline.threshold)
        self.threshold_slider = Scale(threshold_frame, from_=0.1, to=0.8, orient=HORIZONTAL,
                                      variable=self.threshold_var, command=self.update_threshold)
        self.threshold_slider.pack(side=RIGHT, fill=X, expand=True)

        # Anti-spoofing toggle
        self.spoof_var = BooleanVar(value=self.pipeline.config["anti_spoofing"])
        Checkbutton(right_frame, text="Enable Anti-Spoofing", variable=self.spoof_var,
                    command=self.toggle_spoofing).pack(pady=5)

        # Buttons
        btn_frame = Frame(right_frame)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Register New User", command=self.register_user_dialog,
               bg="lightblue", width=20).pack(pady=5)
        Button(btn_frame, text="Delete User", command=self.delete_user_dialog,
               bg="lightcoral", width=20).pack(pady=5)
        Button(btn_frame, text="View Attendance Stats", command=self.show_stats,
               bg="lightgreen", width=20).pack(pady=5)
        Button(btn_frame, text="Stop/Start Recognition", command=self.toggle_recognition,
               bg="orange", width=20).pack(pady=5)

        # Log area
        Label(right_frame, text="Recognition Log", font=("Arial", 12)).pack(pady=5)
        self.log_text = Text(right_frame, height=15, width=40)
        self.log_text.pack(fill=BOTH, expand=True)
        scrollbar = Scrollbar(self.log_text)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

        # Known users list
        Label(right_frame, text="Known Users", font=("Arial", 12)).pack(pady=5)
        self.user_listbox = Listbox(right_frame, height=6)
        self.user_listbox.pack(fill=BOTH, expand=True)
        self.refresh_user_list()

    def refresh_user_list(self):
        self.user_listbox.delete(0, END)
        for name in self.pipeline.get_known_names():
            self.user_listbox.insert(END, name)

    def log_message(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(END, f"[{timestamp}] {msg}\n")
        self.log_text.see(END)

    def update_threshold(self, val):
        self.pipeline.threshold = self.threshold_var.get()
        self.log_message(f"Threshold set to {self.pipeline.threshold:.2f}")

    def toggle_spoofing(self):
        self.pipeline.config["anti_spoofing"] = self.spoof_var.get()
        state = "ON" if self.spoof_var.get() else "OFF"
        self.log_message(f"Anti-spoofing {state}")

    def register_user_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Register New User")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()

        Label(dialog, text="Enter name:", font=("Arial", 11)).pack(pady=10)
        name_entry = Entry(dialog, width=25)
        name_entry.pack(pady=5)

        Label(dialog, text="Number of samples (5-20):", font=("Arial", 11)).pack(pady=5)
        samples_spin = Spinbox(dialog, from_=5, to=20, width=10)
        samples_spin.pack(pady=5)

        def register():
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Name cannot be empty")
                return
            if name in self.pipeline.get_known_names():
                messagebox.showerror("Error", "User already exists")
                return
            num = int(samples_spin.get())
            dialog.destroy()

            try:
                self.log_message(f"Starting registration for {name} (samples={num})")
                # Temporarily stop video capture to use camera
                was_running = self.running
                if was_running:
                    self.stop_video()

                # Give time for camera to release
                time.sleep(0.5)

                self.pipeline.register_user(name, num_samples=num)
                self.log_message(f"User {name} registered successfully")
                self.refresh_user_list()

                if was_running:
                    self.start_video()
            except Exception as e:
                messagebox.showerror("Registration Failed", str(e))
                self.log_message(f"Registration failed: {e}")
                # Try to restart video if it was running
                if was_running:
                    self.start_video()

        Button(dialog, text="Start Registration", command=register, bg="lightgreen", width=20).pack(pady=20)

    def delete_user_dialog(self):
        selected = self.user_listbox.curselection()
        if not selected:
            messagebox.showwarning("No selection", "Please select a user to delete")
            return
        name = self.user_listbox.get(selected[0])
        if messagebox.askyesno("Confirm Delete", f"Delete user '{name}'?"):
            self.pipeline.delete_user(name)
            self.log_message(f"User {name} deleted")
            self.refresh_user_list()

    def show_stats(self):
        stats = self.pipeline.get_stats()
        if not stats:
            messagebox.showinfo("Attendance Stats", "No attendance records yet.")
            return

        # Create a formatted message
        top = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        msg = "📊 Attendance Statistics 📊\n\n"
        msg += f"Total Recognitions: {sum(stats.values())}\n"
        msg += f"Unique People: {len(stats)}\n\n"
        msg += "Most Frequent:\n"
        msg += "━" * 30 + "\n"
        for name, count in top[:10]:
            msg += f"• {name}: {count} times\n"

        messagebox.showinfo("Statistics", msg)

    def toggle_recognition(self):
        if self.running:
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.pipeline.config["camera_id"])
            if not self.cap.isOpened():
                self.log_message("ERROR: Cannot open camera")
                messagebox.showerror("Camera Error", "Cannot access webcam. Please check your camera.")
                return

        self.running = True
        self.log_message("Recognition started")
        self.video_loop()

    def stop_video(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        self.log_message("Recognition stopped")

    def video_loop(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            try:
                # Process frame
                processed = self.pipeline.process_frame(frame)
                # Convert to RGB for Tkinter
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                # Resize to fit the label while maintaining aspect ratio
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            except Exception as e:
                print(f"Video processing error: {e}")

        self.after_id = self.root.after(30, self.video_loop)  # ~33 fps

    def on_close(self):
        self.stop_video()
        self.pipeline.close()
        self.root.destroy()


# ==================== MAIN ENTRY ====================
def main():
    # Load config from file if exists
    config = CONFIG.copy()
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                user_config = json.load(f)
                config.update(user_config)
                print("Loaded configuration from config.json")
        except Exception as e:
            print(f"Error loading config: {e}")

    # Create pipeline
    try:
        pipeline = FaceRecognitionPipeline(config)
        print("Face Recognition Pipeline initialized successfully")
        print(f"Known users: {len(pipeline.get_known_names())}")
        print(f"Anti-spoofing: {config['anti_spoofing']}")
        print(f"Threshold: {config['threshold']}")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        messagebox.showerror("Initialization Error", f"Failed to initialize system: {e}")
        return

    # Start GUI
    root = Tk()
    app = FaceRecognitionGUI(root, pipeline)
    root.mainloop()


if __name__ == "__main__":
    main()