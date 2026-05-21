# Face Recognition Attendance System

An advanced real-time facial recognition attendance system built using Python, OpenCV, MTCNN, DeepFace, and Tkinter. This project automates attendance tracking using live webcam face recognition with anti-spoofing support and a user-friendly GUI interface.

---

## 🚀 Features

* 🎥 Real-time face recognition using webcam
* 👤 Register new users directly from camera
* 🧠 DeepFace + FaceNet512 embeddings
* 🛡️ Anti-spoofing / liveness detection
* 📊 Automatic attendance logging
* 🗂️ SQLite database integration
* 📁 CSV attendance export
* 🖥️ Interactive Tkinter GUI
* ⚡ Optimized frame processing for better performance
* ❌ Unknown face logging system

---

## 🛠️ Technologies Used

* Python
* OpenCV
* DeepFace
* MTCNN
* TensorFlow
* SQLite
* Tkinter
* NumPy
* Scikit-learn
* Pillow

---

## 📂 Project Structure

```bash id="v4x5p7"
Face-Recognition-Attendence-System/
│
├── face_recognition_system.py
├── requirements.txt
├── attendance.csv
├── unknown_log.csv
├── face_db.sqlite
├── config.json
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash id="8z9v6k"
git clone https://github.com/Siddharth190/Face-Recognition-Attendence-System.git
cd Face-Recognition-Attendence-System
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash id="zcn6qy"
python -m venv venv
```

### Activate Environment

#### Windows

```bash id="j3km68"
venv\Scripts\activate
```

#### Linux / macOS

```bash id="vl4l8k"
source venv/bin/activate
```

### 3️⃣ Install Requirements

```bash id="wlw3cr"
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash id="5tnd8r"
python face_recognition_system.py
```

---

## 📸 How It Works

1. Webcam captures live video feed.
2. MTCNN detects faces in each frame.
3. DeepFace generates facial embeddings.
4. Cosine similarity compares embeddings with stored users.
5. Recognized users are marked present automatically.
6. Attendance is saved into a CSV file.

---

## 🧠 Face Recognition Pipeline

* **Face Detection:** MTCNN
* **Face Embeddings:** FaceNet512
* **Matching Algorithm:** Cosine Similarity
* **Database:** SQLite
* **Anti-Spoofing:** DeepFace Liveness Detection

---

## 📊 Attendance System

Attendance records are stored in:

```bash id="7skruv"
attendance.csv
```

Unknown faces are logged in:

```bash id="v8a8l4"
unknown_log.csv
```

---

## 🛡️ Anti-Spoofing

The system includes a basic liveness detection feature using DeepFace anti-spoofing support to reduce fake image/photo attacks.

---

## 🔮 Future Improvements

* 🌐 Web dashboard integration
* ☁️ Cloud database support
* 📱 Mobile app support
* 🎭 Multi-face tracking optimization
* 🚀 GPU acceleration
* 🔔 Email/SMS attendance notifications

---

## 📌 Applications

* Smart Attendance Systems
* College/School Attendance
* Office Employee Tracking
* Secure Access Systems
* Visitor Management

---

## 🖼️ Screenshots

*Add screenshots of your GUI and recognition system here.*

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss improvements.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed by **Siddharth Shetty**

GitHub Repository:
[Face-Recognition-Attendence-System](https://github.com/Siddharth190/Face-Recognition-Attendence-System)

---

Built with Python, debugging sessions at 2 AM, and a webcam that sometimes decides it has stage fright. 📷😅
