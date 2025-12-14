# NeuraFace 🧠📷  
**AI-Based Face Recognition Attendance System**

NeuraFace is a desktop application that uses computer vision and deep learning to automatically recognize faces and record attendance in real time. It eliminates manual attendance by identifying individuals through live webcam input and securely storing attendance data.

---

## ✨ Features
- Real-time face detection and recognition
- Student registration with facial embedding storage
- Automated attendance marking
- Admin-controlled access
- Attendance viewing with database records
- Clean and responsive desktop GUI

---

## 🛠️ Tech Stack
- **Python**
- **PySide6 (Qt for Python)** – GUI
- **OpenCV** – Webcam & image processing
- **DeepFace (ArcFace)** – Face recognition
- **NumPy** – Numerical computations
- **SQLite** – Local database

---

## 🚀 How It Works
1. Students are registered by capturing their face via webcam.
2. Facial embeddings are generated using the ArcFace model.
3. During scanning, live faces are matched against stored embeddings.
4. Attendance is automatically recorded in the database.

---

## ▶️ Installation & Run

```bash
pip install pyside6 opencv-python deepface numpy
python neuraface_final.py
