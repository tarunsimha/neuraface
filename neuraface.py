import sys
import cv2
import sqlite3
import numpy as np
from deepface import DeepFace
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSpacerItem, QGraphicsDropShadowEffect, QLineEdit, QMessageBox,
    QTableWidget, QTableWidgetItem, QDateEdit, QComboBox,
)
from PySide6.QtCore import Qt, QTimer, QDate
from PySide6.QtGui import QPixmap, QFont, QImage, QColor, Qt

DB = "database.db"
THRESHOLD = 4.0

def extract_face(frame):
    try:
        det = DeepFace.extract_faces(frame, detector_backend="opencv")[0]
        region = det["facial_area"]

        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        crop = frame[y:y + h, x:x + w]

        return crop
    except:
        return None


def get_embedding(face):
    try:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        emb = DeepFace.represent(
            face_rgb,
            model_name="ArcFace",
            detector_backend="skip"
        )[0]["embedding"]

        return emb
    except:
        return None


def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            image BLOB NOT NULL,
            embedding BLOB NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            student_id TEXT REFERENCES students(student_id),
            attendance_date TEXT DEFAULT (DATE('now')),
            is_present BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (student_id, attendance_date)
        );
    """)

    conn.commit()
    conn.close()


def save_student_to_db(id, name, image, embedding):
    embedding = np.array(embedding, dtype=np.float32).tobytes()

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO students (student_id, student_name, image, embedding)
        VALUES (?, ?, ?, ?);
    """, (id, name, image, embedding))

    conn.commit()
    conn.close()


def load_all_students_faces():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT student_id, student_name, embedding FROM students")
    rows = cur.fetchall()

    conn.close()

    ids, names, embeddings = [], [], []

    for sid, name, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        ids.append(sid)
        names.append(name)
        embeddings.append(emb)

    return ids, names, np.vstack(embeddings)


def save_student_attendance(student_id):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO attendance (student_id, is_present)
        VALUES (?, TRUE)
    """, (student_id,))

    conn.commit()
    conn.close()


def get_today_attendance():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            s.student_name,
            COALESCE(a.is_present, FALSE) as is_present
        FROM students s
        LEFT JOIN attendance a ON (
            s.student_id = a.student_id 
            AND a.attendance_date = DATE('now')
        )
        ORDER BY s.student_name
    """)

    return cur.fetchall()


init_db()
try:
    ids_, names_, embeddings_ = load_all_students_faces()
except ValueError:
    print("No data in database")


class NeuraFaceHome(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuraFace")

        self.setStyleSheet("background-color: #C7F3FF;")
        self.build_ui()

    def build_ui(self):
        central = QWidget()
        central.setObjectName("bg")
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QWidget#bg {
                background-color: #C7F3FF;
            }
            QLabel {
                color: black;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 40, 0, 40)
        layout.setSpacing(0)
        central.setLayout(layout)

        title = QLabel("NeuraFace", self)
        title.setFont(QFont("Didot", 82, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #000000;")
        layout.addWidget(title)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #DFFBFF,
                    stop:0.5 #C2F1FF,
                    stop:1 #B3E8FF
                );
            }
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))
        title.setGraphicsEffect(shadow)

        layout.addSpacerItem(QSpacerItem(20, 100))

        main_button = QPushButton("Start Scan")
        main_button.setFixedSize(330, 120)
        main_button.setFont(QFont("Didot", 22, QFont.DemiBold))
        main_button.setCursor(Qt.PointingHandCursor)
        main_button.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 32px;
                color: black;
                background-color: qlineargradient(
                    x1:0, y1:0,   x2:1, y2:1,
                    stop:0    #FFF2B0,
                    stop:0.33 #FFD9C2,
                    stop:0.66 #F7C4E8,
                    stop:1    #E5B7FF
                );
            }
        """)

        main_shadow = QGraphicsDropShadowEffect()
        main_shadow.setBlurRadius(25)
        main_shadow.setOffset(0, 8)
        main_shadow.setColor(QColor(0, 0, 0, 70))
        main_button.setGraphicsEffect(main_shadow)
        main_button.clicked.connect(self.open_scan)

        main_shadow = QGraphicsDropShadowEffect()
        main_shadow.setBlurRadius(60)
        main_shadow.setOffset(0, 20)
        main_shadow.setColor(QColor(0, 0, 0, 35))
        main_button.setGraphicsEffect(main_shadow)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(main_button)
        row.addStretch()
        layout.addLayout(row)

        layout.addSpacerItem(QSpacerItem(20, 70))

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(80, 0, 80, 0)
        bottom_row.setSpacing(80)

        left_button = QPushButton("Register Student")
        left_button.setFixedSize(260, 75)
        left_button.setFont(QFont("Didot", 14, QFont.DemiBold))
        left_button.setCursor(Qt.PointingHandCursor)
        left_button.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 36px;
                color: black;
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6C9BFF,
                    stop:1 #FF7ABF
                );
            }
        """)
        left_button.clicked.connect(self.open_admin_login)

        left_shadow = QGraphicsDropShadowEffect()
        left_shadow.setBlurRadius(18)
        left_shadow.setOffset(0, 5)
        left_shadow.setColor(QColor(0, 0, 0, 55))
        left_button.setGraphicsEffect(left_shadow)

        right_button = QPushButton("View Attendance")
        right_button.setFixedSize(260, 75)
        right_button.setFont(QFont("Didot", 14, QFont.DemiBold))
        right_button.setCursor(Qt.PointingHandCursor)
        right_button.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 36px;
                color: black;
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FF7ABF,
                    stop:1 #6C9BFF
                );
            }
        """)
        right_button.clicked.connect(self.open_attendance)

        right_shadow = QGraphicsDropShadowEffect()
        right_shadow.setBlurRadius(18)
        right_shadow.setOffset(0, 5)
        right_shadow.setColor(QColor(0, 0, 0, 55))
        right_button.setGraphicsEffect(right_shadow)

        bottom_row.addWidget(left_button)
        bottom_row.addWidget(right_button)
        layout.addLayout(bottom_row)

        layout.addSpacerItem(QSpacerItem(20, 20))

    def open_admin_login(self):
        self.admin_window = AdminLoginWindow(parent=self)
        self.admin_window.show()
        self.hide()

    def open_scan(self):
        self.scan_window = ScanWindow(self)
        self.scan_window.show()
        self.hide()

    def open_attendance(self):
        self.attendance_window = AttendanceWindow(self)
        self.attendance_window.show()
        self.hide()

    def showEvent(self, event):
        super().showEvent(event)
        self.move(0, 0)


class AdminLoginWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.is_back_navigation = False
        self.setWindowTitle("NeuraFace - Admin Login")
        self.showMaximized()

        central = QWidget()
        central.setObjectName("bg")

        layout = QVBoxLayout()
        layout.setSpacing(35)
        layout.setContentsMargins(0, 60, 0, 60)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QWidget#bg {
                background-color: #C7F3FF;
            }
            QLabel {
                color: black;
            }
        """)

        back_btn = QPushButton("←  Back")
        back_btn.setFixedSize(110, 38)
        back_btn.setFont(QFont("Arial", 13))
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                color: #333333;
            }
            QPushButton:hover {
                color: black;
            }
        """)
        back_btn.clicked.connect(self.go_back)

        back_container = QHBoxLayout()
        back_container.addWidget(back_btn, alignment=Qt.AlignLeft)
        back_container.addStretch()
        layout.addLayout(back_container)

        title = QLabel("NeuraFace")
        title.setFont(QFont("Didot", 58, QFont.Normal))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: black;")
        layout.addWidget(title)

        subtitle = QLabel("Enter Admin Credentials to register a new student")
        subtitle.setFont(QFont("Arial", 18))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #444444;")
        layout.addWidget(subtitle)

        layout.addSpacing(40)

        user_label = QLabel("Admin Username:")
        user_label.setFont(QFont("Arial", 16))
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("color: #333333;")
        layout.addWidget(user_label)

        self.username_input = QLineEdit()
        self.username_input.setFixedSize(480, 48)
        self.username_input.setStyleSheet("""
            QLineEdit {
                background: white;
                color:black;
                border-radius: 24px;
                padding-left: 18px;
                padding-right: 18px;
                font-size: 16px;
            }
        """)
        layout.addWidget(self.username_input, alignment=Qt.AlignCenter)

        layout.addSpacing(30)

        pass_label = QLabel("Password:")
        pass_label.setFont(QFont("Arial", 16))
        pass_label.setStyleSheet("color: #333333;")
        pass_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(pass_label)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFixedSize(480, 48)
        self.password_input.setStyleSheet("""
            QLineEdit {
                background: white;
                color:black;
                border-radius: 24px;
                padding-left: 18px;
                padding-right: 18px;
                font-size: 16px;
            }
        """)
        layout.addWidget(self.password_input, alignment=Qt.AlignCenter)

        layout.addSpacing(40)

        login_button = QPushButton("Login")
        login_button.setFixedSize(200, 55)
        login_button.setFont(QFont("Arial", 17, QFont.DemiBold))
        login_button.setCursor(Qt.PointingHandCursor)
        login_button.setStyleSheet("""
            QPushButton {
                background: white;
                border-radius: 27px;
                color: #333333;
                border: 1px solid #E0E0E0;
            }
            QPushButton:hover {
                background: #F5F5F5;
            }
        """)
        login_button.clicked.connect(self.check_login)
        layout.addWidget(login_button, alignment=Qt.AlignCenter)

        layout.addStretch()

    def check_login(self):
        user = self.username_input.text().strip()
        pwd = self.password_input.text().strip()

        if user == "admin123" and pwd == "1234":
            self.capture_window = NewStudentCaptureWindow(parent=self.parent_window)
            self.capture_window.show()
            self.hide()
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Incorrect username or password.")
            msg.setIcon(QMessageBox.Warning)

            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2B2B2B;
                }
                QLabel {
                    color: white;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #444444;
                    color: white;
                    border-radius: 8px;
                    padding: 6px 14px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
            """)

            msg.exec()

    def closeEvent(self, event):
        if not self.is_back_navigation:
            QApplication.quit()
        event.accept()

    def go_back(self):
        self.is_back_navigation = True

        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        self.close()
        if self.parent_window:
            self.parent_window.show()

    def showEvent(self, event):
        super().showEvent(event)
        self.move(0, 0)


class NewStudentCaptureWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.is_back_navigation = False

        self.setWindowTitle("Register New Student")
        self.showMaximized()

        self.current_frame = None
        self.cam_index = 0

        central = QWidget()
        central.setObjectName("bg")
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QWidget#bg {
                background-color: #C7F3FF;
            }
            QLabel {
                color: black;
            }
        """)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(20)

        back_btn = QPushButton("← Back")
        back_btn.setFixedSize(90, 32)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                color: #333333;
            }
            QPushButton:hover {
                color: black;
            }
        """)
        back_btn.clicked.connect(self.go_back)

        self.cam_selector = QComboBox()
        self.cam_selector.setFixedSize(189, 47)
        self.cam_selector.addItems(["Camera 0", "Camera 1", "Camera 2", "Camera 3"])
        self.cam_selector.currentIndexChanged.connect(self.change_camera)
        self.cam_selector.setStyleSheet("""
            QComboBox {
                border-radius: 10px;
                padding-left: 10px;
                font-size: 14px;
            }
        """)

        top_row = QHBoxLayout()
        top_row.addWidget(back_btn)
        top_row.addStretch()
        top_row.addWidget(self.cam_selector)
        layout.addLayout(top_row)

        title = QLabel("NeuraFace")
        title.setFont(QFont("Didot", 38, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)
        layout.addWidget(title)

        box_container = QWidget()
        box_container.setFixedSize(1000, 550)
        box_container.setStyleSheet("""
            QWidget {
                border: 8px solid #2E2E2E;
                border-radius: 60px;
                background-color: transparent;
            }
        """)

        box_layout = QVBoxLayout(box_container)
        box_layout.setContentsMargins(30, 30, 30, 30)

        self.video_label = QLabel("Camera Loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background: rgba(255,255,255,0.3); border-radius: 20px;"
        )
        box_layout.addWidget(self.video_label)

        cam_row = QHBoxLayout()
        cam_row.addStretch()
        cam_row.addWidget(box_container, alignment=Qt.AlignCenter)
        cam_row.addStretch()
        layout.addLayout(cam_row)

        submit_btn = QPushButton("Submit")
        submit_btn.setFixedSize(180, 55)
        submit_btn.setCursor(Qt.PointingHandCursor)
        submit_btn.clicked.connect(self.register_student)
        submit_btn.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 20px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFF2B0,
                    stop:1 #E5B7FF
                );
                font-size: 16px;
                font-weight: bold;
            }
        """)

        submit_row = QHBoxLayout()
        submit_row.addStretch()
        submit_row.addWidget(submit_btn)
        submit_row.addStretch()
        layout.addLayout(submit_row)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Name")
        self.name_input.setFixedSize(300, 55)

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("ID No.")
        self.id_input.setFixedSize(300, 55)

        for w in (self.name_input, self.id_input):
            w.setStyleSheet("""
                QLineEdit {
                    background: white;
                    color: black;
                    border-radius: 20px;
                    padding-left: 20px;
                    font-size: 16px;
                }
            """)

        form_row = QHBoxLayout()
        form_row.addStretch()
        form_row.addWidget(self.name_input)
        form_row.addSpacing(30)
        form_row.addWidget(self.id_input)
        form_row.addStretch()
        layout.addLayout(form_row)

        self.cap = cv2.VideoCapture(self.cam_index)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def change_camera(self, index):
        self.cam_index = index
        self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Error", f"Camera {index} not available.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                     rgb.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def register_student(self):
        name = self.name_input.text().strip()
        sid = self.id_input.text().strip()

        if not name or not sid:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Fill all Fields")
            msg.setIcon(QMessageBox.Warning)

            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2B2B2B;
                }
                QLabel {
                    color: white;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #444444;
                    color: white;
                    border-radius: 8px;
                    padding: 6px 14px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
            """)

            msg.exec()

            return

        face = extract_face(self.current_frame)
        if face is None:
            msg=QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("No face Detected")
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet("""
            QMessageBox{
                background-color: #2B2B2B;
            }
            QLabel{
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #444444;
                color: white;
                border-radius: 8px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #666666
            }
            """)
            return

        emb = get_embedding(face)
        ok, buf = cv2.imencode(".png", face)
        save_student_to_db(sid, name, buf.tobytes(), emb)

        QMessageBox.information(self, "Success", f"Registered {name}")
        self.name_input.clear()
        self.id_input.clear()

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        if self.parent_window:
            self.parent_window.close()
        if not self.is_back_navigation:
            QApplication.quit()
        event.accept()

    def go_back(self):
        self.is_back_navigation = True

        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        self.hide()
        self.parent_window.show()

    def showEvent(self, event):
        super().showEvent(event)
        self.move(0, 0)


class AttendanceWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.is_back_navigation = False

        self.setWindowTitle("Attendance Viewer")
        self.showMaximized()

        central = QWidget()
        central.setObjectName("bg")
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QWidget#bg {
                background-color: #C7F3FF;
            }
            QLabel {
                color: black;
                background: transparent;
            }
            QPushButton {
                color: white;
                font-size: 14px;
                background-color: black;
            }
        """)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(18)

        back_btn = QPushButton("← Back")
        back_btn.setFixedSize(100, 36)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self.go_back)
        back_btn.setStyleSheet("""
            QPushButton {
                color: black;
                background: transparent;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)

        top_row = QHBoxLayout()
        top_row.addWidget(back_btn)
        top_row.addStretch()
        layout.addLayout(top_row)

        date_row = QHBoxLayout()
        date_row.setSpacing(12)

        date_label = QLabel("Select Date:")
        date_label.setFixedWidth(90)

        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setFixedSize(150, 36)
        self.date_edit.dateChanged.connect(self.load_date_attendance)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_date_attendance)

        date_row.addWidget(date_label)
        date_row.addWidget(self.date_edit)
        date_row.addWidget(load_btn)
        date_row.addStretch()

        layout.addLayout(date_row)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        sql_row = QHBoxLayout()
        sql_row.setSpacing(12)

        sql_label = QLabel("SQL Query:")

        self.sql_input = QLineEdit()
        self.sql_input.setFixedHeight(36)
        self.sql_input.returnPressed.connect(self.execute_sql)

        exec_btn = QPushButton("Execute")
        exec_btn.clicked.connect(self.execute_sql)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_table)

        sql_row.addWidget(sql_label)
        sql_row.addWidget(self.sql_input)
        sql_row.addWidget(exec_btn)
        sql_row.addWidget(clear_btn)

        layout.addLayout(sql_row)

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.status_label)

        self.load_date_attendance()

    def load_date_attendance(self):
        date = self.date_edit.date().toString("yyyy-MM-dd")
        self.status_label.setText(f"Loading {date}...")

        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT s.student_name,
                   COALESCE(a.is_present, 0),
                   CASE WHEN a.is_present THEN '✅ Present' ELSE '❌ Absent' END
            FROM students s
            LEFT JOIN attendance a
            ON s.student_id = a.student_id AND a.attendance_date = ?
            ORDER BY s.student_name
        """, (date,))
        data = cur.fetchall()
        conn.close()

        self.populate_table(data, ["Name", "Present", "Status"])
        self.status_label.setText(f"{len(data)} records loaded")

    def execute_sql(self):
        query = self.sql_input.text().strip()
        if not query:
            return
        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute(query)
            data = cur.fetchall()
            cols = [d[0] for d in cur.description]
            conn.close()
            self.populate_table(data, cols)
        except Exception as e:
            QMessageBox.critical(self, "SQL Error", str(e))

    def populate_table(self, data, columns):
        self.table.setRowCount(len(data))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        for r, row in enumerate(data):
            for c, val in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))

        self.table.resizeColumnsToContents()

    def clear_table(self):
        self.table.clear()
        self.status_label.setText("Cleared")

    def showEvent(self, event):
        super().showEvent(event)
        self.move(0, 0)

    def go_back(self):
        self.is_back_navigation = True

        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        
        self.close()
        self.parent_window.show()

    def closeEvent(self, event):
        if self.parent_window:
            self.parent_window.close()
        if not self.is_back_navigation:
            QApplication.quit()
        event.accept()


class ScanWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        if parent:
            self.showMaximized()

        self.parent_window = parent
        self.is_back_navigation = False

        try:
            ids_, names_, embeddings_ = load_all_students_faces()
            self.ids = ids_
            self.names = names_
            self.embeddings = embeddings_
        except:
            QMessageBox.critical(self, "Error", "Register atleast 1 student to continue")
            self.deleteLater()

        self.current_frame = None

        back_btn = QPushButton("←  Back")
        back_btn.setFixedSize(110, 38)
        back_btn.setFont(QFont("Arial", 13))
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                color: #333333;
            }
            QPushButton:hover {
                color: black;
            }
        """)
        back_btn.clicked.connect(self.go_back)

        self.cam_index = 0
        self.cam_selector = QComboBox()
        self.cam_selector.setFixedSize(189, 47)
        self.cam_selector.addItems(["Camera 0", "Camera 1", "Camera 2", "Camera 3"])
        self.cam_selector.currentIndexChanged.connect(self.change_camera)
        self.cam_selector.setStyleSheet("""
            QComboBox {
                border-radius: 10px;
                padding-left: 10px;
                font-size: 14px;
            }
        """)

        back_row = QHBoxLayout()
        back_row.addWidget(back_btn)
        back_row.addStretch()
        back_row.addWidget(self.cam_selector)

        self.setWindowTitle("NeuraFace – Scan")

        central = QWidget()
        central.setObjectName("bg")
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QWidget#bg {
                background-color: #C7F3FF;
                color: black;
            }
            QLabel {
                color: black;
            }
            QPushButton {
                color: black;
            }
        """)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 40, 0, 40)
        main_layout.setSpacing(20)

        main_layout.addLayout(back_row)

        title = QLabel("NeuraFace")
        title.setFont(QFont("Didot", 55))
        title.setAlignment(Qt.AlignLeft)
        title.setContentsMargins(80, 0, 0, 0)
        main_layout.addWidget(title)

        box_container = QWidget()
        box_container.setFixedSize(1000, 550)
        box_container.setStyleSheet("""
            QWidget {
                border: 8px solid #2E2E2E;
                border-radius: 60px;
                background-color: transparent;
            }
        """)

        box_layout = QVBoxLayout(box_container)
        box_layout.setContentsMargins(30, 30, 30, 30)

        self.video_label = QLabel("Camera Loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background: rgba(255,255,255,0.3); border-radius: 20px;"
        )
        box_layout.addWidget(self.video_label)

        main_layout.addWidget(box_container, alignment=Qt.AlignCenter)

        self.details_label = QLabel("<h3>Not recognized yet</h3>")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setFixedHeight(110)
        self.details_label.setStyleSheet("""
            QLabel {
                color: black;
                background: rgba(255,255,255,0.65);
                border-radius: 25px;
                font-size: 20px;
                padding: 12px;
            }
        """)
        main_layout.addWidget(self.details_label, alignment=Qt.AlignCenter)

        bottom_bar = QWidget()
        bottom_bar.setFixedSize(600, 85)
        bottom_bar.setStyleSheet("""
            QWidget {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #000000,
                    stop: 1 #555555
                );
                border-radius: 40px;
            }
        """)

        btn_layout = QHBoxLayout(bottom_bar)
        btn_layout.setSpacing(40)

        self.accept = QPushButton("Accept")
        self.accept.setFixedSize(180, 40)
        self.accept.setFont(QFont("Arial", 15, QFont.Bold))
        self.accept.setEnabled(False)
        self.accept.clicked.connect(self.accept_result)
        self.accept.setStyleSheet("""
            QPushButton {
                background: #E5E5E5;
                border-radius: 20px;
            }
            QPushButton:hover { background: white; }
        """)

        self.recapture = QPushButton("Recapture")
        self.recapture.setFixedSize(180, 40)
        self.recapture.setFont(QFont("Arial", 15, QFont.Bold))
        self.recapture.setEnabled(False)
        self.recapture.clicked.connect(self.start_capture_again)
        self.recapture.setStyleSheet("""
            QPushButton {
                background: #E5E5E5;
                border-radius: 20px;
            }
            QPushButton:hover { background: white; }
        """)

        btn_layout.addWidget(self.accept)
        btn_layout.addWidget(self.recapture)

        main_layout.addWidget(bottom_bar, alignment=Qt.AlignCenter)

        self.cap = cv2.VideoCapture(self.cam_index)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        self.recognized_student_name = None
        self.recognized_student_id = None

    def accept_result(self):
        save_student_attendance(self.recognized_student_id)

        msg = QMessageBox(self)
        msg.setWindowTitle("Success")
        msg.setText("Attendance saved successfully")
        msg.setIcon(QMessageBox.Information)

        msg.setStyleSheet("""
            QMessageBox {
                background-color: #2B2B2B;
                color: white;
                font-size: 14px;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #444444;
                color: white;
                border-radius: 8px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)

        msg.exec()

    def recognize_frame(self, frame, ids, names, embeddings):
        results = []

        detections = DeepFace.extract_faces(
            frame,
            detector_backend="opencv",
            enforce_detection=False
        )

        for det in detections:
            region = det["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            try:
                emb = DeepFace.represent(
                    face_rgb,
                    model_name="ArcFace",
                    detector_backend="skip"
                )[0]["embedding"]
            except Exception:
                continue

            emb = np.array(emb, dtype=np.float32)

            dists = np.linalg.norm(embeddings - emb, axis=1)
            best_idx = np.argmin(dists)
            best_dist = dists[best_idx]

            name = names[best_idx] if best_dist < THRESHOLD else "Unknown"
            id_ = ids[best_idx] if best_dist < THRESHOLD else "Unknown"

            results.append((x, y, w, h, name, id_, best_dist))

        return results

    def stop_capture(self):
        self.timer.stop()

    def start_capture_again(self):
        self.accept.setEnabled(False)
        self.recapture.setEnabled(False)
        self.details_label.setText("<h3>RECOGNIZED AS:</h3>")
        self.timer.start(10)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.video_label.setText("Camera failed")
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()

        detections = self.recognize_frame(frame, self.ids, self.names, self.embeddings)

        known_count = sum(1 for *_, name, __, ___ in [
            (x, y, w, h, name, id_, dist) for x, y, w, h, name, id_, dist in detections
        ] if name != "Unknown")

        if known_count > 1:
            QMessageBox.warning(self, "Multiple People",
                                "More than one person detected in the frame. Please try again.")

        found_known = False
        for x, y, w, h, name, id_, dist in detections:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if name != "Unknown":
                self.recognized_student_name = name
                self.recognized_student_id = id_
                self.details_label.setText(
                    f"<h3>Name:{name}, ID:{id_}</h3>"
                )
                found_known = True

        if found_known and known_count == 1:
            self.stop_capture()
            self.accept.setEnabled(True)
            self.recapture.setEnabled(True)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def change_camera(self, index):
        self.cam_index = index
        self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Error", f"Camera {index} not available.")

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        if self.parent_window:
            self.parent_window.close()
        if not self.is_back_navigation:
            QApplication.quit()
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        self.move(0, 0)

    
    def go_back(self):
        self.is_back_navigation = True

        self.timer.stop()
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

        self.close()
        self.parent_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = NeuraFaceHome()
    w.showMaximized()

    sys.exit(app.exec())