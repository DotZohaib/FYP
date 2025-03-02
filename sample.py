import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import os
import time
from PIL import Image
from pathlib import Path
import dlib
import face_recognition
import pickle
import csv
import plotly.express as px
import plotly.graph_objects as go

# Configuration
MODEL_DIR = Path("models")
IMAGE_DIR = Path("faces")
DATABASE_FILE = "students_db.pkl"
ATTENDANCE_DIR = Path("attendance")
ATTENDANCE_FILE = ATTENDANCE_DIR / f"attendance_{date.today().strftime('%Y-%m-%d')}.csv"
UNKNOWN_THRESHOLD = 0.6
FRAME_SKIP = 2  # Process every 3rd frame

# Initialize directories
MODEL_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
ATTENDANCE_DIR.mkdir(exist_ok=True)

# Download required model files if they don't exist
def download_dlib_models():
    import urllib.request

    models = {
        "shape_predictor_68_face_landmarks.dat":
        "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2",

        "dlib_face_recognition_resnet_model_v1.dat":
        "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }

    for model_name, url in models.items():
        model_path = MODEL_DIR / model_name

        if not model_path.exists():
            st.info(f"Downloading {model_name}... This may take a few minutes.")

            # Create a temporary file for the compressed model
            temp_path = MODEL_DIR / f"{model_name}.bz2"

            # Download the compressed model
            urllib.request.urlretrieve(url, temp_path)

            # Decompress the model
            import bz2
            with bz2.open(temp_path, 'rb') as source, open(model_path, 'wb') as dest:
                dest.write(source.read())

            # Remove the temporary compressed file
            temp_path.unlink()

            st.success(f"{model_name} downloaded successfully!")

# Streamlit Config
st.set_page_config(
    page_title="Student Attendance System",
    page_icon="👨‍🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .alert {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success { background: #d4edda; border-left: 4px solid #28a745; }
    .warning { background: #fff3cd; border-left: 4px solid #ffc107; }
    .danger { background: #f8d7da; border-left: 4px solid #dc3545; }
    .user-profile {
        background: linear-gradient(to right, #4880EC, #019CAD);
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px 4px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'known_encodings' not in st.session_state:
    st.session_state.known_encodings = []
if 'known_names' not in st.session_state:
    st.session_state.known_names = []
if 'detected_history' not in st.session_state:
    st.session_state.detected_history = []
if 'marked_attendance' not in st.session_state:
    st.session_state.marked_attendance = set()
if 'processing_frame' not in st.session_state:
    st.session_state.processing_frame = None
if 'system_active' not in st.session_state:
    st.session_state.system_active = False
if 'current_student' not in st.session_state:
    st.session_state.current_student = None
if 'notification' not in st.session_state:
    st.session_state.notification = None
if 'notification_time' not in st.session_state:
    st.session_state.notification_time = None

class Student:
    def __init__(self, id, name, course, year, section, email, phone, additional_info=None):
        self.id = id
        self.name = name
        self.course = course
        self.year = year
        self.section = section
        self.email = email
        self.phone = phone
        self.additional_info = additional_info or {}
        self.attendance_record = []
        self.last_detected = None

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "course": self.course,
            "year": self.year,
            "section": self.section,
            "email": self.email,
            "phone": self.phone,
            "additional_info": self.additional_info,
            "attendance_record": self.attendance_record,
            "last_detected": self.last_detected
        }

    @classmethod
    def from_dict(cls, data):
        student = cls(
            data["id"],
            data["name"],
            data["course"],
            data["year"],
            data["section"],
            data["email"],
            data["phone"],
            data["additional_info"]
        )
        student.attendance_record = data.get("attendance_record", [])
        student.last_detected = data.get("last_detected")
        return student

class FaceRecognitionSystem:
    def __init__(self):
        # Download models if needed
        download_dlib_models()

        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = self._load_shape_predictor()
        self.face_recognizer = dlib.face_recognition_model_v1(
            str(MODEL_DIR / "dlib_face_recognition_resnet_model_v1.dat")
        )
        self.students_db = self._load_database()
        self.init_attendance_file()

    def _load_shape_predictor(self):
        path = MODEL_DIR / "shape_predictor_68_face_landmarks.dat"
        return dlib.shape_predictor(str(path))

    def _load_database(self):
     if Path(DATABASE_FILE).exists():
        try:
            with open(DATABASE_FILE, "rb") as f:
                data = pickle.load(f)

                # Convert plain dictionaries to Student objects if needed
                students_db = {}
                for student_id, value in data.items():
                    if isinstance(value, dict) and "encoding" in value and "student" in value:
                        # Convert student dict to Student object if it's not already
                        if isinstance(value["student"], dict):
                            value["student"] = Student.from_dict(value["student"])
                        students_db[student_id] = value
                    else:
                        # Legacy format or incompatible, skip
                        pass

                return students_db
        except (EOFError, _pickle.UnpicklingError, Exception) as e:
            # If there's any error loading the file, log it and create a new database
            print(f"Error loading database: {e}. Creating new database.")
            # Delete the corrupted file
            Path(DATABASE_FILE).unlink(missing_ok=True)
            return {}
    return {}

 # In the FaceRecognitionSystem class:

def save_database(self):
    # Create a temporary file first
    temp_file = f"{DATABASE_FILE}.temp"

    # Convert students to dictionaries before saving
    db_to_save = {}
    for student_id, data in self.students_db.items():
        db_to_save[student_id] = {
            "encoding": data["encoding"],
            "student": data["student"].to_dict()
        }

    try:
        # Write to the temporary file first
        with open(temp_file, "wb") as f:
            pickle.dump(db_to_save, f)

        # If successful, replace the original file
        Path(temp_file).replace(DATABASE_FILE)
    except Exception as e:
        print(f"Error saving database: {e}")
        # Clean up the temporary file if it exists
        Path(temp_file).unlink(missing_ok=True)
        raise

def _load_database(self):
    if Path(DATABASE_FILE).exists():
        try:
            with open(DATABASE_FILE, "rb") as f:
                data = pickle.load(f)
                students_db = {}
                for student_id, value in data.items():
                    # Convert student dict to Student object
                    student = Student.from_dict(value["student"])
                    students_db[student_id] = {
                        "encoding": value["encoding"],
                        "student": student
                    }
                return students_db
        except Exception as e:
            st.error(f"Error loading database: {e}. Creating new database.")
            Path(DATABASE_FILE).unlink(missing_ok=True)
            return {}
    return {}

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(rgb, 1)

        results = []
        for face in faces:
            shape = self.shape_predictor(rgb, face)
            encoding = np.array(self.face_recognizer.compute_face_descriptor(rgb, shape))

            matches = face_recognition.compare_faces(
                st.session_state.known_encodings,
                encoding,
                tolerance=UNKNOWN_THRESHOLD
            )

            name = "Unknown"
            student = None

            if True in matches:
                match_index = matches.index(True)
                student_id = st.session_state.known_names[match_index]
                student_data = self.students_db.get(student_id)

                if student_data:
                    student = student_data["student"]
                    name = student.name

                    # Update last detection time
                    now = datetime.now()
                    student.last_detected = now

                    # Mark attendance if not already marked today
                    today_str = date.today().isoformat()
                    attendance_key = f"{student_id}_{today_str}"

                    if attendance_key not in st.session_state.marked_attendance:
                        self.mark_attendance(student)
                        st.session_state.marked_attendance.add(attendance_key)

                        # Add to notification
                        st.session_state.notification = f"✅ Attendance marked for {name}"
                        st.session_state.notification_time = time.time()

                    # Set as current student for display
                    st.session_state.current_student = student

            results.append({
                "name": name,
                "student": student,
                "encoding": encoding,
                "location": (
                    face.left(), face.top(),
                    face.right(), face.bottom()
                )
            })

            # Add to detection history
            if student:
                st.session_state.detected_history.append({
                    "name": name,
                    "student_id": student.id,
                    "timestamp": time.time()
                })

        return results

    def register_new_student(self, image, student):
        encodings = self.get_face_encodings(image)
        if not encodings:
            return False

        # Save the student data
        self.students_db[student.id] = {
            "encoding": encodings[0],
            "student": student
        }

        # Save student image
        img_path = IMAGE_DIR / f"{student.id}.jpg"
        cv2.imwrite(str(img_path), image)

        self.update_known_faces()
        self.save_database()
        return True

    def update_student(self, student_id, student_data):
        if student_id in self.students_db:
            self.students_db[student_id]["student"] = student_data
            self.save_database()
            return True
        return False

    def get_face_encodings(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(rgb, 1)
        encodings = []

        for face in faces:
            shape = self.shape_predictor(rgb, face)
            encoding = np.array(self.face_recognizer.compute_face_descriptor(rgb, shape))
            encodings.append(encoding)

        return encodings

    def update_known_faces(self):
        st.session_state.known_encodings = []
        st.session_state.known_names = []
        for student_id, data in self.students_db.items():
            st.session_state.known_encodings.append(data["encoding"])
            st.session_state.known_names.append(student_id)

    def init_attendance_file(self):
        if not ATTENDANCE_FILE.exists():
            with open(ATTENDANCE_FILE, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Student ID', 'Name', 'Course', 'Year', 'Section', 'Time', 'Date'])

    def mark_attendance(self, student):
        # Add to CSV file
        now = datetime.now()
        with open(ATTENDANCE_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                student.id,
                student.name,
                student.course,
                student.year,
                student.section,
                now.strftime('%H:%M:%S'),
                now.strftime('%Y-%m-%d')
            ])

        # Add to student record
        student.attendance_record.append(now.isoformat())
        self.save_database()

    def get_attendance_data(self, date_str=None):
        if date_str:
            file_path = ATTENDANCE_DIR / f"attendance_{date_str}.csv"
        else:
            file_path = ATTENDANCE_FILE

        if file_path.exists():
            return pd.read_csv(file_path)
        return pd.DataFrame(columns=['Student ID', 'Name', 'Course', 'Year', 'Section', 'Time', 'Date'])

    def get_attendance_history(self, days=7):
        data = []
        today = date.today()

        for i in range(days):
            day = today - pd.Timedelta(days=i)
            day_str = day.strftime('%Y-%m-%d')
            file_path = ATTENDANCE_DIR / f"attendance_{day_str}.csv"

            if file_path.exists():
                df = pd.read_csv(file_path)
                count = len(df)
            else:
                count = 0

            data.append({
                "Date": day_str,
                "Count": count
            })

        return pd.DataFrame(data)

    def get_course_attendance(self):
        df = self.get_attendance_data()
        if df.empty:
            return pd.DataFrame(columns=['Course', 'Count'])

        return df.groupby('Course').size().reset_index(name='Count')

    def get_student_details(self, student_id):
        if student_id in self.students_db:
            return self.students_db[student_id]["student"]
        return None

def main():
    global UNKNOWN_THRESHOLD
    global FRAME_SKIP
    frs = FaceRecognitionSystem()
    frs.update_known_faces()

    st.title("👨‍🎓 Smart Student Attendance System")

    # Navigation tabs
    tabs = st.tabs(["📹 Attendance", "📊 Analytics", "👥 Student Management", "⚙️ Settings"])

    # Tab 1: Attendance
    with tabs[0]:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Live Camera Feed")
            camera_placeholder = st.empty()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                start_btn = st.button("Start Recognition", key="start_recognition")
            with col_btn2:
                stop_btn = st.button("Stop Recognition", key="stop_recognition")

            if stop_btn:
                st.session_state.system_active = False
                st.session_state.current_student = None
            if start_btn:
                st.session_state.system_active = True

            if st.session_state.system_active:
                cap = cv2.VideoCapture(0)
                frame_count = 0

                while st.session_state.system_active:
                    ret, frame = cap.read()
                    frame_count += 1

                    if not ret:
                        st.error("Failed to capture video")
                        break

                    if frame_count % FRAME_SKIP == 0:
                        processed = frs.process_frame(frame)
                        st.session_state.processing_frame = processed

                        # Check if notification should be cleared
                        if st.session_state.notification_time and time.time() - st.session_state.notification_time > 3:
                            st.session_state.notification = None

                    display_frame = frame.copy()
                    if st.session_state.processing_frame:
                        for result in st.session_state.processing_frame:
                            top, right, bottom, left = result["location"]
                            # Change color based on recognition status
                            color = (0, 255, 0) if result["name"] != "Unknown" else (0, 0, 255)

                            cv2.rectangle(display_frame,
                                (left, top), (right, bottom),
                                color, 2)

                            # Enhanced text display
                            text_bg_height = 20
                            cv2.rectangle(display_frame,
                                (left, bottom - text_bg_height), (right, bottom),
                                color, cv2.FILLED)

                            cv2.putText(display_frame, result["name"],
                                (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                    # Show notification if exists
                    if st.session_state.notification:
                        # Add notification to frame
                        cv2.rectangle(display_frame, (10, 10), (400, 40), (0, 0, 0), cv2.FILLED)
                        cv2.putText(display_frame, st.session_state.notification,
                                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                    camera_placeholder.image(display_frame, channels="BGR")

                cap.release()

        with col2:
            st.subheader("Current Student")
            current_student_placeholder = st.empty()

            if st.session_state.current_student:
                student = st.session_state.current_student
                current_student_placeholder.markdown(f"""
                <div class="user-profile">
                    <h3>{student.name}</h3>
                    <p><strong>ID:</strong> {student.id}</p>
                    <p><strong>Course:</strong> {student.course}</p>
                    <p><strong>Year & Section:</strong> {student.year}-{student.section}</p>
                    <p><strong>Email:</strong> {student.email}</p>
                    <p><strong>Phone:</strong> {student.phone}</p>
                    <hr>
                    <p><small>Last detected: {student.last_detected.strftime('%Y-%m-%d %H:%M:%S') if student.last_detected else 'Never'}</small></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                current_student_placeholder.markdown("""
                <div class="metric-card">
                    <p>No student currently detected.</p>
                    <p>Students will appear here when recognized.</p>
                </div>
                """, unsafe_allow_html=True)

            # Today's attendance
            st.subheader("Today's Attendance")
            today_attendance = frs.get_attendance_data()

            if not today_attendance.empty:
                st.dataframe(today_attendance, use_container_width=True)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Present</h3>
                    <p style="font-size: 24px; margin: 0;">{len(today_attendance)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No attendance records for today yet.")

    # Tab 2: Analytics
    with tabs[1]:
        st.subheader("Attendance Analytics")

        analytics_type = st.radio("Select Analysis",
                                 ["Daily Attendance", "Course-wise Attendance", "Student Details"],
                                 horizontal=True)

        if analytics_type == "Daily Attendance":
            attendance_history = frs.get_attendance_history(days=7)

            fig = px.bar(
                attendance_history,
                x="Date",
                y="Count",
                title="Attendance Over the Last 7 Days",
                labels={"Count": "Number of Students Present", "Date": "Date"},
                color="Count",
                color_continuous_scale=px.colors.sequential.Blues
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Students",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        elif analytics_type == "Course-wise Attendance":
            course_data = frs.get_course_attendance()

            if not course_data.empty:
                fig = px.pie(
                    course_data,
                    values="Count",
                    names="Course",
                    title="Attendance by Course",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No attendance data available for analysis.")

        elif analytics_type == "Student Details":
            # Select a student to view details
            student_options = {
                student_data["student"].name: student_id
                for student_id, student_data in frs.students_db.items()
            }

            if student_options:
                selected_name = st.selectbox("Select Student", list(student_options.keys()))
                selected_id = student_options[selected_name]

                student = frs.get_student_details(selected_id)

                if student:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        img_path = IMAGE_DIR / f"{student.id}.jpg"
                        if img_path.exists():
                            st.image(str(img_path), caption=student.name, width=200)
                        else:
                            st.image("https://www.w3schools.com/howto/img_avatar.png", caption=student.name, width=200)

                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{student.name}</h3>
                            <p><strong>ID:</strong> {student.id}</p>
                            <p><strong>Course:</strong> {student.course}</p>
                            <p><strong>Year & Section:</strong> {student.year}-{student.section}</p>
                            <p><strong>Email:</strong> {student.email}</p>
                            <p><strong>Phone:</strong> {student.phone}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        # Check for attendance data
                        today_attendance = frs.get_attendance_data()
                        student_attendance = today_attendance[today_attendance['Student ID'] == student.id]

                        if not student_attendance.empty:
                            st.success(f"Present today, arrived at {student_attendance.iloc[0]['Time']}")
                        else:
                            st.warning("Not present today")

                        # Show attendance history
                        st.subheader("Attendance History")

                        # Calculate attendance percentage
                        attendance_dates = set()
                        for i in range(30):  # Check last 30 days
                            day = date.today() - pd.Timedelta(days=i)
                            file_path = ATTENDANCE_DIR / f"attendance_{day.strftime('%Y-%m-%d')}.csv"

                            if file_path.exists():
                                df = pd.read_csv(file_path)
                                if not df[df['Student ID'] == student.id].empty:
                                    attendance_dates.add(day.strftime('%Y-%m-%d'))

                        working_days = len(set(frs.get_attendance_history(days=30)['Date']))
                        if working_days > 0:
                            attendance_percentage = (len(attendance_dates) / working_days) * 100
                        else:
                            attendance_percentage = 0

                        # Display gauge chart for attendance percentage
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=attendance_percentage,
                            title={'text': "Attendance Percentage (Last 30 Days)"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 75], 'color': "lightcoral"},
                                    {'range': [75, 90], 'color': "lightyellow"},
                                    {'range': [90, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 75
                                }
                            }
                        ))

                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No students registered in the system.")

    # Tab 3: Student Management
    with tabs[2]:
        st.subheader("Student Management")

        management_type = st.radio("Select Action",
                                  ["Register New Student", "Edit Student", "View All Students"],
                                  horizontal=True)

        if management_type == "Register New Student":
            with st.form("student_registration"):
                st.subheader("Register New Student")

                col1, col2 = st.columns(2)

                with col1:
                    student_id = st.text_input("Student ID")
                    name = st.text_input("Full Name")
                    email = st.text_input("Email")
                    phone = st.text_input("Phone Number")

                with col2:
                    course = st.text_input("Course")
                    year = st.selectbox("Year Level", [1, 2, 3, 4, 5])
                    section = st.text_input("Section")
                    additional_notes = st.text_area("Additional Notes", height=100)

                face_image = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

                submit_button = st.form_submit_button("Register Student")

                if submit_button:
                    if not all([student_id, name, course, section, email]):
                        st.error("Please fill all required fields")
                    elif not face_image:
                        st.error("Please upload a face image")
                    else:
                        # Create the student object
                        student = Student(
                            id=student_id,
                            name=name,
                            course=course,
                            year=year,
                            section=section,
                            email=email,
                            phone=phone,
                            additional_info={"notes": additional_notes}
                        )

                        # Convert the uploaded image
                        image = cv2.imdecode(np.frombuffer(face_image.read(), np.uint8), 1)

                        # Register the student
                        if frs.register_new_student(image, student):
                            st.success(f"{name} registered successfully!")
                        else:
                            st.error("No faces detected in uploaded image")

        elif management_type == "Edit Student":
            student_options = {
                student_data["student"].name: student_id
                for student_id, student_data in frs.students_db.items()
            }

            if student_options:
                selected_name = st.selectbox("Select Student to Edit", list(student_options.keys()))
                selected_id = student_options[selected_name]

                student = frs.get_student_details(selected_id)

                if student:
                    with st.form("student_edit"):
                        st.subheader(f"Edit {student.name}'s Information")

                        col1, col2 = st.columns(2)

                        with col1:
                            student_id = st.text_input("Student ID", value=student.id, disabled=True)
                            name = st.text_input("Full Name", value=student.name)
                            email = st.text_input("Email", value=student.email)
                            phone = st.text_input("Phone Number", value=student.phone)

                        with col2:
                            course = st.text_input("Course", value=student.course)
                            year = st.selectbox("Year Level", [1, 2, 3, 4, 5], index=student.year-1)
                            section = st.text_input("Section", value=student.section)
                            additional_notes = st.text_area("Additional Notes",
                                                          value=student.additional_info.get("notes", ""),
                                                          height=100)

                        update_button = st.form_submit_button("Update Student Information")

                        if update_button:
                            # Update student object
                            updated_student = Student(
                                id=student.id,
                                name=name,
                                course=course,
                                year=year,
                                section=section,
                                email=email,
                                phone=phone,
                                additional_info={"notes": additional_notes}
                            )

                            # Preserve attendance record
                            updated_student.attendance_record = student.attendance_record
                            updated_student.last_detected = student.last_detected

                            # Update the database
                            if frs.update_student(student.id, updated_student):
                                st.success(f"{name}'s information updated successfully!")
                            else:
                                st.error("Failed to update student information")
            else:
                st.warning("No students registered in the system.")

        elif management_type == "View All Students":
            if frs.students_db:
                # Convert students to DataFrame for display
                students_list = []
                for student_id, data in frs.students_db.items():
                    student = data["student"]
                    students_list.append({
                        "ID": student.id,
                        "Name": student.name,
                        "Course": student.course,
                        "Year-Section": f"{student.year}-{student.section}",
                        "Email": student.email,
                        "Phone": student.phone,
                        "Last Detected": student.last_detected.strftime('%Y-%m-%d %H:%M:%S') if student.last_detected else "Never"
                    })

                students_df = pd.DataFrame(students_list)

                # Set the dataframe of students with filter
                st.dataframe(students_df, use_container_width=True)

                # Export option
               # Export option
                if st.button("Export Student List"):
                    csv = students_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"students_list_{date.today().strftime('%Y-%m-%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No students registered in the system.")

    # Tab 4: Settings
    with tabs[3]:
        st.subheader("System Settings")

        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            st.subheader("Recognition Settings")

            new_threshold = st.slider(
                "Face Recognition Threshold",
                min_value=0.4,
                max_value=0.8,
                value=UNKNOWN_THRESHOLD,
                step=0.01,
                help="Lower values are more strict, higher values are more lenient"
            )


            if new_threshold != UNKNOWN_THRESHOLD and st.button("Update Threshold"):
                UNKNOWN_THRESHOLD = new_threshold
                st.success(f"Recognition threshold updated to {UNKNOWN_THRESHOLD}")

            new_frame_skip = st.slider(
                "Frame Processing Rate",
                min_value=1,
                max_value=10,
                value=FRAME_SKIP,
                help="Higher values reduce CPU usage but may be less responsive"
            )

            if new_frame_skip != FRAME_SKIP and st.button("Update Frame Rate"):

                FRAME_SKIP = new_frame_skip
                st.success(f"Frame processing rate updated to process every {FRAME_SKIP} frames")

        with settings_col2:
            st.subheader("Data Management")

            if st.button("Backup Database"):
                # Create backup
                backup_file = f"backup_students_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                with open(backup_file, "wb") as f:
                    pickle.dump(frs.students_db, f)

                # Provide download link
                with open(backup_file, "rb") as f:
                    st.download_button(
                        label="Download Backup File",
                        data=f,
                        file_name=backup_file,
                        mime="application/octet-stream"
                    )

                st.success(f"Database backup created: {backup_file}")

            uploaded_backup = st.file_uploader("Restore Database from Backup", type=["pkl"])
            if uploaded_backup and st.button("Restore Database"):
                try:
                    restored_db = pickle.loads(uploaded_backup.read())
                    # Verify structure
                    valid_backup = True
                    for student_id, data in restored_db.items():
                        if not (isinstance(data, dict) and "encoding" in data and "student" in data):
                            valid_backup = False
                            break

                    if valid_backup:
                        frs.students_db = restored_db
                        frs.save_database()
                        frs.update_known_faces()
                        st.success("Database restored successfully!")
                    else:
                        st.error("Invalid backup file format")
                except Exception as e:
                    st.error(f"Failed to restore database: {str(e)}")

            if st.button("Export Attendance Records"):
                # Create a ZIP file with all attendance CSV files
                import zipfile
                import io

                # Create a BytesIO object to store the ZIP file
                zip_buffer = io.BytesIO()

                # Create a ZIP file
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add all CSV files from the attendance directory
                    for file_path in ATTENDANCE_DIR.glob('*.csv'):
                        zip_file.write(file_path, arcname=file_path.name)

                # Provide download link
                zip_buffer.seek(0)
                st.download_button(
                    label="Download All Attendance Records",
                    data=zip_buffer,
                    file_name=f"attendance_records_{date.today().strftime('%Y-%m-%d')}.zip",
                    mime="application/zip"
                )

            st.divider()

            st.subheader("Danger Zone")
            with st.expander("Reset System"):
                st.warning("This will delete all student data and attendance records!")

                delete_confirmation = st.text_input(
                    "Type 'DELETE' to confirm reset",
                    placeholder="DELETE"
                )

                if delete_confirmation == "DELETE" and st.button("Reset System", type="primary"):
                    # Delete database
                    if Path(DATABASE_FILE).exists():
                        Path(DATABASE_FILE).unlink()

                    # Clear student images
                    for img_file in IMAGE_DIR.glob("*.jpg"):
                        img_file.unlink()

                    # Clear attendance records
                    for csv_file in ATTENDANCE_DIR.glob("*.csv"):
                        csv_file.unlink()

                    # Reset session state
                    st.session_state.known_encodings = []
                    st.session_state.known_names = []
                    st.session_state.detected_history = []
                    st.session_state.marked_attendance = set()
                    st.session_state.system_active = False
                    st.session_state.current_student = None

                    st.success("System reset successfully. Refresh the page to start fresh.")
                    st.experimental_rerun()

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
        <p>Student Attendance System • Powered by AI Face Recognition</p>
        <p style="font-size: 12px;">© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
