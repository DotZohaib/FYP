import streamlit as st
import cv2
import numpy as np
import PIL.Image
from io import BytesIO
import base64
import time
import math
import pandas as pd
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Friend Recognition System",
    page_icon="👁️",
    layout="wide"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .friend-card {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# Configuration flags
USE_DNN_DETECTOR = True      # Toggle between Haar Cascade and DNN detector
ENABLE_EMOTION_DETECTION = True # Enable emotion detection
ENABLE_FRIEND_INFO = True    # Enable friend information display
ENABLE_ANIMATIONS = True     # Enable animation effects

# Animation settings
PULSE_SPEED = 2.0            # Speed of pulsing animations
SLIDE_SPEED = 15             # Speed of sliding animations (pixels per frame)
FADE_SPEED = 0.1             # Speed of fade-in animations (alpha per frame)

# Initialize session state variables
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'face_detected' not in st.session_state:
    st.session_state.face_detected = False
if 'current_friend' not in st.session_state:
    st.session_state.current_friend = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'last_seen_friends' not in st.session_state:
    st.session_state.last_seen_friends = []

# Load emotion labels
emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']

# Sample friend profiles database with expanded information
friend_profiles = {
    "friend1": {
        "id": "FR-2025-001",
        "name": "Zohaib Ali Dayo",
        "age": 21,
        "category": "Me",
        "since": "2018",
        "interests": "Hacking, Coding",
        "relationship_score": 100,
        "last_seen": "2025-02-25",
        "contact": "zohaib@example.com",
        "location": "New York",
        "meetings": 45,
        "recent_activity": "Coding meetup at Tech Hub",
        "conversation_topics": ["AI projects", "Cybersecurity", "New programming languages"]
    },
    "friend2": {
        "id": "FR-2025-002",
        "name": "Sarah Johnson",
        "age": 24,
        "category": "Work Colleague",
        "since": "2022",
        "interests": "Data Science, Hiking",
        "relationship_score": 82,
        "last_seen": "2025-02-20",
        "contact": "sarah.j@example.com",
        "location": "Boston",
        "meetings": 28,
        "recent_activity": "Project collaboration meeting",
        "conversation_topics": ["Data visualization", "Team management", "Project deadlines"]
    },
    "friend3": {
        "id": "FR-2025-003",
        "name": "Michael Chen",
        "age": 29,
        "category": "Study Group",
        "since": "2020",
        "interests": "Machine Learning, Basketball",
        "relationship_score": 78,
        "last_seen": "2025-02-15",
        "contact": "mchen@example.com",
        "location": "Seattle",
        "meetings": 32,
        "recent_activity": "Study session on advanced algorithms",
        "conversation_topics": ["Research papers", "Career advancement", "Industry trends"]
    }
}

# Interaction history database (simulated)
interaction_history = [
    {"friend_id": "FR-2025-001", "date": "2025-02-25", "duration": 45, "location": "Coffee Shop", "notes": "Discussed new coding project ideas"},
    {"friend_id": "FR-2025-001", "date": "2025-02-10", "duration": 120, "location": "Tech Meetup", "notes": "Attended workshop together"},
    {"friend_id": "FR-2025-002", "date": "2025-02-20", "duration": 30, "location": "Office", "notes": "Project planning meeting"},
    {"friend_id": "FR-2025-002", "date": "2025-01-15", "duration": 60, "location": "Conference Call", "notes": "Quarterly review"},
    {"friend_id": "FR-2025-003", "date": "2025-02-15", "duration": 90, "location": "University Library", "notes": "Research collaboration"},
    {"friend_id": "FR-2025-003", "date": "2025-01-30", "duration": 120, "location": "Online Study Group", "notes": "Algorithm practice session"}
]

# Convert to DataFrame for easier filtering
interaction_df = pd.DataFrame(interaction_history)

# Functions for Face Detection and Recognition

@st.cache_resource
def load_face_detection_models():
    """Load the face detection and emotion recognition models"""
    models = {}

    # For DNN face detection
    # In a real app, you'd need to download these models or include them with your app
    try:
        # Placeholder for model loading - in actual implementation, you would load real models here
        # For demo purposes, we'll use OpenCV's built-in models
        models['face_detector'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Placeholder for emotion detection model
        # In a real app, you would load an emotion detection model here
        models['emotion_detector'] = None

        # Placeholder for face recognition model
        # In a real app, you would load a face recognition model here
        models['face_recognizer'] = None

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        models = None

    return models

def detect_faces(img, models):
    """Detect faces in the image and return face locations"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = models['face_detector'].detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces

def detect_emotion(img, face_coords, models):
    """Detect emotion based on facial expression"""
    # This would use a real emotion detection model in a production app
    # For this demo, we'll return a random emotion
    import random
    return random.choice(emotion_labels)

def recognize_friend(img, face_coords, models):
    """Recognize if the face belongs to a known friend"""
    # This would use a real face recognition model in a production app
    # For this demo, we'll pick a random friend from our database
    import random
    friend_key = random.choice(list(friend_profiles.keys()))
    return friend_key

def get_friend_recent_interactions(friend_id):
    """Get recent interactions with this friend"""
    recent_interactions = interaction_df[interaction_df['friend_id'] == friend_id].sort_values(by='date', ascending=False)
    return recent_interactions

def draw_friend_card(img, friend_info, x, y, w, h, frame_count):
    """Draw an animated friend ID card on the image"""
    # Create a copy of the image to draw on
    overlay = img.copy()

    # Card dimensions and position (below the face)
    card_width = max(300, w * 1.5)
    card_height = 160
    card_x = max(10, min(x - (card_width - w) // 2, img.shape[1] - card_width - 10))
    card_y = y + h + 20

    # Animation effects
    if ENABLE_ANIMATIONS:
        # Calculate animation progress (0.0 to 1.0)
        alpha = min(1.0, frame_count * FADE_SPEED)

        # Slide-in animation
        slide_offset = max(0, int((1.0 - alpha) * 50))
        card_y += slide_offset
    else:
        alpha = 1.0

    # Draw main card background
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + card_height),
                 (50, 50, 70), -1)

    # Draw header strip
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + 30),
                 (70, 130, 180), -1)

    # Draw "FRIEND ID" text in header
    cv2.putText(overlay, "FRIEND ID", (card_x + 10, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw ID number in header
    cv2.putText(overlay, friend_info['id'], (card_x + card_width - 110, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw name with larger font
    cv2.putText(overlay, friend_info['name'], (card_x + 15, card_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw additional info
    cv2.putText(overlay, f"Category: {friend_info['category']}", (card_x + 15, card_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(overlay, f"Age: {friend_info['age']}", (card_x + 15, card_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(overlay, f"Since: {friend_info['since']}", (card_x + 15, card_y + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Blend the overlay with the original image
    result = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    return result

def add_to_history(friend_id, friend_name):
    """Add an encounter to the history"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    history_entry = {
        "date": date_str,
        "time": time_str,
        "friend_id": friend_id,
        "friend_name": friend_name,
        "location": "Webcam Detection"
    }

    st.session_state.search_history.append(history_entry)

    # Also add to last seen friends list (keeping most recent at top)
    if friend_id in [f["friend_id"] for f in st.session_state.last_seen_friends]:
        # Remove the old entry
        st.session_state.last_seen_friends = [
            f for f in st.session_state.last_seen_friends if f["friend_id"] != friend_id
        ]

    # Add to the beginning
    st.session_state.last_seen_friends.insert(0, {
        "friend_id": friend_id,
        "friend_name": friend_name,
        "time": f"{date_str} {time_str}"
    })

    # Keep only the last 5 entries
    if len(st.session_state.last_seen_friends) > 5:
        st.session_state.last_seen_friends = st.session_state.last_seen_friends[:5]

# App UI Components
def app_header():
    """Display app header section"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("👁️ Smart Friend Recognition System")
        st.subheader("Identify, track, and manage your social connections")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.session_state.face_detected:
            st.markdown('<h3 style="color:#4CAF50" class="pulse">⚡ ACTIVE</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color:#FFA000">🔍 SEARCHING</h3>', unsafe_allow_html=True)

        st.metric("Friends in Database", len(friend_profiles))
        st.markdown('</div>', unsafe_allow_html=True)

def display_friend_details(friend_key):
    """Display detailed information about a friend"""
    if friend_key and friend_key in friend_profiles:
        friend = friend_profiles[friend_key]

        # Create a two-column layout for friend details
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.subheader(f"📋 {friend['name']} - Profile")
            st.markdown(f"**ID:** {friend['id']}")
            st.markdown(f"**Category:** {friend['category']}")
            st.markdown(f"**Age:** {friend['age']}")
            st.markdown(f"**Friend Since:** {friend['since']}")
            st.markdown(f"**Location:** {friend['location']}")
            st.markdown(f"**Contact:** {friend['contact']}")
            st.markdown(f"**Interests:** {friend['interests']}")

            # Relationship score with color coding
            score = friend['relationship_score']
            if score >= 90:
                score_color = "green"
            elif score >= 70:
                score_color = "blue"
            else:
                score_color = "orange"

            st.markdown(f"**Relationship Score:** <span style='color:{score_color};font-weight:bold'>{score}%</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔄 Recent Interactions")

            # Get recent interactions
            recent = get_friend_recent_interactions(friend['id'])

            if not recent.empty:
                for _, row in recent.head(3).iterrows():
                    st.markdown(f"""
                    <div class='friend-card'>
                        <strong>{row['date']}</strong> | {row['location']} ({row['duration']} mins)<br>
                        <small>{row['notes']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent interactions recorded.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("💬 Conversation Topics")
            for topic in friend['conversation_topics']:
                st.markdown(f"- {topic}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Display metrics row
        st.markdown("<div class='card metrics-container'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Meetings", friend['meetings'])
        with col2:
            st.metric("Last Seen", friend['last_seen'])
        with col3:
            st.metric("Days Since Last Meet", (datetime.now() - datetime.strptime(friend['last_seen'], "%Y-%m-%d")).days)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recent activity
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔔 Recent Activity")
        st.info(friend['recent_activity'])
        st.markdown("</div>", unsafe_allow_html=True)

def display_history():
    """Display interaction history"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📚 Interaction History")

    if not st.session_state.search_history:
        st.info("No history recorded yet. Identify friends to build history.")
    else:
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(history_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

def display_recent_friends():
    """Display recently identified friends"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👥 Recently Identified Friends")

    if not st.session_state.last_seen_friends:
        st.info("No friends identified yet.")
    else:
        for friend in st.session_state.last_seen_friends:
            st.markdown(f"""
            <div class='friend-card'>
                <strong>{friend['friend_name']}</strong> (ID: {friend['friend_id']})<br>
                <small>Last seen: {friend['time']}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def friend_search():
    """Search for friends in the database"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Friend Search")

    search_query = st.text_input("Search by name or ID")

    if search_query:
        results = []
        for key, friend in friend_profiles.items():
            if (search_query.lower() in friend['name'].lower() or
                search_query.lower() in friend['id'].lower()):
                results.append((key, friend))

        if results:
            st.success(f"Found {len(results)} matches")
            for key, friend in results:
                if st.button(f"View: {friend['name']} ({friend['id']})", key=f"btn_{key}"):
                    st.session_state.current_friend = key
                    st.experimental_rerun()
        else:
            st.warning("No matches found")

    st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main application function"""
    # Load models
    models = load_face_detection_models()

    # Display app header
    app_header()

    # Create tabs for different app sections
    tab1, tab2, tab3 = st.tabs(["👁️ Recognition", "👤 Friend Details", "🔍 Search & History"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Face Recognition Camera")

        # Camera activation button
        start_camera = st.button("Start Camera", key="start_cam")
        stop_camera = st.button("Stop Camera", key="stop_cam")

        # Placeholder for camera feed
        camera_placeholder = st.empty()

        # Status message
        status_placeholder = st.empty()

        if start_camera:
            # Initialize webcam
            status_placeholder.info("Starting camera...")
            try:
                # Initialize camera
                cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    status_placeholder.error("Error: Could not open webcam. Please check your camera connection.")
                else:
                    status_placeholder.success("Camera activated! Processing video feed...")

                    # Create a placeholder for the video feed
                    frame_placeholder = camera_placeholder.empty()

                    # Process frames until stop button is clicked
                    while not stop_camera:
                        # Increment frame counter
                        st.session_state.frame_count += 1

                        # Read frame
                        ret, frame = cap.read()

                        if not ret:
                            status_placeholder.error("Error: Could not read from webcam.")
                            break

                        # Convert BGR to RGB (for display in Streamlit)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Detect faces
                        faces = detect_faces(frame_rgb, models)

                        # Set face detected flag
                        st.session_state.face_detected = len(faces) > 0

                        # Process detected faces
                        for i, (x, y, w, h) in enumerate(faces):
                            # Draw rectangle around face
                            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

                            # Detect emotion
                            if ENABLE_EMOTION_DETECTION:
                                emotion = detect_emotion(frame_rgb, (x, y, w, h), models)
                                cv2.putText(frame_rgb, emotion, (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                            # Recognize friend (for the first detected face)
                            if i == 0 and ENABLE_FRIEND_INFO:
                                friend_key = recognize_friend(frame_rgb, (x, y, w, h), models)
                                friend_info = friend_profiles[friend_key]

                                # Draw friend info card
                                frame_rgb = draw_friend_card(frame_rgb, friend_info, x, y, w, h, st.session_state.frame_count)

                                # Set current friend for detailed view
                                if st.session_state.current_friend != friend_key:
                                    st.session_state.current_friend = friend_key
                                    # Add to history
                                    add_to_history(friend_info['id'], friend_info['name'])

                        # Display processed frame
                        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                        # Small sleep to reduce CPU usage
                        time.sleep(0.01)

                    # Release resources
                    cap.release()
                    status_placeholder.info("Camera stopped.")

            except Exception as e:
                status_placeholder.error(f"Error: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        # Display friend details if one is selected
        if st.session_state.current_friend:
            display_friend_details(st.session_state.current_friend)
        else:
            st.info("No friend selected. Start the camera to detect friends or use the search function.")

    with tab3:
        # Create two columns for search and history
        col1, col2 = st.columns(2)

        with col1:
            friend_search()
            display_recent_friends()

        with col2:
            display_history()

if __name__ == "__main__":
    main()
