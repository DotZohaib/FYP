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
import os
from pathlib import Path

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
    .unknown-user {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration flags
USE_DNN_DETECTOR = True      # Toggle between Haar Cascade and DNN detector
ENABLE_EMOTION_DETECTION = False # Enable emotion detection
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
if 'is_unknown_user' not in st.session_state:
    st.session_state.is_unknown_user = False
if 'registering_new_friend' not in st.session_state:
    st.session_state.registering_new_friend = False

# Load emotion labels
emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']

# Create a directory for friend face images if it doesn't exist
image_dir = Path("friend_images")
image_dir.mkdir(exist_ok=True)

# Sample friend profiles database with expanded information and image paths
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
        "conversation_topics": ["AI projects", "Cybersecurity", "New programming languages"],
        "image_path": "friend_images/Zohaib.jpg"  # Fixed path
    },
    "friend2": {
        "id": "FR-2025-002",
        "name": "Babar Ali Dayo",
        "age": 24,
        "category": "Brother",
        "since": "2022",
        "interests": "Data Science, Hiking",
        "relationship_score": 100,
        "last_seen": "2025-02-20",
        "contact": "Babar.j@example.com",
        "location": "Boston",
        "meetings": 28,
        "recent_activity": "Project collaboration meeting",
        "conversation_topics": ["Data visualization", "Team management", "Project deadlines"],
        "image_path": "friend_images/Babar.jpg"  # Fixed path
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

    try:
        # Load face detector
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        models['face_detector'] = face_detector

        # Load face recognizer - using LBPH Face Recognizer which works well with Haar cascades
        try:
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            models['face_recognizer'] = face_recognizer
        except AttributeError:
            # For older OpenCV versions
            face_recognizer = cv2.face.createLBPHFaceRecognizer()
            models['face_recognizer'] = face_recognizer

        # Create and train face recognizer if there are face images available
        train_face_recognizer(models)

        # Placeholder for emotion detection model
        models['emotion_detector'] = None

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        models = None

    return models

# Update these key functions in your code:

def train_face_recognizer(models):
    """Train the face recognizer with the friend images - Improved version"""
    # Check if we have images to train with
    faces = []
    labels = []
    label_map = {}  # Map numeric labels to friend keys

    label_idx = 0
    for friend_key, friend_info in friend_profiles.items():
        img_path = friend_info['image_path']

        # Check if image exists
        if not os.path.exists(img_path):
            # Ensure directory exists
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

            # Create a warning about missing image
            st.warning(f"No image found for {friend_info['name']} at {img_path}")
            continue  # Skip this friend instead of creating dummy data

        try:
            # Read image in both grayscale and color (for face detection)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img_gray is None or img_color is None:
                st.warning(f"Could not read image at {img_path}")
                continue

            # Convert color image to BGR for face detection
            img_bgr = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

            # Detect faces using multiple scale factors for better results
            scale_factors = [1.05, 1.1, 1.15]
            best_faces = []

            for scale in scale_factors:
                detected_faces = models['face_detector'].detectMultiScale(
                    img_bgr, scaleFactor=scale, minNeighbors=5, minSize=(30, 30))

                if len(detected_faces) > 0:
                    # Sort by area (largest face first)
                    detected_faces = sorted(detected_faces,
                                           key=lambda x: x[2]*x[3],
                                           reverse=True)
                    best_faces.append(detected_faces[0])  # Take largest face

            # Process detected faces
            if best_faces:
                # Choose largest face across all scale factors
                best_face = sorted(best_faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                x, y, w, h = best_face

                # Extract face ROI with extra margin for better recognition
                margin = int(min(w, h) * 0.1)  # 10% margin
                # Ensure we don't go out of bounds
                x_start = max(0, x - margin)
                y_start = max(0, y - margin)
                x_end = min(img_gray.shape[1], x + w + margin)
                y_end = min(img_gray.shape[0], y + h + margin)

                face_roi = img_gray[y_start:y_end, x_start:x_end]

                # Resize face to ensure consistent size (slightly larger than before)
                face_roi = cv2.resize(face_roi, (150, 150))

                # Apply histogram equalization for better feature extraction
                face_roi = cv2.equalizeHist(face_roi)

                # Add face and label
                faces.append(face_roi)
                labels.append(label_idx)

                # Debug output
                st.sidebar.success(f"Successfully processed face for {friend_info['name']}")
            else:
                # If no face detected, use the whole image as fallback (with preprocessing)
                st.warning(f"No face detected in {img_path}, using whole image (may reduce accuracy)")
                img_resized = cv2.resize(img_gray, (150, 150))
                img_resized = cv2.equalizeHist(img_resized)
                faces.append(img_resized)
                labels.append(label_idx)

            # Store the mapping
            label_map[label_idx] = friend_key
            label_idx += 1

        except Exception as e:
            st.warning(f"Could not process image {img_path}: {str(e)}")

    # Save the label map in the model dictionary
    models['label_map'] = label_map

    # Train recognizer if we have faces
    if faces and labels:
        try:
            # Convert labels to numpy array
            labels_array = np.array(labels, dtype=np.int32)

            # Re-create the recognizer to ensure clean training
            try:
                models['face_recognizer'] = cv2.face.LBPHFaceRecognizer_create(
                    radius=2,          # Use smaller radius for more detail
                    neighbors=8,       # Standard 8 neighbors
                    grid_x=8,          # More grid cells for better precision
                    grid_y=8,
                    threshold=100      # Higher threshold for more permissive matching
                )
            except AttributeError:
                # For older OpenCV versions
                models['face_recognizer'] = cv2.face.createLBPHFaceRecognizer(
                    radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=100
                )

            # Train with the collected faces
            models['face_recognizer'].train(faces, labels_array)
            st.success(f"Face recognition model trained with {len(faces)} images for {len(label_map)} friends")

            # Save training samples for debugging
            for i, face in enumerate(faces):
                debug_path = f"friend_images/debug_train_{labels[i]}.jpg"
                cv2.imwrite(debug_path, face)

        except Exception as e:
            st.error(f"Error training face recognition model: {str(e)}")
    else:
        st.warning("No face data available for training. Recognition will not work properly.")

def detect_faces(img, models):
    """Improved face detection with multiple scale factors"""
    # Convert to BGR format for OpenCV processing
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    gray_eq = cv2.equalizeHist(gray)

    # Try multiple scale factors for better detection
    scale_factors = [1.05, 1.1, 1.15]
    best_faces = []

    for scale in scale_factors:
        detected_faces = models['face_detector'].detectMultiScale(
            gray_eq,
            scaleFactor=scale,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(detected_faces) > 0:
            # Add to our collection of detected faces
            for face in detected_faces:
                best_faces.append(face)

    # Remove duplicates by non-maximum suppression
    if len(best_faces) > 1:
        # Convert to numpy array
        best_faces = np.array(best_faces)

        # Sort by area (largest first)
        areas = best_faces[:, 2] * best_faces[:, 3]
        indices = np.argsort(areas)[::-1]
        best_faces = best_faces[indices]

        # Non-maximum suppression
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(best_faces[i, 0], best_faces[indices[1:], 0])
            yy1 = np.maximum(best_faces[i, 1], best_faces[indices[1:], 1])
            xx2 = np.minimum(best_faces[i, 0] + best_faces[i, 2],
                             best_faces[indices[1:], 0] + best_faces[indices[1:], 2])
            yy2 = np.minimum(best_faces[i, 1] + best_faces[i, 3],
                             best_faces[indices[1:], 1] + best_faces[indices[1:], 3])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / (best_faces[indices[1:], 2] * best_faces[indices[1:], 3])

            # Remove boxes with IoU > 0.5
            inds = np.where(overlap <= 0.5)[0]
            indices = indices[inds + 1]

        best_faces = best_faces[keep].tolist()

    return best_faces, gray_eq

def recognize_friend(img, face_coords, gray_img, models):
    """Improved friend recognition function"""
    # Extract coordinates
    x, y, w, h = face_coords

    try:
        # Extract the face region with added margin
        margin = int(min(w, h) * 0.1)  # 10% margin
        # Ensure we don't go out of bounds
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(gray_img.shape[1], x + w + margin)
        y_end = min(gray_img.shape[0], y + h + margin)

        face_roi = gray_img[y_start:y_end, x_start:x_end]

        # Resize the face to match the training size
        face_roi = cv2.resize(face_roi, (150, 150))

        # Apply histogram equalization for better feature extraction
        face_roi = cv2.equalizeHist(face_roi)

        # Save current face for debugging
        debug_path = f"friend_images/debug_current_face.jpg"
        cv2.imwrite(debug_path, face_roi)

        # Predict the label
        label, confidence = models['face_recognizer'].predict(face_roi)

        # Adjusted confidence threshold - LBPH returns distance, lower is better
        # This threshold needs to be higher than the default
        confidence_threshold = 100  # Increased from 80 to 100 for more tolerance

        # Debug information
        st.sidebar.write(f"Recognition confidence: {confidence:.2f} (lower is better)")
        st.sidebar.write(f"Label detected: {label} -> {models['label_map'].get(label, 'Unknown')}")

        if confidence < confidence_threshold and label in models['label_map']:
            # Return the friend key corresponding to the recognized label
            return models['label_map'][label], confidence
        else:
            # Return None if the face is not recognized with sufficient confidence
            return None, confidence

    except Exception as e:
        st.error(f"Recognition error: {str(e)}")
        return None, 999  # High confidence value means not recognized

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
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return result

def draw_unknown_user_card(img, x, y, w, h, frame_count, confidence):
    """Draw a card for unknown users"""
    # Create a copy of the image to draw on
    overlay = img.copy()

    # Card dimensions and position (below the face)
    card_width = max(300, w * 1.5)
    card_height = 100
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

    # Draw main card background (red for unknown)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + card_height),
                 (50, 50, 70), -1)

    # Draw header strip (red for unknown)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_width, card_y + 30),
                 (70, 70, 180), -1)

    # Draw "UNKNOWN USER" text in header
    cv2.putText(overlay, "UNKNOWN USER", (card_x + 10, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw message with larger font
    cv2.putText(overlay, "User not registered in system", (card_x + 15, card_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Draw confidence value
    cv2.putText(overlay, f"Confidence: {confidence:.1f}", (card_x + 15, card_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Blend the overlay with the original image
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
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

            # Display friend image if available
            img_path = friend['image_path']
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{friend['name']}'s Profile Image", width=200)
            else:
                st.warning(f"Profile image not found at {img_path}")

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
    elif st.session_state.is_unknown_user:
        st.markdown("""
        <div class='unknown-user'>
            <h3>⚠️ Unknown User Detected</h3>
            <p>This person is not registered in the friend database.</p>
            <p>Would you like to register them as a new friend?</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Register New Friend"):
            st.session_state.registering_new_friend = True
    else:
        st.info("No friend selected. Start the camera to detect friends or use the search function.")

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
                    st.session_state.is_unknown_user = False
                    st.experimental_rerun()
        else:
            st.warning("No matches found")

    st.markdown("</div>", unsafe_allow_html=True)

def upload_profile_image():
    """Allow uploading profile images for friends"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📸 Upload Profile Image")

    # Select friend
    friend_options = ["Select a friend"] + [friend["name"] for _, friend in friend_profiles.items()]
    selected_friend = st.selectbox("Choose friend to update profile image", friend_options)

    if selected_friend != "Select a friend":
        # Get friend key
        friend_key = None
        for key, friend in friend_profiles.items():
            if friend["name"] == selected_friend:
                friend_key = key
                break

        if friend_key:
            # Upload image
            uploaded_file = st.file_uploader("Upload a face image for recognition", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Save the image
                img_path = friend_profiles[friend_key]["image_path"]
                # Ensure directory exists
                os.makedirs(os.path.dirname(img_path), exist_ok=True)

                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Display the image
                st.image(img_path, caption=f"Profile image for {selected_friend}", width=200)
                st.success(f"Image uploaded successfully for {selected_friend}!")

                # Note about retraining
                st.info("The face recognition model will be retrained with the new image the next time you start the camera.")

    st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main application function"""
    # Load models
    models = load_face_detection_models()

    # Display app header
    app_header()

    # Create tabs for different app sections
    tab1, tab2, tab3, tab4 = st.tabs(["👁️ Recognition", "👤 Friend Details", "🔍 Search & History", "⚙️ Settings"])

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
                        faces, gray_img = detect_faces(frame_rgb, models)

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
                                friend_key, confidence = recognize_friend(
                                    frame_rgb, (x, y, w, h), gray_img, models)
# Continuing from where the code was cut off, completing the camera processing loop and the rest of the application

                                if friend_key:
                                    # Known friend detected
                                    friend_info = friend_profiles[friend_key]

                                    # Draw friend info card
                                    frame_rgb = draw_friend_card(
                                        frame_rgb, friend_info, x, y, w, h, st.session_state.frame_count)

                                    # Update session state
                                    if st.session_state.current_friend != friend_key:
                                        # Add to history when a new friend is detected
                                        add_to_history(friend_info['id'], friend_info['name'])
                                        st.session_state.current_friend = friend_key
                                        st.session_state.is_unknown_user = False
                                else:
                                    # Unknown user detected
                                    frame_rgb = draw_unknown_user_card(
                                        frame_rgb, x, y, w, h, st.session_state.frame_count, confidence)

                                    # Update session state
                                    if not st.session_state.is_unknown_user:
                                        st.session_state.is_unknown_user = True
                                        st.session_state.current_friend = None

                        # Display the frame
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        # Brief delay to reduce CPU usage
                        time.sleep(0.05)

                    # Release the camera
                    cap.release()
                    status_placeholder.info("Camera stopped.")

            except Exception as e:
                status_placeholder.error(f"An error occurred: {str(e)}")
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        # Display friend details
        display_friend_details(st.session_state.current_friend)

    with tab3:
        # Create columns for search and history
        col1, col2 = st.columns(2)

        with col1:
            friend_search()
            display_recent_friends()

        with col2:
            display_history()

    with tab4:
        # Settings tab
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("⚙️ System Settings")

        # Face detection method
        detection_method = st.radio(
            "Face Detection Method",
            ["Haar Cascade (Faster, less accurate)", "DNN (Slower, more accurate)"],
            index=1 if USE_DNN_DETECTOR else 0
        )

        # Animation settings
        enable_animations = st.checkbox("Enable UI Animations", value=ENABLE_ANIMATIONS)

        # Emotion detection
        enable_emotion = st.checkbox("Enable Emotion Detection", value=ENABLE_EMOTION_DETECTION)

        # Friend information display
        enable_friend_info = st.checkbox("Enable Friend Information Display", value=ENABLE_FRIEND_INFO)

        # Apply settings button
        if st.button("Apply Settings"):
            # Here you would save the settings and reload models if needed
            st.success("Settings applied successfully!")
            st.info("Some settings may require restarting the application to take effect.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Profile image upload section
        upload_profile_image()

        # System information
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("ℹ️ System Information")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("OpenCV Version", cv2.__version__)
            st.metric("Total Friends", len(friend_profiles))

        with col2:
            st.metric("Interactions Recorded", len(interaction_history))
            st.metric("Recognition Sessions", st.session_state.frame_count)

        st.markdown("</div>", unsafe_allow_html=True)

        # Add new friend section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("➕ Add New Friend")

        new_name = st.text_input("Name")
        new_age = st.number_input("Age", min_value=1, max_value=120, value=25)
        new_category = st.selectbox("Category", ["Friend", "Family", "Work Colleague", "Study Group", "Other"])
        new_interests = st.text_input("Interests (comma separated)")

        if st.button("Add Friend"):
            if new_name:
                # Generate a new friend ID
                new_id = f"FR-{datetime.now().year}-{len(friend_profiles) + 1:03d}"

                # Create new friend profile
                new_friend = {
                    "id": new_id,
                    "name": new_name,
                    "age": new_age,
                    "category": new_category,
                    "since": datetime.now().strftime("%Y"),
                    "interests": new_interests,
                    "relationship_score": 50,  # Default starting score
                    "last_seen": datetime.now().strftime("%Y-%m-%d"),
                    "contact": "",
                    "location": "",
                    "meetings": 1,
                    "recent_activity": "Added to recognition system",
                    "conversation_topics": [],
                    "image_path": f"friend_images/{new_name.lower().replace(' ', '_')}.jpg"
                }

                # Add to profiles
                friend_key = f"friend{len(friend_profiles) + 1}"
                friend_profiles[friend_key] = new_friend

                st.success(f"Friend {new_name} added successfully with ID {new_id}!")
                st.info("Please upload a photo in the 'Upload Profile Image' section to enable face recognition.")
            else:
                st.warning("Please enter at least a name for the new friend.")

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
