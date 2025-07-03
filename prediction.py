import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
import os
import time
import base64

# ==================== Background image ===
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image
add_bg_from_local('image4.jpg')

# Function to detect people using HOG descriptor
def detect_people(frame, hog_descriptor):
    (rects, weights) = hog_descriptor.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    # Draw bounding boxes for detected people
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color rectangle
    return frame

# Load the trained CNN model
def load_cnn_model(model_path='cnn_model.h5'):
    model = load_model(model_path)
    return model

# Function to process video files (detect person using CNN + HOG)
def process_video_file(video_file, model, output_size=(224, 224)):
    # temp_file_path = "uploaded_video.mp4"
    # # Write the file to the temporary path
    # with open(temp_file_path, "wb") as temp_file:
    #     temp_file.write(video_file.getbuffer())
    # Initialize HOG descriptor and people detector
    hog_descriptor = cv2.HOGDescriptor()
    hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Open the video
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        st.error(f"Error: Unable to open video {video_file}")
        return

    # Get video info for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the output video path and codec
    output_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # No more frames to read

        # Resize the frame to the size required by CNN
        frame_resized = cv2.resize(frame, output_size)

        # Convert frame to grayscale (needed for the CNN model)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Expand dimensions to match the input shape expected by the model
        frame_expanded = np.expand_dims(frame_gray, axis=-1)  # Shape becomes (224, 224, 1)
        frame_expanded = np.expand_dims(frame_expanded, axis=0)  # Shape becomes (1, 224, 224, 1)

        # Normalize the frame (optional, depending on how the model was trained)
        frame_expanded = frame_expanded / 255.0  # Normalize to [0, 1]

        # Predict the label using CNN model
        prediction = model.predict(frame_expanded)

        # Check if person detected (assuming prediction returns [0, 1] for binary classification)
        if prediction[0][0] > 0.5:
            # Detect people using HOG and draw bounding boxes
            frame_with_detection = detect_people(frame, hog_descriptor)

            # Convert frame to RGB for displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame_with_detection, cv2.COLOR_BGR2RGB)

            # Write frame to the output video file
            out.write(frame_with_detection)

        else:
            # Write the original frame if no person is detected
            out.write(frame)

    cap.release()
    out.release()

    return output_path

# Main function to upload video file using Streamlit
def main():
    # Set up Streamlit interface
    import streamlit as st
    st.markdown("<h1 style='color: #00FFFF;'>Person Detection in Video</h1>", unsafe_allow_html=True)


    # File uploader widget
    video_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Save uploaded video to a temporary file
        with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        st.video(video_file)  # Display the uploaded video

        # Load CNN model
        model = load_cnn_model('cnn_model.h5')  # Load pre-trained CNN model

        # Process the selected video
        output_video_path = process_video_file(video_path, model)

        # Display the processed video with detections
        if output_video_path:
            # Output processed video with bounding boxes
            st.markdown("<h3 style='color: #00FFFF;'>Processed video with person detection:</h3>", unsafe_allow_html=True)
            #st.video(output_video_path)  # Provide path to the processed video
            
            # Open the processed video for frame-by-frame display
            cap = cv2.VideoCapture(output_video_path)
            
            # Get the total number of frames and FPS
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate the time delay based on FPS for smooth playback
            delay = 1.0 / fps
            
            # Create a placeholder for the video frame
            frame_placeholder = st.empty()
            
            frame_index = 0
            while frame_index < frame_count:
                # Set the video to the current frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB format
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Display the frame
                    frame_placeholder.image(frame, caption=f"Frame {frame_index}", use_column_width=True)

                    # Update frame index
                    frame_index += 1

                    # Delay to control playback speed
                    time.sleep(delay)
                else:
                    break
                
            # Release the video capture object
            cap.release()

if __name__ == "__main__":
    main()

