# Person Re-identification Project

This project implements person re-identification using deep learning techniques, specifically CNNs and DNNs.

## Project Structure
- `model.py`: Contains the model architecture and training code
- `prediction.py`: Streamlit web app for person re-identification
- `login.py`: User authentication module
- `register.py`: User registration module
- `welcome.py`: Welcome screen
- `dataset/`: Directory containing training data
- `test_video/`: Directory containing test videos
- `cnn_model.h5`: Pre-trained CNN model
- `dnn_model.h5`: Pre-trained DNN model

## Setup Instructions

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run welcome.py
   ```

## Usage
1. Register a new account or log in with existing credentials
2. Upload a video file for person re-identification
3. The system will process the video and display the results

## Note
- Make sure you have Python 3.7 or higher installed
- The application requires a webcam for real-time detection (if implemented)
- For best results, use well-lit environments with clear video quality
