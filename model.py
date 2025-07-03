import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import time
def compute_statistics(image):
    """Compute mean, standard deviation, and variance of the image."""
    mean = np.mean(image)
    std_dev = np.std(image)
    variance = np.var(image)
    return mean, std_dev, variance

def compute_lbp(image, P=8, R=1):
    """Compute Local Binary Pattern (LBP) of the image."""
    lbp = local_binary_pattern(image, P, R, method='uniform')
    return lbp
def process_video_files(video_folder, output_size=(224, 224)):
    # Initialize HOG descriptor and people detector
  # hog_descriptor = cv2.HOGDescriptor()
  # hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Lists to store processed images and labels
    image_data = []
    labels = []

    # List all video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # Iterate over each video file
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Check if the video is opened correctly
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_file}")
            continue
        else:
            print(f"Processing video: {video_file}")

        # Read frames one by one
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break  # No more frames to read
            
            # Resize the frame
            frame_resized = cv2.resize(frame, output_size)

            
            # Convert to grayscale
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # Compute statistics (Mean, Standard Deviation, and Variance)
            mean_gray, std_dev_gray, variance_gray = compute_statistics(frame_gray)
            
            # Compute Local Binary Pattern (LBP) for grayscale image
            lbp_gray = compute_lbp(frame_gray)
            
            # Detect people in the frame using HOG
            # frame_with_detection = detect_people(frame_resized, hog_descriptor)
            # Append the processed frame and label
            image_data.append(frame_gray)  # Add your processed frame here
            labels.append(0)  # Replace with the correct label, depending on your dataset
            
             # Print the computed statistics
            print(f"Video: {video_file}, Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
            #print(f"RGB Mean: {mean_rgb}, RGB Std Dev: {std_dev_rgb}, RGB Variance: {variance_rgb}")
            print(f"Gray Mean: {mean_gray}, Gray Std Dev: {std_dev_gray}, Gray Variance: {variance_gray}")
            print("-----")
            
            #Create subplots with 1 row and 4 columns (including detection results)
            fig, axes = plt.subplots(1, 2, figsize=(5, 5))
            # Display grayscale frame in the second column
            axes[0].imshow(frame_gray, cmap='gray')
            axes[0].set_title("Grayscale Frame")
            axes[0].axis('off')  # Hide axis
            
            # Display LBP image in the third column
            axes[1].imshow(lbp_gray, cmap='gray')
            axes[1].set_title("LBP Frame")
            axes[1].axis('off')  # Hide axis
            plt.show()

        # Release the video capture object
        cap.release()

    return image_data, labels
def split_data(image_data, labels, test_size=0.2):
    # Convert lists to numpy arrays for easier manipulation
    image_data = np.array(image_data)
    labels = np.array(labels)

    # Check the size of the data
    print(f"Number of images processed: {len(image_data)}")
    print(f"Number of labels processed: {len(labels)}")

    if len(image_data) == 0:
        print("Error: No images processed. Check your video files or processing logic.")
        return None, None, None, None

    # Encode labels to integers (if they are not already integers)
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels_encoded = np.array([label_map[label] for label in labels])

    # One-hot encode the labels
    labels_encoded = to_categorical(labels_encoded, num_classes=len(unique_labels))
    #reshape the data
    image_data = image_data.reshape(image_data.shape[0], image_data.shape[1], image_data.shape[2], 1)
    image_data = np.array(image_data).reshape((len(image_data), 224, 224, 1))

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(image_data, labels_encoded, test_size=test_size, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")


    return X_train, X_test, y_train, y_test

# --- Simplified DNN and CNN Models ---
def build_dnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))  
    model.add(layers.Dropout(0.4))  # Increased dropout for regularization
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer
    return model


def build_cnn_model(input_shape):
    """Build a CNN model."""
    model = models.Sequential()
    
    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))  # Number of classes
    
    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate both DNN and CNN models."""
    input_shape = X_train.shape[1:]  # The input shape should match the image dimensions

    # DNN Model
    dnn_model = build_dnn_model(input_shape)
    dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    dnn_test_loss, dnn_test_accuracy = dnn_model.evaluate(X_test, y_test)
    dnn_model = build_dnn_model(input_shape)
    dnn_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
                      loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    dnn_history = dnn_model.fit(X_train, y_train, epochs=4, batch_size=2, validation_split=0.10)  # Reduced epochs and batch size
    dnn_train_time = time.time() - start_time
    dnn_test_loss, dnn_test_accuracy = dnn_model.evaluate(X_test, y_test)
    print(f"DNN Test Accuracy: {dnn_test_accuracy:.4f}")
    print(f"DNN Test Precision: {dnn_test_accuracy:.4f}")
    # Save DNN model
    dnn_model.save('dnn_model.h5')  # Save the model to a .h5 file
    
    # CNN Model
    cnn_model = build_cnn_model(input_shape)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test, y_test)
    print(f"CNN Test Accuracy: {cnn_test_accuracy:.4f}")
    print(f"CNN Test Precision: {cnn_test_accuracy:.4f}")
    # Save CNN model
    cnn_model.save('cnn_model.h5')  # Save the model to a .h5 file

# Example usage:
video_folder_path = 'dataset/'
image_data, labels = process_video_files(video_folder_path)
X_train, X_test, y_train, y_test = split_data(image_data, labels)
# Train and evaluate models
train_and_evaluate_model(X_train, X_test, y_train, y_test)
