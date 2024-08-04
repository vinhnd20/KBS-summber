import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Base directory containing the data
base_dir = "preprocessing"

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load data and labels for fake voice recognition
def load_data_for_fake_voice_recognition(base_dir):
    X = []
    y = []
    fake_voice_folders = set(map(str, range(80)))  # Folders 0 to 79 are fake voices

    for speaker_folder in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_folder)
        if os.path.isdir(speaker_path):
            is_fake_voice = speaker_folder in fake_voice_folders
            label = 1 if is_fake_voice else 0  # 1 for fake, 0 for real
            print(f"Processing speaker folder: {speaker_path} (Fake: {is_fake_voice})")

            for type_folder in os.listdir(speaker_path):
                type_path = os.path.join(speaker_path, type_folder)
                if os.path.isdir(type_path):
                    for file_name in os.listdir(type_path):
                        if file_name.endswith(".wav"):
                            file_path = os.path.join(type_path, file_name)
                            print(f"Found audio file: {file_path}")
                            try:
                                features = extract_features(file_path)
                                X.append(features)
                                y.append(label)
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)

# Load data
X, y = load_data_for_fake_voice_recognition(base_dir)
print(f"Loaded {len(X)} samples.")

# Check if the dataset is empty
if len(X) == 0:
    print("No data found. Please check the dataset directory and file paths.")
else:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train an SVM classifier
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
