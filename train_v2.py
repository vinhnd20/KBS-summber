import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Base directory containing the data
base_dir = "preprocessing"

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load data for speaker verification
def load_data_for_speaker_verification(base_dir):
    X = []
    y = []
    label_counter = 0
    for speaker_folder in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_folder)
        if os.path.isdir(speaker_path):
            for type_folder in os.listdir(speaker_path):
                type_path = os.path.join(speaker_path, type_folder)
                if os.path.isdir(type_path):
                    for file_name in os.listdir(type_path):
                        if file_name.endswith(".wav"):
                            file_path = os.path.join(type_path, file_name)
                            features = extract_features(file_path)
                            X.append(features)
                            y.append(label_counter)
            label_counter += 1
    return np.array(X), np.array(y)

# Load data for fake voice recognition
def load_data_for_fake_voice_recognition(base_dir):
    X = []
    y = []
    fake_voice_folders = set(map(str, range(80)))  # Folders 0 to 79 are fake voices
    for speaker_folder in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_folder)
        if os.path.isdir(speaker_path):
            is_fake_voice = speaker_folder in fake_voice_folders
            label = 1 if is_fake_voice else 0  # 1 for fake, 0 for real
            for type_folder in os.listdir(speaker_path):
                type_path = os.path.join(speaker_path, type_folder)
                if os.path.isdir(type_path):
                    for file_name in os.listdir(type_path):
                        if file_name.endswith(".wav"):
                            file_path = os.path.join(type_path, file_name)
                            features = extract_features(file_path)
                            X.append(features)
                            y.append(label)
    return np.array(X), np.array(y)

# Load data for command detection
def load_data_for_command_detection(base_dir):
    X = []
    y = []
    label_mapping = {"Type A": 1, "Type B": 1, "Type C": 0, "Type D": 0}
    for user_dir in os.listdir(base_dir):
        user_path = os.path.join(base_dir, user_dir)
        if os.path.isdir(user_path):
            for type_dir in label_mapping:
                type_path = os.path.join(user_path, type_dir)
                if os.path.isdir(type_path):
                    for file_name in os.listdir(type_path):
                        file_path = os.path.join(type_path, file_name)
                        if os.path.isfile(file_path) and file_path.endswith('.wav'):
                            y_audio, sr = librosa.load(file_path, sr=None)
                            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                            mfcc_mean = np.mean(mfcc, axis=1)
                            X.append(mfcc_mean)
                            y.append(label_mapping[type_dir])
    return np.array(X), np.array(y)

# Train models for each task and save them
def train_and_save_models():
    # Speaker Verification Model
    X_sv, y_sv = load_data_for_speaker_verification(base_dir)
    X_train_sv, X_test_sv, y_train_sv, y_test_sv = train_test_split(X_sv, y_sv, test_size=0.2, random_state=42)
    scaler_sv = StandardScaler()
    X_train_sv = scaler_sv.fit_transform(X_train_sv)
    X_test_sv = scaler_sv.transform(X_test_sv)
    model_sv = SVC(kernel='linear', probability=True)
    model_sv.fit(X_train_sv, y_train_sv)
    joblib.dump(model_sv, 'model_sv.pkl')
    joblib.dump(scaler_sv, 'scaler_sv.pkl')

    # Fake Voice Recognition Model
    X_fv, y_fv = load_data_for_fake_voice_recognition(base_dir)
    X_train_fv, X_test_fv, y_train_fv, y_test_fv = train_test_split(X_fv, y_fv, test_size=0.2, random_state=42)
    scaler_fv = StandardScaler()
    X_train_fv = scaler_fv.fit_transform(X_train_fv)
    X_test_fv = scaler_fv.transform(X_test_fv)
    model_fv = SVC(kernel='linear', probability=True)
    model_fv.fit(X_train_fv, y_train_fv)
    joblib.dump(model_fv, 'model_fv.pkl')
    joblib.dump(scaler_fv, 'scaler_fv.pkl')

    # Command Detection Model
    X_cd, y_cd = load_data_for_command_detection(base_dir)
    X_train_cd, X_test_cd, y_train_cd, y_test_cd = train_test_split(X_cd, y_cd, test_size=0.2, random_state=42)
    scaler_cd = StandardScaler()
    X_train_cd = scaler_cd.fit_transform(X_train_cd)
    X_test_cd = scaler_cd.transform(X_test_cd)
    model_cd = SVC(kernel='linear', probability=True)
    model_cd.fit(X_train_cd, y_train_cd)
    joblib.dump(model_cd, 'model_cd.pkl')
    joblib.dump(scaler_cd, 'scaler_cd.pkl')

# Train and save models
train_and_save_models()
