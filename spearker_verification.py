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
    speakers = {}
    label_counter = 0
    for speaker_folder in os.listdir(base_dir):
        if speaker_folder.isdigit() and int(speaker_folder) < 80:
            continue
        speaker_path = os.path.join(base_dir, speaker_folder)
        if os.path.isdir(speaker_path):
            if speaker_folder not in speakers:
                speakers[speaker_folder] = label_counter
                label_counter += 1
            for type_folder in os.listdir(speaker_path):
                type_path = os.path.join(speaker_path, type_folder)
                if os.path.isdir(type_path):
                    for file_name in os.listdir(type_path):
                        if file_name.endswith(".wav"):
                            file_path = os.path.join(type_path, file_name)
                            features = extract_features(file_path)
                            X.append(features)
                            y.append(speakers[speaker_folder])
    return np.array(X), np.array(y), speakers

# Load data and train speaker verification model
def train_speaker_verification_model():
    X, y, speakers = load_data_for_speaker_verification(base_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'model_sv.pkl')
    joblib.dump(scaler, 'scaler_sv.pkl')
    joblib.dump(speakers, 'speakers.pkl')
    
    print(f"Model trained with accuracy: {model.score(X_test, y_test):.2f}")

# Function to verify if the new audio file belongs to the correct speaker
def verify_speaker(audio_file_path):
    model = joblib.load('model_sv.pkl')
    scaler = joblib.load('scaler_sv.pkl')
    speakers = joblib.load('speakers.pkl')
    
    features = extract_features(audio_file_path)
    features = scaler.transform([features])
    
    probabilities = model.predict_proba(features)[0]
    predicted_speaker_idx = np.argmax(probabilities)
    
    for speaker, idx in speakers.items():
        if idx == predicted_speaker_idx:
            predicted_speaker = speaker
            break
    
    return predicted_speaker, probabilities[predicted_speaker_idx]

# Train the speaker verification model
train_speaker_verification_model()

# Example usage for verifying a new audio file
audio_file_path = '/home/vinh/KBS/preprocessing/B19DCCN259_Hieu_Dep/Type B/1.wav'
predicted_speaker, confidence = verify_speaker(audio_file_path)
print(f"Predicted Speaker: {predicted_speaker}, Confidence: {confidence:.2f}")
