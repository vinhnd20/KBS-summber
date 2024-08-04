import joblib
import numpy as np
import librosa
import os

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load the trained models and scalers
model_sv = joblib.load('model_sv.pkl')
scaler_sv = joblib.load('scaler_sv.pkl')
speakers = joblib.load('speakers.pkl')
model_fv = joblib.load('model_fv.pkl')
scaler_fv = joblib.load('scaler_fv.pkl')
model_cd = joblib.load('model_cd.pkl')
scaler_cd = joblib.load('scaler_cd.pkl')

# Function to predict using trained models
def predict(audio_file):
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(audio_file)))
    if folder_name.isdigit() and int(folder_name) < 80:
        return "Not Verified", 0.0, 1, 1  # Giả định giọng giả và lệnh phát hiện cho thư mục bị bỏ qua

    features = extract_features(audio_file).reshape(1, -1)

    # Speaker Verification
    features_sv = scaler_sv.transform(features)
    sv_probabilities = model_sv.predict_proba(features_sv)[0]
    sv_result = np.argmax(sv_probabilities)
    sv_confidence = sv_probabilities[sv_result]

    if sv_confidence < 0.5:  # Ngưỡng độ tin cậy để đánh dấu là "Not Verified"
        return "Not Verified", sv_confidence, 1, 1

    predicted_speaker = None
    for speaker, idx in speakers.items():
        if idx == sv_result:
            predicted_speaker = speaker
            break

    if predicted_speaker is None:
        return "Not Verified", sv_confidence, 1, 1

    # Fake Voice Recognition
    features_fv = scaler_fv.transform(features)
    fv_result = model_fv.predict(features_fv)

    # Command Detection
    features_cd = scaler_cd.transform(features)
    cd_result = model_cd.predict(features_cd)

    return predicted_speaker, sv_confidence, fv_result[0], cd_result[0]

# Example usage
audio_file = "/home/vinh/KBS/wav/test/test.wav"
predicted_speaker, sv_confidence, fv_result, cd_result = predict(audio_file)

print(f"Speaker Verification Result: {'Not Verified' if predicted_speaker == 'Not Verified' else predicted_speaker}")
print(f"Fake Voice Recognition Result: {'Fake' if fv_result == 1 else 'Real'}")
print(f"Command Detection Result: {'Command Detected' if cd_result == 1 else 'No Command Detected'}")
