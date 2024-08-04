import joblib
import numpy as np
import librosa

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load the trained models and scalers
model_sv = joblib.load('model_sv.pkl')
scaler_sv = joblib.load('scaler_sv.pkl')
model_fv = joblib.load('model_fv.pkl')
scaler_fv = joblib.load('scaler_fv.pkl')
model_cd = joblib.load('model_cd.pkl')
scaler_cd = joblib.load('scaler_cd.pkl')

# Function to predict using trained models
def predict(audio_file):
    features = extract_features(audio_file).reshape(1, -1)

    # Speaker Verification
    features_sv = scaler_sv.transform(features)
    sv_result = model_sv.predict(features_sv)

    # Fake Voice Recognition
    features_fv = scaler_fv.transform(features)
    fv_result = model_fv.predict(features_fv)

    # Command Detection
    features_cd = scaler_cd.transform(features)
    cd_result = model_cd.predict(features_cd)

    return sv_result[0], fv_result[0], cd_result[0]

# Example usage
audio_file = "/home/vinh/KBS/preprocessing/B19DCCN133_Dung_Linh/Type D/1.wav"
sv_result, fv_result, cd_result = predict(audio_file)

print(f"Speaker Verification Result: {'Verified' if sv_result == 1 else 'Not Verified'}")
print(f"Fake Voice Recognition Result: {'Fake' if fv_result == 1 else 'Real'}")
print(f"Command Detection Result: {'Command Detected' if cd_result == 1 else 'No Command Detected'}")
