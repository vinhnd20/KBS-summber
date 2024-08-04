import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Define the base directory
base_dir = "preprocessing"

# Initialize lists to hold MFCC features and labels
X = []
y = []

# Label mapping
label_mapping = {"Type A": 0, "Type B": 1, "Type C": 1, "Type D": 1}

# Iterate through each user directory
for user_dir in os.listdir(base_dir):
    user_path = os.path.join(base_dir, user_dir)
    if os.path.isdir(user_path):
        # Iterate through each type directory
        for type_dir, label in label_mapping.items():
            type_path = os.path.join(user_path, type_dir)
            if os.path.isdir(type_path):
                # Iterate through each file in the type directory
                for file_name in os.listdir(type_path):
                    file_path = os.path.join(type_path, file_name)
                    if os.path.isfile(file_path) and file_path.endswith('.wav'):
                        # Load the audio file
                        y_audio, sr = librosa.load(file_path, sr=None)
                        # Extract MFCC features
                        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                        # Average MFCC features over time
                        mfcc_mean = np.mean(mfcc, axis=1)
                        # Append the MFCC features and label to the lists
                        X.append(mfcc_mean)
                        y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVC model
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Make predictions
y_pred = svc.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Now you have trained an SVC model and evaluated its performance
