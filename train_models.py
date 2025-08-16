import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from preprocess import preprocess_audio
from feature_extraction import extract_features
import joblib

# Path to Mozilla Common Voice Dataset
mozilla_tsv = r'data\Mozilla Common Voice Dataset\validated.tsv'
mozilla_clips_dir = r'data\Mozilla Common Voice Dataset\clips'

# Path to CREMA-D Dataset
crema_dir = r'data\CREMA-D Dataset'

def load_mozilla_dataset(tsv_file, clips_dir):
    """
    Load Mozilla Common Voice dataset for gender and age classification.
    """
    # Load metadata
    print("Loading Mozilla Common Voice dataset...")
    df = pd.read_csv(tsv_file, sep='\t')

    # Drop rows with missing gender or age values
    df = df.dropna(subset=['gender', 'age'])

    # Filter for gender classification (include variations like male_masculine and female_feminine)
    valid_genders = ['male', 'female', 'male_masculine', 'female_feminine']
    gender_data = df[df['gender'].isin(valid_genders)]

    # Filter for age classification (include valid age ranges)
    valid_ages = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties']
    age_data = df[df['age'].isin(valid_ages)]

    print(f"Total rows in dataset: {len(df)}")
    print(f"Rows with valid gender labels: {len(gender_data)}")
    print(f"Rows with valid age labels: {len(age_data)}")

    # Extract features for gender classification
    X_gender = []
    y_gender = []
    for _, row in gender_data.iterrows():
        file_path = os.path.join(clips_dir, row['path'])
        if not os.path.exists(file_path):  # Check if file exists
            # print(f"File not found: {file_path}")
            continue  # Skip to the next file if missing
        audio = preprocess_audio(file_path)
        if audio is None:
            # print(f"Skipping file due to preprocessing errors: {file_path}")
            continue  # Skip to the next file if preprocessing fails
        features = extract_features(audio)
        X_gender.append(features)
        y_gender.append(row['gender'])

    # Encode gender labels as binary values (0 for male, 1 for female)
    label_encoder = LabelEncoder()
    y_gender = label_encoder.fit_transform(y_gender)

    # Extract features for age classification
    X_age = []
    y_age = []
    for _, row in age_data.iterrows():
        file_path = os.path.join(clips_dir, row['path'])
        if not os.path.exists(file_path):  # Check if file exists
            # print(f"File not found: {file_path}")
            continue  # Skip to the next file if missing
        audio = preprocess_audio(file_path)
        if audio is None:
            # print(f"Skipping file due to preprocessing errors: {file_path}")
            continue  # Skip to the next file if preprocessing fails
        features = extract_features(audio)
        X_age.append(features)
        y_age.append(row['age'])

    print(f"Gender samples after feature extraction: {len(X_gender)}")
    print(f"Age samples after feature extraction: {len(X_age)}")

    return np.array(X_gender), np.array(y_gender), np.array(X_age), np.array(y_age), label_encoder

def load_crema_dataset(data_dir):
    """
    Load CREMA-D dataset for emotion classification.
    """
    emotions = {
        'NEU': 'neutral',
        'HAP': 'happy',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fear',
        'DIS': 'disgust'
    }
    X = []
    y_emotion = []
    print("Loading CREMA-D dataset...")
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            emotion_code = file_name.split('_')[2]
            emotion = emotions.get(emotion_code)
            if emotion:
                file_path = os.path.join(data_dir, file_name)
                audio = preprocess_audio(file_path)
                if audio is None:
                    # print(f"Skipping file due to errors: {file_path}")
                    continue  # Skip to the next file if preprocessing fails
                features = extract_features(audio)
                X.append(features)
                y_emotion.append(emotion)
    return np.array(X), np.array(y_emotion)

# Train Gender Classification Model (LSTM)
def train_gender_model(X_train, X_test, y_train, y_test):
    """
    Train an LSTM model for gender classification.
    """
    print("Training gender classification model...")
    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        X_train, y_train,
        epochs=10, batch_size=32,
        validation_data=(X_test, y_test)
    )
    return model

# Train Age Classification Model (Random Forest)
def train_age_model(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model for age classification.
    """
    print("Training age classification model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train Emotion Classification Model (SVM)
def train_emotion_model(X_train, X_test, y_train, y_test):
    """
    Train an SVM model for emotion classification.
    """
    print("Training emotion classification model...")
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    # Load Mozilla dataset
    X_gender, y_gender, X_age, y_age, label_encoder = load_mozilla_dataset(mozilla_tsv, mozilla_clips_dir)

    # Split data for gender classification
    X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(
        X_gender, y_gender, test_size=0.2, random_state=42
    )

    # Split data for age classification
    X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(
        X_age, y_age, test_size=0.2, random_state=42
    )

    # Load CREMA-D dataset
    X_emotion, y_emotion = load_crema_dataset(crema_dir)
    X_emotion_train, X_emotion_test, y_emotion_train, y_emotion_test = train_test_split(
        X_emotion, y_emotion, test_size=0.2, random_state=42
    )

    # Train models
    gender_model = train_gender_model(
        X_gender_train.reshape(X_gender_train.shape[0], X_gender_train.shape[1], 1),
        X_gender_test.reshape(X_gender_test.shape[0], X_gender_test.shape[1], 1),
        y_gender_train, y_gender_test
    )
    age_model = train_age_model(X_age_train, X_age_test, y_age_train, y_age_test)
    emotion_model = train_emotion_model(X_emotion_train, X_emotion_test, y_emotion_train, y_emotion_test)

    # Save models using the latest Keras format
    os.makedirs('models', exist_ok=True)
    gender_model.save('models/gender_model.keras')  # Use .keras instead of .h5
    joblib.dump(age_model, 'models/age_model.pkl')
    joblib.dump(emotion_model, 'models/emotion_model.pkl')

    # Save the LabelEncoder for gender decoding
    joblib.dump(label_encoder, 'models/gender_label_encoder.pkl')

    print("All models trained and saved successfully!")