from preprocess import preprocess_audio
from feature_extraction import extract_features
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load trained models
gender_model = load_model('models/gender_model.keras')  # Updated to .keras format
age_model = joblib.load('models/age_model.pkl')
emotion_model = joblib.load('models/emotion_model.pkl')

# Load the LabelEncoder for gender decoding
label_encoder = joblib.load('models/gender_label_encoder.pkl')

def predict(file_path):
    try:
        audio = preprocess_audio(file_path)
        if audio is None:
            return {"error": "Failed to process audio file."}

        features = extract_features(audio)
        features = np.expand_dims(features, axis=0)  # Add batch dimension

        # Predict gender
        gender_prob = gender_model.predict(features)[0][0]
        gender_numeric = 1 if gender_prob > 0.5 else 0
        gender = label_encoder.inverse_transform([gender_numeric])[0]

        # Predict age
        age = age_model.predict(features)[0]

        # Predict emotion
        emotion = emotion_model.predict(features)[0]

        return {
            'gender': gender,
            'gender_prob': float(gender_prob),
            'age': age,
            'emotion': emotion
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "An error occurred during prediction."}