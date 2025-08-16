🎙️ Audio-based Emotion, Age, and Gender Recognition using Deep Learning

📌 Project Overview

This project focuses on building an audio-based recognition system that can classify a speaker’s emotion, age group, and gender using deep learning and machine learning techniques. The system leverages speech signals as input and extracts meaningful features to train robust classification models.

🔑 Key Features

.Emotion Recognition: Identifies emotions such as Happy, Sad, Angry, Neutral from speech.

.Age Classification: Predicts the speaker’s age group based on voice characteristics.

.Gender Classification: Detects whether the speaker is Male or Female.

.Multi-model Approach: Utilizes both Deep Learning (LSTM) and Machine Learning models (Random Forest, SVM) for comparison.

🛠️ Technologies & Tools

 .Programming Language: Python

Libraries Used:

 .Librosa → Feature extraction (MFCCs, spectral features, etc.)

 .TensorFlow / Keras → LSTM model implementation

 .Scikit-learn → Random Forest, SVM classifiers

 .[Pandas, NumPy, Matplotlib, Seaborn] → Data preprocessing & visualization

📂 Datasets

.CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset (for emotion recognition).

.Mozilla Common Voice: Large-scale open-source dataset for age and gender classification.

🧠 Methodology

1.Data Preprocessing:

.Cleaned and normalized audio samples.

.Extracted features such as MFCCs, Chroma, Spectral Contrast using Librosa.

2.Model Development:

.Trained LSTM networks to capture temporal dependencies in audio signals.

.Implemented Random Forest and SVM for baseline comparisons.

3.Evaluation:

.Compared accuracy, precision, recall, and F1-score across different models.

.Visualized results using confusion matrices and plots.

📊 Results

.LSTM models showed higher performance for emotion detection due to their ability to capture temporal features.

.Machine learning models (Random Forest, SVM) provided competitive results for gender and age classification. 
