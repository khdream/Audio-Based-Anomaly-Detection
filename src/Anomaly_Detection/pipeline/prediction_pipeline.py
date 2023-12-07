import os
import joblib
import librosa
import numpy as np
import tensorflow as tf
from Anomaly_Detection import logger


class AudioPredictionPipeline:
    def __init__(self, model_path, top_features_path, scaler_path, feature_names, scores_path):
        self.model = tf.keras.models.load_model(model_path)
        self.top_features = joblib.load(top_features_path)
        self.scaler = joblib.load(scaler_path)
        self.scores = joblib.load(scores_path)
        
        # Create a mapping of feature names to their indices
        self.feature_name_to_index = {name: idx for idx, name in enumerate(feature_names)}
        # Convert self.top_features to a list of indices
        self.top_features_indices = [self.feature_name_to_index[feature] for feature in self.top_features if feature in self.feature_name_to_index]

    def extract_mfccs(self,audio, sample_rate, n_mfcc=13):
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    
    def extract_spectral_features(self,audio, sample_rate):
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
        return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)
    
    def extract_temporal_features(self,audio):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        autocorrelation = librosa.autocorrelate(audio)
        return np.mean(zero_crossing_rate), np.mean(autocorrelation)
    
    def extract_additional_features(self,audio, sample_rate):
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spec_flatness = librosa.feature.spectral_flatness(y=audio)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        rms = librosa.feature.rms(y=audio)
        
        return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)
    
    def extract_all_features(self, audio, sample_rate):
    # No need to loop over audio_data, as it's a single sample
        mfccs = self.extract_mfccs(audio, sample_rate)
        spectral_features = self.extract_spectral_features(audio, sample_rate)
        temporal_features = self.extract_temporal_features(audio)
        additional_features = self.extract_additional_features(audio, sample_rate)
        all_features = np.concatenate([mfccs, spectral_features, temporal_features, additional_features])
        return all_features

    def preprocess_audio(self, audio_files):
        all_preprocessed_data = []

        for audio_file in audio_files:
            # Load the audio file
            audio, sample_rate = librosa.load(audio_file, sr=None)

            # Extract features for the current audio file
            features = self.extract_all_features(audio, sample_rate)

            # Scale the features
            scaled_features = self.scaler.transform([features])

            # Select only the top features used by the model using indices
            selected_features = scaled_features[0, self.top_features_indices]  # Remove the extra dimension

            all_preprocessed_data.append(selected_features)

        return np.array(all_preprocessed_data)
    
    def predict(self, audio_files):
        threshold = self.scores['Optimal Threshold']

        predictions = []
        for audio_file in audio_files:
            preprocessed_data = self.preprocess_audio([audio_file])  # Process one file at a time

            # Get the model's reconstruction of the input
            reconstructed_data = self.model.predict(preprocessed_data)

            # Calculate the mean squared error between input and reconstruction
            mse = np.mean(np.square(preprocessed_data - reconstructed_data), axis=1)

            # Classify as abnormal (1) if mse exceeds threshold, else normal (0)
            classification = 1 if mse > threshold else 0
            predictions.append(classification)

        return predictions



# Paths to the model, top features list, and scaler
model_path = 'artifacts/training/Encoder_Model.keras'
top_features_path = 'artifacts/training/top_features_list.pkl'
scaler_path = 'artifacts/data_transformation/scaler.pkl'  # Replace with the actual path to the scaler
scores_path= 'artifacts/evaluation/scores.pkl'
#threshold=scores['Optimal Threshold']



# List of all features
all_feature_names = ['MFCC_1','MFCC_2','MFCC_3','MFCC_4','MFCC_5','MFCC_6','MFCC_7','MFCC_8','MFCC_9','MFCC_10','MFCC_11','MFCC_12','MFCC_13','Spectral Centroid','Spectral Rolloff','Spectral Contrast','Zero Crossing Rate','Autocorrelation','Chroma Features','Spectral Bandwidth','Spectral Flatness','Spectral Roll-off Frequency','Root Mean Square Energy']

# Create an instance of the prediction pipeline
prediction_pipeline = AudioPredictionPipeline(model_path, top_features_path, scaler_path, all_feature_names, scores_path)

# Example usage
audio_files = ['artifacts/data_ingestion/abnormal/00000000.wav', 'artifacts/data_ingestion/abnormal/00000001.wav', 'artifacts/data_ingestion/abnormal/00000002.wav','artifacts/data_ingestion/normal/00000000.wav','artifacts/data_ingestion/normal/00000002.wav']  # List of file paths
predictions = prediction_pipeline.predict(audio_files)
print("Prediction:", predictions)
