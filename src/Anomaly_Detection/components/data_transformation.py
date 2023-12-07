import os
from Anomaly_Detection import logger
from Anomaly_Detection.entity.config_entity import DataTransformationConfig
from Anomaly_Detection.utils.common import get_size
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_audio_files(self,path, label):
        audio_files = []
        labels = []
        logger.info(f"Loading audio data from {path} into file.")
        for filename in os.listdir(path):
            if filename.endswith('.wav'):
                file_path = os.path.join(path, filename)
                audio, sample_rate = librosa.load(file_path, sr=None)
                audio_files.append(audio)
                labels.append(label)
        return audio_files, labels, sample_rate

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

    def extract_all_features(self,audio_data, sample_rate):
        features = []
        logger.info(f"Extracting features from audio_data.")
        for audio in audio_data:
            mfccs = self.extract_mfccs(audio, sample_rate)
            spectral_features = self.extract_spectral_features(audio, sample_rate)
            temporal_features = self.extract_temporal_features(audio)
            additional_features = self.extract_additional_features(audio, sample_rate)
            all_features = np.concatenate([mfccs, spectral_features, temporal_features, additional_features])
            features.append(all_features)
        return np.array(features)
    
    def feature_list(self):
        # Assuming you have 13 MFCCs
        n_mfcc = 13
        mfcc_labels = [f'MFCC_{i+1}' for i in range(n_mfcc)]

        # We have 3 spectral features
        spectral_labels = ['Spectral Centroid', 'Spectral Rolloff', 'Spectral Contrast']

        # We have 2 temporal features
        temporal_labels = ['Zero Crossing Rate', 'Autocorrelation']

        # Adding additional features to features list
        additional_features = ['Chroma Features', 'Spectral Bandwidth', 'Spectral Flatness', 'Spectral Roll-off Frequency', 'Root Mean Square Energy']

        feature_names = mfcc_labels + spectral_labels + temporal_labels + additional_features
        return  feature_names


    def data_preprocessing(self):

        logger.info(f"Starting Data preprocessing")
        abnormal_pump_path = self.config.abnormal_data_path
        normal_pump_path= self.config.normal_data_path

        print(abnormal_pump_path)
        print(normal_pump_path)
  
        # Load the datasets
        abnormal_audio, abnormal_labels, _ = self.load_audio_files(abnormal_pump_path, label=1)
        normal_audio, normal_labels, sample_rate = self.load_audio_files(normal_pump_path, label=0)

        logger.info(f"Size of abnormal_audio: {len(abnormal_audio)}.")
        logger.info(f"Size of normal_audio: {len(normal_audio)}.")

        # Extract features for both normal and abnormal data
        normal_features = self.extract_all_features(normal_audio, sample_rate)
        abnormal_features = self.extract_all_features(abnormal_audio, sample_rate)
        feature_names = self.feature_list()

        joblib.dump(normal_features,(os.path.join(self.config.root_dir, "normal_features.pkl")))
        joblib.dump(abnormal_features,(os.path.join(self.config.root_dir, "abnormal_features.pkl")))
        joblib.dump(feature_names,(os.path.join(self.config.root_dir, "feature_names.pkl")))
        logger.info(f"Data preprocessing completed.")


    def train_test_spliting(self):

        # Load normal_features.pkl
        normal_features = joblib.load(os.path.join(self.config.root_dir, "normal_features.pkl"))
        logger.info(f"Loaded normal features {normal_features.shape}.")
        # Load abnormal_features.pkl
        abnormal_features = joblib.load(os.path.join(self.config.root_dir, "abnormal_features.pkl"))
        logger.info(f"Loaded abnormal features {abnormal_features.shape}.")

        X_train, X_val = train_test_split(normal_features, test_size=0.2, random_state=42)
        X_test = abnormal_features
        scaler = StandardScaler()
        # Fit the scaler on the training data and transform both training, validation, and test sets
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        # Combine normal and abnormal data
        X_combined_test = np.concatenate((X_val_scaled, X_test_scaled))
        y_combined_test = np.concatenate((np.zeros(len(X_val_scaled)), np.ones(len(X_test_scaled))))  # 0 for normal, 1 for abnormal

        joblib.dump(scaler,(os.path.join(self.config.root_dir, "scaler.pkl")))
        joblib.dump(X_train_scaled,(os.path.join(self.config.root_dir, "X_train_scaled.pkl")))
        joblib.dump(X_val_scaled,(os.path.join(self.config.root_dir, "X_val_scaled.pkl")))
        joblib.dump(X_combined_test,(os.path.join(self.config.root_dir, "X_combined_test.pkl")))
        joblib.dump(y_combined_test,(os.path.join(self.config.root_dir, "y_combined_test.pkl")))
        
        logger.info("Splited data into training and test sets")
        logger.info(f"Saved X_train_scaled {X_train_scaled.shape} into file.")
        logger.info(f"Saved X_train_scaled {X_val_scaled.shape} into file.")
        logger.info(f"Saved X_combined_test {X_combined_test.shape} into file.")
        logger.info(f"Saved y_combined_test {y_combined_test.shape} into file.")
       