import os
import urllib.request as request
import tensorflow as tf
from Anomaly_Detection import logger
from Anomaly_Detection.entity.config_entity import PrepareBaseModelConfig
from Anomaly_Detection.utils.common import get_size
import pandas as pd
import numpy as np
import librosa
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_ind
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def enhanced_autoencoder(self,input_dim):
        input_layer = Input(shape=(input_dim,))

        # Encoder
        encoder = Dense(128, activation='relu')(input_layer)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.1)(encoder)
        encoder = Dense(64, activation='relu')(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.1)(encoder)
        encoder = Dense(32, activation='relu')(encoder)

        # Decoder
        decoder = Dense(64, activation='relu')(encoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.1)(decoder)
        decoder = Dense(128, activation='relu')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.1)(decoder)
        output_layer = Dense(input_dim, activation='sigmoid')(decoder)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder
    
    def model_training(self,X_train_scaled, X_val_scaled):
        # Adjusting input_dim based on your feature dimensions
        input_dim = X_train_scaled.shape[1]
        autoencoder = self.enhanced_autoencoder(input_dim)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=400,  # Increase epochs if necessary
            batch_size=256,
            shuffle=True,
            validation_data=(X_val_scaled, X_val_scaled),
            callbacks=[early_stopping],
            verbose=0
            )
        return autoencoder

    

    def model_evaluation(self,autoencoder,X_combined_test, y_combined_test):
        reconstructed_combined = autoencoder.predict(X_combined_test)
        mse_combined = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=1)
        precisions, recalls, thresholds = precision_recall_curve(y_combined_test, mse_combined)
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_combined_test, mse_combined)

        # Calculate F1 score for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Use the optimal threshold to define anomalies
        optimal_predictions = (mse_combined > optimal_threshold).astype(int)

        # Calculate metrics using the optimal threshold
        optimal_accuracy = accuracy_score(y_combined_test, optimal_predictions)
        optimal_precision = precision_score(y_combined_test, optimal_predictions)
        optimal_recall = recall_score(y_combined_test, optimal_predictions)
        optimal_f1 = f1_score(y_combined_test, optimal_predictions)
        optimal_cm = confusion_matrix(y_combined_test, optimal_predictions)
        # Print metrics using the optimal threshold
        logger.info(f"Optimal Threshold: {optimal_threshold}")
        logger.info(f"Accuracy: {optimal_accuracy}")
        logger.info(f"Precision: {optimal_precision}")
        logger.info(f"Recall: {optimal_recall}")
        logger.info(f"F1 Score: {optimal_f1}")
        logger.info(f"confusion_matrix: {optimal_cm}")


    def feature_importance(self,autoencoder, X_combined_test):
        # Predict the reconstructed sounds for the combined test set
        reconstructed_combined = autoencoder.predict(X_combined_test)

        # Calculate the mean squared reconstruction error for each feature
        mse_features = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=0)

        # Rank features by reconstruction error
        feature_importance_ranking = np.argsort(mse_features)[::-1]  # Features with the highest error first
        logger.info(f"feature_importance_ranking: {feature_importance_ranking}")
        return feature_importance_ranking
    
    def buiding_base_model(self):
        logger.info(f"Starting Building Base Model")
        feature_names = joblib.load(self.config.feature_names_path)
        X_train_scaled = joblib.load(self.config.X_train_scaled_path)
        X_val_scaled = joblib.load(self.config.X_val_path)
        X_combined_test = joblib.load(self.config.X_combined_test_path)
        y_combined_test = joblib.load(self.config.y_combined_test_path)
        
        autoencoder = self.model_training(X_train_scaled, X_val_scaled)
        self.model_evaluation(autoencoder,X_combined_test, y_combined_test)
        feature_importance_ranking = self.feature_importance(autoencoder,X_combined_test)

        joblib.dump(autoencoder,(os.path.join(self.config.root_dir, "autoencoder.pkl")))
        joblib.dump(feature_importance_ranking,(os.path.join(self.config.root_dir, "feature_importance_ranking.pkl")))
        autoencoder.save((os.path.join(self.config.root_dir, 'Encoder_Model.keras')))        


