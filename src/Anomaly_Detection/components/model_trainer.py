import os
import urllib.request as request
import tensorflow as tf

from Anomaly_Detection import logger
from Anomaly_Detection.entity.config_entity import TrainingConfig
from Anomaly_Detection.utils.common import get_size
import pandas as pd

import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Training:
    def __init__(self, config: TrainingConfig):
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

    def feature_importance(self,autoencoder, X_combined_test):
        # Predict the reconstructed sounds for the combined test set
        reconstructed_combined = autoencoder.predict(X_combined_test)

        # Calculate the mean squared reconstruction error for each feature
        mse_features = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=0)

        # Rank features by reconstruction error
        feature_importance_ranking = np.argsort(mse_features)[::-1]  # Features with the highest error first
        logger.info(f"feature_importance_ranking: {feature_importance_ranking}")
        return feature_importance_ranking
    
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

    
    def feature_selection(self, N, feature_importance_ranking, feature_names):
        top_features_indices = feature_importance_ranking[:N]
        top_features=[]
        for rank in feature_importance_ranking[:N]:
            top_features.append(feature_names[rank])

        return top_features,top_features_indices

    def train_test_spliting(self,top_features_indices):

        # Load normal_features.pkl
        normal_features = joblib.load(self.config.normal_features_path)
        logger.info(f"Loaded normal features {normal_features.shape}.")
        # Load abnormal_features.pkl
        abnormal_features = joblib.load(self.config.abnormal_features_path)
        logger.info(f"Loaded abnormal features {abnormal_features.shape}.")
        
        # Subset the features for both normal and abnormal data
        normal_features = normal_features[:, top_features_indices]
        abnormal_features = abnormal_features[:, top_features_indices]

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

        joblib.dump(X_train_scaled,(os.path.join(self.config.root_dir, "X_train_scaled.pkl")))
        joblib.dump(X_val_scaled,(os.path.join(self.config.root_dir, "X_val_scaled.pkl")))
        joblib.dump(X_combined_test,(os.path.join(self.config.root_dir, "X_combined_test.pkl")))
        joblib.dump(y_combined_test,(os.path.join(self.config.root_dir, "y_combined_test.pkl")))
        
        logger.info("Splited data into training and test sets")
        logger.info(f"Saved X_train_scaled {X_train_scaled.shape} into file.")
        logger.info(f"Saved X_train_scaled {X_val_scaled.shape} into file.")
        logger.info(f"Saved X_combined_test {X_combined_test.shape} into file.")
        logger.info(f"Saved y_combined_test {y_combined_test.shape} into file.")
        return X_train_scaled,X_val_scaled,X_combined_test,y_combined_test

    def train(self):
        logger.info(f"Starting Model Building")
        feature_names = joblib.load(self.config.feature_names_path)
        abnormal_features_path = joblib.load(self.config.abnormal_features_path)
        normal_features_path = joblib.load(self.config.normal_features_path)
        n = self.config.params_feature_count
        feature_importance_ranking= joblib.load(self.config.feature_importance_path)
        
        top_features,top_features_indices = self.feature_selection(n, feature_importance_ranking, feature_names)
        X_train_scaled,X_val_scaled,X_combined_test,y_combined_test=self.train_test_spliting(top_features_indices)
        
        autoencoder = self.model_training(X_train_scaled, X_val_scaled)
        self.model_evaluation(autoencoder,X_combined_test, y_combined_test)
        feature_importance_ranking = self.feature_importance(autoencoder,X_combined_test)

        joblib.dump(autoencoder,(os.path.join(self.config.root_dir, "autoencoder.pkl")))
        joblib.dump(feature_importance_ranking,(os.path.join(self.config.root_dir, "updated_feature_importance_ranking.pkl")))
        autoencoder.save((os.path.join(self.config.root_dir, 'Encoder_Model.keras')))        
