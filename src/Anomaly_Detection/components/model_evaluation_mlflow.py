import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import os
import urllib.request as request
import tensorflow as tf
from tensorflow.keras.models import Model
from Anomaly_Detection import logger
from Anomaly_Detection.entity.config_entity import EvaluationConfig
from Anomaly_Detection.utils.common import get_size,save_json
from tensorflow.keras.models import load_model
import numpy as np
import os
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def model_evaluation(self,autoencoder,X_combined_test, y_combined_test,top_features):
        reconstructed_combined = autoencoder.predict(X_combined_test)
        mse_combined = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=1)
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

        self.save_score(optimal_threshold,optimal_accuracy,optimal_precision,optimal_recall,optimal_f1)

        self.feature_importance(reconstructed_combined,X_combined_test,top_features)
        
        # Print metrics using the optimal threshold
        logger.info(f"Optimal Threshold: {optimal_threshold}")
        logger.info(f"Accuracy: {optimal_accuracy}")
        logger.info(f"Precision: {optimal_precision}")
        logger.info(f"Recall: {optimal_recall}")
        logger.info(f"F1 Score: {optimal_f1}")
        logger.info(f"confusion_matrix: {optimal_cm}")   

    def feature_importance(self,reconstructed_combined,X_combined_test,feature_names):
        # Calculate the mean squared reconstruction error for each feature
        mse_features = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=0)
        # Rank features by reconstruction error
        feature_importance_ranking = np.argsort(mse_features)[::-1]  # Features with the highest error first
        # Create a dictionary to store feature names and their importance scores
        feature_importance = {}
        for idx, feature_idx in enumerate(feature_importance_ranking):
            feature_name = feature_names[feature_idx] 
            importance_score = mse_features[feature_idx]
            feature_importance[feature_name] = importance_score
        joblib.dump(feature_importance,(os.path.join(self.config.root_dir,"feature_importance.pkl")))
        

    def evaluation(self):
        logger.info(f"Starting Model Evaluation")
        feature_names = joblib.load(self.config.feature_names_path)
        X_combined_test = joblib.load(self.config.X_combined_test_path)
        y_combined_test = joblib.load(self.config.y_combined_test_path)


        autoencoder = load_model(self.config.trained_model_path)
        self.model_evaluation(autoencoder,X_combined_test, y_combined_test,feature_names)        
   

    def save_score(self,optimal_threshold,optimal_accuracy,optimal_precision,optimal_recall,optimal_f1):
        scores = {"Optimal Threshold": optimal_threshold, "Accuracy": optimal_accuracy,
                  "Precision": optimal_precision , "Recall": optimal_recall,
                  "F1 Score": optimal_f1
                  }

        joblib.dump(scores,self.config.scores_path)
        save_json(path=Path((os.path.join(self.config.root_dir, "scores.json"))), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            feature_importance=joblib.load((os.path.join(self.config.root_dir,"feature_importance.pkl")))
            mlflow.log_params(feature_importance)

            scores=joblib.load(self.config.scores_path)

            # Load the model
            model = load_model(self.config.trained_model_path)
            
            mlflow.log_metrics(
                scores
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(model, "model", registered_model_name="Top 10 Features")
            else:
                mlflow.keras.log_model(model, "model")