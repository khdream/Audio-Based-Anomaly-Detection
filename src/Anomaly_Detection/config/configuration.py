from Anomaly_Detection.constants import *
from Anomaly_Detection.utils.common import read_yaml, create_directories
from Anomaly_Detection.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, EvaluationConfig, PrepareBaseModelConfig, TrainingConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            normal_data_path=config.normal_data_path,
            abnormal_data_path=config.abnormal_data_path,
            normal_features_path= config.normal_features_path,
            abnormal_features_path= config.abnormal_features_path,
            feature_names= config.feature_names
        )

        return data_transformation_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model
            
            create_directories([config.root_dir])

            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                feature_names_path=Path(config.feature_names_path),
                X_train_scaled_path=Path(config.X_train_scaled_path),
                X_val_path=Path(config.X_val_path),
                X_combined_test_path=Path(config.X_combined_test_path),
                y_combined_test_path=Path(config.y_combined_test_path),
                feature_importance_path=Path(config.feature_importance_path)
                
            )

            return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params.training
        create_directories([
            Path(training.root_dir)
        ])

        training_config  = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            normal_features_path=Path(training.normal_features_path),
            abnormal_features_path=Path(training.abnormal_features_path),
            feature_names_path=Path(training.feature_names_path),
            feature_importance_path=Path(training.feature_importance_path),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_feature_count=params.FEATURE_COUNT
            
        )

        return training_config
        
    def get_evaluation_config(self) -> EvaluationConfig:
            evaluation = self.config.evaluation
            params = self.params.training
            create_directories([
                Path(evaluation.root_dir)
            ])
            eval_config = EvaluationConfig(

                mlflow_uri="https://dagshub.com/JAISON14/Audio-Based-Anomaly-Detection-for-Industrial-Machinery-End-to-End-Project-using-MLflow-DVC.mlflow",
                root_dir=Path(evaluation.root_dir),
                feature_names_path=Path(evaluation.feature_names_path),
                trained_model_path=Path(evaluation.trained_model_path),
                feature_importance_path=Path(evaluation.feature_importance_path),
                X_combined_test_path=Path(evaluation.X_combined_test_path),
                y_combined_test_path=Path(evaluation.y_combined_test_path),
                scores_path=Path(evaluation.scores_path),
                all_params=self.params,
                params_epochs=params.EPOCHS,
                params_batch_size=params.BATCH_SIZE,
                params_feature_count=params.FEATURE_COUNT
            )
            return eval_config