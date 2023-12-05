from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    normal_data_path: Path
    abnormal_data_path: Path
    normal_features_path: Path
    abnormal_features_path: Path
    feature_names: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    feature_names_path: Path
    X_train_scaled_path: Path
    X_val_path: Path
    X_combined_test_path: Path
    y_combined_test_path: Path
    base_model_path: Path
    feature_importance_path: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    normal_features_path: Path
    abnormal_features_path: Path
    feature_names_path: Path
    feature_importance_path: Path
    params_epochs: int
    params_batch_size: int
    params_feature_count: int

@dataclass(frozen=True)
class EvaluationConfig:
    trained_model_path: Path
    root_dir: Path
    all_params: dict
    mlflow_uri: str
    feature_names_path: Path
    feature_importance_path: Path
    X_combined_test_path: Path
    y_combined_test_path: Path
    scores_path : Path
    params_epochs: int
    params_batch_size: int
    params_feature_count: int