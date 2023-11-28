from Anomaly_Detection import logger
from Anomaly_Detection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from Anomaly_Detection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
# from Anomaly_Detection.pipeline.stage_03_model_trainer import ModelTrainingPipeline
# from Anomaly_Detection.pipeline.stage_04_model_evaluation import EvaluationPipeline



STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

