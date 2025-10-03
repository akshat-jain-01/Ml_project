from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
from src.mlproject.components.model_trainer import ModelTrainerConfig, ModelTrainer
import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        #data_ingestion.initiate_data_ingestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr,_ = data_transformation.initaite_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logging.exception("An error occurred during execution")
        raise CustomException(e,sys)