from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        #data_ingestion.initiate_data_ingestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initaite_data_transformation(train_data_path, test_data_path)

    except Exception as e:
        logging.exception("An error occurred during execution")
        raise CustomException(e,sys)