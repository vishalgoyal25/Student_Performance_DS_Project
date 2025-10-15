import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import (DataTransformation, DataTransformationConfig)

from src.components.model_trainer import (ModelTrainer, ModelTrainerConfig)

# Here dataclass is used for configuration classes or 
# objects that only store data. No need to write __init__()
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            
            df=pd.read_csv('notebook/data/stud.csv')   # Data File
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
            X_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            X_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train_arr, X_test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    print("R2_Score:-", model_trainer.initiate_model_trainer(X_train_arr, X_test_arr))
