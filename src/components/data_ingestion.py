import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact', 'train.csv')
    test_data_path: str=os.path.join('artifact', 'test.csv')
    raw_data_path: str=os.path.join('artifact', 'raw.csv')
    extra_data_path: str=os.path.join('artifact', 'extra_cols.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read from source
            df = pd.read_csv(
                '/home/dk/code_base/ml_projects/ds-project/notebook/student-survey.csv'
                )
            
            # Create artifact dir
            logging.info('Read the dataset as dataframe')
            os.makedirs( 
                os.path.dirname(self.ingestion_config.train_data_path), 
                exist_ok=True
                )
            
            # Segregating String cols
            logging.info('Dropping Extra columns') 
            str_cols = ['Preferred-job', 'Preferred-industry', 'Redo-subjects', 'Current-skills', 'Job-portals', 'Missing-subjects', 'Certification-domain', 'Domain', 'Location']
            extra_cols = df[str_cols].copy()
            df = df.drop(str_cols, axis=1)
            
            # split the dataset
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Create CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #raw
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) #train
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) #test
            extra_cols.to_csv(self.ingestion_config.extra_data_path, index=False, header=True) #extra
            
            logging.info('Ingestion completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()