import sys
from dataclasses import dataclass
import os
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.utils import ZeroReplacer, OutlierCapper
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')
    
    @staticmethod
    def categorical_mapping (keys=False):
        with open('./json/categorical_map.json', 'r') as file:
            data = json.load(file)
            
            if keys:
                return list(data.keys())
            else :
                return data
        


class DataTransformation:
    def __init__(self):
        self.transform_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        
        This function is responsible for Data transformation (categorical & numerical)
        
        '''
        try:
            # Only include Score in numerical cols, since 'Exp-month-salary(INR )' is the target
            numerical_cols = ['Score']  
            categorical_cols = self.transform_config.categorical_mapping(keys=True)
            logging.info( ' Numerical and Categorical cols loaded' )
            
            
            # Constructing Pipelines
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("zero_replacer", ZeroReplacer(replace_with='median')),
                    ("outlier_capper", OutlierCapper(method='IQR')),
                    ("scaler", StandardScaler())
                ]                
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)),  # Change to non-sparse output
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(" Pipeline Constructed ")
            
            
            # Combining The 2 Pipelines
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_cols),
                    ("categorical_pipeline", cat_pipeline, categorical_cols)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def init_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info( ' Read train and test completed ' )
            
                        
            # The Pre-processor Object
            processor = self.get_data_transformer_object()
            target_col = 'Exp-month-salary(INR )'  # Space after INR
            
            # Pass target column as a list to drop method
            X_train = train_df.drop(columns=[target_col], axis=1)
            Y_train = train_df[target_col]
            
            X_test = test_df.drop(columns=[target_col], axis=1)
            Y_test = test_df[target_col]
            
            logging.info(" X and Y features segregated. ")
            
            # Applying the processor object
            logging.info(
                " Applying Pre-processor object on training dataframe and testing dataframe"
                )
            input_train_arr = processor.fit_transform(X_train)
            input_test_arr = processor.fit_transform(X_test)
            
            train_arr = np.c_[input_train_arr, np.array(Y_train)]
            test_arr = np.c_[input_test_arr, np.array(Y_test)]
            
            # Saving the Pre-processor object as .pkl
            save_object(
                file_path = self.transform_config.preprocessor_obj_file_path,
                obj = processor
            )
            logging.info(" Saved preprocessing object to .pkl")
            
            return (
                train_arr,
                test_arr,
                self.transform_config.preprocessor_obj_file_path,
            )            
        except Exception as e:
            raise CustomException(e, sys)
            

if __name__ == '__main__':
    dt = DataTransformation()
    dt.get_data_transformer_object()