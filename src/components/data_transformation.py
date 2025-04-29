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

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    obj_file_path = os.path.join('artifact', 'preprocessor.pkl')
    
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
        try:
            numerical_cols = ['Exp-month-salary(INR )', 'Score']
            categorical_cols = self.transform_config.categorical_mapping(keys=True)
            
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    dt = DataTransformation()
    dt.get_data_transformer_object()