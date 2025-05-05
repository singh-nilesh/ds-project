import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin


# Function to Save file object
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
            
    except Exception as e:
        raise CustomException(e, sys)




# Custom transformer to replace zeros with median
class ZeroReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, replace_with='median'):
        self.columns = columns
        self.replace_with = replace_with
        self.replacement_values = {}
        
    def fit(self, X, y=None):
        # Convert to DataFrame if X is numpy array
        if isinstance(X, np.ndarray):
            if self.columns is None:
                # If no columns provided and X is numpy array, use indices as column names
                self.columns = list(range(X.shape[1]))
            X_copy = pd.DataFrame(X, columns=self.columns)
        else:
            X_copy = X.copy()
            if self.columns is None:
                self.columns = list(X_copy.columns)
            
        for col in self.columns:
            non_zero_values = X_copy[X_copy[col] != 0][col]
            if len(non_zero_values) == 0:  # Handle case where all values are 0
                self.replacement_values[col] = 0
            elif self.replace_with == 'median':
                self.replacement_values[col] = non_zero_values.median()
            elif self.replace_with == 'mean':
                self.replacement_values[col] = non_zero_values.mean()
            else:
                self.replacement_values[col] = self.replace_with
        return self
    
    def transform(self, X):
        # Convert to DataFrame if X is numpy array
        input_is_ndarray = isinstance(X, np.ndarray)
        if input_is_ndarray:
            X_copy = pd.DataFrame(X, columns=self.columns)
        else:
            X_copy = X.copy()
            
        for col in self.columns:
            X_copy.loc[X_copy[col] == 0, col] = self.replacement_values[col]
            
        # Return numpy array if input was numpy array
        if input_is_ndarray:
            return X_copy.values
        return X_copy

# Custom transformer to handle outliers
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, method='IQR', threshold=1.5):
        self.columns = columns
        self.method = method
        self.threshold = threshold
        self.caps = {}
        
    def fit(self, X, y=None):
        # Convert to DataFrame if X is numpy array
        if isinstance(X, np.ndarray):
            if self.columns is None:
                # If no columns provided and X is numpy array, use indices as column names
                self.columns = list(range(X.shape[1]))
            X_copy = pd.DataFrame(X, columns=self.columns)
        else:
            X_copy = X.copy()
            if self.columns is None:
                self.columns = list(X_copy.columns)
            
        for col in self.columns:
            if self.method == 'IQR':
                Q1 = X_copy[col].quantile(0.25)
                Q3 = X_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
            elif self.method == 'std':
                mean = X_copy[col].mean()
                std = X_copy[col].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
            else:
                raise ValueError("Method not supported. Use 'IQR' or 'std'")
                
            self.caps[col] = {'lower': lower_bound, 'upper': upper_bound}
        return self
    
    def transform(self, X):
        # Convert to DataFrame if X is numpy array
        input_is_ndarray = isinstance(X, np.ndarray)
        if input_is_ndarray:
            X_copy = pd.DataFrame(X, columns=self.columns)
        else:
            X_copy = X.copy()
            
        for col in self.columns:
            X_copy[col] = X_copy[col].clip(
                lower=self.caps[col]['lower'], 
                upper=self.caps[col]['upper']
            )
            
        # Return numpy array if input was numpy array
        if input_is_ndarray:
            return X_copy.values
        return X_copy