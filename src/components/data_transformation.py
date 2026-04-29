import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config = config
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Tenure Months",
                                 "Monthly Charges",
                                "CLTV" ]
            
            categorical_columns = [
                "Gender",
                "Senior Citizen",
                "Partner",
                "Dependents",
                "Phone Service",
                "Multiple Lines",
                "Internet Service",
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Contract",
                "Paperless Billing",
                "Payment Method"
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    
                ]
            )
            
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipelin",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ]
         )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("Read train and test data ")
            
            target_column = "Churn Value"
            drop_columns = [
                "CustomerID",
                "Lat Long",
                "Latitude",
                "Longitude",
                "Churn Label",
                "Churn Score",
                "Churn Reason",
                "Count"
            ]
            
            train_df = train_df.drop(columns=drop_columns)
            test_df = test_df.drop(columns=drop_columns)
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            preprocessing_obj = self.get_data_transformer_object()
            
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)
            
            train_arr = np.c_[X_train_arr,np.array(y_train)]
            test_arr = np.c_[X_test_arr,np.array(y_test)]
            
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path),exist_ok=True)
            
            joblib.dump(preprocessing_obj,self.config.preprocessor_obj_file_path)
            
            logger.info("Saved preprocessing object")
            
            return(
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
            
