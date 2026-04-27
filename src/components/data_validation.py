import os
import sys
import pandas as pd
import yaml

from src.exception import CustomException
from src.logger import logger
from src.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self,config:DataValidationConfig):
        self.config = config
    
    def validate_all_columns(self):
        try:
            validation_status = True
            
            data = pd.read_csv(self.config.unzip_data_dir)
            
            with open("config/schema.yaml") as file:
                schema = yaml.safe_load(file)
            
            all_cols = schema["columns"]
            
            for col in data.columns:
                if col not in all_cols:
                    validation_status = False
            
            with open(self.config.STATUS_FILE,"w") as file:
                file.write(f"Validation status: {validation_status}")
            
            return validation_status
        
        except Exception as e:
            logger.info("Error occurred in data validation method")
            raise CustomException(e,sys)
        
        