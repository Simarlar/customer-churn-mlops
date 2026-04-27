import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config
    
    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method")
        
        try:
            # Load dataset
            df = pd.read_csv("notebook/data/Telco_customer_churn.csv")
            logger.info("Read dataset as pandas dataframe")
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            
            # Save raw data
            df.to_csv(self.config.raw_data_path,index=False)
            logger.info("Raw dataset saved")
            
            # Train test split
            train_set , test_set = train_test_split(df,
                                                    test_size=0.2,
                                                    random_state=42)
            # Save train and test dataset
            train_set.to_csv(self.config.train_data_path,index=False)
            test_set.to_csv(self.config.test_data_path,index=False)
            logger.info("Train and test dataset saved")
            
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        
        except Exception as e:
            logger.info("Error occurred in data ingestion method")
            raise CustomException(e,sys)
        
            
            
            
            
        
        
