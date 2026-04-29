from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig

if __name__ == "__main__":
    
    # Data Ingestion
    ingestion_config = DataIngestionConfig(
        raw_data_path="artifacts/raw_data.csv",
        train_data_path = "artifacts/train_data.csv",
        test_data_path= "artifacts/test_data.csv"
    )
    
    ingestion = DataIngestion(config = ingestion_config)
    # Data Validation
    
    validation_config = DataValidationConfig(
        unzip_data_dir="artifacts/raw_data.csv",
        STATUS_FILE="artifacts/validation_status.txt"
    )
    
    validation = DataValidation(config=validation_config)
    
     # Data Transformation
     
    transformation_config = DataTransformationConfig(
        preprocessor_obj_file_path="artifacts/preprocessor.pkl"
    )
    
    transformation = DataTransformation(transformation_config)
    
    train_arr,test_arr,_ = transformation.initiate_data_transformation(
        train_path = "artifacts/train_data.csv",
        test_path = "artifacts/test_data.csv"
    )
    
    
    
    
    