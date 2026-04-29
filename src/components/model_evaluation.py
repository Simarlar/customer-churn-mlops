import os
import sys
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)

from src.exception import CustomException
from src.logger import logger
from src.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config = config
        
    def initiate_model_evaluation(self,test_arr,model_path):
        
        try:
            
            logger.info("Starting model evaluation")
            
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            
            model = joblib.load(model_path)
            
            y_pred = model.predict(X_test)
            
            metrics = {
                "accuracy" : accuracy_score(y_test,y_pred),
                "precision" : precision_score(y_test,y_pred),
                "recall" : recall_score(y_test,y_pred),
                "f1_score" : f1_score(y_test,y_pred),
            }
            
            # ROC-AUC only if probability exists
            if hasattr(model,"predict_proba"):
                y_prob = model.predict_proba(X_test)[:,1]
                
                metrics["roc_auc"] = roc_auc_score(y_test,y_prob)
            
            os.makedirs(os.path.dirname(self.config.metric_file_path),
                        exist_ok=True
            )
            
            with open(self.config.metric_file_path,"w") as f:
                json.dump(metrics,f,indent=4)
            
            logger.info(f"Evaluation metrics saved")
            
            return metrics
        
        except Exception as e:
            raise CustomException(e,sys)