import os
import sys
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, train_arr, test_arr):

        try:
            logger.info("Splitting training and testing arrays")

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier()
            }

            model_report = {}

            mlflow.set_experiment("Customer-Churn-Prediction")

            best_model_name = None
            best_f1 = -1

            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    logger.info(f"Training {model_name}")

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # MLflow logging
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("f1_score", f1)

                    
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name="CustomerChurnModel"
                )

                    model_report[model_name] = {
                        "accuracy": accuracy,
                        "f1_score": f1
                    }

                    # track best model
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_name = model_name

            # AFTER loop → select best model
            best_model = models[best_model_name]

            logger.info(f"Best model found: {best_model_name}")

            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path),
                exist_ok=True
            )

            joblib.dump(best_model, self.config.trained_model_file_path)

            logger.info("Best model saved successfully")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
        
