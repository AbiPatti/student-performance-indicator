import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    """Configuration class for model training artifact paths"""
    trained_model_file_path: str = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    """Handles model training and selection of best performing model"""
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path=None):
        """
        Trains multiple models and selects the best one based on R² score.
        
        Args:
            train_array: Training data with features and target
            test_array: Test data with features and target
            preprocessor_path: Path to the preprocessor object
            
        Returns:
            float: R² score of the best model
            
        Raises:
            CustomException: If no model performs well enough or training fails
        """
        try:
            logging.info("Splitting training and test input data")
            # Split features and target from arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define candidate models to evaluate
            models: dict = {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(random_state=42),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42)
            }

            model_params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[3,5,7,9],
                    'weights':['uniform','distance']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            # Train and evaluate all models
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, model_params)

            # Get best model's score (highest R² score) from model report
            best_model_score = max(sorted(model_report.values()))

            # Get best model's name from model report
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Get best model
            best_model = models[best_model_name]

            # Check if best model meets minimum performance threshold
            if best_model_score < 0.6:
                raise CustomException("No model with an R2 score greater than 0.6 found")
            
            logging.info("Best model found: ", best_model_name)

            # Save best model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Calculate final R² score on test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as ex:
            raise CustomException(ex, sys)