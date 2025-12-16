import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Saves a Python object to file using dill serialization.
    
    Args:
        file_path: Path where object will be saved
        obj: Python object to serialize
        
    Raises:
        CustomException: If saving fails
    """
    try:
        # Create directory if it doesn't exist
        dir_path: str = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as ex:
        raise CustomException(ex, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Trains and evaluates multiple models, returning their R² scores.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model names and instances
        
    Returns:
        dict: Model names mapped to their test R² scores
        
    Raises:
        CustomException: If model evaluation fails
    """
    try:
        report: dict = {}

        # Loop through each model
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            # Train model on training data
            model.fit(X_train, y_train)
            
            # Make predictions on train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as ex:
        raise CustomException(ex, sys)