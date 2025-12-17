import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    """
    Handles prediction workflow using trained model and preprocessor.
    Loads saved artifacts and generates predictions for new data.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Generates predictions using trained model and preprocessor.
        
        Args:
            features: DataFrame containing student features for prediction
            
        Returns:
            numpy array of predicted math scores
            
        Raises:
            CustomException: If prediction fails
        """
        try:
            # Define paths to saved model and preprocessor artifacts
            model_path = 'artifact\model.pkl'
            preprocessor_path = 'artifact\preprocessor.pkl'

            # Load trained model and preprocessor from disk
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Transform features using preprocessor (scaling, encoding)
            data_scaled = preprocessor.transform(features)

            # Generate predictions using trained model
            preds = model.predict(data_scaled)
            return preds
        except Exception as ex:
            raise CustomException(ex, sys)

class CustomData:
    """
    Encapsulates student data for prediction.
    Stores individual student features and converts them to DataFrame format.
    """
    def __init__(self,
                gender,
                race_ethnicity,
                parental_level_of_education,
                lunch,
                test_preparation_course,
                reading_score,
                writing_score):
        
        # Store all student demographic and academic features
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_dataframe(self):
        """
        Converts stored student data into pandas DataFrame format.
        Returns DataFrame with single row containing all features for prediction.
        """
        try:
            # Create dictionary with feature names as keys
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as ex:
            raise CustomException(ex, sys)