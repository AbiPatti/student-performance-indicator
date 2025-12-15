import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation artifact paths"""
    preprocessor_fil_path: str = os.path.join("artifact", "preprocessor.pkl")

class DataTransformation:
    """Handles feature engineering and data preprocessing"""
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        """
        Creates preprocessing pipeline for numerical and categorical features.
        
        Returns:
            ColumnTransformer: Sklearn preprocessor with scaling and encoding pipelines
            
        Raises:
            CustomException: If pipeline creation fails
        """
        try:
            # Separate numerical and categorical features
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Create pipeline for numerical features (impute + scale)
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Create pipeline for categorical features (impute + encode)
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns pipeline completed")
            logging.info("Categorical columns pipeline completed")

            # Combine pipelines into single preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as ex:
            raise CustomException(ex, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing transformations to train and test datasets.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            
        Returns:
            tuple: Transformed train array, test array, and preprocessor file path
            
        Raises:
            CustomException: If transformation fails
        """
        try:
            # Read train and test data
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Train and test sets have been read into DataFrames")

            # Get preprocessing object
            preprocessor = self.get_data_transformer()

            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate features and target for training data
            df_input_feature_train = df_train.drop(columns=[target_column], axis=1)
            df_target_feature_train = df_train[target_column]

            # Separate features and target for test data
            df_input_feature_test = df_test.drop(columns=[target_column], axis=1)
            df_target_feature_test = df_test[target_column]

            # Apply transformations
            arr_input_feature_train = preprocessor.fit_transform(df_input_feature_train)
            arr_input_feature_test = preprocessor.transform(df_input_feature_test)

            # Combine features and target into single arrays
            train_arr = np.c_[arr_input_feature_train, np.array(df_target_feature_train)]
            test_arr = np.c_[arr_input_feature_test, np.array(df_target_feature_test)]

            logging.info("Saved preprocessing object")

            # Save object
            save_object(
                file_path=self.data_transformation_config.preprocessor_fil_path,
                obj=preprocessor
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_fil_path)

        except Exception as ex:
            raise CustomException(ex, sys)