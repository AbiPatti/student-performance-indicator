import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

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