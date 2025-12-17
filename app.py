from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
app = Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction - handles both form display and prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Display the prediction form
        return render_template('home.html')
    else:
        # Collect form data and create CustomData object
        data = CustomData(
                    gender = request.form.get("gender"),
                    race_ethnicity = request.form.get("race_ethnicity"),
                    parental_level_of_education = request.form.get("parental_level_of_education"),
                    lunch = request.form.get("lunch"),
                    test_preparation_course = request.form.get("test_preparation_course"),
                    reading_score = float(request.form.get("reading_score")),
                    writing_score = float(request.form.get("writing_score"))
        )
        # Convert CustomData object to DataFrame format for model input
        df_pred = data.get_data_as_dataframe()
        print(df_pred)

        # Initialize prediction pipeline and generate math score prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(df_pred)

        # Render template with predicted score (extract first element from array)
        return render_template('home.html', results=results[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)