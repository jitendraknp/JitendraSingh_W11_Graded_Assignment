# Local imports
import datetime

# Third part imports
from flask import request
import pandas as pd

from ms import app
from ms.functions import load_model,get_model_response


model_name = "Breast Cancer Wisconsin (Diagnostic)"
model_file = 'model/model_binary.dat.gz'
version = "v1.0.0"

data_file="data/breast_cancer.csv"
model=load_model(model_file)
@app.route('/info', methods=['GET'])
def info():
    """Return model information, version, how to call"""
    result = {}

    result["name"] = model_name
    result["version"] = version

    return result


@app.route('/health', methods=['GET'])
def health():
    try:
            # Load the dataset
            df = pd.read_csv(data_file)

            # Simple check to ensure the dataset is loaded and contains expected columns
            expected_columns = ["radius_mean", "texture_mean","perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
            if not all(column in df.columns for column in expected_columns):
                return {'status': 'error', 'message': 'Dataset is missing expected columns'}, 500

            # Sample data for prediction
            sample_data = df[expected_columns].iloc[0].to_dict()

            # Perform a prediction to ensure the model is working
            response = get_model_response(model,sample_data)
            if 'error' in response:
                return {'status': response, 'message': 'Model prediction failed'}, 500

    except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

    return {'status': 'ok'}, 200


@app.route('/predict', methods=['POST'])
def predict():
    feature_dict = request.get_json()
    if not feature_dict:
        return {
            'error': 'Body is empty.'
        }, 500

    try:
        response = get_model_response(feature_dict)
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500

    return response, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')
