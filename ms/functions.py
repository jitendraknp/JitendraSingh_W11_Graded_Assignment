import pandas as pd
from ms import model
import joblib
import numpy as np
def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = predict(X, model)
    if prediction == 1:
        label = "M"
    else:
        label = "B"
    return {
        'status': 200,
        'label': label,
        'prediction': int(prediction)
    }

def load_model(model_path):
    """Load the pre-trained model from the specified path."""
    return joblib.load(model_path)

def get_model_response(model, feature_dict):
    """
    Generate model prediction response based on input features.

    Parameters:
    model: The pre-trained model.
    feature_dict (dict): A dictionary containing feature values.

    Returns:
    dict: Prediction result.
    """
    try:
        features = np.array([feature_dict[col] for col in feature_dict])
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        return {'prediction': int(prediction[0])}
    except Exception as e:
        return {'error': str(e)}