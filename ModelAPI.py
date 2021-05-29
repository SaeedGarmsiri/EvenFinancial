import joblib
import os
import pandas as pd

from flask import Flask, jsonify, request


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({'prediction': prediction.tolist()})


@app.route('/predict_single', methods=['POST'])
def predict_single():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    data_path = os.path.join(ROOT_PATH, 'data/')
    loaded_path = os.path.join(data_path, 'loaded/')
    model_path = os.path.join(ROOT_PATH, 'model/')
    model_file_name = os.path.join(model_path, 'LR_Model.pkl')
    model = joblib.load(model_file_name)
    app.run(debug=True)
