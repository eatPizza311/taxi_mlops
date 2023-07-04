import os

import mlflow
from flask import Flask, jsonify, request

app = Flask("duration-prediction")


RUN_ID = os.getenv("RUN_ID")
logged_model = f"s3://taxi-mlops/1/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    preds = model.predict(features)
    return preds[0]


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {"duration": pred}
    return jsonify(result)


@app.route("/", methods=["GET"])
def index():
    return "The server is running!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4444)
