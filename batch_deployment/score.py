#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import uuid
from typing import Dict, List

import mlflow
import pandas as pd


def read_dataframe(filename: str) -> pd.DataFrame:
    """Read DataFrame and target column from secs to mins."""
    df = pd.read_parquet(filename)

    df["duration"] = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60.0)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df["ride_id"] = generate_uuids(len(df))
    return df


def prepare_dictionaries(df: pd.DataFrame) -> List[Dict]:
    """Turn DataFrame into list of dictionary."""
    # convert data type
    categorical_str = ["PULocationID", "DOLocationID"]
    df[categorical_str] = df[categorical_str].astype(str)

    # Feature engineering
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    return dicts


def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def load_model(run_id):
    logged_model = f"s3://taxi-mlops/1/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    model = load_model(run_id)
    y_pred = model.predict(dicts)

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result["predicted_duration"] = y_pred
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = run_id
    df_result.to_parquet(output_file, index=False)

    return df_result


def run(year, month, taxi_type, run_id):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Log an informational message
    logging.info("Starting the run...")

    # Setup input and output file
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"

    # Setup AWS credential
    os.environ["AWS_PROFILE"] = "Profile1"

    # Log another message with variables included
    logging.info(f"Processing data for {taxi_type} taxis in {year}/{month:02d}")

    result = apply_model(input_file, run_id, output_file)

    # Log a final message
    logging.info("Run completed successfully.")
    logging.info(f"The mean difference is {result['diff'].mean():.2f} mins.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="The year of the input data")
    parser.add_argument("--month", type=int, help="The month of the input data")
    parser.add_argument("--taxi_type", type=str, help="The taxi type", default="green")
    # "95c848791a7642ff8c26794d43e410a8"
    parser.add_argument(
        "--run_id", type=str, help="MLflow run id for model in S3 bucket"
    )
    args = parser.parse_args()
    # run(**vars(args))
    result = run(args.year, args.month, args.taxi_type, args.run_id)
