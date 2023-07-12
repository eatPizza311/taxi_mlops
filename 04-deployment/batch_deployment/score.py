#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import mlflow
import pandas as pd

# from dateutil.relativedelta import relativedelta
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.context import get_run_context


def generate_uuids(length: int) -> List[str]:
    """Generate uuid for each record."""
    return [str(uuid.uuid4()) for _ in range(length)]


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


def load_model(run_id):
    """Load model from S3 bucket."""
    logged_model = f"s3://taxi-mlops/1/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def save_results(
    df: pd.DataFrame, y_pred: List[float], run_id: str, output_file: str
) -> pd.DataFrame:
    """Create result dataframe and save it to parquet."""
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


@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info(f"Reading the data from {input_file}...")
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f"Loading the model with RUN_ID={run_id}...")
    model = load_model(run_id)

    logger.info("applying the model...")
    y_pred = model.predict(dicts)

    logger.info(f"Saving the result to {output_file}...")

    result = save_results(df, y_pred, run_id, output_file)

    start_index = (
        input_file.rfind("_") + 1
    )  # Find the index of the underscore before the desired portion
    end_index = input_file.rfind(
        "."
    )  # Find the index of the dot before the file extension
    year_month = input_file[
        start_index:end_index
    ]  # Extract the substring between the underscore and the dot

    markdown_report = f"""# Prediction Report
        ## Summary

        Duration Prediction

        ## Random Forest Model
        |   Year/Month    | Mean difference |
        |:----------------|----------------:|
        |    {year_month}      |          {result["diff"].mean():.2f} |
        """

    create_markdown_artifact(key="duration-report", markdown=markdown_report)


def get_paths(run_date: datetime, taxi_type: str, run_id: str) -> Tuple[str]:
    """Getting the path for input and ouput file"""
    # prev_month = run_date - relativedelta(months=1)
    year = run_date.year
    month = run_date.month

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"s3://taxi-mlops/output/taxi_type={taxi_type}/year={year:04d}/month={month:02d}/{run_id}.parquet"

    return input_file, output_file


@flow(name="inference")
def ride_duration_prediction(taxi_type: str, run_id: str, run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    input_file, output_file = get_paths(run_date, taxi_type, run_id)
    os.environ["AWS_PROFILE"] = "Profile1"

    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", type=int, help="The year of the input data", default=None
    )
    parser.add_argument(
        "--month", type=int, help="The month of the input data", default=None
    )
    parser.add_argument("--taxi_type", type=str, help="The taxi type", default="green")
    # "95c848791a7642ff8c26794d43e410a8"
    parser.add_argument(
        "--run_id", type=str, help="MLflow run id for model in S3 bucket"
    )

    args = parser.parse_args()

    if args.month is None:
        run_date = None
    else:
        run_date = datetime(year=args.year, month=args.month, day=1)

    ride_duration_prediction(
        taxi_type=args.taxi_type,
        run_id=args.run_id,
        run_date=run_date,
    )
