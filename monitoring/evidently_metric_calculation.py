import datetime
import random
import time

import joblib
import pandas as pd
import psycopg2
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report

from prefect import flow, get_run_logger, task

SEND_TIMEOUT = 10
rand = random.Random()

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS evidently_metrics (
      id SERIAL PRIMARY KEY,
      timestamp TIMESTAMP,
      prediction_drift FLOAT,
      num_drifted_columns INTEGER,
      SHARE_MISSING_VALUES FLOAT
);
"""

reference_data = pd.read_parquet("./data/reference.parquet")
with open("./model/lin_reg.bin", "rb") as f_in:
    model = joblib.load(f_in)

raw_data = pd.read_parquet("./data/green_tripdata_2022-02.parquet")

begin = datetime.datetime(2022, 2, 1, 0, 0)
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]
column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


@task
def prep_db():
    logger = get_run_logger()
    conn_string_default_db = (
        "host=localhost port=5432 dbname=postgres user=postgres password=example"
    )
    conn = psycopg2.connect(conn_string_default_db)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    if len(cursor.fetchall()) == 0:
        cursor.execute("CREATE DATABASE test;")
    cursor.close()
    conn.close()
    conn_string = "host=localhost port=5432 dbname=test user=postgres password=example"
    with psycopg2.connect(conn_string) as conn:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_TABLE_STATEMENT)
    logger.info("Table created!")


@task
def calculate_metrics_postgresql(i):
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i)))
        & (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))
    ]
    current_data["prediction"] = model.predict(
        current_data[num_features + cat_features].fillna(0)
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]

    return prediction_drift, num_drifted_columns, share_missing_values


@flow
def batch_monitoring_backfill():
    prep_db()
    conn_string = "host=localhost port=5432 dbname=test user=postgres password=example"
    logger = get_run_logger()
    with psycopg2.connect(conn_string) as conn:
        for i in range(0, 27):
            with conn.cursor() as cursor:
                (
                    prediction_drift,
                    num_drifted_columns,
                    share_missing_values,
                ) = calculate_metrics_postgresql(i)
                query = (
                    f"INSERT INTO evidently_metrics(timestamp, prediction_drift, num_drifted_columns, "
                    f"share_missing_values) VALUES ('{begin + datetime.timedelta(i)}', "
                    f"{prediction_drift}, '{num_drifted_columns}', {share_missing_values});"
                )
                cursor.execute(query)
            logger.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()
