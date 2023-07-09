import datetime
import logging
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

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


def prep_db():
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


def calculate_metrics_postgresql(curr, i):
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

    query = (
        f"INSERT INTO evidently_metrics(timestamp, prediction_drift, num_drifted_columns, "
        f"share_missing_values) VALUES ('{begin + datetime.timedelta(i)}', "
        f"{prediction_drift}, '{num_drifted_columns}', {share_missing_values})"
    )
    curr.execute(query)


def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    conn_string = "host=localhost port=5432 dbname=test user=postgres password=example"
    with psycopg2.connect(conn_string) as conn:
        for i in range(0, 27):
            with conn.cursor() as cursor:
                calculate_metrics_postgresql(cursor, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send += datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()
