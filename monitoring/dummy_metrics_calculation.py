import datetime
import io
import logging
import random
import time
import uuid

import pandas as pd
import psycopg2
import pytz

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS dummy_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    value1 INTEGER,
    value2 UUID,
    value3 FLOAT
);
"""


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


def calculate_dummy_metrics_postgresql(curr):
    value1 = random.randint(0, 1000)
    value2 = str(uuid.uuid4())
    value3 = random.random()
    timestamp = datetime.datetime.now(pytz.timezone("Asia/Taipei"))

    query = f"INSERT INTO dummy_metrics(timestamp, value1, value2, value3) VALUES ('{timestamp}', {value1}, '{value2}', {value3})"
    curr.execute(query)


def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    conn_string = "host=localhost port=5432 dbname=test user=postgres password=example"
    with psycopg2.connect(conn_string) as conn:
        for _ in range(0, 100):
            with conn.cursor() as curr:
                calculate_dummy_metrics_postgresql(curr)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == "__main__":
    main()
