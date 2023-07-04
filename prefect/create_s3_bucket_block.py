import os
from time import sleep

from prefect_aws import AwsCredentials, S3Bucket


def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name=os.environ["S3_BUCKET_BLOCK_NAME"], credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="my-s3-bucket", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()
