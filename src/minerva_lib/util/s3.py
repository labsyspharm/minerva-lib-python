from .progress import ProgressPercentage
import boto3
import logging

class S3Uploader:
    def __init__(self, region):
        self.region = region

    def upload_file(self, filepath, bucket, object_name, credentials, callback: ProgressPercentage=None):
        try:
            logging.info("Uploading file %s", filepath)
            s3 = boto3.client("s3", aws_access_key_id=credentials["AccessKeyId"],
                              aws_secret_access_key=credentials["SecretAccessKey"],
                              aws_session_token=credentials["SessionToken"],
                              region_name=self.region)

            s3.upload_file(filepath, bucket, object_name, Callback=callback)
        except Exception as e:
            logging.error(e)

    def upload_data(self, data, bucket, object_name, credentials):
        try:
            logging.info("Uploading object %s", object_name)
            s3 = boto3.client("s3", aws_access_key_id=credentials["AccessKeyId"],
                              aws_secret_access_key=credentials["SecretAccessKey"],
                              aws_session_token=credentials["SessionToken"],
                              region_name=self.region)
            s3.put_object(Body=data, Bucket=bucket, Key=object_name)

        except Exception as e:
            logging.error(e)
