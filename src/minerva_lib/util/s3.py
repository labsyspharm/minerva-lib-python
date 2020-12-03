from .progress import ProgressPercentage
import boto3
import logging
import s3transfer
import s3transfer.manager
import s3transfer.subscribers
import concurrent.futures

class S3Uploader:
    def __init__(self, region):
        self.region = region
        self.transfer_config = s3transfer.manager.TransferConfig()
        self.transfer_manager = None
        self.executor = concurrent.futures.ThreadPoolExecutor()

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

    def async_upload(self, buf, bucket, object_name, credentials, tile_content_type="image/tiff"):
        if self.transfer_manager is None:
            s3 = boto3.client("s3", aws_access_key_id=credentials["AccessKeyId"],
                              aws_secret_access_key=credentials["SecretAccessKey"],
                              aws_session_token=credentials["SessionToken"],
                              region_name=self.region)

            self.transfer_manager = s3transfer.manager.TransferManager(s3, config=self.transfer_config)

        future = self.executor.submit(self.upload_data, buf, bucket, object_name, credentials)
        return future

    def wait_upload(self):
        self.executor.shutdown()