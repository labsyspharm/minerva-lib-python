import base64
import re
from io import BytesIO

import boto3
import logging
import requests
import json
import botocore
import tifffile


class InvalidUsernameOrPassword(Exception):
    pass


class InvalidCognitoClientId(Exception):
    pass

class MinervaClient:
    def __init__(self, endpoint, region, cognito_client_id):
        self.endpoint = endpoint
        self.region = region
        self.cognito_client_id = cognito_client_id
        self.id_token = None
        self.token_type = None
        self.refresh_token = None
        self.session = None
        self.credentials_cache = {}

    def authenticate(self, username, password):
        try:
            logging.info("Logging in as %s", username)
            config = botocore.config.Config(signature_version=botocore.UNSIGNED, region_name=self.region)
            client = boto3.client('cognito-idp', config=config)
            response = client.initiate_auth(
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                },
                ClientId=self.cognito_client_id
            )
            self.id_token = response["AuthenticationResult"]["IdToken"]
            self.token_type = response["AuthenticationResult"]["TokenType"]
            self.refresh_token = response["AuthenticationResult"]["RefreshToken"]
            logging.debug("Authenticated successfully")
        except client.exceptions.NotAuthorizedException:
            raise InvalidUsernameOrPassword
        except client.exceptions.ResourceNotFoundException as e:
            logging.error(e)
            raise InvalidCognitoClientId

    def request(self, method, path, body=None, parameters=None, json_response=True):
        if self.session is None:
            self.session = requests.Session()

        self.session.headers.update({
            "Authorization": self.token_type + " " + self.id_token,
            "Content-Type": "application/json"
        })
        url = self.endpoint + path

        if body is not None:
            body = json.dumps(body)

        response = self.session.request(method=method, url=url, data=body, params=parameters)

        if response.status_code >= 400:
            logging.error(response.text)

        response.raise_for_status()
        logging.debug(response)
        if json_response:
            return response.json()
        else:
            return response.text

    def list_repositories(self):
        return self.request('GET', '/repository')

    def create_repository(self, name, raw_storage="Destroy"):
        body = {
            "name": name,
            "raw_storage": raw_storage
        }
        return self.request('POST', '/repository', body)

    def create_import(self, name, repository_uuid):
        body = {
            "name": name,
            "repository_uuid": repository_uuid
        }
        return self.request('POST', '/import', body)

    def get_import_credentials(self, import_uuid):
        return self.request('GET', '/import/' + import_uuid + '/credentials')

    def get_image_credentials(self, image_uuid):
        res = self.request('GET', '/image/' + image_uuid + '/credentials')

        m = re.match(r"^s3://([A-z0-9\-]+)/([A-z0-9\-]+)/$", res["data"]["image_url"])
        bucket = m.group(1)
        prefix = m.group(2)
        credentials = res["data"]["credentials"]
        self.credentials_cache[image_uuid] = {
            "credentials": credentials,
            "bucket": bucket,
            "prefix": prefix
        }
        return credentials, bucket, prefix

    def get_raw_tile(self, uuid, x, y, z, t, c, level, format=".tif"):
        if uuid not in self.credentials_cache:
            self.get_image_credentials(uuid)

        credentials = self.credentials_cache[uuid]["credentials"]
        bucket = self.credentials_cache[uuid]["bucket"]

        s3 = boto3.client("s3", aws_access_key_id=credentials["AccessKeyId"],
                          aws_secret_access_key=credentials["SecretAccessKey"],
                          aws_session_token=credentials["SessionToken"],
                          region_name=self.region)

        key = f'{uuid}/C{c}-T{t}-Z{z}-L{level}-Y{y}-X{x}{format}'
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj['Body']
        data = body.read()
        stream = BytesIO(data)
        return tifffile.imread(stream)

    def get_image_metadata(self, image_uuid):
        return base64.b64decode(self.request('GET', '/image/' + image_uuid + '/metadata', json_response=False))

    def get_image(self, image_uuid):
        return self.request('GET', '/image/' + image_uuid)

    def mark_import_complete(self, import_uuid):
        body = {
            "complete": True
        }
        return self.request('PUT', '/import/' + import_uuid, body)

    def list_filesets_in_import(self, import_uuid):
        return self.request('GET', '/import/' + import_uuid + '/filesets')

    def list_images_in_fileset(self, fileset_uuid):
        return self.request('GET', '/fileset/' + fileset_uuid + '/images')

    def get_image_dimensions(self, image_uuid):
        return self.request('GET', '/image/' + image_uuid + '/dimensions')

    def list_incomplete_imports(self):
        return self.request('GET', '/import/incomplete')

    def cognito_details(self):
        return self.request('GET', '/cognito_details')

    def create_image(self, name, repository_uuid, pyramid_levels=1):
        body = {
            "name": name,
            "pyramid_levels": pyramid_levels,
            "repository_uuid": repository_uuid
        }
        return self.request('POST', '/image', body)
