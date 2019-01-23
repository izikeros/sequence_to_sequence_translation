# -*- coding: utf-8 -*-
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tensorflow.python.client import device_lib  # for checking GPU


# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
# !pip install -U -q PyDrive

def g_authenticate():
    # Authenticate and create the PyDrive client.
    # This only needs to be done once per notebook.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive


def check_gpu():
    print(device_lib.list_local_devices())


def load_from_gdrive(file_name, gdrive):
    file_list = gdrive.ListFile(
        {'q': "'1SS2LbNox9DwPeTE0IxlLhJUVH5SKeTv6' in parents"}).GetList()
    downloaded = None
    for f in file_list:
        if f['title'] == file_name:
            file_id = f['id']
            downloaded = gdrive.CreateFile({'id': file_id})
    return downloaded.GetContentString()

# drive = g_authenticate()
