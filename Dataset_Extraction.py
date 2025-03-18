import zipfile
import os

# Unzip dataset
with zipfile.ZipFile("/content/predata.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/predata")