import os
import zipfile
from pathlib import Path

SOURCE_PATH = 'row_data\\avidreaders.ru\\book'
TARGET_FOLDER = 'row_data\\avidreaders.ru\\unzip'

for path in os.listdir(SOURCE_PATH):
    file_path = os.path.join(SOURCE_PATH, path, "data_set.fb2.zip")
    if os.path.isfile(file_path):
        try:

            with zipfile.ZipFile(file_path,"r") as zip_ref:
                target_folder = os.path.join(TARGET_FOLDER, path)
                Path(target_folder).mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(target_folder)
        except Exception:
            print(f'Error {file_path}')
