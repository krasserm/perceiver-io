import zipfile
from pathlib import Path

import requests


def download_file(uri: str, target_file: Path):
    response = requests.get(uri, stream=True)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise FileNotFoundError(f"Could not find file at {uri}")
        else:
            raise e

    with open(target_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_file(dataset_file: Path, target_dir: Path):
    with zipfile.ZipFile(dataset_file, "r") as zip_file:
        zip_file.extractall(target_dir)
