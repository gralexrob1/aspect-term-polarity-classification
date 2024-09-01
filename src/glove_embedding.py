import os
import urllib.request
import zipfile


def download_glove(url: str, filename: str):
    print(f"Downloading GloVe embeddings from {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded GloVe embeddings to {filename}")


def extract_glove(zip_file_path: str, extract_to: str):
    print(f"Extracting GloVe embeddings from {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted GloVe embeddings to {extract_to}")


def setup_glove(glove_file_path: str):
    if not os.path.isfile(glove_file_path):
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_file_path = "glove.6B.zip"

        if not os.path.isfile(zip_file_path):
            download_glove(url, zip_file_path)

        extract_glove(zip_file_path, ".")

        if not os.path.isfile(glove_file_path):
            raise FileNotFoundError(f"Expected file {glove_file_path} not found after extraction.")
