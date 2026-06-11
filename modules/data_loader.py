"""Downloads and manages FIFA dataset from Kaggle"""

import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


class DataSet:
    def __init__(
        self,
        dataset_signature: str = "harrachimustapha/fifa-world-cup-team-dataset",
        download_path: str = "./datasets/fifa",
    ):
        """
        Initializes the DataSet configuration.
        The constructor is clean, light, and parameterized.
        """
        self.dataset_signature = dataset_signature
        self.download_path = download_path

        # Define target file paths for the dataset
        self.train_file_path = os.path.join(self.download_path, "train.csv")
        self.test_file_path = os.path.join(self.download_path, "test.csv")

        # Placeholders for dataframes (initially empty)
        self.train = None
        self.test = None

    def load_data(self, force_download: bool = False):
        """
        Main orchestration method. Loads data from local files.
        Triggers download only if files are missing or force_download is True.
        """
        # Check if the required CSV files already physically exist on the disk
        files_exist = os.path.exists(self.train_file_path) and os.path.exists(
            self.test_file_path
        )

        if not files_exist or force_download:
            print("Local files missing or force_download=True. Fetching from Kaggle...")
            self._download_and_extract()
        else:
            print("Data found locally. Skipping redundant download to save bandwidth.")

        # Read data from disk into RAM (Pandas DataFrames)
        self.train = pd.read_csv(self.train_file_path)
        self.test = pd.read_csv(self.test_file_path)
        print("Data loaded successfully into attributes `.train` and `.test`.")

    def _download_and_extract(self):
        """
        Internal helper method handling Kaggle API requests.
        """
        # Initialize API - automatically looks for KAGGLE_USERNAME and KAGGLE_KEY in os.environ
        api = KaggleApi()
        api.authenticate()

        # Download the dataset ZIP package
        print(f"Downloading {self.dataset_signature}...")
        api.dataset_download_files(
            self.dataset_signature, path=self.download_path, unzip=False
        )

        # Extract the archive
        zip_file_path = os.path.join(
            self.download_path, "fifa-world-cup-team-dataset.zip"
        )
        print("Extracting files...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.download_path)

        # Optional: Clean up the ZIP file after extraction to preserve disk space
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)


if __name__ == "__main__":
    data_set = DataSet()

    # Trigger the explicit data loading pipeline
    data_set.load_data()

    # Now it is safe to operate on the DataFrame attributes
    print("\nPreview of the test dataset (first 5 rows):")
    print(data_set.test.head())
