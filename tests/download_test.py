import unittest
from unittest import TestCase
import sys
sys.path.append('..')
import src.download as download
import tempfile
import os


class TestTrain(TestCase):
    def test_download(self):
        # Directory of the dataset
        data_directory = tempfile.mkdtemp()

        # Downlaod
        download.download_dataset('https://zenodo.org/record/3678171/files/dev_data_fan.zip', dataset_path=data_directory)

        # Check the path of the dataset content
        self.assertIn('data', os.listdir(data_directory))
        self.assertIn('fan', os.listdir(f"{data_directory}/data"))
        self.assertIn('train', os.listdir(f"{data_directory}/data/fan"))
        self.assertIn('test', os.listdir(f"{data_directory}/data/fan"))


if __name__ == '__main__':
    unittest.main()
