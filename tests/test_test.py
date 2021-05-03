from unittest import TestCase
import unittest
import sys
sys.path.append('..')
import src.test as test
import tempfile
import os
import pathlib


DATA_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../data/test/minidataset/"
MODEL_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../data/test/minidataset_model/"


class TestTrain(TestCase):
    def test_test(self):
        # Results directory
        results_directory = tempfile.mkdtemp()

        # Labels directory
        labels_directory = tempfile.mkdtemp()

        # Anomaly directory
        anomaly_directory = tempfile.mkdtemp()

        # Metrics file
        metrics_file = tempfile.NamedTemporaryFile()

        # Test call
        test.test(dataset_path=DATA_DIR, feature_frames=5, feature_hop_length=512, feature_n_fft=1024, feature_n_mels=128,
                  feature_power=2.0, fit_batch_size=512, fit_compile_loss="mean_squared_error", fit_compile_optimizer="adam",
                  fit_epochs=2, fit_shuffle=True, fit_validation_split=0.15, fit_verbose=1, max_fpr=0.1, models_dir=MODEL_DIR,
                  anomaly_dir=anomaly_directory, results_dir=results_directory, mlpipelinemetrics_path=metrics_file.name,
                  labels_dir=labels_directory)

        # Check the labels are correctly created in the directory
        self.assertIn("y_scores.txt", os.listdir(labels_directory))
        self.assertIn("y_labels.txt", os.listdir(labels_directory))

        # Check anomaly directory is not empty
        self.assertNotEqual(list(os.listdir(anomaly_directory)), 0)

        # Check results directory is has result.csv
        self.assertIn("results.csv", os.listdir(results_directory))

        # Closing all open files
        metrics_file.close()


if __name__ == '__main__':
    unittest.main()
