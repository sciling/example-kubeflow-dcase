from unittest import TestCase
import unittest
import sys
sys.path.append('..')
import src.train as train
import tempfile
import os
import pathlib
import json


DATA_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../data/test/minidataset/"


class TestTrain(TestCase):
    def test_train(self):
        # Directory of the model
        model_directory = tempfile.mkdtemp()

        # Loss plot file
        loss_plot_file = tempfile.NamedTemporaryFile()

        # Train call
        lossplot_web_app = train.train(dataset_path=DATA_DIR, feature_frames=5, feature_hop_length=512, feature_n_fft=1024, feature_n_mels=128, feature_power=2.0,
                                       fit_batch_size=512, fit_compile_loss="mean_squared_error", fit_compile_optimizer="adam", fit_epochs=2, fit_shuffle=True,
                                       fit_validation_split=0.15, fit_verbose=1, max_fpr=0.1, lossplot_path=loss_plot_file.name, models_dir=model_directory)

        # Check model directory has all files
        self.assertIn('model', os.listdir(model_directory))
        self.assertNotEqual(0, len(os.listdir(f"{model_directory}/model")))

        # Check the lossplot file content
        self.assertIn('outputs', list(json.loads(lossplot_web_app[0]).keys()))
        self.assertIsInstance(json.loads(lossplot_web_app[0])['outputs'], list)
        self.assertEqual(len(json.loads(lossplot_web_app[0])['outputs']), 1)

        # Closing all open files
        loss_plot_file.close()


if __name__ == '__main__':
    unittest.main()
