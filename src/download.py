import kfp.components as comp


def download_dataset(dataset_url, dataset_path: comp.OutputPath(str)):
    import os
    import tempfile
    import zipfile

    import requests

    os.makedirs(dataset_path + "/data", exist_ok=True)
    r = requests.get(dataset_url, stream=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r.iter_content(chunk_size=128):
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(dataset_path + "/data")
