import typing

from kfp import components
from kfp import dsl


# **3 Download dataset**
#
# One pipeline parameter will be the url of the dataset we are going to focus. That url should lead to a dataset that has the following structure:
#
#
# /machine( Toy_Car / fan / slider ...)
#
#     /train (Only normal data for all Machine IDs are included.)
#         /normal_id_01_00000000.wav
#         ...
#         /normal_id_01_00000999.wav
#         /normal_id_02_00000000.wav
#         ...
#         /normal_id_04_00000999.wav
#     /test (Normal and anomaly data for all Machine IDs are included.)
#         /normal_id_01_00000000.wav
#         ...
#         /normal_id_01_00000349.wav
#         /anomaly_id_01_00000000.wav
#         ...
#         /anomaly_id_01_00000263.wav
#         /normal_id_02_00000000.wav
#         ...
#         /anomaly_id_04_00000264.wav
#
# Some options for the dataset url are:
#
# * Fan dataset: https://zenodo.org/record/3678171/files/dev_data_fan.zip
# * Toy-car dataset:https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip
# * Slider rail dataset: https://zenodo.org/record/3678171/files/dev_data_slider.zip
# * Toy-conveyor dataset: https://zenodo.org/record/3678171/files/dev_data_ToyConveyor.zip
# * Valve dataset: https://zenodo.org/record/3678171/files/dev_data_valve.zip
# * Pump dataset: https://zenodo.org/record/3678171/files/dev_data_pump.zip
#
# The following python function will become a component later. This component will download the dataset in a shared space between pods, so it can be accessed by the others components.

# In[ ]:


def download_dataset(dataset_url, dataset_path: components.OutputPath(str)):
    import os
    import tempfile
    import zipfile

    import requests

    # Dataset for development
    os.makedirs(dataset_path + "/data", exist_ok=True)
    r = requests.get(dataset_url, stream=True)
    with tempfile.TemporaryFile() as tf:
        for chunk in r.iter_content(chunk_size=128):
            tf.write(chunk)
        with zipfile.ZipFile(tf, "r") as f:
            f.extractall(dataset_path + "/data")


# **4 Training component**
#
# The following python function will become a component later. This component gets the reference from the dataset and another pipeline parameters.
# Using all the parameters, trains a model and saves it in a temporal shared directory. It also generates a loss plot.

# In[ ]:


def train(
    dataset_path: components.InputPath(str),
    feature_frames,
    feature_hop_length,
    feature_n_fft,
    feature_n_mels,
    feature_power,
    fit_batch_size,
    fit_compile_loss,
    fit_compile_optimizer,
    fit_epochs,
    fit_shuffle,
    fit_validation_split,
    fit_verbose,
    max_fpr,
    lossplot_path: components.OutputPath(str),
    models_dir: components.OutputPath(str),
) -> typing.NamedTuple(
    "loss_plot", [("mlpipeline_ui_metadata", "UI_metadata")]  # noqa: F821
):
    import base64
    import glob
    import logging
    import os
    import sys

    import librosa
    import librosa.core
    import librosa.feature
    import matplotlib.pyplot as plt
    import numpy

    # Configure logger
    logger = logging.getLogger(" ")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Parse arguments from pipeline
    feature_frames = int(feature_frames)
    feature_hop_length = int(feature_hop_length)
    feature_n_fft = int(feature_n_fft)
    feature_n_mels = int(feature_n_mels)
    feature_power = float(feature_power)
    fit_batch_size = int(fit_batch_size)
    fit_epochs = int(fit_epochs)
    fit_validation_split = float(fit_validation_split)
    fit_verbose = int(fit_verbose)
    max_fpr = float(max_fpr)

    def select_dirs():
        """
        return :
                dirs :  list [ str ]
                    load base directory list of data
        """

        logger.info("load_directory <- data")
        dir_path = os.path.abspath(dataset_path + "{base}/*".format(base="/data"))
        dirs = sorted(glob.glob(dir_path))
        return dirs

    def file_list_generator(target_dir, dir_name="train", ext="wav"):
        """
        target_dir : str
            base directory path of the data
        dir_name : str (default="train")
            directory name containing training data
        ext : str (default="wav")
            file extension of audio files
        return :
            train_files : list [ str ]
                file list for training
        """

        logger.info("target_dir : {}".format(target_dir))
        # generate training list
        training_list_path = os.path.abspath(
            "{dir}/{dir_name}/*.{ext}".format(
                dir=target_dir, dir_name=dir_name, ext=ext
            )
        )
        files = sorted(glob.glob(training_list_path))
        if len(files) == 0:
            logger.exception("no_wav_file!!")

        logger.info("train_file num : {num}".format(num=len(files)))
        return files

    def file_load(wav_name, mono=False):
        """
        load .wav file.
        wav_name : str
            target .wav file
        sampling_rate : int
            audio file sampling_rate
        mono : boolean
            When load a multi channels file and this param True, the returned data will be merged for mono data
        return : numpy.array( float )
        """

        try:
            return librosa.load(wav_name, sr=None, mono=mono)
        except Exception:
            logger.error("file_broken or not exists!! : {}".format(wav_name))

    def file_to_vector_array(
        file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0
    ):
        """
        convert file_name to a vector array.
        file_name : str
            target .wav file
        return : numpy.array( numpy.array( float ) )
            vector array
            * dataset.shape = (dataset_size, feature_vector_length)
        """

        dims = n_mels * frames
        y, sr = file_load(file_name)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
        )

        log_mel_spectrogram = (
            20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
        )

        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

        if vector_array_size < 1:
            return numpy.empty((0, dims))

        vector_array = numpy.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[
                :, t : t + vector_array_size
            ].T

        return vector_array

    def list_to_vector_array(
        file_list,
        msg="calc...",
        n_mels=64,
        frames=5,
        n_fft=1024,
        hop_length=512,
        power=2.0,
    ):
        """
        convert the file_list to a vector array.
        file_to_vector_array() is iterated, and the output vector array is concatenated.
        file_list : list [ str ]
            .wav filename list of dataset
        msg : str ( default = "calc..." )
            description for tqdm.
            this parameter will be input into "desc" param at tqdm.
        return : numpy.array( numpy.array( float ) )
            vector array for training (this function is not used for test.)
            * dataset.shape = (number of feature vectors, dimensions of feature vectors)
        """
        dims = n_mels * frames
        for idx in range(len(file_list)):
            vector_array = file_to_vector_array(
                file_list[idx],
                n_mels=n_mels,
                frames=frames,
                n_fft=n_fft,
                hop_length=hop_length,
                power=power,
            )
            if idx == 0:
                dataset = numpy.zeros(
                    (vector_array.shape[0] * len(file_list), dims), float
                )
            dataset[
                vector_array.shape[0] * idx : vector_array.shape[0] * (idx + 1), :
            ] = vector_array
        return dataset

    def get_model(inputDim):
        import tensorflow.keras.models

        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model

        """
        define the keras model
        the model based on the simple dense auto encoder
        (128*128*128*128*8*128*128*128*128)
        """
        inputLayer = Input(shape=(inputDim,))

        h = Dense(128)(inputLayer)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(8)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)

        h = Dense(inputDim)(h)

        return Model(inputs=inputLayer, outputs=h)

    def loss_plot(loss, val_loss):
        fig = plt.figure(figsize=(30, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure():
        """
        Save figure.
        return : encoded bytes of the plot"""
        with open(lossplot_path, "wb") as fd:
            plt.savefig(fd)
        encoded = base64.b64encode(open(lossplot_path, "rb").read()).decode("latin1")
        return encoded

    dirs = select_dirs()
    # loop of the base directory
    outputs = []
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print(
            "[{idx}/{total}] {dirname}".format(
                dirname=target_dir, idx=idx + 1, total=len(dirs)
            )
        )
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(
            model=models_dir + "/model", machine_type=machine_type
        )
        if not os.path.exists(models_dir + "/model"):
            os.makedirs(models_dir + "/model")
        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir)
        train_data = list_to_vector_array(
            files,
            msg="generate train_dataset",
            n_mels=feature_n_mels,
            frames=feature_frames,
            n_fft=feature_n_fft,
            hop_length=feature_hop_length,
            power=feature_power,
        )
        # train model
        print("============== MODEL TRAINING ==============")
        model = get_model(feature_n_mels * feature_frames)
        model.summary()
        model.compile(
            optimizer=fit_compile_optimizer,
            loss=fit_compile_loss,
        )
        history = model.fit(
            train_data,
            train_data,
            epochs=fit_epochs,
            batch_size=fit_batch_size,
            shuffle=fit_shuffle,
            validation_split=fit_validation_split,
            verbose=fit_verbose,
        )
        loss_plot(history.history["loss"], history.history["val_loss"])
        encoded = save_figure()
        model.save(model_file_path)
        outputs.append(
            {
                "type": "web-app",
                "storage": "inline",
                "source": f"""<img width="100%" src="data:image/png;base64,{encoded}"/>""",
            }
        )
    print("============== END TRAINING ==============")
    import json

    from collections import namedtuple

    metadata = {"outputs": outputs}
    loss_plot = namedtuple("loss_plot", ["mlpipeline_ui_metadata"])
    return loss_plot(json.dumps(metadata))


# **5 Testing component**
#
# The following python function will become a component later. This component gets the reference from the dataset, the reference from the model and another pipeline parameters.
# Using all the parameters, tests the model and saves score results temporal shared directory.

# In[ ]:


def test(  # noqa: C901
    dataset_path: components.InputPath(str),
    feature_frames,
    feature_hop_length,
    feature_n_fft,
    feature_n_mels,
    feature_power,
    fit_batch_size,
    fit_compile_loss,
    fit_compile_optimizer,
    fit_epochs,
    fit_shuffle,
    fit_validation_split,
    fit_verbose,
    max_fpr,
    models_dir: components.InputPath(str),
    anomaly_dir: components.OutputPath(str),
    results_dir: components.OutputPath(str),
    mlpipelinemetrics_path: components.OutputPath(str),
):

    import csv
    import glob
    import itertools
    import json
    import logging
    import os
    import re
    import sys
    import typing

    import librosa
    import librosa.core
    import librosa.feature
    import numpy

    from sklearn import metrics
    from sklearn.metrics import confusion_matrix

    # Configure logger
    logger = logging.getLogger(" ")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Parse pipeline parameters
    feature_frames = int(feature_frames)
    feature_hop_length = int(feature_hop_length)
    feature_n_fft = int(feature_n_fft)
    feature_n_mels = int(feature_n_mels)
    feature_power = float(feature_power)
    fit_batch_size = int(fit_batch_size)
    fit_epochs = int(fit_epochs)
    fit_validation_split = float(fit_validation_split)
    fit_verbose = int(fit_verbose)
    max_fpr = float(max_fpr)

    def select_dirs():
        """
        return :
                dirs :  list [ str ]
                    load base directory list of data
        """
        logger.info("load_directory <- data")
        dir_path = os.path.abspath(dataset_path + "{base}/*".format(base="/data"))
        dirs = sorted(glob.glob(dir_path))
        return dirs

    def load_model(file_path):
        """
        return:
            model loaded from file_path
        """
        import tensorflow.keras.models

        return tensorflow.keras.models.load_model(file_path)

    def get_machine_id_list_for_test(target_dir, dir_name="test", ext="wav"):
        """
        target_dir : str
            base directory path of "dev_data" or "eval_data"
        test_dir_name : str (default="test")
            directory containing test data
        ext : str (default="wav)
            file extension of audio files

        return :
            machine_id_list : list [ str ]
                list of machine IDs extracted from the names of test files
        """
        # create test files
        dir_path = os.path.abspath(
            "{dir}/{dir_name}/*.{ext}".format(
                dir=target_dir, dir_name=dir_name, ext=ext
            )
        )
        file_paths = sorted(glob.glob(dir_path))
        machine_id_list = sorted(
            list(
                set(
                    itertools.chain.from_iterable(
                        [re.findall("id_[0-9][0-9]", ext_id) for ext_id in file_paths]
                    )
                )
            )
        )
        return machine_id_list

    def test_file_list_generator(
        target_dir,
        id_name,
        dir_name="test",
        prefix_normal="normal",
        prefix_anomaly="anomaly",
        ext="wav",
    ):
        """
        target_dir : str
            base directory path of the dev_data or eval_data
        id_name : str
            id of wav file in <<test_dir_name>> directory
        dir_name : str (default="test")
            directory containing test data
        prefix_normal : str (default="normal")
            normal directory name
        prefix_anomaly : str (default="anomaly")
            anomaly directory name
        ext : str (default="wav")
            file extension of audio files

        return :
            if the mode is "development":
                test_files : list [ str ]
                    file list for test
                test_labels : list [ boolean ]
                    label info. list for test
                    * normal/anomaly = 0/1
            if the mode is "evaluation":
                test_files : list [ str ]
                    file list for test
        """
        logger.info("target_dir : {}".format(target_dir + "_" + id_name))
        normal_files = sorted(
            glob.glob(
                "{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(
                    dir=target_dir,
                    dir_name=dir_name,
                    prefix_normal=prefix_normal,
                    id_name=id_name,
                    ext=ext,
                )
            )
        )
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob(
                "{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(
                    dir=target_dir,
                    dir_name=dir_name,
                    prefix_anomaly=prefix_anomaly,
                    id_name=id_name,
                    ext=ext,
                )
            )
        )
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")
        return files, labels

    def file_to_vector_array(
        file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0
    ):
        """
        convert file_name to a vector array.

        file_name : str
            target .wav file

        return : numpy.array( numpy.array( float ) )
            vector array
            * dataset.shape = (dataset_size, feature_vector_length)
        """
        dims = n_mels * frames
        y, sr = file_load(file_name)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
        )
        log_mel_spectrogram = (
            20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
        )
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
        if vector_array_size < 1:
            return numpy.empty((0, dims))
        vector_array = numpy.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[
                :, t : t + vector_array_size
            ].T
        return vector_array

    def save_csv(save_file_path, save_data):
        """
        Write csv data to specified path
        """
        with open(save_file_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(save_data)

    def file_load(wav_name, mono=False):
        """
        load .wav file.

        wav_name : str
            target .wav file
        sampling_rate : int
            audio file sampling_rate
        mono : boolean
            When load a multi channels file and this param True, the returned data will be merged for mono data

        return : numpy.array( float )
        """
        try:
            return librosa.load(wav_name, sr=None, mono=mono)
        except Exception:
            logger.exception("file_broken or not exists!! : {}".format(wav_name))

    dirs = select_dirs()
    csv_lines = []
    metrics_list = []
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print(
            "[{idx}/{total}] {dirname}".format(
                dirname=target_dir, idx=idx + 1, total=len(dirs)
            )
        )
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(
            model=models_dir + "/model", machine_type=machine_type
        )

        # load model file
        print("============== MODEL LOAD ==============")
        if not os.path.exists(model_file_path):
            print("{} model not found ".format(machine_type))
            sys.exit(-1)

        model = load_model(model_file_path)
        model.summary()

        # results by type
        csv_lines.append([machine_type])
        csv_lines.append(["id", "AUC", "pAUC"])
        performance = []

        machine_id_list = get_machine_id_list_for_test(target_dir)
        print("Machine_id_list: " + str(machine_id_list))

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = test_file_list_generator(target_dir, id_str)
            anomaly_score_list = []
            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            y_pred = [0.0 for k in test_files]
            for file_idx, file_path in enumerate(test_files):
                try:
                    # Computing error
                    data = file_to_vector_array(
                        file_path,
                        n_mels=feature_n_mels,
                        frames=feature_frames,
                        n_fft=feature_n_fft,
                        hop_length=feature_hop_length,
                        power=feature_power,
                    )
                    y_pred_for_this_file = model.predict(data)

                    # print(y_pred_for_this_file)
                    errors = numpy.mean(
                        numpy.square(data - y_pred_for_this_file), axis=1
                    )
                    y_pred[file_idx] = numpy.mean(errors)
                    anomaly_score_list.append(
                        [os.path.basename(file_path), y_pred[file_idx]]
                    )

                except Exception as e:
                    print(str(e))
                    print("file broken!!: {}".format(file_path))

            print(y_pred)
            # save anomaly score
            if not os.path.exists(anomaly_dir):
                os.makedirs(anomaly_dir)
            anomaly_csv = os.path.join(
                anomaly_dir, "anomaly_score_" + machine_type + "_" + id_str
            )
            save_csv(save_file_path=anomaly_csv, save_data=anomaly_score_list)

            # append AUC and pAUC to lists
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])
            print("AUC : {}".format(auc))
            print("pAUC : {}".format(p_auc))
            metrics_list.append(
                {
                    "name": machine_type + "_" + id_str + "_AUC",
                    "numberValue": auc,
                    "format": "PERCENTAGE",
                }
            )
            metrics_list.append(
                {
                    "name": machine_type + "_" + id_str + "_pAUC",
                    "numberValue": p_auc,
                    "format": "PERCENTAGE",
                }
            )

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        # calculate averages for AUCs and pAUCs
        averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])

    # output results
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_csv = os.path.join(results_dir, "results.csv")
    save_csv(save_file_path=results_csv, save_data=csv_lines)

    with open(mlpipelinemetrics_path, "w") as f:
        json.dump(metrics_list, f)


# **6. Metrics component**
#
# The following python function will become a component later.
# This component gets the reference from the file written by the test component and transforms it into metrics the kubeflow ui can understand.


def ui_metrics(
    mlpipelinemetrics_path: components.InputPath(str),
) -> typing.NamedTuple("Outputs", [("mlpipeline_metrics", "Metrics")]):  # noqa: F821
    import json

    with open(mlpipelinemetrics_path, "r") as f:
        metrics = json.load(f)

    return [json.dumps({"metrics": metrics})]


# **7. The pipeline **


# Define the pipeline
@dsl.pipeline(
    name="anomalous-sound-detection-pipeline",
    description="Pipeline for detecting anomalous sounds",
)
def pipeline(
    dataset_url: str,
    max_fpr: float,
    feature_n_mels: int,
    feature_frames: int,
    feature_n_fft: int,
    feature_hop_length: int,
    feature_power: float,
    fit_compile_optimizer: str,
    fit_compile_loss: str,
    fit_epochs: int,
    fit_batch_size: int,
    fit_shuffle: bool,
    fit_validation_split: float,
    fit_verbose: int,
):
    packages_to_install = [
        "pathlib",
        "pyunpack",
        "patool",
        "keras==2.1.6",
        "Keras-Applications==1.0.8",
        "Keras-Preprocessing==1.1.0",
        "numpy==1.16.0",
        "PyYAML==5.1",
        "scikit-learn==0.20.2",
        "librosa==0.6.0",
        "numba==0.48",
        "audioread==2.1.5",
        "setuptools==41.0.0",
        "matplotlib",
    ]
    download_op = components.func_to_container_op(
        download_dataset, base_image="tensorflow/tensorflow:latest-gpu-py3"
    )
    train_op = components.func_to_container_op(
        train,
        base_image="tensorflow/tensorflow:latest-gpu-py3",
        packages_to_install=packages_to_install,
    )
    test_op = components.func_to_container_op(
        test,
        base_image="tensorflow/tensorflow:latest-gpu-py3",
        packages_to_install=packages_to_install,
    )
    ui_metrics_op = components.func_to_container_op(ui_metrics)

    result = download_op(dataset_url)
    train_process = train_op(
        result.output,
        feature_frames,
        feature_hop_length,
        feature_n_fft,
        feature_n_mels,
        feature_power,
        fit_batch_size,
        fit_compile_loss,
        fit_compile_optimizer,
        fit_epochs,
        fit_shuffle,
        fit_validation_split,
        fit_verbose,
        max_fpr,
    ).after(result)
    testing_process = test_op(
        result.output,
        feature_frames,
        feature_hop_length,
        feature_n_fft,
        feature_n_mels,
        feature_power,
        fit_batch_size,
        fit_compile_loss,
        fit_compile_optimizer,
        fit_epochs,
        fit_shuffle,
        fit_validation_split,
        fit_verbose,
        max_fpr,
        train_process.outputs["models_dir"],
    ).after(train_process)
    ui_metrics_op(testing_process.outputs["mlpipelinemetrics"])


# ARGUMENTS DEFINITION
# arguments = {
#     "dataset_url" : "https://zenodo.org/record/3678171/files/dev_data_fan.zip",
#     "max_fpr" : 0.1,
#     "feature_n_mels" : 128,
#     "feature_frames" : 5,
#     "feature_n_fft" : 1024,
#     "feature_hop_length" : 512,
#     "feature_power" : 2.0,
#     "fit_compile_optimizer" : "adam",
#     "fit_compile_loss" : "mean_squared_error",
#     "fit_epochs" : 2,
#     "fit_batch_size" : 512,
#     "fit_shuffle" : True,
#     "fit_validation_split" : 0.1,
#     "fit_verbose" : 1
# }
