import typing

import kfp.components as comp


def train(
    dataset_path: comp.InputPath(str),
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
    lossplot_path: comp.OutputPath(str),
    models_dir: comp.OutputPath(),
) -> typing.NamedTuple(
    "loss_plot", [("mlpipeline_ui_metadata", "UI_metadata")]  # noqa: F821
):
    import base64
    import glob
    import json
    import os
    import sys

    from collections import namedtuple

    import librosa
    import librosa.core
    import librosa.feature
    import matplotlib.pyplot as plt
    import numpy

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

    def select_dirs(dataset_path):
        """
        return :
                dirs :  list [ str ]
                    load base directory list of data
        """
        print("load_directory <- data")
        dir_path = os.path.abspath(dataset_path + "{base}/*".format(base="/data"))
        dirs = sorted(glob.glob(dir_path))
        return dirs

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
            print("Error: file_broken or not exists!! : {}".format(wav_name))

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

        print("target_dir : {}".format(target_dir))
        training_list_path = os.path.abspath(
            "{dir}/{dir_name}/*.{ext}".format(
                dir=target_dir, dir_name=dir_name, ext=ext
            )
        )
        files = sorted(glob.glob(training_list_path))
        if len(files) == 0:
            print("Exception: no_wav_file!!")

        print("train_file num : {num}".format(num=len(files)))
        return files

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

    def save_loss_plot(history, plot_path):
        """
        history: History object from keras. Its History.history attribute is a record of training loss and validation loss values.
        plot_path: path where plot image will be saved.
        """
        # Creation of the plot
        loss, val_loss = history.history["loss"], history.history["val_loss"]
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

        # Saving plot in specified path
        with open(plot_path, "wb") as fd:
            plt.savefig(fd)

    def get_web_app_from_loss_plot(plot_path):
        """
        plot_path: path where plot image is saved.
        return: JSON object representing kubeflow output viewer for web-app.
        """
        # Retrieve encoded bytes of the specified image path
        encoded = base64.b64encode(open(plot_path, "rb").read()).decode("latin1")

        web_app_json = {
            "type": "web-app",
            "storage": "inline",
            "source": f"""<img width="100%" src="data:image/png;base64,{encoded}"/>""",
        }
        return web_app_json

    dirs = select_dirs(dataset_path)

    # loop of the base directory
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
        print(train_data)
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

        model.save(model_file_path)

    save_loss_plot(history, lossplot_path)
    loss_plot = [get_web_app_from_loss_plot(lossplot_path)]

    print("============== END TRAINING ==============")

    metadata = {"outputs": loss_plot}
    loss_plot = namedtuple("loss_plot", ["mlpipeline_ui_metadata"])
    return loss_plot(json.dumps(metadata))
