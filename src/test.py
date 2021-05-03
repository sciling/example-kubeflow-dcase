import kfp.components as comp


def test(  # noqa: C901
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
    models_dir: comp.InputPath(),
    anomaly_dir: comp.OutputPath(str),
    results_dir: comp.OutputPath(str),
    mlpipelinemetrics_path: comp.OutputPath(),
    labels_dir: comp.OutputPath(),
):

    import csv
    import glob
    import itertools
    import json
    import os
    import re
    import sys

    import librosa
    import librosa.core
    import librosa.feature
    import numpy
    import tensorflow as tf

    from sklearn import metrics

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

    def load_model(file_path):
        """
        return:
            model loaded from file_path
        """
        return tf.keras.models.load_model(file_path)

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
        print("target_dir : {}".format(target_dir + "_" + id_name))
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
        print("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("Exception: no_wav_file!!")
        print("\n========================================")
        return files, labels

    def save_csv(save_file_path, save_data):
        """
        Write csv data to specified path
        """
        with open(save_file_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(save_data)

    dirs = select_dirs(dataset_path)
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
            y_scores = [0.0 for k in test_files]
            for file_idx, file_path in enumerate(test_files):
                try:
                    data = file_to_vector_array(
                        file_path,
                        n_mels=feature_n_mels,
                        frames=feature_frames,
                        n_fft=feature_n_fft,
                        hop_length=feature_hop_length,
                        power=feature_power,
                    )
                    errors = numpy.mean(
                        numpy.square(data - model.predict(data)), axis=1
                    )
                    y_scores[file_idx] = numpy.mean(errors)
                    anomaly_score_list.append(
                        [os.path.basename(file_path), y_scores[file_idx]]
                    )

                except Exception as e:
                    print(str(e))
                    print("file broken!!: {}".format(file_path))

            # save anomaly score
            if not os.path.exists(anomaly_dir):
                os.makedirs(anomaly_dir)
            anomaly_csv = os.path.join(
                anomaly_dir, "anomaly_score_" + machine_type + "_" + id_str
            )
            save_csv(save_file_path=anomaly_csv, save_data=anomaly_score_list)

            if not os.path.exists(labels_dir):
                os.makedirs(labels_dir)

            # Save true labels and computed scores for metric generation
            with open(f"{labels_dir}/y_labels.txt", "w") as ft:
                ft.write(str(list(y_true)))

            with open(f"{labels_dir}/y_scores.txt", "w") as fp:
                fp.write(str(y_scores))

            # append AUC and pAUC to lists
            auc = metrics.roc_auc_score(y_true, y_scores)
            p_auc = metrics.roc_auc_score(y_true, y_scores, max_fpr=max_fpr)
            csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])
            print("AUC : {}".format(auc))
            print("pAUC : {}".format(p_auc))

            metrics_list.append(
                {
                    "name": machine_type + "_" + id_str + "_AUC",
                    "numberValue": str(auc),
                    "format": "PERCENTAGE",
                }
            )
            metrics_list.append(
                {
                    "name": machine_type + "_" + id_str + "_pAUC",
                    "numberValue": str(p_auc),
                    "format": "PERCENTAGE",
                }
            )

            # append precision score
            precision = metrics.average_precision_score(y_true, y_scores)
            metrics_list.append(
                {
                    "name": machine_type + "_" + id_str + "_precision",
                    "numberValue": str(precision),
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
