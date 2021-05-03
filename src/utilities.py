import glob
import os
import sys

import librosa
import librosa.core
import librosa.feature
import numpy


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
