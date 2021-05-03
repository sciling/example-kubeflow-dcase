# Anomalous sound detection (ASD) with Kubeflow Pipeline #
Anomalous sound detection (ASD) is the task of identifying whether the sound emitted from a machine is normal or anomalous. Automatic detection of mechanical failure is essential technology in the fourth industrial revolution, including artificial intelligence (AI)-based factory automation. Prompt detection of machine anomalies by observing sounds is useful for machine condition monitoring.

In this repository, we try to adapt the code provided by the github project [dcase2020_task2_baseline](https://github.com/y-kawagu/dcase2020_task2_baseline) under the [MIT License](https://github.com/y-kawagu/dcase2020_task2_baseline/blob/master/LICENSE) whose main goals are "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring". Our aim will be to convert this code into a kubeflow pipeline.

# Pipeline parameters #
| Pipeline parameter | Description |
| ------ | ------ |
|dataset_url | url of the dataset (e.g https://zenodo.org/record/3678171/files/dev_data_fan.zip)|
|max_fpr| Float between 0 and 1. The standardized partial AUC over the range [0, max_fpr] is returned|
|feature_n_mels| Integer representing the number of mels (e.g 128)|
|feature_frames| Integer representing the number of frames of the FFT window(e.g 5)|
|feature_n_fft| Integer representing the length of the FFT window (e.g 1024)|
|feature_hop_length| Integer representing the number of samples between successive frames (e.g 512)|
|feature_power| Float representing the exponent for the magnitude melspectrogram. (e.g., 1.0 for energy, 2.0 for power, etc )|
|fit_compile_optimizer| String (name of optimizer) or optimizer instance. See tf.keras.optimizers. (e.g "adam")|
|fit_compile_loss| String (name of objective function) or tf.keras.losses.Loss instance. See tf.keras.losses (e.g "mean_squared_error")|
|fit_epochs| Integer. Number of epochs to train the model. (e.g 50)|
|fit_batch_size| Integer or None. Number of samples per gradient update(e.g 512)|
|fit_shuffle| Boolean (whether to shuffle the training data before each epoch)|
|fit_validation_split| Float between 0 and 1. Fraction of the training data to be used as validation data(e.g 0.1)|
|fit_verbose| 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch|

The url of the dataset should lead to a dataset that has the following structure:

    /machine( Toy_Car / fan / slider ...)
        /train (Only normal data for all Machine IDs are included.)
            /normal_id_01_00000000.wav
            ...
            /normal_id_01_00000999.wav
            /normal_id_02_00000000.wav
            ...
            /normal_id_04_00000999.wav
        /test (Normal and anomaly data for all Machine IDs are included.)
            /normal_id_01_00000000.wav
            ...
            /normal_id_01_00000349.wav
            /anomaly_id_01_00000000.wav
            ...
            /anomaly_id_01_00000263.wav
            /normal_id_02_00000000.wav
            ...
            /anomaly_id_04_00000264.wav

Some options for the dataset url are:

* Fan dataset: https://zenodo.org/record/3678171/files/dev_data_fan.zip
* Toy-car dataset:https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip
* Slider rail dataset: https://zenodo.org/record/3678171/files/dev_data_slider.zip
* Toy-conveyor dataset: https://zenodo.org/record/3678171/files/dev_data_ToyConveyor.zip
* Valve dataset: https://zenodo.org/record/3678171/files/dev_data_valve.zip
* Pump dataset: https://zenodo.org/record/3678171/files/dev_data_pump.zip

# Pipeline stages #

![pipeline.png](./data/images/pipeline.png)

##### 1. Download dataset ([code](./src/download.py))
This component, given the dataset url, downloads all its contents inside an OutputPath Artifact.

##### 2. Train ([code](./src/train.py))
This component performs the following operations:

    1. Given an InputPath containing the previously downloaded dataset, extracts all the training files (audio), converting them into numeric arrays.
    2. Uses those arrays, trains a model with the specified parameters.
    3. Save the model in an OutputPath Artifact.
    4. Generate a loss plot, saves it in an OutputArtifact and embed its visualization inside a web-app component.

##### 3. Test ([code](./src/test.py))
This component performs the following operations:

    1. Loads the previously saved model through an InputPath Artifact.
    2. Given an InputPath containing the previously downloaded dataset, extracts all the testing files (audio), converting them into numeric arrays.
    3. Uses those arrays to test the model.
    4. Saves the  inside a file generated as an OutputPath Artifact(results_path).
    5. Saves true labels and predicted scores to pass it later to the ROC curve.
    6. Saves the name, AUC and pAUC for each subgroup of the test into a results OutputPath Artifact.
    7. Saves the scores for the anomalies files of the test into a anomaly_dir OutputPath Artifact.
    6. Saves accuracy as metrics that will later be passed to the Metrics component.

##### 4.1. ROC Curve ([code](./src/roc_curve.py))
This component is passed the labels directory, which contains true labels and predicted scores, and generates a roc curve that the kubeflow UI can understand. This function can be reused in other pipelines if given the appropiate parameters.

##### 4.2. Metrics ([code](./src/metrics.py))
This component is passed the mlpipelinemetrics which contains metrics and generates a visualization of them that the kubeflow UI can understand.


# File generation #
To generate the pipeline from the python file, execute the following command:

```python3 pipeline.py```

pipeline.py is located inside src folder. The pipeline will be created at the same directory that the command is executed.

Also, if you want to run all tests locally, execute:
```python3 -m unittest tests/*_test.py```

Once the pipeline has been created, we can upload the generated zip file in kubeflow UI and create runs of it.

# Experimental results #

In this section we will replicate the results for the pump dataset in the [DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring"](https://github.com/y-kawagu/dcase2020_task2_baseline/README.md).
The pipeline outputs are a loss plot, a roc curve, and different metrics, from which metrics can be directly compared.
We can see them in the visualizations of the pipeline or in the Run Output Tab of the Run.

In order to check the validity of the pipeline, we are going to execute a run with the same parameters as the original experiment and compare the outputs with the ones obtained in [the original code](https://github.com/y-kawagu/dcase2020_task2_baseline).

### Input parameters ###
| Pipeline parameter | Value |
| ------ | ------ |
|dataset_url |
https://zenodo.org/record/3678171/files/dev_data_pump.zip|
|max_fpr|0.1|
|feature_n_mels|128|
|feature_frames|5|
|feature_n_fft|1024|
|feature_hop_length|512|
|feature_power|2.0|
|fit_compile_optimizer|adam|
|fit_compile_loss|mean_squared_error|
|fit_epochs|100|
|fit_batch_size|512|
|fit_shuffle|True|
|fit_validation_split|0.1|
|fit_verbose|1|

### Loss plot ###

![lossplot.png](./data/images/lossplot.png)

### ROC Curve ###

![roccurve.png](./data/images/roccurve.png)

### Metrics ###
The original results are shown in https://github.com/y-kawagu/dcase2020_task2_baseline#7-check-results. In particular, the results for the pump task are:

| id | AUC | pAUC
| ------ | ------ | ------ |
|0	|	0.670769 |	0.57269 |
|2	|	0.609369 |	0.58037 |
|4	|	0.8886	 |	0.676842 |
|6	|	0.734902 |	0.570175 |

In our replication, we get similar results (our results are in percentage format):

| id | AUC | pAUC
| ------ | ------ | ------ |
|0	|	67.126%	|	56.570% |
|2	|	60.991%	 |	58.037%	|
|4	|	88.490%	 |  66.947% |
|6	|	73.324%	 |  56.966%	|

If we increase the number of epochs to 150, and the validation split to 0.15, the results improve a little:

| id | AUC | pAUC
| ------ | ------ | ------ |
|0	|	68.336% |	57.085%	|
|2	|	63.468% |	58.321% |
|4	|	87.260% |   64.158%	|
|6	|	72.716% |   58.153% |

