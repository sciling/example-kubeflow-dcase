import kfp
import kfp.dsl as dsl
import kfp.components as comp


# Define the pipeline
@dsl.pipeline(
    name="anomalous-sound-detection-pipeline",
    description="Pipeline for detecting anomalous sounds",
)
def pipeline(
    dataset_url: str = 'https://zenodo.org/record/3678171/files/dev_data_pump.zip',
    max_fpr: float = 0.1,
    feature_n_mels: int = 128,
    feature_frames: int = 5,
    feature_n_fft: int = 1024,
    feature_hop_length: int = 512,
    feature_power: float = 2.0,
    fit_compile_optimizer: str = "adam",
    fit_compile_loss: str = "mean_squared_error",
    fit_epochs: int = 50,
    fit_batch_size: int = 512,
    fit_shuffle: bool = True,
    fit_validation_split: float = 0.15,
    fit_verbose: int = 1,
):
    from train import train
    from test import test
    from download import download_dataset
    from metrics import generate_metrics
    from roc_curve import roc_curve

    # Create train and predict lightweight components.
    packages_to_install = ['pathlib', 'pyunpack', 'patool', 'keras==2.1.6', 'Keras-Applications==1.0.8', 'Keras-Preprocessing==1.1.0', 'numpy==1.16.0', 'PyYAML==5.1', 'scikit-learn==0.20.2', 'librosa==0.6.0', 'numba==0.48', 'audioread==2.1.5', 'setuptools==41.0.0', 'matplotlib']
    download_op = comp.func_to_container_op(download_dataset, base_image='tensorflow/tensorflow:latest-gpu-py3')
    train_op = comp.func_to_container_op(train, base_image='tensorflow/tensorflow:latest-gpu-py3', packages_to_install=packages_to_install)
    test_op = comp.func_to_container_op(test, base_image='tensorflow/tensorflow:latest-gpu-py3', packages_to_install=packages_to_install)
    generate_metrics_op = comp.func_to_container_op(generate_metrics)
    roc_curve_op = comp.func_to_container_op(roc_curve, packages_to_install=['scikit-learn'])

    result = download_op(dataset_url)
    result.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_process = train_op(result.output, feature_frames, feature_hop_length, feature_n_fft, feature_n_mels, feature_power, fit_batch_size, fit_compile_loss, fit_compile_optimizer, fit_epochs, fit_shuffle, fit_validation_split, fit_verbose, max_fpr)
    testing_process = test_op(result.output, feature_frames, feature_hop_length, feature_n_fft, feature_n_mels, feature_power, fit_batch_size, fit_compile_loss, fit_compile_optimizer, fit_epochs, fit_shuffle, fit_validation_split, fit_verbose, max_fpr, train_process.outputs['models_dir']).after(train_process)
    generate_metrics_op(testing_process.outputs['mlpipelinemetrics'])
    roc_curve_op(testing_process.outputs['labels_dir'])


if __name__ == '__main__':

    # Compile pipeline to generate compressed YAML definition of the pipeline.
    kfp.compiler.Compiler().compile(pipeline, '{}.zip'.format('dcase-pipeline'))
