# SAME Example: Anomolous Sound Detection

> **This is a work in progress!**

Anomalous sound detection (ASD) is the task of identifying whether the sound emitted from a machine is normal or anomalous. Automatic detection of mechanical failure is essential technology in the fourth industrial revolution, including artificial intelligence (AI)-based factory automation. Prompt detection of machine anomalies by observing sounds is useful for machine condition monitoring.

In this example, we try to adapt the code provided by the project [github](https://github.com/y-kawagu/dcase2020_task2_baseline) whose main goals are the ones stated in the [dcase community](http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds): This task is the follow-up to DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring".

## Usage

Create a working SAME installation by [following instructions found in the wiki](https://github.com/azure-octo/same-cli/wiki/Epic-Sprint-1-Demo), but stop before the "Run a program" section. Then run the following commands:

```bash
git clone https://github.com/SAME-Project/example-kubeflow-dcase
cd example-kubeflow-dcase
same program create -f same.yaml
same program run -f same.yaml --experiment-name dcase --run-name default
```

Now browse to your kubeflow installation and you should be able to see an experiment and a run.

## Testing

This repo is not a library, nor is it meant to run with different permutations of Python or library versions. It is not guaranteed to work with different Python or library versions, but it might. There is limited matrix testing in the github action CI/CD.
