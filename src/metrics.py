import kfp.components as comp
import typing


def generate_metrics(mlpipelinemetrics_path: comp.InputPath()) -> typing.NamedTuple('Outputs', [('mlpipeline_metrics', 'Metrics')]):
    import json
    with open(mlpipelinemetrics_path, 'r') as f:
        metrics = json.load(f)

    return [json.dumps({'metrics': metrics})]
