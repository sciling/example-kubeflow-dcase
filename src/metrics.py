import typing

import kfp.components as comp


def generate_metrics(
    mlpipelinemetrics_path: comp.InputPath(),
) -> typing.NamedTuple("Outputs", [("mlpipeline_metrics", "Metrics")]):  # noqa: F821
    import json

    with open(mlpipelinemetrics_path, "r") as f:
        metrics = json.load(f)

    return [json.dumps({"metrics": metrics})]
