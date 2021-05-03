import kfp.components as comp
import typing


def roc_curve(labels_dir: comp.InputPath()) -> typing.NamedTuple('roc_curve', [('mlpipeline_ui_metadata', 'UI_metadata')]):
    import json
    from collections import namedtuple
    from sklearn import metrics

    # Load test labels and predicted labels
    with open(f'{labels_dir}/y_labels.txt', 'r') as fl:
        y_true = eval(fl.read())

    with open(f'{labels_dir}/y_scores.txt', 'r') as fs:
        y_scores = eval(fs.read())

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)

    csv_literal_roc_curve = ""
    for i in range(len(fpr)):
        csv_literal_roc_curve += "{fpr},{tpr},{thresholds}\n".format(fpr=fpr[i], tpr=tpr[i], thresholds=thresholds[i])

    kf_literal_roc_curve = {
        'outputs' : [{
            'type': 'roc',
            'format': 'csv',
            'schema': [
                {'name': 'fpr', 'type': 'NUMBER'},
                {'name': 'tpr', 'type': 'NUMBER'},
                {'name': 'thresholds', 'type': 'NUMBER'},
            ],
            'storage': 'inline',
            'source': csv_literal_roc_curve,
        }]
    }

    roc_curve_result = namedtuple('roc_curve', ['mlpipeline_ui_metadata'])
    return roc_curve_result(json.dumps(kf_literal_roc_curve))
