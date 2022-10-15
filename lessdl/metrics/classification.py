from sklearn import metrics


def binary_auc(labels, predictions):
    """
    labels: [0, 1, ...]
    predictions: [0.1, 0.2, ...]
    """
    if len(labels) == 0:
        return 0
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    return metrics.auc(fpr, tpr)


def binary_ctr_metrics(labels, predictions):
    """
    n, pos, ctr, pctr, bias
    """
    assert len(labels) == len(predictions)
    n = len(labels)
    if n == 0:
        return {
            'n': 0, 'ctr': 0, 'pctr': 0, 'bias': 0
        }
    ctr_sum = sum(labels)
    pctr_sum = sum(predictions)
    bias = pctr_sum / (ctr_sum + 0.000001) - 1
    return {
        'n': n,
        'ctr': ctr_sum / n,
        'pctr': pctr_sum / n,
        'bias': bias
    }
