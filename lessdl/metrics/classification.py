import torch
from sklearn import metrics


def binary_auc(labels, predictions):
    """
    labels: [0, 1, ...], numpy array
    predictions: [0.1, 0.2, ...], numpy array
    """
    if len(labels) == 0:
        return 0
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    return metrics.auc(fpr, tpr)


def binary_ctr_metrics(labels, predictions):
    """
    n, pos, ctr, pctr, bias
    labels, predictions: numpy array 
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    return accuracy percentage, such as [90, 95] when topk = (1,5)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print('output.device', output.device, 'pred.device', pred.device, 'target.device', target.device)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

