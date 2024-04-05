from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, average_precision_score


def accuracy(labels, pred, average):
    return accuracy_score(labels, pred)


def precision(labels, pred, average):
    return precision_score(labels, pred, average=average)


def recall(labels, pred, average):
    return recall_score(labels, pred, average=average)


def f1(labels, pred, average):
    return f1_score(labels, pred, average=average)


def roc_score(labels, scores):
    return roc_auc_score(labels, scores)


def calc_roc_curve(labels, scores):
    return roc_curve(labels, scores)


def average_precision(labels, score, average):
    return average_precision_score(labels, score, average=average)


metrics_def = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

roc = {
    'roc': roc_score
}
