import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import wandb


def compute_metrics(prediction, target, n_classes, ignored_labels):
    """Compute and print metrics (OA, PA, AA, Kappa)

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes: number of classes, max(target) by default
        ignored_labels: list of labels to ignore, e.g. for undef

    Returns:
        {Confusion Matrix, OA, PA, AA, Kappa}
    """
    mask = np.ones(target.shape[:2], dtype=np.bool)
    for k in ignored_labels:
        mask[target == k] = False
    target = target[mask]
    pred = prediction[mask]

    results = {}

    # compute Overall Accuracy
    cm = confusion_matrix(target, pred, labels=range(n_classes + 1))
    results['Confusion matrix'] = cm

    # compute Overall Accuracy (OA)
    oa = 1. * np.trace(cm) / np.sum(cm)
    results['OA'] = oa

    # compute Producer Accuracy (PA)
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) for i in range(n_classes + 1)])
    results['PA'] = pa

    # compute Average Accuracy (AA)
    aa = np.mean(pa)
    results['AA'] = aa

    # compute kappa coefficient
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm))
    kappa = (oa - pe) / (1 - pe)
    results['Kappa'] = kappa

    return results


def metrics2text(results, labels_text, replica=0):
    cm = results["Confusion matrix"]
    oa = results['OA']
    pa = results['PA']
    aa = results['AA']
    kappa = results['Kappa']

    text = ""
    text += "**Overall Accuracy**: {:.04f}  \n".format(oa)
    text += "  \n"
    text += "**Producer's Accuracy**:  \n"
    for label, acc in zip(labels_text, pa):
        text += "{}: {:.04f}  \n".format(label.center(13), acc)
    text += "  \n"
    text += "**Average Accuracy**: {:.04f}  \n".format(aa)
    text += "  \n"
    text += "**Kappa**: {:.04f}  \n".format(kappa)
    text += "  \n"

    # Console output
    text = ""
    text += "Confusion matrix:\n"
    text += str(cm)
    text += "\n---\n"

    text += "Overall Accuracy: {:.04f}\n".format(oa)
    text += "---\n"

    text += "Producer's Accuracy:\n"
    for label, acc in zip(labels_text, pa):
        text += "\t{}: {:.04f}\n".format(label, acc)
    text += "---\n"

    text += "Average Accuracy: {:.04f}\n".format(aa)
    text += "---\n"

    text += "Kappa: {:.04f}\n".format(kappa)
    text += "---\n"

    print(text)
    # wandb.log({"OA": oa, "AA": aa, "Kappa":kappa})


def cm_viz(cm, labels_text, replica):
    cm = pd.DataFrame(data=cm / np.sum(cm, axis=1, keepdims=True), index=labels_text, columns=labels_text)
    plt.figure(figsize=(12, 7))
    Img = wandb.Image(sns.heatmap(data=cm, annot=True).get_figure(), caption=f"Confusion Matrix {replica}")
    wandb.log({"Confusion Matrix": Img})
    

def show_statistics(statistics, labels_text):
    OAs = np.array([statistics[rep]['OA'] for rep in range(len(statistics))])
    PAs = np.array([statistics[rep]['PA'] for rep in range(len(statistics))])
    AAs = np.array([statistics[rep]['AA'] for rep in range(len(statistics))])
    Kappas = np.array([statistics[rep]['Kappa'] for rep in range(len(statistics))])

    # Console output
    text = "===== Summary =====\n"

    text += "Producer's Accuracy:\n"
    for label, acc in zip(labels_text, PAs.T):
        text += "\t{}: {:.04f} ± {:.04f}\n".format(label, acc.mean(), acc.std())
    text += "---\n"

    text += f"Overall Accuracy: {OAs.mean():.04f} ± {OAs.std():.04f}\n"
    text += "---\n"

    text += f"Average Accuracy: {AAs.mean():.04f} ± {AAs.std():.04f}\n"
    text += "---\n"

    text += f"Kappa: {Kappas.mean():.04f} ± {Kappas.std():.04f}\n"
    text += "---\n"

    print(text)

    wandb.log({"OA": OAs.mean(), "AA": AAs.mean(), "Kappa":Kappas.mean()})
