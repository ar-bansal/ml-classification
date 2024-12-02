import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def print_confusion_matrix(y_true, y_pred, normalize=None, colorbar=True, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize).T
    labels = np.unique(y_true)

    fig, ax = plt.subplots(figsize=(2 * len(labels), 2 * len(labels)))

    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=colorbar, cmap=cmap)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("True Label")
    ax.xaxis.set_label_position("top") 
    ax.xaxis.tick_top()

    ax.set_ylabel("Predict Label")

    return cm


def effect_encode(variable, baseline, prefix):
    categories = variable.unique()
    df = pd.DataFrame()
    
    for category in categories:
        if category != baseline:
            df[f"{prefix}[{category}]"] = (variable == category).astype(int) - (variable == baseline).astype(int)

    return df
         



if __name__ == "__main__":
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 0, 0, 0]
    print_confusion_matrix(y_true=y_true, y_pred=y_pred)