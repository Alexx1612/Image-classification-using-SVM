import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

def plot_class_distribution(y_train_path, report_dir):
    """
    Generate and save class distribution visualization.
    """
    y_train = pd.read_csv(y_train_path)["Label"]
    plt.figure(figsize=(8, 6))
    y_train.value_counts().sort_index().plot(kind="bar", color="skyblue")
    plt.title("Class distribution in y_train")
    plt.xlabel("Class labels (0-9)")
    plt.ylabel("Number of samples")
    plt.savefig(os.path.join(report_dir, "class_distribution.png"))
    plt.close()
    print("Class distribution plot saved.")

def plot_confusion_matrix(y_true, y_pred, report_dir):
    """
    Generate and save confusion matrix visualization.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close()
    print("Confusion matrix saved.")
