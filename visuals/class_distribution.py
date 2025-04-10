# File: visuals/class_distribution.py
# -----------------------------------
# Bar charts for class label counts (true vs predicted)

import matplotlib.pyplot as plt
import pandas as pd

def plot_class_distribution(y_true: pd.Series, y_pred: pd.Series):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    y_true.value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title("True Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    y_pred.value_counts().sort_index().plot(kind='bar', color='salmon')
    plt.title("Predicted Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
