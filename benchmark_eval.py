# ==========================
# benchmark_eval.py
# ==========================
# Evaluation and benchmarking tools for model predictions.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ----------------------------------------------
# Benchmark Accuracy Plot
# ----------------------------------------------
def benchmark_accuracy_chart(y_test, preds):
    y_true = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(preds).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Label', alpha=0.7, linewidth=1.2)
    plt.plot(y_pred, label='Predicted Label', alpha=0.7, linestyle='--', linewidth=1.2)
    plt.title('Prediction Accuracy Benchmark')
    plt.xlabel('Samples')
    plt.ylabel('Class Label')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------
# Confusion Matrix Visualization
# ----------------------------------------------
def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix of Predictions')
    plt.grid(False)
    plt.show()

# ----------------------------------------------
# Text-Based Summary Report
# ----------------------------------------------
def print_classification_summary(report):
    print("\n--- Classification Report ---\n")
    print(report)
