"""
Classification Metrics Implementation

This module provides implementations of classification metrics including
precision, recall, F1 score, confusion matrix, and ROC curves.

Author: Cavin Otieno Ouma
Registration: SDS6/46982/2024
Course: SDS 6217 Advanced Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


class ClassificationMetrics:
    """
    Collection of classification performance metrics.
    
    This class provides implementations for:
    - True Positives, False Positives, True Negatives, False Negatives
    - Precision, Recall, F1 Score
    - Accuracy, Specificity
    - False Positive Rate, False Negative Rate
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize with true and predicted labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = y_pred_proba
        
        # Calculate confusion matrix values
        self.tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
    
    @property
    def confusion_matrix(self):
        """Return confusion matrix as 2x2 array."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])
    
    @property
    def accuracy(self):
        """Overall accuracy: (TP + TN) / (TP + TN + FP + FN)"""
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0
    
    @property
    def precision(self):
        """Precision: TP / (TP + FP) - What proportion of positive predictions is correct"""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0
    
    @property
    def recall(self):
        """Recall (Sensitivity): TP / (TP + FN) - What proportion of actual positives is identified"""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0
    
    @property
    def specificity(self):
        """Specificity: TN / (TN + FP) - What proportion of actual negatives is identified"""
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0 else 0
    
    @property
    def f1_score(self):
        """F1 Score: Harmonic mean of precision and recall"""
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator > 0 else 0
    
    @property
    def false_positive_rate(self):
        """FPR: FP / (FP + TN) - Proportion of negatives incorrectly classified as positive"""
        denominator = self.fp + self.tn
        return self.fp / denominator if denominator > 0 else 0
    
    @property
    def false_negative_rate(self):
        """FNR: FN / (FN + TP) - Proportion of positives incorrectly classified as negative"""
        denominator = self.fn + self.tp
        return self.fn / denominator if denominator > 0 else 0
    
    def print_metrics(self):
        """Print all metrics in a formatted way."""
        print("\n" + "=" * 50)
        print("Classification Metrics Report")
        print("=" * 50)
        
        print("\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"              Neg     Pos")
        print(f"Actual Neg   {self.tn:4d}   {self.fp:4d}")
        print(f"Actual Pos   {self.fn:4d}   {self.tp:4d}")
        
        print("\n" + "-" * 50)
        print("Detailed Metrics:")
        print("-" * 50)
        print(f"True Positives (TP):     {self.tp:4d}")
        print(f"True Negatives (TN):     {self.tn:4d}")
        print(f"False Positives (FP):    {self.fp:4d}")
        print(f"False Negatives (FN):    {self.fn:4d}")
        
        print("\n" + "-" * 50)
        print("Performance Metrics:")
        print("-" * 50)
        print(f"Accuracy:          {self.accuracy:.4f}")
        print(f"Precision:         {self.precision:.4f}")
        print(f"Recall/Sensitivity: {self.recall:.4f}")
        print(f"Specificity:       {self.specificity:.4f}")
        print(f"F1 Score:          {self.f1_score:.4f}")
        print(f"False Pos Rate:    {self.false_positive_rate:.4f}")
        print(f"False Neg Rate:    {self.false_negative_rate:.4f}")
        print("=" * 50)


def calculate_recall_example():
    """
    Demonstrate recall calculation from the quiz question.
    
    Question: Model outputs 5 TP, 6 TN, 3 FP, and 2 FN. Calculate the recall.
    Answer: 0.714
    """
    print("\n" + "=" * 60)
    print("Example: Recall Calculation")
    print("=" * 60)
    
    tp = 5
    fn = 2
    
    recall = tp / (tp + fn)
    
    print(f"\nGiven:")
    print(f"  True Positives (TP): {tp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"\nRecall = TP / (TP + FN)")
    print(f"Recall = {tp} / ({tp} + {fn})")
    print(f"Recall = {tp} / {tp + fn}")
    print(f"Recall = {recall:.3f}")
    print(f"\nAnswer: {recall} ✓")
    

def calculate_precision_example():
    """
    Demonstrate precision calculation from the quiz question.
    
    Question: Model outputs 3 TP, 4 TN, 2 FP, and 1 FN. Calculate the precision.
    Answer: 0.6
    """
    print("\n" + "=" * 60)
    print("Example: Precision Calculation")
    print("=" * 60)
    
    tp = 3
    fp = 2
    
    precision = tp / (tp + fp)
    
    print(f"\nGiven:")
    print(f"  True Positives (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"\nPrecision = TP / (TP + FP)")
    print(f"Precision = {tp} / ({tp} + {fp})")
    print(f"Precision = {tp} / {tp + fp}")
    print(f"Precision = {precision:.3f}")
    print(f"\nAnswer: {precision} ✓")


def demonstrate_threshold_effect():
    """
    Demonstrate how changing the classification threshold affects
    precision, recall, true positives, and false positives.
    
    This visualizes the trade-offs discussed in the quiz.
    """
    print("\n" + "=" * 60)
    print("Demonstrating Threshold Effects on Classification")
    print("=" * 60)
    
    # Generate sample classification data
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=3,
                               weights=[0.6, 0.4], random_state=42)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("\nThreshold Analysis:")
    print("-" * 70)
    print(f"{'Threshold':<12} {'TP':>8} {'FP':>8} {'TN':>8} {'FN':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 70)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = ClassificationMetrics(y_test, y_pred)
        
        results.append({
            'threshold': threshold,
            'tp': metrics.tp,
            'fp': metrics.fp,
            'tn': metrics.tn,
            'fn': metrics.fn,
            'precision': metrics.precision,
            'recall': metrics.recall
        })
        
        print(f"{threshold:<12.1f} {metrics.tp:>8} {metrics.fp:>8} {metrics.tn:>8} "
              f"{metrics.fn:>8} {metrics.precision:>10.3f} {metrics.recall:>8.3f}")
    
    print("-" * 70)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: TP and FP vs Threshold
    ax1 = axes[0, 0]
    ax1.plot([r['threshold'] for r in results], [r['tp'] for r in results], 
             'b-o', linewidth=2, label='True Positives')
    ax1.plot([r['threshold'] for r in results], [r['fp'] for r in results], 
             'r-s', linewidth=2, label='False Positives')
    ax1.set_xlabel('Classification Threshold', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('True Positives and False Positives vs Threshold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: TN and FN vs Threshold
    ax2 = axes[0, 1]
    ax2.plot([r['threshold'] for r in results], [r['tn'] for r in results], 
             'g-o', linewidth=2, label='True Negatives')
    ax2.plot([r['threshold'] for r in results], [r['fn'] for r in results], 
             'm-s', linewidth=2, label='False Negatives')
    ax2.set_xlabel('Classification Threshold', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('True Negatives and False Negatives vs Threshold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision and Recall vs Threshold
    ax3 = axes[1, 0]
    ax3.plot([r['threshold'] for r in results], [r['precision'] for r in results], 
             'b-o', linewidth=2, label='Precision')
    ax3.plot([r['threshold'] for r in results], [r['recall'] for r in results], 
             'r-s', linewidth=2, label='Recall')
    ax3.set_xlabel('Classification Threshold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Precision and Recall vs Threshold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ROC Curve
    ax4 = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax4.set_title('ROC Curve', fontsize=12)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/advanced-machine-learning-sds6217/code_examples/threshold_effects.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to threshold_effects.png")
    
    # Summary of quiz answers
    print("\n" + "=" * 60)
    print("Quiz Question Answers:")
    print("=" * 60)
    print("1. Legitimate website classified as malware = FALSE POSITIVE")
    print("2. Threshold increases → TP decreases, FP decreases")
    print("3. Threshold increases → TN increases, FN increases")
    print(f"4. Recall with TP=5, FN=2 = {5/(5+2):.3f} ✓")
    print(f"5. Precision with TP=3, FP=2 = {3/(3+2):.3f} ✓")
    print("6. Insect detection: Optimize RECALL (false alarms are acceptable)")


def demonstrate_insect_trap_scenario():
    """
    Demonstrate the recall optimization scenario for insect detection.
    
    Scenario: Invasive species detection where false alarms are easy to handle.
    Priority: Maximize recall to catch all invasive species.
    """
    print("\n" + "=" * 60)
    print("Insect Trap Classification Scenario")
    print("=" * 60)
    
    # Simulate the scenario
    np.random.seed(42)
    
    # Generate data: mostly negative (harmless), few positive (invasive)
    n_samples = 1000
    n_invasive = 50  # 5% invasive species
    
    # Simulate model predictions
    # For invasive species: lower confidence (harder to detect)
    invasive_confidence = np.random.beta(2, 5, n_invasive)  # Biased toward low
    # For harmless: higher confidence
    harmless_confidence = np.random.beta(5, 2, n_samples - n_invasive)  # Biased toward high
    
    y_true = np.array([1] * n_invasive + [0] * (n_samples - n_invasive))
    y_proba = np.concatenate([invasive_confidence, harmless_confidence])
    
    print(f"\nDataset: {n_samples} samples")
    print(f"  - Invasive species (positive): {n_invasive}")
    print(f"  - Harmless (negative): {n_samples - n_invasive}")
    print(f"  - Imbalance ratio: {(n_samples - n_invasive) / n_invasive}:1")
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n" + "-" * 80)
    print(f"{'Threshold':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>8} {'Comment':>20}")
    print("-" * 80)
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = ClassificationMetrics(y_true, y_pred)
        
        if threshold <= 0.4:
            comment = "Higher recall, more alerts"
        elif threshold >= 0.6:
            comment = "Lower recall, fewer alerts"
        else:
            comment = "Balanced"
        
        print(f"{threshold:<12.1f} {metrics.tp:>6} {metrics.fp:>6} {metrics.fn:>6} "
              f"{metrics.precision:>10.3f} {metrics.recall:>8.3f} {comment:>20}")
    
    print("-" * 80)
    
    print("\nRecommendation:")
    print("  Since false alarms (FP) are easy to handle and missing invasive")
    print("  species (FN) has severe consequences, optimize for RECALL.")
    print("  Lower threshold (e.g., 0.3-0.4) catches more invasive species,")
    print("  even at the cost of some false alarms.")


if __name__ == "__main__":
    print("=" * 60)
    print("Classification Metrics Demonstration")
    print("Course: SDS 6217 Advanced Machine Learning")
    print("Student: Cavin Otieno Ouma")
    print("=" * 60)
    
    # Quiz calculations
    calculate_recall_example()
    calculate_precision_example()
    
    # Threshold effects demonstration
    demonstrate_threshold_effect()
    
    # Real-world scenario
    demonstrate_insect_trap_scenario()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
