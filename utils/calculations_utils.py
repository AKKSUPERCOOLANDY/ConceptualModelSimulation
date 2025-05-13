from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np

class CalculationUtils:
    @staticmethod
    def calculate_classification_metrics(true_labels, predictions):
        cm = confusion_matrix(true_labels, predictions)
        # Calculate precision, recall, f1 scores
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=['B', 'M']
        )
        
        # Calculate macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro'
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Standard accuracy using scikit-learn's accuracy_score
        # This is the primary accuracy metric used for evaluation
        standard_accuracy = accuracy_score(true_labels, predictions)
        
        # For binary classification with 'M' as positive class and 'B' as negative
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Convert all NumPy values to native Python types to ensure JSON serialization works
        return {
            # Using scikit-learn's accuracy_score as the main accuracy metric
            'accuracy': float(standard_accuracy),
            'precision_by_class': [float(p) for p in precision.tolist()],
            'recall_by_class': [float(r) for r in recall.tolist()],
            'f1_by_class': [float(f) for f in f1.tolist()],
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
            'false_negatives': int(fn) if fn is not None else 0,
            'false_positives': int(fp) if fp is not None else 0
        }