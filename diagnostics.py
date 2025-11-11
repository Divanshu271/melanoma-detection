"""
Diagnostics and Performance Reporting for Heavy Ensemble
Tracks per-threshold metrics and model contributions for analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from pathlib import Path
import json


class EnsembleDiagnostics:
    """
    Track and report detailed per-model and per-threshold metrics.
    """
    
    def __init__(self, output_dir='results/diagnostics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.threshold_metrics = []
        self.model_contributions = {}
        self.calibration_metrics = {}
    
    def record_threshold_sweep(self, thresholds, precisions, recalls, f1s, balanced_accs, accuracies, model_name='ensemble'):
        """
        Record metrics at each threshold tested.
        
        Args:
            thresholds: array of threshold values tested
            precisions: precision at each threshold
            recalls: recall at each threshold
            f1s: F1 score at each threshold
            balanced_accs: balanced accuracy at each threshold
            accuracies: accuracy at each threshold
            model_name: identifier for model
        """
        df = pd.DataFrame({
            'threshold': thresholds,
            'precision': precisions,
            'recall': recalls,
            'f1_score': f1s,
            'balanced_accuracy': balanced_accs,
            'accuracy': accuracies,
            'model': model_name,
            'min_prec_rec': np.minimum(precisions, recalls)  # for balanced optimization
        })
        
        self.threshold_metrics.append(df)
        
        # Save intermediate
        output_file = self.output_dir / f'threshold_sweep_{model_name}.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Saved threshold sweep to {output_file}")
        
        return df
    
    def record_model_contribution(self, model_name, individual_probs, ensemble_probs, y_true):
        """
        Analyze contribution of individual model to ensemble.
        
        Args:
            model_name: identifier (e.g., 'qnn', 'qsvc', 'svm')
            individual_probs: probabilities from this model
            ensemble_probs: final ensemble probabilities
            y_true: true labels
        """
        # Correlation between model and ensemble
        correlation = np.corrcoef(individual_probs, ensemble_probs)[0, 1]
        
        # Prediction agreement
        model_preds = (individual_probs >= 0.5).astype(int)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        agreement = np.mean(model_preds == ensemble_preds)
        
        # Individual performance
        individual_accuracy = accuracy_score(y_true, model_preds)
        individual_f1 = f1_score(y_true, model_preds, zero_division=0)
        
        self.model_contributions[model_name] = {
            'correlation_with_ensemble': float(correlation),
            'prediction_agreement': float(agreement),
            'individual_accuracy': float(individual_accuracy),
            'individual_f1': float(individual_f1),
            'contribution_score': float(correlation * agreement)  # simple composite
        }
        
        print(f"\n{model_name.upper()} Contribution Analysis:")
        print(f"  Correlation with ensemble: {correlation:.4f}")
        print(f"  Prediction agreement: {agreement:.4f} ({agreement*100:.1f}%)")
        print(f"  Individual accuracy: {individual_accuracy:.4f}")
        print(f"  Individual F1: {individual_f1:.4f}")
        print(f"  Contribution score: {self.model_contributions[model_name]['contribution_score']:.4f}")
    
    def record_calibration_metrics(self, y_true, model_name, uncalibrated_probs=None, calibrated_probs=None):
        """
        Analyze calibration improvement.
        
        Args:
            y_true: true labels
            model_name: model identifier
            uncalibrated_probs: probabilities before calibration
            calibrated_probs: probabilities after calibration
        """
        if uncalibrated_probs is not None and calibrated_probs is not None:
            # Expected calibration error
            def compute_ece(y_true, probs, n_bins=10):
                bins = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bins[:-1]
                bin_uppers = bins[1:]
                
                ece = 0
                total = len(y_true)
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (probs >= bin_lower) & (probs < bin_upper)
                    bin_accuracy = np.mean(y_true[in_bin]) if np.any(in_bin) else 0
                    bin_confidence = np.mean(probs[in_bin]) if np.any(in_bin) else 0
                    bin_weight = np.sum(in_bin) / total
                    ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
                
                return ece
            
            ece_uncal = compute_ece(y_true, uncalibrated_probs)
            ece_cal = compute_ece(y_true, calibrated_probs)
            
            self.calibration_metrics[model_name] = {
                'ece_uncalibrated': float(ece_uncal),
                'ece_calibrated': float(ece_cal),
                'calibration_improvement': float(ece_uncal - ece_cal)
            }
            
            print(f"\n{model_name.upper()} Calibration:")
            print(f"  ECE (uncalibrated): {ece_uncal:.4f}")
            print(f"  ECE (calibrated): {ece_cal:.4f}")
            print(f"  Improvement: {ece_uncal - ece_cal:.4f}")
    
    def generate_report(self, fold_idx=None):
        """
        Generate comprehensive diagnostics report.
        
        Returns:
            dict with all diagnostics
        """
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'fold': fold_idx,
            'model_contributions': self.model_contributions,
            'calibration_metrics': self.calibration_metrics
        }
        
        # Save
        if fold_idx is not None:
            output_file = self.output_dir / f'report_fold_{fold_idx}.json'
        else:
            output_file = self.output_dir / 'report_overall.json'
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Diagnostics report saved to {output_file}")
        
        return report
    
    def generate_summary_table(self, all_fold_results, output_file='results/diagnostics/cross_validation_summary.csv'):
        """
        Create summary table of all CV folds.
        
        Args:
            all_fold_results: list of dicts with fold metrics
            output_file: where to save
        """
        df = pd.DataFrame(all_fold_results)
        
        # Add mean/std rows
        summary_rows = []
        for col in df.select_dtypes(include=[np.number]).columns:
            summary_rows.append({
                'fold': 'MEAN',
                col: df[col].mean()
            })
            summary_rows.append({
                'fold': 'STD',
                col: df[col].std()
            })
        
        if summary_rows:
            df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
        
        df.to_csv(output_file, index=False)
        print(f"\n✓ CV summary saved to {output_file}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        return df
    
    def generate_threshold_optimization_plot(self, df_sweep, output_file=None):
        """
        Create visualization of threshold optimization.
        
        Args:
            df_sweep: DataFrame from record_threshold_sweep
            output_file: where to save (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("⚠️  matplotlib not available; skipping plot generation")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Precision vs Threshold
        axes[0, 0].plot(df_sweep['threshold'], df_sweep['precision'], label='Precision', linewidth=2)
        axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% target')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall vs Threshold
        axes[0, 1].plot(df_sweep['threshold'], df_sweep['recall'], label='Recall', color='green', linewidth=2)
        axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% target')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall vs Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 vs Threshold
        axes[1, 0].plot(df_sweep['threshold'], df_sweep['f1_score'], label='F1', color='orange', linewidth=2)
        axes[1, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% target')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Min(Precision, Recall) vs Threshold
        axes[1, 1].plot(df_sweep['threshold'], df_sweep['min_prec_rec'], label='Min(Prec, Rec)', color='purple', linewidth=2)
        axes[1, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% target')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Min(Precision, Recall)')
        axes[1, 1].set_title('Balanced Metric vs Threshold (Optimization Target)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / 'threshold_optimization.png'
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to {output_file}")
        
        plt.close()
        
        return output_file


class MetricsRecorder:
    """Simple recorder for fold-level metrics."""
    
    def __init__(self):
        self.folds = []
    
    def record_fold(self, fold_idx, **metrics):
        """Record metrics for a fold."""
        record = {'fold': fold_idx}
        record.update(metrics)
        self.folds.append(record)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.folds)
    
    def summary_stats(self):
        """Compute summary statistics."""
        df = self.to_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {}
        for col in numeric_cols:
            summary[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return summary


if __name__ == '__main__':
    # Example usage
    diag = EnsembleDiagnostics()
    
    # Simulate threshold sweep
    thresholds = np.linspace(0.3, 0.7, 50)
    precisions = 0.85 + 0.05 * np.sin(thresholds * np.pi)
    recalls = 0.88 + 0.07 * np.cos(thresholds * np.pi)
    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    balanced_accs = (precisions + recalls) / 2
    accuracies = 0.9 + 0.02 * np.random.randn(len(thresholds))
    
    df = diag.record_threshold_sweep(
        thresholds, precisions, recalls, f1s, balanced_accs, accuracies,
        model_name='ensemble'
    )
    
    print("\nThreshold sweep example:")
    print(df.head())
    
    # Simulate model contributions
    y_true = np.concatenate([np.zeros(100), np.ones(10)])
    individual_probs = np.random.beta(2, 5, 110)
    ensemble_probs = individual_probs * 0.8 + 0.1
    
    diag.record_model_contribution('qnn', individual_probs, ensemble_probs, y_true)
    
    # Generate report
    report = diag.generate_report(fold_idx=0)
    print("\nDiagnostics report:")
    print(json.dumps(report, indent=2))
