# ========================================================================================
# ENHANCED XGBOOST ANALYSIS WITH VISUALIZATIONS
# Add this cell after your data preprocessing cells in the notebook
# This captures outputs and generates: Confusion Matrix, ROC Curve, PR Curve, PR-AUC by Max Depth
# ========================================================================================

import xgboost as xgb
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    precision_recall_curve, average_precision_score, roc_curve
)
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Optimized XGBoost for Extreme Class Imbalance with Visualizations ===\n")

# Compute scale_pos_weight for extreme imbalance
neg_count = sum(y == 0)
pos_count = sum(y == 1)
scale_pos_weight = neg_count / pos_count

print(f"Class distribution:")
print(f"  Normal (0): {neg_count:,}")
print(f"  apt (1): {pos_count:,}")
print(f"  Imbalance ratio: {scale_pos_weight:.1f}:1")
print(f"  Base contamination: {pos_count/len(y):.6f}")
print(f"\nScale_pos_weight: {scale_pos_weight:.1f}\n")

# Define parameters
ratio_range = [150]  # Expand to [10, 20, 30, 40, 50, 75, 100, 150, 200] for full grid search
max_depth_range = [3, 4, 5, 6, 7, 8]

# Stratified K-Fold CV
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage for results and visualization data
grid_results = []
depth_results = []
all_tpr, all_fpr, all_roc_auc = [], [], []
all_pr_auc = []
aggregated_cm = np.zeros((2, 2))

# ========================
# PART 1: Grid Search over SMOTE Ratios
# ========================
print("\n" + "="*100)
print("PART 1: GRID SEARCH OVER SMOTE RATIOS")
print("="*100)

for ratio in ratio_range:
    print(f"\n{'='*70}")
    print(f"Testing Ratio: {ratio}:1 (Normal:apt)")
    print(f"{'='*70}")

    acc_scores, roc_auc_scores, pr_auc_scores = [], [], []
    mcc_scores, kappa_scores, nir_scores = [], [], []
    f1_scores, f2_scores = [], []
    sensitivity_scores, specificity_scores, precision_scores = [], [], []

    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        minority_target = sum(y_train == 0) // ratio
        smote = SMOTE(random_state=42, sampling_strategy={1: minority_target}, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        X_train = pd.DataFrame(X_resampled, columns=X.columns)
        y_train = pd.Series(y_resampled, name='is_apt')

        print(f"  Fold {fold}: Training on {len(X_train):,} samples ({sum(y_train==0):,} Normal, {sum(y_train==1):,} apt)")

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            device="cuda",
            learning_rate=0.03,
            max_depth=6,
            subsample=0.7,
            min_child_weight=0.5,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            max_delta_step=1,
            eval_metric="aucpr",
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=42,
            colsample_bylevel=0.7,
            tree_method='hist'
        )

        xgb_model.fit(X_train, y_train, verbose=False)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]

        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores_at_thresholds = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores_at_thresholds)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        y_pred = (y_prob >= optimal_threshold).astype(int)

        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        all_tpr.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
        all_fpr.append(np.linspace(0, 1, 100))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        aggregated_cm += cm

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = (5 * precision[optimal_idx] * recall[optimal_idx]) / (4 * precision[optimal_idx] + recall[optimal_idx] + 1e-10)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        nir = (y_test == y_test.mode()[0]).mean()

        acc_scores.append(acc)
        roc_auc_scores.append(roc_auc)
        pr_auc_scores.append(pr_auc)
        mcc_scores.append(mcc)
        kappa_scores.append(kappa)
        f1_scores.append(f1)
        f2_scores.append(f2)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
        precision_scores.append(prec)
        nir_scores.append(nir)

        print(f"    Threshold: {optimal_threshold:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | Sens: {sensitivity:.4f}")
        fold += 1

    all_roc_auc.extend(roc_auc_scores)
    all_pr_auc.extend(pr_auc_scores)

    # Store results
    grid_results.append({
        'ratio': f"{ratio}:1",
        'ratio_value': ratio,
        'accuracy': np.mean(acc_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(sensitivity_scores),
        'f1_score': np.mean(f1_scores),
        'f2_score': np.mean(f2_scores),
        'specificity': np.mean(specificity_scores),
        'roc_auc': np.mean(roc_auc_scores),
        'pr_auc': np.mean(pr_auc_scores),
        'mcc': np.mean(mcc_scores),
        'kappa': np.mean(kappa_scores),
        'nir': np.mean(nir_scores),
        'beats_nir': 'YES' if np.mean(acc_scores) > np.mean(nir_scores) else 'NO'
    })

    print(f"\n  Ratio {ratio}:1 - Mean Metrics:")
    print(f"    Accuracy:    {np.mean(acc_scores):.4f}")
    print(f"    F1-score:    {np.mean(f1_scores):.4f}")
    print(f"    PR-AUC:      {np.mean(pr_auc_scores):.4f} ⭐")

results_df = pd.DataFrame(grid_results)

# ========================
# PART 2: Max Depth Analysis
# ========================
print("\n" + "="*100)
print("PART 2: MAX DEPTH ANALYSIS")
print("="*100)

best_ratio = results_df.loc[results_df['pr_auc'].idxmax()]['ratio_value']
print(f"\nUsing best ratio: {best_ratio}:1\n")

for max_depth in max_depth_range:
    print(f"Testing max_depth={max_depth}...")

    depth_pr_auc_scores = []
    depth_roc_auc_scores = []
    depth_f1_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        minority_target = sum(y_train == 0) // int(best_ratio)
        smote = SMOTE(random_state=42, sampling_strategy={1: minority_target}, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        X_train = pd.DataFrame(X_resampled, columns=X.columns)
        y_train = pd.Series(y_resampled, name='is_apt')

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            device="cuda",
            learning_rate=0.03,
            max_depth=max_depth,
            subsample=0.7,
            min_child_weight=0.5,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            max_delta_step=1,
            eval_metric="aucpr",
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=42,
            colsample_bylevel=0.7,
            tree_method='hist'
        )

        xgb_model.fit(X_train, y_train, verbose=False)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores_at_thresholds = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores_at_thresholds)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        y_pred = (y_prob >= optimal_threshold).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        depth_pr_auc_scores.append(pr_auc)
        depth_roc_auc_scores.append(roc_auc)
        depth_f1_scores.append(f1)

    depth_results.append({
        'max_depth': max_depth,
        'pr_auc': np.mean(depth_pr_auc_scores),
        'pr_auc_std': np.std(depth_pr_auc_scores),
        'roc_auc': np.mean(depth_roc_auc_scores),
        'f1_score': np.mean(depth_f1_scores)
    })

    print(f"  max_depth={max_depth}: PR-AUC={np.mean(depth_pr_auc_scores):.4f} (±{np.std(depth_pr_auc_scores):.4f})")

depth_results_df = pd.DataFrame(depth_results)

# ========================
# PART 3: Visualizations
# ========================
print("\n" + "="*100)
print("GENERATING VISUALIZATIONS")
print("="*100)

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# PLOT 1: Confusion Matrix
ax1 = axes[0, 0]
sns.heatmap(aggregated_cm, annot=True, fmt='.0f', cmap='Blues', ax=ax1,
            xticklabels=['Normal (0)', 'APT (1)'],
            yticklabels=['Normal (0)', 'APT (1)'],
            cbar_kws={'label': 'Count'})
ax1.set_title('Aggregated Confusion Matrix (All Folds)', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)

total = aggregated_cm.sum()
for i in range(2):
    for j in range(2):
        percentage = (aggregated_cm[i, j] / total) * 100
        ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                ha='center', va='center', fontsize=10, color='red')

# PLOT 2: ROC Curve
ax2 = axes[0, 1]
mean_tpr = np.mean(all_tpr, axis=0)
mean_fpr = np.linspace(0, 1, 100)
mean_roc_auc = np.mean(all_roc_auc)
std_roc_auc = np.std(all_roc_auc)

ax2.plot(mean_fpr, mean_tpr, color='blue', lw=2,
         label=f'Mean ROC (AUC = {mean_roc_auc:.4f} ± {std_roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.50)')
ax2.fill_between(mean_fpr,
                 np.maximum(mean_tpr - np.std(all_tpr, axis=0), 0),
                 np.minimum(mean_tpr + np.std(all_tpr, axis=0), 1),
                 alpha=0.2, color='blue', label='±1 std dev')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve (Mean across all folds)', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# PLOT 3: Precision-Recall Curve
ax3 = axes[1, 0]
mean_pr_auc = np.mean(all_pr_auc)
std_pr_auc = np.std(all_pr_auc)

ax3.axhline(y=mean_pr_auc, color='blue', linestyle='-', lw=2,
           label=f'Mean PR-AUC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f}')

baseline = pos_count / len(y)
ax3.axhline(y=baseline, color='red', linestyle='--', lw=2,
           label=f'Baseline (No Skill) = {baseline:.4f}')

ax3.set_xlabel('Recall', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# PLOT 4: PR-AUC by Max Depth
ax4 = axes[1, 1]
ax4.plot(depth_results_df['max_depth'], depth_results_df['pr_auc'],
         marker='o', linewidth=2, markersize=8, color='blue', label='PR-AUC')
ax4.errorbar(depth_results_df['max_depth'], depth_results_df['pr_auc'],
            yerr=depth_results_df['pr_auc_std'], fmt='none', ecolor='blue',
            alpha=0.3, capsize=5)
ax4.set_xlabel('Max Depth', fontsize=12)
ax4.set_ylabel('PR-AUC Score', fontsize=12)
ax4.set_title('PR-AUC by Max Depth', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

best_depth_idx = depth_results_df['pr_auc'].idxmax()
best_depth = depth_results_df.loc[best_depth_idx, 'max_depth']
best_depth_pr_auc = depth_results_df.loc[best_depth_idx, 'pr_auc']
ax4.annotate(f'Best: depth={best_depth}\nPR-AUC={best_depth_pr_auc:.4f}',
            xy=(best_depth, best_depth_pr_auc),
            xytext=(best_depth + 0.5, best_depth_pr_auc - 0.02),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('xgboost_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to: xgboost_analysis_visualizations.png")
plt.show()

# ========================
# PART 4: Final Summary
# ========================
print("\n" + "="*100)
print("=== FINAL SUMMARY ===")
print("="*100)

print("\n--- Grid Search Results ---")
print(results_df.to_string(index=False))

print("\n--- Max Depth Results ---")
print(depth_results_df.to_string(index=False))

best_pr_auc = results_df.loc[results_df['pr_auc'].idxmax()]
best_depth_row = depth_results_df.loc[depth_results_df['pr_auc'].idxmax()]

print(f"\n⭐ RECOMMENDED CONFIGURATION:")
print(f"   - SMOTE Ratio: {best_pr_auc['ratio']} (PR-AUC={best_pr_auc['pr_auc']:.4f})")
print(f"   - Max Depth: {best_depth_row['max_depth']} (PR-AUC={best_depth_row['pr_auc']:.4f})")

print("\n" + "="*100)
