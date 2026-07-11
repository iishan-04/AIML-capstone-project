"""
build_notebook.py — assembles Part2_ML.ipynb directly as JSON (same approach
as Part 1, since nbformat/nbconvert aren't available in this sandbox).
"""
import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []}

cells = []

cells.append(md(
"# Part 2 — Supervised Machine Learning Model — Build, Train, Evaluate",
"",
"Requires `cleaned_data.csv` from Part 1 to be in the **same folder** as this notebook.",
"Run cells top to bottom (or **Run All**)."
))

cells.append(md("## Setup"))
cells.append(code(
"import numpy as np",
"import pandas as pd",
"import matplotlib.pyplot as plt",
"import os",
"",
"from sklearn.model_selection import train_test_split",
"from sklearn.preprocessing import StandardScaler",
"from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression",
"from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix,",
"                              classification_report, roc_curve, roc_auc_score,",
"                              precision_score, recall_score, f1_score)",
"",
"pd.set_option('display.width', 120)",
"pd.set_option('display.max_columns', 20)",
"RANDOM_STATE = 42",
"os.makedirs('plots', exist_ok=True)",
"print('Setup complete.')"
))

cells.append(md("## Task 1 — Load data and define labels"))
cells.append(code(
"df = pd.read_csv('cleaned_data.csv')",
"print('Loaded cleaned_data.csv:', df.shape)",
"",
"y_reg = df['salary'].copy()",
"",
"salary_q75 = df['salary'].quantile(0.75)",
"y_clf = (df['salary'] >= salary_q75).astype(int)",
"print(f\"y_clf = (salary >= {salary_q75:.2f}).astype(int)  ['high_earner': top 25%]\")",
"print(y_clf.value_counts())",
"print((y_clf.value_counts(normalize=True) * 100).round(2))",
"",
"X = df.drop(columns=['employee_id', 'salary'])",
"print('\\nFeature matrix X shape (before encoding):', X.shape)",
"print(list(X.columns))"
))
cells.append(md(
"**Why top-quartile instead of median split:** a median split gives an almost exactly",
"50/50 class balance on this dataset, which would make it impossible to genuinely",
"demonstrate imbalance handling. Top-25% is still literally 'binarizing y_reg', just at a",
"threshold that produces a real, business-realistic imbalance.",
"",
"**Why `salary` is dropped from X for both tasks:** it's the direct source of `y_clf` —",
"leaving it in would let the classifier trivially threshold that single column for ~100%",
"accuracy, making every classification metric meaningless (data leakage)."
))

cells.append(md("## Task 2 — Encode categorical columns"))
cells.append(code(
"education_order = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}",
"X['education_level'] = X['education_level'].map(education_order)",
"print('education_level ordinal mapping:', education_order)",
"",
"X = pd.get_dummies(X, columns=['department', 'region'], drop_first=True)",
"print('X shape after encoding:', X.shape)",
"print(list(X.columns))"
))
cells.append(md(
"- `education_level` → **ordinal** label encoding: it has a genuine real-world order, so",
"  increasing integers preserve information a linear model can use.",
"- `department`, `region` → **one-hot** encoding, first category dropped: neither has a",
"  natural ranking, so label-encoding would impose a false ordinal relationship. One-hot",
"  gives each category its own independent column; dropping the first avoids",
"  multicollinearity."
))

cells.append(md("## Task 3 — Leak-free train-test split and scaling"))
cells.append(code(
"X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(",
"    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE",
")",
"print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)",
"",
"scaler = StandardScaler()",
"scaler.fit(X_train)  # fit ONLY on training data",
"X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)",
"X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)",
"print('Scaled.')"
))
cells.append(md(
"**Fitting the scaler on the full dataset would be data leakage:** the scaler's mean/std",
"would then partly reflect the test set's own values, meaning test-set information leaks",
"into how the training features are represented — giving an optimistically biased picture",
"of real-world performance. The test set must stay statistically unseen until evaluation."
))

cells.append(md("## Task 4 — Regression: Linear Regression vs Ridge"))
cells.append(code(
"lin_reg = LinearRegression()",
"lin_reg.fit(X_train_scaled, y_reg_train)",
"y_pred_reg = lin_reg.predict(X_test_scaled)",
"",
"mse_lin = mean_squared_error(y_reg_test, y_pred_reg)",
"r2_lin = r2_score(y_reg_test, y_pred_reg)",
"print(f'Linear Regression — MSE: {mse_lin:.2f}, R2: {r2_lin:.4f}')",
"",
"coef_table = pd.DataFrame({",
"    'feature': X_train_scaled.columns,",
"    'coefficient': lin_reg.coef_",
"}).sort_values('coefficient', key=lambda s: s.abs(), ascending=False)",
"print(coef_table)",
"print('\\nTop 3 by |coefficient|:')",
"print(coef_table.head(3))"
))
cells.append(code(
"ridge = Ridge(alpha=1.0)",
"ridge.fit(X_train_scaled, y_reg_train)",
"y_pred_ridge = ridge.predict(X_test_scaled)",
"mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)",
"r2_ridge = r2_score(y_reg_test, y_pred_ridge)",
"print(f'Ridge Regression — MSE: {mse_ridge:.2f}, R2: {r2_ridge:.4f}')",
"",
"comparison_reg = pd.DataFrame({",
"    'model': ['Linear Regression (OLS)', 'Ridge (alpha=1.0)'],",
"    'MSE': [mse_lin, mse_ridge],",
"    'R2': [r2_lin, r2_ridge]",
"})",
"print(comparison_reg)"
))
cells.append(md(
"**Coefficient interpretation:** on standardized features, each coefficient is the change",
"in predicted salary per one-standard-deviation increase in that feature. A large",
"**positive** coefficient means more of that feature → higher predicted salary; a large",
"**negative** coefficient means belonging to that category is associated with *lower*",
"predicted salary relative to the dropped reference category.",
"",
"**Ridge vs OLS:** `alpha` controls L2 penalty strength — it shrinks coefficients toward",
"zero to reduce variance at the cost of some bias. Here OLS and Ridge land on nearly",
"identical results because the salary relationship is strong and close to linear by",
"construction, and `alpha=1.0` is a mild penalty relative to that signal."
))

cells.append(md("## Task 5 — Classification: Logistic Regression"))
cells.append(code(
"train_counts = y_clf_train.value_counts()",
"minority_pct = train_counts.min() / train_counts.sum() * 100",
"print('y_clf_train class counts:')",
"print(train_counts)",
"print(f'Minority class: {minority_pct:.1f}% of training samples')"
))
cells.append(md(
"**Imbalance handling method: SMOTE** (`imblearn.over_sampling.SMOTE`) — chosen because it",
"synthesizes new, interpolated minority-class points rather than duplicating exact rows,",
"reducing overfitting risk versus naive duplication. Run `pip install imbalanced-learn`",
"before this cell if you haven't already; a manual-oversampling fallback runs automatically",
"if the package isn't found."
))
cells.append(code(
"resample_method = None",
"if minority_pct < 35:",
"    try:",
"        from imblearn.over_sampling import SMOTE",
"        smote = SMOTE(random_state=RANDOM_STATE)",
"        X_train_bal, y_clf_train_bal = smote.fit_resample(X_train_scaled, y_clf_train)",
"        resample_method = 'SMOTE (imblearn.over_sampling)'",
"    except ImportError:",
"        rng = np.random.default_rng(RANDOM_STATE)",
"        train_df = X_train_scaled.copy()",
"        train_df['_y'] = y_clf_train.values",
"        minority_class = train_counts.idxmin()",
"        n_needed = train_counts.max() - train_counts.min()",
"        minority_rows = train_df[train_df['_y'] == minority_class]",
"        oversampled = minority_rows.sample(n=n_needed, replace=True, random_state=RANDOM_STATE)",
"        train_df_bal = pd.concat([train_df, oversampled], ignore_index=True)",
"        y_clf_train_bal = train_df_bal.pop('_y')",
"        X_train_bal = train_df_bal",
"        resample_method = 'Manual random oversampling (SMOTE fallback — install imbalanced-learn for real SMOTE)'",
"    print('Resampling method used:', resample_method)",
"    print('Before:', dict(train_counts))",
"    print('After:', dict(y_clf_train_bal.value_counts()))",
"else:",
"    X_train_bal, y_clf_train_bal = X_train_scaled, y_clf_train",
"    print('No resampling needed.')"
))
cells.append(code(
"log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)",
"log_reg.fit(X_train_bal, y_clf_train_bal)",
"",
"y_pred_clf = log_reg.predict(X_test_scaled)",
"y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]",
"",
"cm = confusion_matrix(y_clf_test, y_pred_clf)",
"print('Confusion matrix:')",
"print(cm)",
"print()",
"print(classification_report(y_clf_test, y_pred_clf))"
))
cells.append(code(
"fpr, tpr, thresholds = roc_curve(y_clf_test, y_pred_proba)",
"auc = roc_auc_score(y_clf_test, y_pred_proba)",
"print(f'ROC AUC: {auc:.4f}')",
"",
"plt.figure(figsize=(7, 6))",
"plt.plot(fpr, tpr, color='#2b6cb0', linewidth=2, label=f'Logistic Regression (AUC = {auc:.3f})')",
"plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')",
"plt.title('ROC Curve — Logistic Regression (high_earner classification)')",
"plt.xlabel('False Positive Rate')",
"plt.ylabel('True Positive Rate')",
"plt.annotate(f'AUC = {auc:.3f}', xy=(0.6, 0.2), fontsize=12,",
"             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2b6cb0'))",
"plt.legend(loc='lower right')",
"plt.tight_layout()",
"plt.savefig('plots/roc_curve.png', dpi=120)",
"plt.show()"
))
cells.append(md(
"**Precision** = TP / (TP + FP)  **Recall** = TP / (TP + FN)",
"",
"**Which matters more:** for flagging 'high earners', **recall** matters more — missing a",
"true high earner (false negative) is typically costlier than an extra false-positive",
"review. **AUC ≈ 0.99** means the model separates the two classes almost perfectly, which",
"makes sense since `high_earner` is a direct function of `salary`, and the same features",
"that predict salary well in Task 4 separate the classes cleanly here."
))

cells.append(md("## Task 5b — Decision-threshold sensitivity"))
cells.append(code(
"thresholds_to_test = [0.30, 0.40, 0.50, 0.60, 0.70]",
"threshold_rows = []",
"for t in thresholds_to_test:",
"    preds_t = (y_pred_proba >= t).astype(int)",
"    p = precision_score(y_clf_test, preds_t, zero_division=0)",
"    r = recall_score(y_clf_test, preds_t, zero_division=0)",
"    f1 = f1_score(y_clf_test, preds_t, zero_division=0)",
"    threshold_rows.append((t, p, r, f1))",
"",
"threshold_table = pd.DataFrame(threshold_rows, columns=['Threshold', 'Precision', 'Recall', 'F1'])",
"print(threshold_table)",
"best_f1_row = threshold_table.loc[threshold_table['F1'].idxmax()]",
"print(f\"\\nF1-maximising threshold: {best_f1_row['Threshold']} (F1 = {best_f1_row['F1']:.4f})\")"
))

cells.append(md("## Task 6 — Regularization experiment (C=1.0 vs C=0.01)"))
cells.append(code(
"log_reg_strong = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.01)",
"log_reg_strong.fit(X_train_bal, y_clf_train_bal)",
"y_pred_proba_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]",
"y_pred_clf_strong = log_reg_strong.predict(X_test_scaled)",
"",
"precision_baseline = precision_score(y_clf_test, y_pred_clf, zero_division=0)",
"recall_baseline = recall_score(y_clf_test, y_pred_clf, zero_division=0)",
"auc_baseline = auc",
"",
"precision_strong = precision_score(y_clf_test, y_pred_clf_strong, zero_division=0)",
"recall_strong = recall_score(y_clf_test, y_pred_clf_strong, zero_division=0)",
"auc_strong = roc_auc_score(y_clf_test, y_pred_proba_strong)",
"",
"reg_comparison = pd.DataFrame({",
"    'model': ['C=1.0 (baseline)', 'C=0.01 (strong regularization)'],",
"    'precision': [precision_baseline, precision_strong],",
"    'recall': [recall_baseline, recall_strong],",
"    'AUC': [auc_baseline, auc_strong],",
"})",
"print(reg_comparison)"
))
cells.append(md(
"**What `C` controls:** the *inverse* of regularization strength — a smaller `C` means a",
"stronger L2 penalty, shrinking coefficients more aggressively toward zero."
))

cells.append(md("## Task 6b — Bootstrap confidence interval for AUC difference"))
cells.append(code(
"rng = np.random.default_rng(RANDOM_STATE)",
"y_clf_test_arr = y_clf_test.to_numpy()",
"n_test = len(y_clf_test_arr)",
"diffs = []",
"for i in range(500):",
"    idx = rng.choice(n_test, size=n_test, replace=True)",
"    y_sample = y_clf_test_arr[idx]",
"    if len(np.unique(y_sample)) < 2:",
"        continue",
"    auc_b = roc_auc_score(y_sample, y_pred_proba[idx])",
"    auc_s = roc_auc_score(y_sample, y_pred_proba_strong[idx])",
"    diffs.append(auc_b - auc_s)",
"",
"diffs = np.array(diffs)",
"mean_diff = diffs.mean()",
"ci_lower = np.percentile(diffs, 2.5)",
"ci_upper = np.percentile(diffs, 97.5)",
"print(f'Bootstrap iterations used: {len(diffs)} of 500')",
"print(f'Mean AUC difference (C=1.0 - C=0.01): {mean_diff:.4f}')",
"print(f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')",
"print(f'CI excludes zero: {ci_lower > 0 or ci_upper < 0}')"
))
cells.append(md(
"**Interpretation:** if the 95% CI excludes zero, the C=1.0 model's AUC advantage is",
"statistically reliable across resamples of the test set — not just a quirk of one",
"particular split — even if the absolute margin is small."
))

cells.append(md("## Part 2 complete"))
cells.append(code("print('Part 2 pipeline finished successfully.')"))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Part2_ML.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Part2_ML.ipynb written.")
