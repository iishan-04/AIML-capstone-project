"""
ml_pipeline.py — Part 2: Supervised ML — Build, Train, Evaluate
==================================================================
Loads cleaned_data.csv from Part 1. Runs top-to-bottom, no arguments needed.
Produces printed results for every task plus plots/roc_curve.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix,
                              classification_report, roc_curve, roc_auc_score,
                              precision_score, recall_score, f1_score)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
RANDOM_STATE = 42

# =========================================================================
# TASK 1 — Load data and define X, y_reg, y_clf
# =========================================================================
print("=" * 80)
print("TASK 1 — Load data and define labels")
print("=" * 80)
df = pd.read_csv("cleaned_data.csv")
print(f"\nLoaded cleaned_data.csv: {df.shape}")

# Regression target: salary (continuous)
y_reg = df["salary"].copy()

# Classification target: "high_earner" = top 25% of salaries (NOT a median
# split — a median split on this dataset gives an almost perfect 50/50
# class balance, which defeats the purpose of the imbalance-handling task
# below. Top-quartile is still literally "binarizing y_reg", just at a
# threshold that produces a genuinely imbalanced, business-realistic label
# ("identify the top-earning segment").
salary_q75 = df["salary"].quantile(0.75)
y_clf = (df["salary"] >= salary_q75).astype(int)
print(f"\ny_reg = df['salary'] (continuous)")
print(f"y_clf = (df['salary'] >= {salary_q75:.2f}).astype(int)  [top-25% 'high_earner' flag]")
print(f"\ny_clf class balance:\n{y_clf.value_counts()}")
print(f"\ny_clf class balance (%):\n{(y_clf.value_counts(normalize=True) * 100).round(2)}")

# Feature matrix: all columns except identifiers and the target itself.
# salary is dropped from X for BOTH tasks — it is the direct source of
# y_clf, so leaving it in would leak the classification label straight
# into the features (the model could just threshold this one column and
# get 100% accuracy, which would make every downstream metric in this
# part meaningless).
X = df.drop(columns=["employee_id", "salary"])
print(f"\nFeature matrix X shape (before encoding): {X.shape}")
print(f"X columns: {list(X.columns)}")

# =========================================================================
# TASK 2 — Encode categorical columns
# =========================================================================
print("\n" + "=" * 80)
print("TASK 2 — Encode categorical columns")
print("=" * 80)

# education_level has a natural order -> ordinal label encoding
education_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
X["education_level"] = X["education_level"].map(education_order)
print(f"\neducation_level ordinal mapping: {education_order}")
print("Justification: education level has a genuine real-world ordering "
      "(each level represents strictly more education than the last), so "
      "mapping to increasing integers preserves meaningful information a "
      "linear model can use (e.g. a monotonic relationship with salary).")

# department, region have no natural order -> one-hot encoding, drop first
# to avoid multicollinearity (the dropped category becomes the reference
# level, implicitly represented when all dummy columns are 0)
print(f"\ndepartment categories: {sorted(df['department'].unique())}")
print(f"region categories: {sorted(df['region'].unique())}")
print("Justification: department and region have no inherent ranking (e.g. "
      "'Engineering' is not numerically greater than 'Sales'). Label-encoding "
      "them would impose a false ordinal relationship the model would "
      "wrongly treat as meaningful (e.g. implying HR < Engineering < "
      "Finance). One-hot encoding avoids this by giving each category its "
      "own independent binary column.")

X = pd.get_dummies(X, columns=["department", "region"], drop_first=True)
print(f"\nX shape after encoding: {X.shape}")
print(f"X columns after encoding: {list(X.columns)}")

# =========================================================================
# TASK 3 — Leak-free train-test split and scaling
# =========================================================================
print("\n" + "=" * 80)
print("TASK 3 — Leak-free train-test split and scaling")
print("=" * 80)

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

scaler = StandardScaler()
scaler.fit(X_train)  # fit ONLY on training data
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print("\nIMPORTANT: the scaler was fit only on X_train, then used to transform "
      "both X_train and X_test. Fitting it on the full dataset (train+test "
      "combined) would be data leakage: the scaler's mean/std would then "
      "partly reflect the test set's own statistics, meaning information "
      "about the test set would leak into how the training features are "
      "represented, giving an optimistic bias to every metric computed "
      "later. The test set must stay statistically 'unseen' until evaluation.")

# =========================================================================
# TASK 4 — Regression: Linear Regression + Ridge
# =========================================================================
print("\n" + "=" * 80)
print("TASK 4 — Linear Regression")
print("=" * 80)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)
print(f"\nLinear Regression — MSE: {mse_lin:.2f}, R2: {r2_lin:.4f}")

coef_table = pd.DataFrame({
    "feature": X_train_scaled.columns,
    "coefficient": lin_reg.coef_
}).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
print("\nCoefficients (sorted by absolute value):\n", coef_table)
print(f"\nTop 3 features by |coefficient|:\n{coef_table.head(3)}")

print("\nRidge Regression (alpha=1.0)")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"Ridge Regression — MSE: {mse_ridge:.2f}, R2: {r2_ridge:.4f}")

comparison_reg = pd.DataFrame({
    "model": ["Linear Regression (OLS)", "Ridge (alpha=1.0)"],
    "MSE": [mse_lin, mse_ridge],
    "R2": [r2_lin, r2_ridge]
})
print("\nComparison table:\n", comparison_reg)

# =========================================================================
# TASK 5 — Classification: Logistic Regression with imbalance handling
# =========================================================================
print("\n" + "=" * 80)
print("TASK 5 — Logistic Regression (with class-imbalance handling)")
print("=" * 80)

train_counts = y_clf_train.value_counts()
minority_pct = train_counts.min() / train_counts.sum() * 100
print(f"\ny_clf_train class counts:\n{train_counts}")
print(f"Minority class: {minority_pct:.1f}% of training samples")

resample_method = None
if minority_pct < 35:
    print(f"\nMinority class is below 35% -> imbalance handling required.")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_bal, y_clf_train_bal = smote.fit_resample(X_train_scaled, y_clf_train)
        resample_method = "SMOTE (imblearn.over_sampling)"
    except ImportError:
        # Fallback used only when imbalanced-learn isn't installed in this
        # environment. Manual random oversampling with replacement achieves
        # the same class-balancing goal (matching minority count to
        # majority count), though SMOTE is preferred when available since it
        # synthesizes new interpolated points rather than duplicating exact
        # rows, which reduces overfitting risk on the resampled data.
        print("imbalanced-learn not installed here -> falling back to "
              "manual random oversampling (SMOTE is the primary/preferred "
              "method — install with `pip install imbalanced-learn` to use it).")
        rng = np.random.default_rng(RANDOM_STATE)
        train_df = X_train_scaled.copy()
        train_df["_y"] = y_clf_train.values
        majority_class = train_counts.idxmax()
        minority_class = train_counts.idxmin()
        n_needed = train_counts.max() - train_counts.min()
        minority_rows = train_df[train_df["_y"] == minority_class]
        oversampled = minority_rows.sample(n=n_needed, replace=True, random_state=RANDOM_STATE)
        train_df_bal = pd.concat([train_df, oversampled], ignore_index=True)
        y_clf_train_bal = train_df_bal.pop("_y")
        X_train_bal = train_df_bal
        resample_method = "Manual random oversampling (SMOTE fallback)"

    print(f"\nResampling method used: {resample_method}")
    print(f"Class counts BEFORE resampling:\n{train_counts}")
    print(f"Class counts AFTER resampling:\n{y_clf_train_bal.value_counts()}")
else:
    X_train_bal, y_clf_train_bal = X_train_scaled, y_clf_train
    print("\nMinority class already >= 35% — no resampling applied.")

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_bal, y_clf_train_bal)

y_pred_clf = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_clf_test, y_pred_clf)
print(f"\nConfusion matrix:\n{cm}")
print(f"\nClassification report:\n{classification_report(y_clf_test, y_pred_clf)}")

fpr, tpr, thresholds = roc_curve(y_clf_test, y_pred_proba)
auc = roc_auc_score(y_clf_test, y_pred_proba)
print(f"\nROC AUC: {auc:.4f}")

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#2b6cb0", linewidth=2, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.title("ROC Curve — Logistic Regression (high_earner classification)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.annotate(f"AUC = {auc:.3f}", xy=(0.6, 0.2), fontsize=12,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="#2b6cb0"))
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=120)
plt.close()

# =========================================================================
# TASK 5b — Decision-threshold sensitivity
# =========================================================================
print("\n" + "=" * 80)
print("TASK 5b — Decision-threshold sensitivity")
print("=" * 80)

thresholds_to_test = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_rows = []
for t in thresholds_to_test:
    preds_t = (y_pred_proba >= t).astype(int)
    p = precision_score(y_clf_test, preds_t, zero_division=0)
    r = recall_score(y_clf_test, preds_t, zero_division=0)
    f1 = f1_score(y_clf_test, preds_t, zero_division=0)
    threshold_rows.append((t, p, r, f1))

threshold_table = pd.DataFrame(threshold_rows, columns=["Threshold", "Precision", "Recall", "F1"])
print("\n", threshold_table)
best_f1_row = threshold_table.loc[threshold_table["F1"].idxmax()]
print(f"\nF1-maximising threshold: {best_f1_row['Threshold']} (F1 = {best_f1_row['F1']:.4f})")

# =========================================================================
# TASK 6 — Regularization experiment (C=0.01 vs C=1.0)
# =========================================================================
print("\n" + "=" * 80)
print("TASK 6 — Regularization experiment")
print("=" * 80)

log_reg_strong = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.01)
log_reg_strong.fit(X_train_bal, y_clf_train_bal)
y_pred_proba_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]
y_pred_clf_strong = log_reg_strong.predict(X_test_scaled)

precision_baseline = precision_score(y_clf_test, y_pred_clf, zero_division=0)
recall_baseline = recall_score(y_clf_test, y_pred_clf, zero_division=0)
auc_baseline = auc

precision_strong = precision_score(y_clf_test, y_pred_clf_strong, zero_division=0)
recall_strong = recall_score(y_clf_test, y_pred_clf_strong, zero_division=0)
auc_strong = roc_auc_score(y_clf_test, y_pred_proba_strong)

reg_comparison = pd.DataFrame({
    "model": ["C=1.0 (baseline)", "C=0.01 (strong regularization)"],
    "precision": [precision_baseline, precision_strong],
    "recall": [recall_baseline, recall_strong],
    "AUC": [auc_baseline, auc_strong],
})
print("\n", reg_comparison)

# =========================================================================
# TASK 6b — Bootstrap confidence interval for AUC difference
# =========================================================================
print("\n" + "=" * 80)
print("TASK 6b — Bootstrap CI for AUC difference (C=1.0 minus C=0.01)")
print("=" * 80)

rng = np.random.default_rng(RANDOM_STATE)
y_clf_test_arr = y_clf_test.to_numpy()
n_test = len(y_clf_test_arr)
diffs = []
for i in range(500):
    idx = rng.choice(n_test, size=n_test, replace=True)
    y_sample = y_clf_test_arr[idx]
    if len(np.unique(y_sample)) < 2:
        continue  # AUC undefined if the bootstrap sample has only one class
    proba_baseline_sample = y_pred_proba[idx]
    proba_strong_sample = y_pred_proba_strong[idx]
    auc_b = roc_auc_score(y_sample, proba_baseline_sample)
    auc_s = roc_auc_score(y_sample, proba_strong_sample)
    diffs.append(auc_b - auc_s)

diffs = np.array(diffs)
mean_diff = diffs.mean()
ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
print(f"\nBootstrap iterations used: {len(diffs)} (of 500 requested)")
print(f"Mean AUC difference (C=1.0 - C=0.01): {mean_diff:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"CI excludes zero: {ci_lower > 0 or ci_upper < 0}")

print("\n" + "=" * 80)
print("Part 2 complete.")
print("=" * 80)
