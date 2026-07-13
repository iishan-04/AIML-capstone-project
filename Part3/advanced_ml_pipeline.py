"""
advanced_ml_pipeline.py — Part 3: Ensembles, Tuning, and Full ML Pipeline
============================================================================
Loads cleaned_data.csv from Part 1. Rebuilds the exact same preprocessing
and train/test split as Part 2 (same random_state=42), so this script is
fully self-contained and reproducible on its own. Runs top-to-bottom.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import clone

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
RANDOM_STATE = 42

# =========================================================================
# Rebuild the exact Part 2 preprocessing (same random_state -> identical split)
# =========================================================================
print("=" * 80)
print("SETUP — Rebuilding Part 2's preprocessing (same random_state=42)")
print("=" * 80)
df = pd.read_csv("cleaned_data.csv")

y_reg = df["salary"].copy()
salary_q75 = df["salary"].quantile(0.75)
y_clf = (df["salary"] >= salary_q75).astype(int)

X = df.drop(columns=["employee_id", "salary"])
education_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
X["education_level"] = X["education_level"].map(education_order)
X = pd.get_dummies(X, columns=["department", "region"], drop_first=True)

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
print(f"y_clf_train balance:\n{y_clf_train.value_counts()}")

# =========================================================================
# TASK 1 — Decision Tree baseline (unconstrained)
# =========================================================================
print("\n" + "=" * 80)
print("TASK 1 — Decision Tree baseline (unconstrained)")
print("=" * 80)

tree_unconstrained = DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE)
tree_unconstrained.fit(X_train_scaled, y_clf_train)
train_acc_unc = accuracy_score(y_clf_train, tree_unconstrained.predict(X_train_scaled))
test_acc_unc = accuracy_score(y_clf_test, tree_unconstrained.predict(X_test_scaled))
print(f"Unconstrained tree — Train accuracy: {train_acc_unc:.4f}, Test accuracy: {test_acc_unc:.4f}")
print(f"Train/test gap: {train_acc_unc - test_acc_unc:.4f}")

# =========================================================================
# TASK 2 — Controlled Decision Tree
# =========================================================================
print("\n" + "=" * 80)
print("TASK 2 — Controlled Decision Tree (max_depth=5, min_samples_split=20)")
print("=" * 80)

tree_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE)
tree_controlled.fit(X_train_scaled, y_clf_train)
train_acc_ctrl = accuracy_score(y_clf_train, tree_controlled.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, tree_controlled.predict(X_test_scaled))
print(f"Controlled tree — Train accuracy: {train_acc_ctrl:.4f}, Test accuracy: {test_acc_ctrl:.4f}")
print(f"Train/test gap: {train_acc_ctrl - test_acc_ctrl:.4f}")

# =========================================================================
# TASK 3 — Gini vs Entropy
# =========================================================================
print("\n" + "=" * 80)
print("TASK 3 — Gini vs Entropy (max_depth=5)")
print("=" * 80)

tree_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=RANDOM_STATE)
tree_gini.fit(X_train_scaled, y_clf_train)
acc_gini = accuracy_score(y_clf_test, tree_gini.predict(X_test_scaled))

tree_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=RANDOM_STATE)
tree_entropy.fit(X_train_scaled, y_clf_train)
acc_entropy = accuracy_score(y_clf_test, tree_entropy.predict(X_test_scaled))

print(f"Gini test accuracy:    {acc_gini:.4f}")
print(f"Entropy test accuracy: {acc_entropy:.4f}")

# =========================================================================
# TASK 4 — Random Forest
# =========================================================================
print("\n" + "=" * 80)
print("TASK 4 — Random Forest (n_estimators=100, max_depth=10)")
print("=" * 80)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"Random Forest — Train accuracy: {rf_train_acc:.4f}, Test accuracy: {rf_test_acc:.4f}, ROC-AUC: {rf_auc:.4f}")

importances = pd.Series(rf.feature_importances_, index=X_train_scaled.columns).sort_values(ascending=False)
print(f"\nTop 5 features by importance:\n{importances.head(5)}")

# =========================================================================
# TASK 4a — Gradient Boosting
# =========================================================================
print("\n" + "=" * 80)
print("TASK 4a — Gradient Boosting (n_estimators=100, lr=0.1, max_depth=3)")
print("=" * 80)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print(f"Gradient Boosting — Train accuracy: {gb_train_acc:.4f}, Test accuracy: {gb_test_acc:.4f}, ROC-AUC: {gb_auc:.4f}")

# =========================================================================
# TASK 4b — Feature ablation study
# =========================================================================
print("\n" + "=" * 80)
print("TASK 4b — Feature ablation study")
print("=" * 80)

lowest5 = importances.tail(5).index.tolist()
print(f"5 lowest-importance features: {lowest5}")

X_train_reduced = X_train_scaled.drop(columns=lowest5)
X_test_reduced = X_test_scaled.drop(columns=lowest5)

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf_reduced.fit(X_train_reduced, y_clf_train)
rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print(f"Full model  ({X_train_scaled.shape[1]} features)  test AUC: {rf_auc:.4f}")
print(f"Reduced model ({X_train_reduced.shape[1]} features) test AUC: {rf_reduced_auc:.4f}")
print(f"AUC change: {rf_reduced_auc - rf_auc:+.4f}")

# =========================================================================
# TASK 5 — Cross-validated comparison
# =========================================================================
print("\n" + "=" * 80)
print("TASK 5 — Cross-validated comparison (5-fold StratifiedKFold, ROC-AUC)")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Logistic Regression here uses class_weight='balanced' rather than Part 2's
# SMOTE-resampled training set. Resampling before cross_val_score would leak
# information across folds (synthetic points derived from data that ends up
# split across train/validation folds); class_weight is safe to embed
# directly in the estimator passed to cross_val_score.
log_reg_cv = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

models_for_cv = {
    "Logistic Regression": log_reg_cv,
    "Decision Tree (controlled)": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
}

cv_results = {}
for name, model in models_for_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name}: mean AUC = {scores.mean():.4f}, std = {scores.std():.4f}")

# =========================================================================
# TASK 6 — GridSearchCV hyperparameter tuning
# =========================================================================
print("\n" + "=" * 80)
print("TASK 6 — GridSearchCV (Random Forest)")
print("=" * 80)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=RANDOM_STATE)
)

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc", n_jobs=-1
)
# NOTE: fit on the UNSCALED X_train — the pipeline has its own imputer + scaler
grid_search.fit(X_train, y_clf_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
n_folds = 5
print(f"\nTotal grid configurations: {n_configs} (x {n_folds} folds = {n_configs * n_folds} total fits)")

best_pipeline = grid_search.best_estimator_

# =========================================================================
# TASK 7 — Manual learning curve
# =========================================================================
print("\n" + "=" * 80)
print("TASK 7 — Manual learning curve (best pipeline from Task 6)")
print("=" * 80)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_rows = []
for f in fractions:
    n_rows = int(f * len(X_train))
    X_subset = X_train.iloc[:n_rows]
    y_subset = y_clf_train.iloc[:n_rows]

    pipeline_f = clone(best_pipeline)
    pipeline_f.fit(X_subset, y_subset)

    train_auc = roc_auc_score(y_subset, pipeline_f.predict_proba(X_subset)[:, 1])
    test_auc = roc_auc_score(y_clf_test, pipeline_f.predict_proba(X_test)[:, 1])
    learning_curve_rows.append((f, n_rows, train_auc, test_auc))
    print(f"Fraction {f:.1f} ({n_rows} rows) — Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")

learning_curve_table = pd.DataFrame(learning_curve_rows, columns=["Fraction", "N rows", "Train AUC", "Test AUC"])
print("\n", learning_curve_table)

# =========================================================================
# TASK 8 — Serialize the best model
# =========================================================================
print("\n" + "=" * 80)
print("TASK 8 — Serialize best model")
print("=" * 80)

joblib.dump(best_pipeline, "best_model.pkl")
print("Saved best_pipeline to best_model.pkl")

# Reload-and-predict demonstration block
loaded_model = joblib.load("best_model.pkl")
hand_crafted_rows = X_test.iloc[:2].copy()
preds = loaded_model.predict(hand_crafted_rows)
probas = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
print(f"\nReload-and-predict demo on 2 hand-picked test rows:")
print(f"Predictions: {preds}")
print(f"Probabilities: {probas}")

# =========================================================================
# TASK 9 — Summary comparison table
# =========================================================================
print("\n" + "=" * 80)
print("TASK 9 — Summary comparison table")
print("=" * 80)

test_auc_lookup = {
    "Logistic Regression": roc_auc_score(
        y_clf_test,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        .fit(X_train_scaled, y_clf_train).predict_proba(X_test_scaled)[:, 1]
    ),
    "Decision Tree (controlled)": roc_auc_score(y_clf_test, tree_controlled.predict_proba(X_test_scaled)[:, 1]),
    "Random Forest": rf_auc,
    "Gradient Boosting": gb_auc,
}

summary_rows = []
for name in models_for_cv:
    mean_auc, std_auc = cv_results[name]
    summary_rows.append((name, mean_auc, std_auc, test_auc_lookup[name]))
summary_table = pd.DataFrame(summary_rows, columns=["Model", "CV Mean AUC", "CV Std AUC", "Test AUC"])
summary_table = summary_table.sort_values("Test AUC", ascending=False)
print("\n", summary_table)

print("\n" + "=" * 80)
print("Part 3 complete.")
print("=" * 80)
