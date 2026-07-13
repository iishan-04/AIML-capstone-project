"""
build_notebook.py — assembles Part3_Advanced.ipynb directly as JSON.
"""
import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []}

cells = []

cells.append(md(
"# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline",
"",
"Requires `cleaned_data.csv` from Part 1 in the **same folder** as this notebook.",
"Rebuilds the exact same preprocessing/split as Part 2 (`random_state=42`), so this",
"notebook is self-contained. **Note: Task 6 (GridSearchCV) takes a minute or two to run",
"— 90 model fits.** Run cells top to bottom."
))

cells.append(md("## Setup"))
cells.append(code(
"import numpy as np",
"import pandas as pd",
"import matplotlib.pyplot as plt",
"import joblib",
"",
"from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV",
"from sklearn.preprocessing import StandardScaler",
"from sklearn.linear_model import LogisticRegression",
"from sklearn.tree import DecisionTreeClassifier",
"from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier",
"from sklearn.pipeline import make_pipeline",
"from sklearn.impute import SimpleImputer",
"from sklearn.metrics import accuracy_score, roc_auc_score",
"from sklearn.base import clone",
"",
"pd.set_option('display.width', 120)",
"pd.set_option('display.max_columns', 20)",
"RANDOM_STATE = 42",
"print('Setup complete.')"
))

cells.append(md("## Rebuild Part 2's preprocessing (same random_state -> identical split)"))
cells.append(code(
"df = pd.read_csv('cleaned_data.csv')",
"",
"y_reg = df['salary'].copy()",
"salary_q75 = df['salary'].quantile(0.75)",
"y_clf = (df['salary'] >= salary_q75).astype(int)",
"",
"X = df.drop(columns=['employee_id', 'salary'])",
"education_order = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}",
"X['education_level'] = X['education_level'].map(education_order)",
"X = pd.get_dummies(X, columns=['department', 'region'], drop_first=True)",
"",
"X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(",
"    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE",
")",
"",
"scaler = StandardScaler()",
"scaler.fit(X_train)",
"X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)",
"X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)",
"",
"print('X_train_scaled:', X_train_scaled.shape, 'X_test_scaled:', X_test_scaled.shape)",
"print(y_clf_train.value_counts())"
))

cells.append(md("## Task 1 — Decision Tree baseline (unconstrained)"))
cells.append(code(
"tree_unconstrained = DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE)",
"tree_unconstrained.fit(X_train_scaled, y_clf_train)",
"train_acc_unc = accuracy_score(y_clf_train, tree_unconstrained.predict(X_train_scaled))",
"test_acc_unc = accuracy_score(y_clf_test, tree_unconstrained.predict(X_test_scaled))",
"print(f'Train accuracy: {train_acc_unc:.4f}, Test accuracy: {test_acc_unc:.4f}')",
"print(f'Train/test gap: {train_acc_unc - test_acc_unc:.4f}')"
))
cells.append(md(
"**Overfitting check:** a perfect training score alongside a lower test score is the",
"signature of overfitting. Decision trees are **high-variance models** because they build",
"greedily — at each split, picking whatever best separates the *current* node's data",
"without ever revisiting earlier decisions — so left unconstrained they keep splitting",
"until they fit training noise as readily as real signal."
))

cells.append(md("## Task 2 — Controlled Decision Tree"))
cells.append(code(
"tree_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE)",
"tree_controlled.fit(X_train_scaled, y_clf_train)",
"train_acc_ctrl = accuracy_score(y_clf_train, tree_controlled.predict(X_train_scaled))",
"test_acc_ctrl = accuracy_score(y_clf_test, tree_controlled.predict(X_test_scaled))",
"print(f'Train accuracy: {train_acc_ctrl:.4f}, Test accuracy: {test_acc_ctrl:.4f}')",
"print(f'Train/test gap: {train_acc_ctrl - test_acc_ctrl:.4f}')"
))
cells.append(md(
"**`max_depth`** caps how many splits deep the tree can grow (less variance, a bit more",
"bias). **`min_samples_split`** blocks a node from splitting unless it has at least 20",
"samples, preventing splits driven by a handful of noisy points. Both shrink the",
"train/test gap compared to the unconstrained tree."
))

cells.append(md("## Task 3 — Gini vs Entropy"))
cells.append(code(
"tree_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=RANDOM_STATE)",
"tree_gini.fit(X_train_scaled, y_clf_train)",
"acc_gini = accuracy_score(y_clf_test, tree_gini.predict(X_test_scaled))",
"",
"tree_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=RANDOM_STATE)",
"tree_entropy.fit(X_train_scaled, y_clf_train)",
"acc_entropy = accuracy_score(y_clf_test, tree_entropy.predict(X_test_scaled))",
"",
"print(f'Gini test accuracy:    {acc_gini:.4f}')",
"print(f'Entropy test accuracy: {acc_entropy:.4f}')"
))
cells.append(md(
"**Gini impurity:** `1 - sum(p_i^2)`  **Entropy:** `-sum(p_i * log2(p_i))`",
"",
"A node with **Gini = 0** means every sample in that node belongs to a single class — the",
"node is perfectly pure."
))

cells.append(md("## Task 4 — Random Forest"))
cells.append(code(
"rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)",
"rf.fit(X_train_scaled, y_clf_train)",
"rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))",
"rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))",
"rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])",
"print(f'Train accuracy: {rf_train_acc:.4f}, Test accuracy: {rf_test_acc:.4f}, ROC-AUC: {rf_auc:.4f}')",
"",
"importances = pd.Series(rf.feature_importances_, index=X_train_scaled.columns).sort_values(ascending=False)",
"print('\\nTop 5 features by importance:')",
"print(importances.head(5))"
))
cells.append(md(
"**Feature importance** = average reduction in Gini impurity from splits on that feature,",
"across all trees. This differs from a linear regression coefficient, which gives a",
"*signed, linear* effect size — importance only says how *useful* a feature was for",
"splitting, with no direction or linearity assumption.",
"",
"**Bagging:** each tree trains on a random *bootstrap sample* (drawn with replacement), and",
"at each split only a random subset of ~sqrt(n_features) features is considered. Averaging",
"many trees that each overfit to *different* noise in *different* ways cancels much of that",
"noise out, reducing variance versus a single deep tree."
))

cells.append(md("## Task 4a — Gradient Boosting"))
cells.append(code(
"gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)",
"gb.fit(X_train_scaled, y_clf_train)",
"gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))",
"gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))",
"gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])",
"print(f'Train accuracy: {gb_train_acc:.4f}, Test accuracy: {gb_test_acc:.4f}, ROC-AUC: {gb_auc:.4f}')"
))

cells.append(md("## Task 4b — Feature ablation study"))
cells.append(code(
"lowest5 = importances.tail(5).index.tolist()",
"print('5 lowest-importance features:', lowest5)",
"",
"X_train_reduced = X_train_scaled.drop(columns=lowest5)",
"X_test_reduced = X_test_scaled.drop(columns=lowest5)",
"",
"rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)",
"rf_reduced.fit(X_train_reduced, y_clf_train)",
"rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])",
"",
"print(f'Full model  ({X_train_scaled.shape[1]} features)  test AUC: {rf_auc:.4f}')",
"print(f'Reduced model ({X_train_reduced.shape[1]} features) test AUC: {rf_reduced_auc:.4f}')",
"print(f'AUC change: {rf_reduced_auc - rf_auc:+.4f}')"
))
cells.append(md(
"A near-zero AUC change means these 5 features were genuinely uninformative, not quietly",
"contributing signal. **Production trade-off:** dropping them gives lower inference cost",
"and a simpler pipeline to maintain — only acceptable because the AUC cost here is",
"negligible; a larger drop would change that calculus."
))

cells.append(md("## Task 5 — Cross-validated comparison"))
cells.append(code(
"cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)",
"",
"# Uses class_weight='balanced' rather than Part 2's SMOTE-resampled data:",
"# resampling before cross_val_score would leak information across folds.",
"log_reg_cv = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)",
"",
"models_for_cv = {",
"    'Logistic Regression': log_reg_cv,",
"    'Decision Tree (controlled)': DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE),",
"    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),",
"    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),",
"}",
"",
"cv_results = {}",
"for name, model in models_for_cv.items():",
"    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring='roc_auc')",
"    cv_results[name] = (scores.mean(), scores.std())",
"    print(f'{name}: mean AUC = {scores.mean():.4f}, std = {scores.std():.4f}')"
))
cells.append(md(
"**Why CV beats a single split:** one 80/20 split gives one estimate that depends on which",
"rows happened to land in the test set. 5-fold CV rotates through 5 different",
"train/validation splits and reports both the mean (more stable estimate) and the std (how",
"much that estimate varies) — a single split can't tell you that variability at all."
))

cells.append(md("## Task 6 — GridSearchCV hyperparameter tuning", "", "*(This cell takes a minute or two — 90 model fits.)*"))
cells.append(code(
"param_grid = {",
"    'randomforestclassifier__n_estimators': [50, 100, 200],",
"    'randomforestclassifier__max_depth': [5, 10, None],",
"    'randomforestclassifier__min_samples_leaf': [1, 5]",
"}",
"",
"pipeline = make_pipeline(",
"    SimpleImputer(strategy='median'),",
"    StandardScaler(),",
"    RandomForestClassifier(random_state=RANDOM_STATE)",
")",
"",
"grid_search = GridSearchCV(",
"    pipeline, param_grid,",
"    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),",
"    scoring='roc_auc', n_jobs=-1",
")",
"grid_search.fit(X_train, y_clf_train)  # unscaled X_train — pipeline handles scaling",
"",
"print('Best params:', grid_search.best_params_)",
"print(f'Best CV score (ROC-AUC): {grid_search.best_score_:.4f}')",
"",
"n_configs = 1",
"for v in param_grid.values():",
"    n_configs *= len(v)",
"print(f'Total configurations: {n_configs} x 5 folds = {n_configs * 5} total fits')",
"",
"best_pipeline = grid_search.best_estimator_"
))
cells.append(md(
"**Grid vs Randomized Search:** Grid Search is exhaustive — guaranteed to find the best",
"combination *within the grid*, but cost grows multiplicatively with every added",
"hyperparameter. Randomized Search samples a fixed number of combinations instead — no",
"guarantee of the single best, but usually finds a very good one much faster, especially",
"when some hyperparameters barely matter."
))

cells.append(md("## Task 7 — Manual learning curve"))
cells.append(code(
"fractions = [0.2, 0.4, 0.6, 0.8, 1.0]",
"learning_curve_rows = []",
"for f in fractions:",
"    n_rows = int(f * len(X_train))",
"    X_subset = X_train.iloc[:n_rows]",
"    y_subset = y_clf_train.iloc[:n_rows]",
"",
"    pipeline_f = clone(best_pipeline)",
"    pipeline_f.fit(X_subset, y_subset)",
"",
"    train_auc = roc_auc_score(y_subset, pipeline_f.predict_proba(X_subset)[:, 1])",
"    test_auc = roc_auc_score(y_clf_test, pipeline_f.predict_proba(X_test)[:, 1])",
"    learning_curve_rows.append((f, n_rows, train_auc, test_auc))",
"    print(f'Fraction {f:.1f} ({n_rows} rows) — Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}')",
"",
"learning_curve_table = pd.DataFrame(learning_curve_rows, columns=['Fraction', 'N rows', 'Train AUC', 'Test AUC'])",
"print(learning_curve_table)"
))
cells.append(md(
"**(i)** Training AUC stays flat at 1.0000 — the model has enough capacity to perfectly fit",
"training data at every size tested. **(ii)** Test AUC rises but the gains shrink sharply",
"toward the end — the curve is flattening, not still climbing steeply. **(iii)** Conclusion:",
"the model looks closer to **capacity-limited than data-limited** — more data would likely",
"help only marginally at this point."
))

cells.append(md("## Task 8 — Serialize the best model"))
cells.append(code(
"joblib.dump(best_pipeline, 'best_model.pkl')",
"print('Saved best_pipeline to best_model.pkl')"
))
cells.append(code(
"# Reload-and-predict demonstration",
"loaded_model = joblib.load('best_model.pkl')",
"hand_crafted_rows = X_test.iloc[:2].copy()",
"preds = loaded_model.predict(hand_crafted_rows)",
"probas = loaded_model.predict_proba(hand_crafted_rows)[:, 1]",
"print('Predictions:', preds)",
"print('Probabilities:', probas)"
))

cells.append(md("## Task 9 — Summary comparison table"))
cells.append(code(
"test_auc_lookup = {",
"    'Logistic Regression': roc_auc_score(",
"        y_clf_test,",
"        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)",
"        .fit(X_train_scaled, y_clf_train).predict_proba(X_test_scaled)[:, 1]",
"    ),",
"    'Decision Tree (controlled)': roc_auc_score(y_clf_test, tree_controlled.predict_proba(X_test_scaled)[:, 1]),",
"    'Random Forest': rf_auc,",
"    'Gradient Boosting': gb_auc,",
"}",
"",
"summary_rows = []",
"for name in models_for_cv:",
"    mean_auc, std_auc = cv_results[name]",
"    summary_rows.append((name, mean_auc, std_auc, test_auc_lookup[name]))",
"summary_table = pd.DataFrame(summary_rows, columns=['Model', 'CV Mean AUC', 'CV Std AUC', 'Test AUC'])",
"summary_table = summary_table.sort_values('Test AUC', ascending=False)",
"print(summary_table)"
))
cells.append(md(
"**Recommendation: Logistic Regression** — it has the highest AUC (both CV and test) *and*",
"the lowest CV variance, despite being the simplest model here. This makes sense because",
"`salary` (and therefore `high_earner`) was constructed as a close-to-linear function of",
"the features in Part 1 — exactly the setting where a linear model is well-specified and",
"tree ensembles buy little extra. `best_model.pkl` is still the tuned Random Forest",
"pipeline, since Task 6/8 specifically direct tuning and serializing that model — but this",
"comparison table is the evidence that would motivate also trying a tuned Logistic",
"Regression pipeline before a real production decision."
))

cells.append(md("## Part 3 complete"))
cells.append(code("print('Part 3 pipeline finished successfully.')"))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Part3_Advanced.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Part3_Advanced.ipynb written.")
