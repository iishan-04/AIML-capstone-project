# Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate

## Label definitions

- **`y_reg` (regression target):** `salary` — the continuous column from Part 1's cleaned dataset.
- **`y_clf` (classification target): `high_earner`**, defined as `(salary >= salary.quantile(0.75)).astype(int)` — the **top 25% of earners** vs everyone else.

**Why top-quartile instead of a median split:** a median split (`salary > salary.median()`) produces an almost exactly 50/50 class balance on this dataset, which would make the imbalance-handling requirement in Task 5 impossible to genuinely demonstrate. The top-quartile threshold is still literally "binarizing `y_reg`" as the brief specifies, just at a business-realistic cutoff (identifying a minority high-earning segment) that produces a real ~25/75 imbalance worth handling properly.

**Feature matrix `X`:** all columns except `employee_id` (an identifier, not a feature) and `salary`. `salary` is dropped for **both** tasks — it's the direct source of `y_clf`, so leaving it in the features would let the classifier trivially threshold that one column for ~100% accuracy, making every classification metric in this Part meaningless. `X` starts with 9 columns and grows to 16 after one-hot encoding (see Task 2).

---

## Task 2 — Categorical encoding

- **`education_level` → ordinal label encoding:** `{'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}`. Justified because education level has a genuine real-world ordering — each level represents strictly more education than the last — so mapping to increasing integers preserves information a linear model can use directly (e.g. a monotonic salary relationship), rather than discarding the ordering.
- **`department`, `region` → one-hot encoding, first category dropped** (`pd.get_dummies(..., drop_first=True)`). Neither has an inherent ranking — "Engineering" isn't numerically greater than "Sales." Label-encoding them would impose a false ordinal relationship the model would wrongly treat as meaningful (implying, say, HR < Engineering < Finance in some scaled sense). One-hot encoding avoids this by giving each category an independent binary column; dropping the first category avoids multicollinearity (the dropped category becomes the implicit reference level when all its dummy columns are 0).

Final `X`: 16 columns (7 numeric/ordinal + 5 department dummies + 4 region dummies for a base of 6 departments / 5 regions).

---

## Task 3 — Leak-free split and scaling

`X`, `y_reg`, and `y_clf` were split together (`train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)`) so the same 1,600 training / 400 test rows are used for both tasks. `StandardScaler` was **fit only on `X_train`**, then used to transform both `X_train` and `X_test`.

**Why fitting on the full dataset would be leakage:** the scaler's mean and standard deviation would then be computed using the test set's own values. Every "unseen" test row would be scaled using statistics partly derived from itself, which means information about the test set has leaked into how the training data is represented before the model has even been trained. This gives an optimistically biased picture of how well the model would perform on genuinely new data.

---

## Task 4 — Regression: Linear Regression vs Ridge

**Linear Regression:** MSE = **24,982,139.95**, R² = **0.9540**

**Top 3 coefficients by absolute value** (on standardized features):

| Feature | Coefficient |
|---|---|
| years_experience | +15,241.60 |
| education_level | +7,786.82 |
| age | +5,362.01 |

**Interpretation:** these coefficients are on standardized (z-scored) features, so each one represents "the change in predicted salary for a one-standard-deviation increase in that feature, holding others constant." A large **positive** coefficient (e.g. `years_experience`) means more of that feature is associated with higher predicted salary; a large **negative** coefficient (e.g. `department_Support`, −5,349.73) means belonging to that category is associated with *lower* predicted salary relative to the dropped reference department (Engineering).

**Ridge Regression (alpha=1.0):** MSE = **24,979,257.44**, R² = **0.9540**

| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | 24,982,139.95 | 0.9540 |
| Ridge (alpha=1.0) | 24,979,257.44 | 0.9540 |

**Why Ridge barely differs here, and what alpha controls:** `alpha` controls the strength of the L2 penalty added to the loss function — it shrinks coefficients toward zero to reduce variance, at the cost of a small amount of bias, which is especially useful when features are correlated or the model is overfitting. On this dataset the relationship between features and salary is strong and close to linear by construction, and `alpha=1.0` is a relatively mild penalty relative to the size of the signal, so OLS and Ridge land on almost identical coefficients and near-identical MSE/R². Ridge would diverge more noticeably from OLS with a larger `alpha`, or on noisier/more collinear data.

---

## Task 5 — Classification: Logistic Regression

**Imbalance check:** `y_clf_train.value_counts()` → class 0 (not high earner): 1,194, class 1 (high earner): 406 → **minority class = 25.4%** of training data, below the 35% threshold, so imbalance handling was applied.

**Method chosen: SMOTE** (`imblearn.over_sampling.SMOTE`), because it synthesizes new, interpolated minority-class examples rather than duplicating exact existing rows, which reduces the risk of the model simply overfitting to repeated points the way naive duplication can. *(Note: this sandbox environment has no internet access to install `imbalanced-learn`, so the code includes an automatic fallback to manual random oversampling with replacement if SMOTE's package isn't available — that fallback is what actually ran to produce the numbers below. On your machine, run `pip install imbalanced-learn` first and the real SMOTE path will be used automatically.)*

Class counts before → after resampling: **1,194 / 406 → 1,194 / 1,194** (minority class brought up to match the majority).

**Results on the held-out test set (400 rows):**

Confusion matrix:
```
                Predicted 0   Predicted 1
Actual 0            284           22
Actual 1              3           91
```

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (not high earner) | 0.99 | 0.93 | 0.96 |
| 1 (high earner) | 0.81 | 0.97 | 0.88 |

Accuracy: 0.94 · **ROC AUC: 0.9937**

**Formulas:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

**Which metric matters more here:** for identifying "high earners" (e.g. for a targeted retention or promotion-review program), **recall matters more** — missing a genuine high earner (a false negative) means the business fails to flag someone it should be paying special attention to, which is typically costlier than a false positive (flagging someone who turns out not to be a high earner, which just means one extra review). This model already achieves strong recall (0.97) for the high-earner class.

**What AUC = 0.9937 means:** the model separates the two classes almost perfectly — if you picked one random true high-earner and one random true non-high-earner, the model would assign the high-earner a higher predicted probability about 99.4% of the time. This near-ceiling AUC makes sense given how the label was constructed: `high_earner` is a direct function of `salary`, and `salary` itself has a strong, largely linear relationship with `years_experience`, `education_level`, and `department` (Part 1's correlation heat map already showed `years_experience`–`salary` at r=0.878) — so the same features that predict salary well in Task 4 separate the high/low earner classes cleanly here too.

---

## Task 5b — Decision-threshold sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.762 | 0.989 | 0.861 |
| 0.40 | 0.786 | 0.979 | 0.872 |
| 0.50 | 0.805 | 0.968 | 0.879 |
| 0.60 | 0.858 | 0.968 | 0.910 |
| **0.70** | **0.909** | **0.957** | **0.933** |

**Formulas:** Precision = TP / (TP + FP), Recall = TP / (TP + FN)

**F1-maximizing threshold: 0.70** (F1 = 0.933).

**Which metric to prioritize:** as discussed above, recall is more important for this task (missing a true high earner is costlier than a false alarm). Even at the F1-optimal threshold of 0.70, recall stays high (0.957), so in this specific case the F1-maximizing threshold also happens to be a good choice for the business priority — but if recall needed to be pushed even higher, **lowering the threshold** (e.g. to 0.30) would raise recall to 0.989 at the cost of precision dropping to 0.762 (more false positives — more people incorrectly flagged as high earners, each costing a wasted review). The choice depends on how expensive a false positive review actually is versus a missed high earner.

---

## Task 6 — Regularization experiment (C=1.0 vs C=0.01)

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.805 | 0.968 | 0.9937 |
| C=0.01 (strong regularization) | 0.715 | 0.989 | 0.9884 |

**What `C` controls:** `C` is the *inverse* of regularization strength in scikit-learn's `LogisticRegression` — a **smaller** `C` means a **stronger** L2 penalty on the coefficients, shrinking them more aggressively toward zero. Reducing `C` from 1.0 to 0.01 slightly **worsened** performance here: precision dropped from 0.805 to 0.715 and AUC dropped from 0.9937 to 0.9884, though recall actually improved slightly (0.968 → 0.989). This pattern — heavier regularization trading precision for recall while lowering overall discriminative power (AUC) — makes sense because strong regularization pulls the decision boundary toward flatter, more conservative predictions, which here means predicting "high earner" more liberally (catching more true positives, but also more false positives).

---

## Task 6b — Bootstrap confidence interval for the AUC difference

500 bootstrap samples were drawn (with replacement) from the 400-row test set; for each sample, the AUC of the C=1.0 model and the C=0.01 model were both recomputed and the difference (C=1.0 minus C=0.01) recorded.

- **Mean AUC difference:** 0.0053
- **95% CI:** [0.0006, 0.0123]
- **CI excludes zero: Yes**

**Interpretation:** since the 95% confidence interval for the AUC difference (C=1.0 minus C=0.01) lies entirely above zero, the C=1.0 model's advantage over the C=0.01 model is **statistically reliable across resamples of the test set** — it isn't just a fluke of this particular train/test split. That said, the interval is narrow and close to zero (0.0006 to 0.0123), so while the difference is real, it's also small in absolute terms — both models perform strongly, and the practical cost of choosing C=0.01 (e.g. for its slightly better recall) would be a very small, well-quantified AUC trade-off.

---

## Files in this folder

- `ml_pipeline.py` — full Task 1–6b pipeline; run top-to-bottom, needs `Part1`'s `cleaned_data.csv` (already copied into this folder)
- `cleaned_data.csv` — copied from Part 1 (unchanged)
- `plots/roc_curve.png` — ROC curve with AUC annotated
- `run_log.txt` — full console output from the last verified run

**To reproduce:** `python3 ml_pipeline.py` (optionally `pip install imbalanced-learn` first so the real SMOTE path runs instead of the manual-oversampling fallback)
