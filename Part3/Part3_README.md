# Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

This script rebuilds the exact same preprocessing and train/test split as Part 2 (same `random_state=42`, same encoding, same 80/20 split), so it can be run and graded independently while still landing on identical `X_train_scaled` / `X_test_scaled` / `y_clf_train` / `y_clf_test` arrays. Classification target throughout: `high_earner` (top 25% of salaries, as defined in Part 2).

---

## Task 1 — Decision Tree baseline (unconstrained)

| | Train accuracy | Test accuracy |
|---|---|---|
| Unconstrained tree (`max_depth=None`) | **1.0000** | 0.9075 |

**Train/test gap: 0.0925.** A perfect training score alongside a noticeably lower test score is the textbook signature of overfitting — the tree has memorized the training data (growing branches deep enough to isolate individual noisy training points) rather than learning the general pattern. Decision trees are described as **high-variance models** because they build greedily: at each split, the algorithm picks whatever feature/threshold best separates the *current* node's data, without ever revisiting or correcting earlier decisions. Left unconstrained, this greedy process keeps splitting until nodes are pure (or tiny), which fits training noise as readily as it fits real signal.

---

## Task 2 — Controlled Decision Tree

| | Train accuracy | Test accuracy |
|---|---|---|
| Controlled tree (`max_depth=5, min_samples_split=20`) | 0.9363 | 0.9025 |

**Train/test gap: 0.0338** — much smaller than the unconstrained tree's 0.0925 gap, at the cost of a small drop in raw training accuracy.

- **`max_depth`** caps how many splits deep the tree can grow, directly limiting how finely it can carve up the feature space — this trades away some ability to fit intricate patterns (a bit more bias) in exchange for not chasing noise (less variance).
- **`min_samples_split`** blocks a node from splitting further unless it has at least 20 samples, which prevents the tree from creating splits based on a handful of possibly-noisy points.

Both constraints pull the train and test accuracy closer together — exactly the overfitting-control effect they're designed for.

---

## Task 3 — Gini vs Entropy

| Criterion | Test accuracy |
|---|---|
| Gini | 0.9025 |
| Entropy | 0.9075 |

Nearly identical — a common outcome, since both are just different ways of measuring node "impurity" and usually agree on which splits are best.

- **Gini impurity:** `1 − Σ pᵢ²` (sum over each class's proportion in the node)
- **Entropy:** `−Σ pᵢ log₂(pᵢ)`

A node with **Gini = 0** means every sample in that node belongs to a single class — the node is perfectly "pure," and no further split could improve it.

---

## Task 4 — Random Forest

`n_estimators=100, max_depth=10, random_state=42`

| Train accuracy | Test accuracy | ROC-AUC |
|---|---|---|
| 0.9981 | 0.9225 | 0.9764 |

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| years_experience | 0.4135 |
| age | 0.2477 |
| education_level | 0.0842 |
| performance_score | 0.0584 |
| monthly_hours | 0.0490 |

**How Random Forest computes feature importance:** for each feature, it averages how much that feature reduces Gini impurity every time it's used for a split, across every tree in the forest. This differs fundamentally from a linear regression coefficient — a coefficient tells you the *direction and magnitude* of a linear relationship (holding other features fixed), while a Gini-based importance only tells you *how useful* a feature was for splitting the data, with no sign and no assumption of linearity. A feature could have high importance in a Random Forest purely by being useful for non-linear splits, even with a near-zero linear correlation to the target.

**Bagging concept:** each of the 100 trees in the forest is trained on a different **bootstrap sample** — a random sample of the training rows drawn *with replacement*, so each tree sees a slightly different (and imperfect) view of the data. On top of that, at every single split, only a random subset of √(number of features) features is even considered as a candidate — so different trees end up relying on different features too. Averaging predictions across many trees that each overfit to *different* noise in *different* ways cancels a lot of that noise out, which is why the forest's test accuracy (0.9225) is meaningfully more stable than a single deep tree's, even though any individual tree in the forest is itself high-variance.

---

## Task 4a — Gradient Boosting

`n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42`

| Train accuracy | Test accuracy | ROC-AUC |
|---|---|---|
| 0.9744 | 0.9500 | 0.9881 |

Gradient Boosting outperforms the Random Forest here on both test accuracy (0.9500 vs 0.9225) and AUC (0.9881 vs 0.9764) — consistent with boosting's core idea of building trees *sequentially*, where each new (shallow, `max_depth=3`) tree is trained specifically to correct the errors of the trees before it, rather than averaging many independent trees.

---

## Task 4b — Feature ablation study

**5 lowest-importance features (from the Random Forest):** `region_North`, `region_West`, `department_Marketing`, `region_South`, `region_East`

| Model | # features | Test AUC |
|---|---|---|
| Full | 16 | 0.9764 |
| Reduced (5 lowest dropped) | 11 | 0.9755 |

**AUC change: −0.0009** — essentially no change. This is strong evidence the 5 removed features were **genuinely uninformative** rather than quietly contributing signal the importance scores understated — all five are `region` dummy variables (plus one `department` dummy), consistent with Part 1's correlation analysis, which never found `region` meaningfully related to `salary` in the first place.

**Production implication:** since dropping these 5 features costs essentially nothing in predictive power, deploying the **11-feature reduced model** would be a reasonable choice in production — fewer features means lower inference cost, a smaller/simpler feature pipeline to maintain, and one less place for upstream data quality issues (e.g. a missing `region` field) to break the model. This trade-off is only acceptable because the AUC drop (0.0009) is negligible; if it had been, say, 0.02–0.03, the calculus would shift toward keeping the full feature set.

---

## Task 5 — Cross-validated comparison (5-fold StratifiedKFold, ROC-AUC)

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | **0.9887** | 0.0027 |
| Decision Tree (controlled) | 0.9427 | 0.0109 |
| Random Forest | 0.9709 | 0.0024 |
| Gradient Boosting | 0.9803 | 0.0044 |

*(Note: the Logistic Regression compared here uses `class_weight='balanced'` rather than Part 2's SMOTE-resampled training data — resampling the data before handing it to `cross_val_score` would leak information across folds, since synthetic minority points get created from rows that could end up split across train and validation folds. `class_weight` avoids this by being safely embeddable directly in the estimator.)*

**Why cross-validation is more reliable than a single train-test split:** a single 80/20 split gives one estimate of generalization performance that depends on exactly which rows happened to land in the test set — a "lucky" or "unlucky" split can over- or under-state true performance. 5-fold cross-validation instead rotates through 5 different train/validation splits and averages the result, giving both a more stable estimate (the mean) and a sense of how much that estimate varies (the std) — a model with a high mean AUC but a large std is less trustworthy than one with a similar mean and small std, and a single train-test split can't tell you that difference at all.

---

## Task 6 — GridSearchCV hyperparameter tuning

**Grid:**
```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```
**Total configurations evaluated:** 3 × 3 × 2 = **18** combinations × 5 CV folds = **90 total model fits**.

**Best params:** `{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 200}`
**Best CV score (ROC-AUC):** **0.9731**

**Grid Search vs Randomized Search trade-off:** Grid Search is exhaustive — it tries every combination in the grid, guaranteeing you find the best combination *within that grid*, but the cost grows multiplicatively with every hyperparameter you add (an 18-combination grid becomes 180 combinations if you add one more parameter with 10 values). Randomized Search instead samples a fixed number of random combinations from the specified ranges — it won't guarantee finding the single best combination, but in practice it finds a very good one far faster, especially when some hyperparameters barely matter (Randomized Search doesn't waste fits exhaustively varying a parameter that has little effect, the way Grid Search does).

---

## Task 7 — Manual learning curve

Pipeline: best GridSearchCV pipeline from Task 6.

| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 (320 rows) | 1.0000 | 0.9549 |
| 0.4 (640 rows) | 1.0000 | 0.9666 |
| 0.6 (960 rows) | 1.0000 | 0.9715 |
| 0.8 (1280 rows) | 1.0000 | 0.9738 |
| 1.0 (1600 rows) | 1.0000 | 0.9744 |

**(i) Training AUC:** stays flat at a perfect 1.0000 across every fraction — it does **not** decrease as the training set grows, which is actually the *un*-expected pattern for a genuinely high-variance model overfitting small datasets (normally you'd expect training AUC to start near-perfect on tiny datasets and ease down slightly as more data forces the model to generalize a bit). Here, the unconstrained Random Forest (`max_depth=None`) is expressive enough to perfectly separate the training data at every single size tested, even at full size — a sign the model has more capacity than it strictly needs for this problem.

**(ii) Test AUC:** rises steadily from 0.9549 → 0.9744 as training data grows, but the gains shrink sharply toward the end (+0.0117 from 20%→40%, but only +0.0006 from 80%→100%) — the curve is **flattening out**, not still climbing steeply.

**(iii) Conclusion:** the model looks closer to **capacity/ceiling-limited than data-limited** at this point. Test AUC has nearly plateaued by 80–100% of the training data (a gain of only 0.0006 for the last 20% of data added), while training AUC never moved off a perfect 1.0000 at all — more data would likely yield only marginal further improvement; a meaningfully better result would more likely come from better features or a different model class (consistent with Logistic Regression actually outperforming this Random Forest overall — see Task 9).

---

## Task 8 — Model serialization

The tuned Random Forest pipeline (`best_pipeline` from Task 6's `GridSearchCV.best_estimator_`) was saved with `joblib.dump(best_pipeline, 'best_model.pkl')`. File size: **4.3 MB** (well under the 100 MB limit, committed directly to the repo).

Reload-and-predict demonstration (also in the script):
```python
import joblib
loaded_model = joblib.load('best_model.pkl')
hand_crafted_rows = X_test.iloc[:2]        # 2 hand-picked feature rows
preds = loaded_model.predict(hand_crafted_rows)
probas = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
```
Output: predictions `[0, 0]`, probabilities `[0.005, 0.000]` — both test rows correctly predicted as "not a high earner" with high confidence.

---

## Task 9 — Summary comparison table and recommendation

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| **Logistic Regression** | **0.9887** | 0.0027 | **0.9941** |
| Gradient Boosting | 0.9803 | 0.0045 | 0.9881 |
| Random Forest | 0.9709 | 0.0024 | 0.9764 |
| Decision Tree (controlled) | 0.9427 | 0.0109 | 0.9559 |

**Recommendation: Logistic Regression.** This is a the simplest model in the comparison has the *highest* AUC, both in cross-validation and on the held-out test set, and the lowest CV variance (std = 0.0027, tightest of all four). This result is expected because `salary` (and therefore `high_earner`) is a close-to-linear function of `years_experience`, `education_level`, `age`, and `department`, with only moderate noise — exactly the setting where a linear model is well-specified and the extra flexibility of tree ensembles buys little to nothing. Logistic Regression is also faster to train, easier to interpret (coefficients directly show each feature's effect), and cheaper to serve in production than a 200-tree Random Forest or a boosted ensemble.

**Why `best_model.pkl` is still the tuned Random Forest pipeline, not the Logistic Regression:** Task 6/8 of this brief specifically direct tuning and serializing the Random Forest pipeline via `GridSearchCV`, so that's what's saved. In a real client engagement, the honest next step (flagged here rather than quietly skipped) would be to also wrap Logistic Regression in the same `GridSearchCV`-tuned pipeline and compare before committing to a final production model — this comparison table is exactly the evidence that would motivate that follow-up.

---

## Files in this folder

- `advanced_ml_pipeline.py` — full Task 1–9 pipeline; run top-to-bottom, needs `cleaned_data.csv` (already copied into this folder)
- `cleaned_data.csv` — copied from Part 1 (unchanged)
- `best_model.pkl` — serialized tuned Random Forest pipeline (GridSearchCV winner)
- `run_log.txt` — full console output from the last verified run

**To reproduce:** `python3 advanced_ml_pipeline.py`
