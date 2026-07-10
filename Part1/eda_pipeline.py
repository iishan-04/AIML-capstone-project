"""
eda_pipeline.py — Part 1: Data Acquisition, Cleaning, and Exploratory Analysis
================================================================================
Runs top-to-bottom, no arguments needed. Produces:
  - cleaned_data.csv
  - plots/*.png  (all 6 required visualizations)
  - printed console output for every task (also mirrored into README.md
    findings by hand after running once)

Dataset: synthetic Employee Compensation & Performance data (see
generate_raw_data.py for full generation logic and column descriptions).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
sns.set_theme(style="whitegrid")

# =========================================================================
# TASK 1 — Load and inspect
# =========================================================================
print("=" * 80)
print("TASK 1 — Load and inspect")
print("=" * 80)
df = pd.read_csv("data/raw_data.csv")
print("\nFirst 5 rows:\n", df.head())
print("\nColumn dtypes:\n", df.dtypes)
print("\nShape:", df.shape)

# =========================================================================
# TASK 2 — Null value analysis
# =========================================================================
print("\n" + "=" * 80)
print("TASK 2 — Null value analysis")
print("=" * 80)
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct.round(2)})
print("\nNull count/percentage per column:\n", null_table)

exceed_20 = null_table[null_table["null_pct"] > 20]
print("\nColumns exceeding 20% null rate:\n", exceed_20)

# Drop columns that exceed the 20% threshold (too unreliable to impute
# confidently) — documented reasoning in README.
cols_to_drop = list(exceed_20.index)
if cols_to_drop:
    print(f"\nDropping columns exceeding 20% nulls: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# Fill remaining numeric columns below 20% nulls with the median
numeric_cols_with_nulls = [c for c in df.columns
                            if df[c].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[c])]
print(f"\nNumeric columns to median-fill: {numeric_cols_with_nulls}")
for col in numeric_cols_with_nulls:
    df[col] = df[col].fillna(df[col].median())

print("\nNulls remaining after fill:\n", df.isnull().sum())

# =========================================================================
# TASK 3 — Duplicate detection and removal
# =========================================================================
print("\n" + "=" * 80)
print("TASK 3 — Duplicate detection and removal")
print("=" * 80)
n_dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {n_dupes}")
null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
df = df.drop_duplicates()
null_pct_after = (df.isnull().sum() / df.shape[0]) * 100
print(f"Rows removed: {n_dupes}. New shape: {df.shape}")
print("\nNull % before vs after duplicate removal:\n",
      pd.DataFrame({"before": null_pct_before.round(3), "after": null_pct_after.round(3)}))

# =========================================================================
# TASK 4 — Data type correction
# =========================================================================
print("\n" + "=" * 80)
print("TASK 4 — Data type correction")
print("=" * 80)
mem_before = df.memory_usage(deep=True).sum()
print(f"\nMemory usage before conversion: {mem_before} bytes ({mem_before/1024:.1f} KB)")

# years_experience is stored as object but is really numeric (a few "N/A"
# placeholders in the raw export coerce to NaN and are median-filled here,
# since this null only becomes visible after the dtype conversion itself)
print(f"\nyears_experience dtype before conversion: {df['years_experience'].dtype}")
df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce")
new_nulls = df["years_experience"].isnull().sum()
print(f"New NaNs introduced by coercion (e.g. 'N/A' placeholders): {new_nulls}")
df["years_experience"] = df["years_experience"].fillna(df["years_experience"].median())
print(f"years_experience dtype after conversion:  {df['years_experience'].dtype}")

# department is a repetitive string column -> category dtype
df["department"] = df["department"].astype("category")
df["education_level"] = df["education_level"].astype("category")
df["region"] = df["region"].astype("category")

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory usage after conversion:  {mem_after} bytes ({mem_after/1024:.1f} KB)")
print(f"Memory reduction: {mem_before - mem_after} bytes "
      f"({(1 - mem_after/mem_before)*100:.1f}% smaller)")
print("\nDtypes after correction:\n", df.dtypes)

# =========================================================================
# TASK 5 — Descriptive statistics and skewness
# =========================================================================
print("\n" + "=" * 80)
print("TASK 5 — Descriptive statistics and skewness")
print("=" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "employee_id"]
print("\nDescribe:\n", df[numeric_cols].describe())

skew_vals = df[numeric_cols].skew().sort_values(key=lambda s: s.abs(), ascending=False)
print("\nSkewness (sorted by absolute value):\n", skew_vals)
most_skewed_col = skew_vals.index[0]
print(f"\nMost skewed column: {most_skewed_col} (skew = {skew_vals.iloc[0]:.3f})")

# =========================================================================
# TASK 6 — Outlier detection with IQR
# =========================================================================
print("\n" + "=" * 80)
print("TASK 6 — Outlier detection with IQR")
print("=" * 80)
iqr_cols = ["distance_from_office_km", "monthly_hours"]
iqr_report = {}
for col in iqr_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    iqr_report[col] = dict(Q1=Q1, Q3=Q3, IQR=IQR, lower=lower, upper=upper, n_outliers=n_outliers)
    print(f"\n{col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
          f"bounds=({lower:.2f}, {upper:.2f}), outliers={n_outliers} "
          f"({n_outliers/len(df)*100:.1f}% of rows)")

# =========================================================================
# TASK 7 — Visualizations
# =========================================================================
print("\n" + "=" * 80)
print("TASK 7 — Visualizations")
print("=" * 80)

# 7.1 Line plot
plt.figure(figsize=(10, 5))
plt.plot(df.index[:200], df["salary"].values[:200], color="#2b6cb0", linewidth=1)
plt.title("Salary by Row Index (first 200 rows)")
plt.xlabel("Row Index")
plt.ylabel("Salary ($)")
plt.tight_layout()
plt.savefig("plots/01_line_salary.png", dpi=120)
plt.close()

# 7.2 Bar chart — mean salary by department
plt.figure(figsize=(9, 5))
dept_means = df.groupby("department", observed=True)["salary"].mean().sort_values(ascending=False)
dept_means.plot.bar(color="#2f855a")
plt.title("Mean Salary by Department")
plt.xlabel("Department")
plt.ylabel("Mean Salary ($)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/02_bar_mean_salary_by_department.png", dpi=120)
plt.close()

# 7.3 Histogram of most skewed column
plt.figure(figsize=(8, 5))
sns.histplot(df[most_skewed_col], bins=20, color="#c05621")
plt.title(f"Distribution of {most_skewed_col} (skew = {skew_vals.iloc[0]:.2f})")
plt.xlabel(most_skewed_col)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/03_histogram_most_skewed.png", dpi=120)
plt.close()

# 7.4 Scatter plot — years_experience vs salary
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="years_experience", y="salary", alpha=0.4, color="#2b6cb0")
plt.title("Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.tight_layout()
plt.savefig("plots/04_scatter_experience_salary.png", dpi=120)
plt.close()
scatter_corr = df["years_experience"].corr(df["salary"])
print(f"\nPearson correlation (years_experience, salary) = {scatter_corr:.3f}")

# 7.5 Box plot — salary by education_level
plt.figure(figsize=(9, 5))
order = ["High School", "Bachelors", "Masters", "PhD"]
sns.boxplot(data=df, x="education_level", y="salary", order=order,
            hue="education_level", palette="Blues", legend=False)
plt.title("Salary Distribution by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Salary ($)")
plt.tight_layout()
plt.savefig("plots/05_box_salary_by_education.png", dpi=120)
plt.close()
edu_medians = df.groupby("education_level", observed=True)["salary"].median().reindex(order)
print("\nMedian salary by education level:\n", edu_medians)

# 7.6 Correlation heat map
plt.figure(figsize=(9, 7))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heat Map (numeric columns)")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png", dpi=120)
plt.close()

corr_unstacked = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).unstack().dropna()
top_pair = corr_unstacked.abs().idxmax()
top_pair_val = corr_unstacked[top_pair]
print(f"\nHighest |correlation| pair: {top_pair} = {top_pair_val:.3f}")
print("\nFull correlation matrix:\n", corr_matrix.round(3))

# =========================================================================
# TASK 8 — Imputation strategy comparison (mean vs median) for 2 most-skewed cols
# =========================================================================
print("\n" + "=" * 80)
print("TASK 8 — Imputation strategy comparison")
print("=" * 80)
top2_skew_cols = skew_vals.index[:2].tolist()
for col in top2_skew_cols:
    mean_val = df[col].mean()
    median_val = df[col].median()
    print(f"\n{col}: mean = {mean_val:.3f}, median = {median_val:.3f}, "
          f"skew = {skew_vals[col]:.3f}")
    # (these two columns already had their nulls filled with median in Task 2;
    #  this section demonstrates the comparison methodology and confirms
    #  no nulls remain)
    print(f"Remaining nulls in {col}: {df[col].isnull().sum()}")

# =========================================================================
# TASK 8b — Spearman vs Pearson
# =========================================================================
print("\n" + "=" * 80)
print("TASK 8b — Spearman rank correlation vs Pearson")
print("=" * 80)
pearson_matrix = df[numeric_cols].corr(method="pearson")
spearman_matrix = df[numeric_cols].corr(method="spearman")
print("\nPearson matrix:\n", pearson_matrix.round(3))
print("\nSpearman matrix:\n", spearman_matrix.round(3))

diff_matrix = (spearman_matrix - pearson_matrix).abs()
diff_unstacked = diff_matrix.where(~np.eye(len(diff_matrix), dtype=bool)).unstack().dropna()
diff_unstacked = diff_unstacked[~diff_unstacked.index.duplicated()]
# keep only unique unordered pairs
seen = set()
rows = []
for (a, b), val in diff_unstacked.sort_values(ascending=False).items():
    key = frozenset([a, b])
    if key in seen:
        continue
    seen.add(key)
    rows.append((a, b, pearson_matrix.loc[a, b], spearman_matrix.loc[a, b], val))
    if len(rows) == 3:
        break

diff_table = pd.DataFrame(rows, columns=["col_a", "col_b", "pearson", "spearman", "abs_diff"])
print("\nTop 3 pairs by |Spearman - Pearson|:\n", diff_table)

# =========================================================================
# TASK 8c — Grouped aggregation
# =========================================================================
print("\n" + "=" * 80)
print("TASK 8c — Grouped aggregation")
print("=" * 80)
grouped = df.groupby("department", observed=True)["salary"].agg(["mean", "std", "count"])
print("\nSalary by department (mean, std, count):\n", grouped)
highest_mean_group = grouped["mean"].idxmax()
highest_std_group = grouped["std"].idxmax()
mean_ratio = grouped["mean"].max() / grouped["mean"].min()
print(f"\nHighest mean group: {highest_mean_group}")
print(f"Highest std group: {highest_std_group}")
print(f"Ratio of highest to lowest group mean: {mean_ratio:.3f}")

# =========================================================================
# Save cleaned dataset
# =========================================================================
df.to_csv("cleaned_data.csv", index=False)
print("\n" + "=" * 80)
print(f"cleaned_data.csv saved. Final shape: {df.shape}")
print("=" * 80)
