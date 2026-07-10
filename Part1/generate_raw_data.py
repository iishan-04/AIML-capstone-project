"""
generate_raw_data.py
---------------------
Generates a realistic synthetic "Employee Compensation & Performance" dataset
that stands in for a client-supplied raw data file. This is used because the
project environment has no internet access to download an external dataset;
the generation logic is fully documented here for transparency and
reproducibility (random_state fixed throughout).

Dataset satisfies the brief's minimum requirements:
  - 2000 rows (>= 500)
  - 7 numeric columns (>= 5): age, years_experience, performance_score,
    monthly_hours, distance_from_office_km, num_projects, bonus_pct, salary
  - 3 categorical columns (>= 2): education_level (ordinal), department
    (nominal), region (nominal)
  - salary = continuous regression target
  - salary binarized at median = classification target (Part 2)

Intentional data-quality issues (for Part 1 cleaning tasks):
  - years_experience stored as object/string dtype (needs correction)
  - department stored as plain object strings (repetitive -> category dtype)
  - missingness: performance_score (~7%), monthly_hours (~5%),
    distance_from_office_km (~6%), bonus_pct (~35%, deliberately > 20%)
  - 15 exact duplicate rows appended
  - distance_from_office_km is exponentially distributed -> strong positive
    skew and genuine IQR outliers
  - age and years_experience are strongly correlated by construction (both
    driven by "career start age"), used later for the alternative-explanation
    discussion in the correlation heat map task
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
N = 2000

# ---- categorical columns -----------------------------------------------
education_levels = ["High School", "Bachelors", "Masters", "PhD"]
education_probs = [0.25, 0.40, 0.25, 0.10]
education_level = rng.choice(education_levels, size=N, p=education_probs)

departments = ["Sales", "Engineering", "Marketing", "HR", "Finance", "Support"]
department_probs = [0.22, 0.28, 0.14, 0.10, 0.14, 0.12]
department = rng.choice(departments, size=N, p=department_probs)

regions = ["North", "South", "East", "West", "Central"]
region = rng.choice(regions, size=N)

# ---- numeric columns, built with real relationships ---------------------
age = rng.integers(22, 61, size=N).astype(float)

# years_experience is correlated with age (career start age varies by person)
career_start_age = rng.normal(23, 2.5, size=N)
years_experience = np.clip(age - career_start_age + rng.normal(0, 1.5, size=N), 0, 40)

performance_score = np.clip(rng.normal(3.2, 0.7, size=N), 1, 5)

# right-skewed via gamma distribution (occasional heavy-overtime employees)
monthly_hours = rng.gamma(shape=6.0, scale=25.0, size=N)

# strongly right-skewed via exponential distribution -> designed to be the
# most-skewed numeric column, with genuine IQR outliers
distance_from_office_km = rng.exponential(scale=8.0, size=N)

num_projects = rng.poisson(4, size=N).clip(0, 15)

bonus_pct = np.clip(rng.normal(10 + performance_score * 2, 4, size=N), 0, 30)

education_bump = {"High School": 0, "Bachelors": 8000, "Masters": 15000, "PhD": 25000}
department_bump = {"Sales": 0, "Engineering": 12000, "Marketing": 3000,
                    "HR": -2000, "Finance": 9000, "Support": -4000}

salary = (
    30000
    + years_experience * 1800
    + np.array([education_bump[e] for e in education_level])
    + np.array([department_bump[d] for d in department])
    + performance_score * 3000
    + rng.normal(0, 4000, size=N)
)
salary = np.round(salary, 2)

years_experience_str = np.round(years_experience, 1).astype(str)
# Simulate a messy client export: ~1.5% of entries recorded as "unknown"
# instead of a number. Deliberately NOT one of pandas' default na_values
# strings (unlike "N/A"/"NA"/"NULL"), so the column genuinely loads as
# object dtype and pd.to_numeric(errors='coerce') has real work to do in
# Task 4, rather than being silently absorbed as a null at load time.
na_idx = rng.choice(N, size=int(N * 0.015), replace=False)
years_experience_str[na_idx] = "unknown"

df = pd.DataFrame({
    "employee_id": np.arange(1, N + 1),
    "age": age,
    "years_experience": years_experience_str,  # dtype issue: stored as object, some "N/A" placeholders
    "education_level": education_level,
    "department": department,
    "region": region,
    "performance_score": np.round(performance_score, 2),
    "monthly_hours": np.round(monthly_hours, 1),
    "distance_from_office_km": np.round(distance_from_office_km, 2),
    "num_projects": num_projects,
    "bonus_pct": np.round(bonus_pct, 2),
    "salary": salary,
})

# ---- inject missingness --------------------------------------------------
def inject_nulls(frame, col, frac):
    idx = rng.choice(frame.index, size=int(len(frame) * frac), replace=False)
    frame.loc[idx, col] = np.nan

inject_nulls(df, "performance_score", 0.07)
inject_nulls(df, "monthly_hours", 0.05)
inject_nulls(df, "distance_from_office_km", 0.06)
inject_nulls(df, "bonus_pct", 0.35)  # deliberately exceeds the 20% threshold

# ---- inject duplicate rows ------------------------------------------------
dupes = df.sample(n=15, random_state=RANDOM_STATE)
df = pd.concat([df, dupes], ignore_index=True)

# shuffle so duplicates aren't trivially at the tail
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

df.to_csv("data/raw_data.csv", index=False)
print(f"raw_data.csv written: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.dtypes)
