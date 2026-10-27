
from pathlib import Path
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT = Path("week1_outputs")
OUTPUT.mkdir(exist_ok=True)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)
df.to_csv(OUTPUT / "titanic_raw_copy.csv", index=False)

# 2. Initial inspection
print("\\n--- Shape ---")
print(df.shape)
print("\\n--- First rows ---")
print(df.head())
print("\\n--- Data types and non-null counts ---")
df.info()
print("\\n--- Summary statistics ---")
print(df.describe(include="all").T)
print("\\n--- Missing values ---")
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_percent": missing_pct})
print(missing_report[missing_report["missing_count"] > 0])
missing_report.to_csv(OUTPUT / "missing_values_before.csv")

# 3. Duplicate and consistency checks
duplicate_rows = int(df.duplicated().sum())
print(f"\\nExact duplicate rows: {duplicate_rows}")
print("\\nUnique values in categorical columns:")
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"{col}: {df[col].dropna().astype(str).str.strip().value_counts().head(20).to_dict()}")

# Normalize whitespace in text columns (does not change meaningful category labels).
clean = df.copy()
for col in clean.select_dtypes(include=["object"]).columns:
    clean[col] = clean[col].str.strip()

# Check plausible ranges for selected numeric fields.
# Age should be non-negative; fare should be non-negative; sibsp/parch are counts.
range_checks = {
    "age": lambda s: s < 0,
    "fare": lambda s: s < 0,
    "sibsp": lambda s: s < 0,
    "parch": lambda s: s < 0,
}
for col, invalid_condition in range_checks.items():
    if col in clean.columns:
        n_invalid = int(invalid_condition(clean[col]).fillna(False).sum())
        print(f"Invalid values in {col} (negative): {n_invalid}")

# 4. Missing-value treatment
# Preserve the original dataset. Add an explicit missingness indicator for age.
if "age" in clean.columns:
    clean["age_was_missing"] = clean["age"].isna().astype(int)
    age_median = clean["age"].median()
    clean["age"] = clean["age"].fillna(age_median)

# Embark town is categorical; use the mode because only a small number are missing.
if "embark_town" in clean.columns:
    mode = clean["embark_town"].mode(dropna=True)
    if not mode.empty:
        clean["embark_town"] = clean["embark_town"].fillna(mode.iloc[0])

# 'deck' has substantial missingness in the common Titanic version.
# Do not invent a deck value. Preserve missingness explicitly as "Unknown".
if "deck" in clean.columns:
    clean["deck"] = clean["deck"].cat.add_categories(["Unknown"]) if isinstance(clean["deck"].dtype, pd.CategoricalDtype) else clean["deck"]
    clean["deck"] = clean["deck"].fillna("Unknown")

# For other columns, report missingness rather than blindly filling everything.
# 'embarked' can also have a small number of missing values; use mode if present.
if "embarked" in clean.columns:
    mode = clean["embarked"].mode(dropna=True)
    if not mode.empty:
        clean["embarked"] = clean["embarked"].fillna(mode.iloc[0])

# 5. Outlier detection using IQR on fare.
# Outliers are flagged, not automatically deleted: high fares may be genuine first-class tickets.
if "fare" in clean.columns:
    q1 = clean["fare"].quantile(0.25)
    q3 = clean["fare"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clean["fare_outlier_iqr"] = ((clean["fare"] < lower) | (clean["fare"] > upper)).astype(int)
    print(f"\\nFare IQR bounds: {lower:.3f} to {upper:.3f}")
    print("Fare rows flagged by IQR:", int(clean["fare_outlier_iqr"].sum()))

# 6. Remove exact duplicate records only if verified.
# This dataset does not provide a guaranteed unique passenger ID in all versions.
# We report duplicates and retain them unless the project owner confirms they are erroneous.
print("\\nExact duplicate rows retained pending verification:", duplicate_rows)

# 7. Feature engineering / encoding example.
# Keep the cleaned categorical fields readable in the main output.
# Create a simple family-size feature from sibling/spouse and parent/child counts.
if {"sibsp", "parch"}.issubset(clean.columns):
    clean["family_size"] = clean["sibsp"].fillna(0) + clean["parch"].fillna(0) + 1

# 8. Save cleaned dataset and audit outputs.
clean.to_csv(OUTPUT / "titanic_cleaned.csv", index=False)
after_missing = pd.DataFrame({
    "missing_count": clean.isna().sum(),
    "missing_percent": (clean.isna().mean() * 100).round(2)
})
after_missing.to_csv(OUTPUT / "missing_values_after.csv")

# 9. Create plots suitable for inserting into the report.
if "age" in df.columns:
    plt.figure(figsize=(8, 4))
    df["age"].dropna().plot(kind="hist", bins=25, edgecolor="black")
    plt.title("Age distribution before imputation")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT / "age_distribution_before.png", dpi=150)
    plt.close()

if "fare" in clean.columns:
    plt.figure(figsize=(8, 4))
    clean.boxplot(column="fare")
    plt.title("Fare boxplot (IQR outliers flagged, not removed)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "fare_boxplot.png", dpi=150)
    plt.close()

print("\\n--- Final dataset shape ---")
print(clean.shape)
print("\\n--- Remaining missing values ---")
print(after_missing[after_missing["missing_count"] > 0])
print(f"\\nOutputs saved in: {OUTPUT.resolve()}")