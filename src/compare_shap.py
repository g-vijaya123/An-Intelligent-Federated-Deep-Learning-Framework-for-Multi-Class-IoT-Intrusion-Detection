import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE PATHS
# ============================================================

WRONG_FILE = "report/record_47_shap_explanation.csv"
CORRECT_FILE = "report/record_49_shap_explanation.csv"

OUTPUT_FILE = "report/dos_shap_comparison.png"


# ============================================================
# LOAD SHAP RESULTS
# ============================================================

wrong_df = pd.read_csv(WRONG_FILE)
correct_df = pd.read_csv(CORRECT_FILE)


# ============================================================
# SELECT TOP 8 FEATURES
# ============================================================

wrong_top = (
    wrong_df
    .sort_values("Absolute_SHAP", ascending=False)
    .head(8)
    .copy()
)

correct_top = (
    correct_df
    .sort_values("Absolute_SHAP", ascending=False)
    .head(8)
    .copy()
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("report", exist_ok=True)


# ============================================================
# GRAPH 1:
# WRONG PREDICTION
# Actual DoS -> Predicted Exploits
# ============================================================

plt.figure(figsize=(10, 6))

wrong_plot = wrong_top.sort_values(
    "Absolute_SHAP",
    ascending=True
)

plt.barh(
    wrong_plot["Feature"],
    wrong_plot["SHAP_Value"]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.title(
    "Record 47: Actual DoS -> Predicted Exploits (95.03%)"
)

plt.xlabel(
    "SHAP Value for Predicted Exploits Class"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

wrong_output = (
    "report/"
    "record_47_wrong_dos_shap.png"
)

plt.savefig(
    wrong_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2:
# CORRECT PREDICTION
# Actual DoS -> Predicted DoS
# ============================================================

plt.figure(figsize=(10, 6))

correct_plot = correct_top.sort_values(
    "Absolute_SHAP",
    ascending=True
)

plt.barh(
    correct_plot["Feature"],
    correct_plot["SHAP_Value"]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.title(
    "Record 49: Actual DoS -> Predicted DoS (49.12%)"
)

plt.xlabel(
    "SHAP Value for Predicted DoS Class"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

correct_output = (
    "report/"
    "record_49_correct_dos_shap.png"
)

plt.savefig(
    correct_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMBINED COMPARISON
# ============================================================

# Features appearing in either top-8 list
features = list(
    dict.fromkeys(
        list(wrong_top["Feature"]) +
        list(correct_top["Feature"])
    )
)


wrong_values = (
    wrong_df
    .set_index("Feature")["SHAP_Value"]
    .reindex(features)
    .fillna(0)
)

correct_values = (
    correct_df
    .set_index("Feature")["SHAP_Value"]
    .reindex(features)
    .fillna(0)
)


comparison_df = pd.DataFrame({
    "Feature": features,
    "Wrong_DoS_to_Exploits": wrong_values.values,
    "Correct_DoS_to_DoS": correct_values.values
})


comparison_df["Importance"] = (
    comparison_df["Wrong_DoS_to_Exploits"].abs()
    +
    comparison_df["Correct_DoS_to_DoS"].abs()
)


comparison_df = comparison_df.sort_values(
    "Importance",
    ascending=True
)


# ============================================================
# PLOT COMPARISON
# ============================================================

ax = comparison_df.plot(
    x="Feature",
    y=[
        "Wrong_DoS_to_Exploits",
        "Correct_DoS_to_DoS"
    ],
    kind="barh",
    figsize=(11, 8)
)

ax.axvline(
    x=0,
    linewidth=1
)

ax.set_title(
    "SHAP Comparison: Misclassified vs Correct DoS Record"
)

ax.set_xlabel(
    "SHAP Value"
)

ax.set_ylabel(
    "Feature"
)

ax.legend([
    "Record 47: DoS -> Exploits",
    "Record 49: DoS -> DoS"
])

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

csv_output = (
    "report/"
    "dos_shap_comparison.csv"
)

comparison_df.to_csv(
    csv_output,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("=" * 65)
print("SHAP COMPARISON COMPLETED")
print("=" * 65)

print()
print("Created files:")

print(
    "1.",
    wrong_output
)

print(
    "2.",
    correct_output
)

print(
    "3.",
    OUTPUT_FILE
)

print(
    "4.",
    csv_output
)