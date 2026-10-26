import os
import pandas as pd

os.makedirs("report", exist_ok=True)

results = {
    "Metric": [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "Without SMOTE-Tomek (%)": [
        79.43,
        77.77,
        79.43,
        76.69,
        54.00,
        42.00,
        42.00
    ],
    "Controlled SMOTE-Tomek (%)": [
        78.18,
        77.96,
        78.18,
        75.66,
        49.56,
        53.73,
        44.57
    ]
}

df = pd.DataFrame(results)

output_path = os.path.join(
    "report",
    "final_model_comparison.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nFINAL MODEL COMPARISON")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)

print(
    "\nResults saved ->",
    output_path
)