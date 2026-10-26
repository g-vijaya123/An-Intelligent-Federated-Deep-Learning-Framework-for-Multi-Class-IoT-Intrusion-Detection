import os
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# Create report folder
# ==================================================

os.makedirs("report", exist_ok=True)


# ==================================================
# GRAPH 1: Federated Learning Accuracy Across Rounds
# ==================================================

rounds = [1, 2, 3, 4, 5]

balanced_accuracy = [
    38.68,
    53.34,
    77.57,
    78.10,
    78.18
]

plt.figure(figsize=(8, 5))

plt.plot(
    rounds,
    balanced_accuracy,
    marker="o",
    linewidth=2
)

plt.title(
    "Federated Learning Accuracy Across Rounds"
)

plt.xlabel("Federated Round")
plt.ylabel("Global Accuracy (%)")

plt.xticks(rounds)
plt.ylim(0, 100)

plt.grid(
    True,
    alpha=0.3
)

# Display accuracy above each point
for x, y in zip(
    rounds,
    balanced_accuracy
):
    plt.text(
        x,
        y + 2,
        f"{y:.2f}%",
        ha="center"
    )

plt.tight_layout()

output_path = os.path.join(
    "report",
    "federated_round_accuracy.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Graph saved ->",
    output_path
)


# ==================================================
# GRAPH 2: Baseline vs Controlled SMOTE-Tomek
# ==================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

# Original federated model results
baseline = [
    79.43,
    77.77,
    79.43,
    76.69
]

# Controlled SMOTE-Tomek federated model results
smote_tomek = [
    78.18,
    77.96,
    78.18,
    75.66
]

x = np.arange(
    len(metrics)
)

width = 0.35

plt.figure(
    figsize=(9, 5)
)

bars1 = plt.bar(
    x - width / 2,
    baseline,
    width,
    label="Without SMOTE-Tomek"
)

bars2 = plt.bar(
    x + width / 2,
    smote_tomek,
    width,
    label="Controlled SMOTE-Tomek"
)

plt.xlabel(
    "Evaluation Metric"
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Federated IDS: Before vs After Controlled SMOTE-Tomek"
)

plt.xticks(
    x,
    metrics
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

# Display values above bars
for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=8
        )

plt.tight_layout()

comparison_path = os.path.join(
    "report",
    "baseline_vs_smote_tomek_metrics.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Comparison graph saved ->",
    comparison_path
)


# ==================================================
# GRAPH 3: Attack-wise Recall Comparison
# ==================================================

attack_classes = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Reconnaissance",
    "Shellcode",
    "Worms"
]

# Federated model WITHOUT Controlled SMOTE-Tomek
baseline_recall = [
    1,
    0,
    4,
    91,
    49,
    97,
    91,
    67,
    24,
    0
]

# Federated model WITH Controlled SMOTE-Tomek
smote_tomek_recall = [
    9,
    12,
    1,
    84,
    33,
    97,
    93,
    75,
    75,
    57
]

x = np.arange(
    len(attack_classes)
)

width = 0.35

plt.figure(
    figsize=(14, 6)
)

bars1 = plt.bar(
    x - width / 2,
    baseline_recall,
    width,
    label="Without SMOTE-Tomek"
)

bars2 = plt.bar(
    x + width / 2,
    smote_tomek_recall,
    width,
    label="Controlled SMOTE-Tomek"
)

plt.title(
    "Attack-wise Recall: Before vs After Controlled SMOTE-Tomek"
)

plt.xlabel(
    "Attack Category"
)

plt.ylabel(
    "Recall (%)"
)

plt.xticks(
    x,
    attack_classes,
    rotation=35,
    ha="right"
)

plt.ylim(
    0,
    110
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

# Display recall percentage above bars
for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.0f}%",
            ha="center",
            fontsize=8
        )

plt.tight_layout()

recall_path = os.path.join(
    "report",
    "attack_recall_before_vs_smote_tomek.png"
)

plt.savefig(
    recall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Attack recall graph saved ->",
    recall_path
)


# ==================================================
# Finished
# ==================================================

print("\n" + "=" * 60)
print("ALL RESULT GRAPHS GENERATED SUCCESSFULLY")
print("=" * 60)

print(
    "1. report/federated_round_accuracy.png"
)

print(
    "2. report/baseline_vs_smote_tomek_metrics.png"
)

print(
    "3. report/attack_recall_before_vs_smote_tomek.png"
)