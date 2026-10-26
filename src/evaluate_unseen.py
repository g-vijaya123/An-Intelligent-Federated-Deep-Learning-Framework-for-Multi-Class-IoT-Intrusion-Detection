import os
import sys
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/verified_unseen_with_labels.csv"

MODEL_FILE = (
    "models/"
    "federated_cnn_lstm_multiclass_smote.keras"
)

ARTIFACT_FILE = (
    "models/"
    "preprocessing_multiclass_stratified.joblib"
)

RESULT_FILE = (
    "report/"
    "unseen_actual_vs_predicted.csv"
)

CONFUSION_MATRIX_FILE = (
    "report/"
    "unseen_confusion_matrix.png"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EVALUATING FEDERATED MODEL ON NON-OVERLAPPING RECORDS")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    INPUT_FILE,
    MODEL_FILE,
    ARTIFACT_FILE
]:

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

artifacts = joblib.load(
    ARTIFACT_FILE
)

categorical_encoders = artifacts[
    "categorical_encoders"
]

scaler = artifacts["scaler"]

selector = artifacts["selector"]

feature_names = list(
    artifacts["all_feature_names"]
)

selected_feature_names = list(
    artifacts["selected_feature_names"]
)

class_names = list(
    artifacts["class_names"]
)


print()
print(
    f"Original features : {len(feature_names)}"
)

print(
    f"Selected features : {len(selected_feature_names)}"
)

print(
    f"Classes           : {len(class_names)}"
)


# ============================================================
# LOAD NON-OVERLAPPING DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print()
print(
    f"Evaluation records loaded : {len(df)}"
)


if df.empty:
    raise ValueError(
        "Evaluation dataset is empty."
    )


# ============================================================
# CHECK ACTUAL LABEL
# ============================================================

if "Actual_Category" not in df.columns:

    raise ValueError(
        "Actual_Category column is missing."
    )


actual_labels = (
    df["Actual_Category"]
    .astype(str)
    .str.strip()
    .to_numpy()
)


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in feature_names
    if feature not in df.columns
]


if missing_features:

    print()
    print("Missing features:")

    for feature in missing_features:
        print(
            f" - {feature}"
        )

    raise ValueError(
        "Required model features are missing."
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

input_df = df[
    feature_names
].copy()


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

for column_name, encoder in (
    categorical_encoders.items()
):

    if column_name not in input_df.columns:
        continue

    encoded_values = []

    known_categories = {
        str(value)
        for value in encoder.classes_
    }

    for row_number, raw_value in enumerate(
        input_df[column_name],
        start=1
    ):

        value = str(raw_value).strip()

        if value in known_categories:

            encoded_value = int(
                encoder.transform(
                    [value]
                )[0]
            )

        else:

            print(
                f"WARNING: Record {row_number}: "
                f"unknown {column_name}='{value}'. "
                "Using 0."
            )

            encoded_value = 0

        encoded_values.append(
            encoded_value
        )

    input_df[column_name] = np.asarray(
        encoded_values,
        dtype=np.int64
    )


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

input_df = input_df.apply(
    pd.to_numeric,
    errors="coerce"
)


missing_value_count = int(
    input_df.isna().sum().sum()
)


if missing_value_count > 0:

    print(
        f"WARNING: {missing_value_count} "
        "invalid values replaced with 0."
    )

    input_df = input_df.fillna(0)


# ============================================================
# APPLY SAME TRAINING PREPROCESSING
# ============================================================

scaled_input = scaler.transform(
    input_df
)

selected_input = selector.transform(
    scaled_input
)


model_input = selected_input.reshape(
    selected_input.shape[0],
    selected_input.shape[1],
    1
)


print(
    f"Model input shape : {model_input.shape}"
)


# ============================================================
# LOAD FEDERATED MODEL
# ============================================================

model = load_model(
    MODEL_FILE,
    compile=False
)


print(
    "Federated CNN-LSTM model loaded."
)


# ============================================================
# MODEL PREDICTION
# ============================================================

probabilities = model.predict(
    model_input,
    verbose=0
)


predicted_indices = np.argmax(
    probabilities,
    axis=1
)


predicted_labels = np.array(
    [
        str(class_names[index])
        for index in predicted_indices
    ]
)


confidence_values = (
    np.max(
        probabilities,
        axis=1
    )
    * 100
)


# ============================================================
# CORRECT / INCORRECT
# ============================================================

correct_values = (
    actual_labels
    == predicted_labels
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)


correct_count = int(
    np.sum(correct_values)
)

incorrect_count = (
    len(df) - correct_count
)


print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Total records     : {len(df)}"
)

print(
    f"Correct           : {correct_count}"
)

print(
    f"Incorrect         : {incorrect_count}"
)

print(
    f"Overall Accuracy  : {accuracy * 100:.2f}%"
)


# ============================================================
# CLASS-WISE RESULTS
# ============================================================

print()
print("=" * 70)
print("CLASS-WISE RESULTS")
print("=" * 70)


for class_name in class_names:

    mask = (
        actual_labels == class_name
    )

    total = int(
        np.sum(mask)
    )

    if total == 0:
        continue

    correct = int(
        np.sum(
            predicted_labels[mask]
            == class_name
        )
    )

    class_accuracy = (
        correct / total
    ) * 100

    print(
        f"{class_name:<18} "
        f"{correct:>2}/{total:<2} "
        f"({class_accuracy:6.2f}%)"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=class_names,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CREATE RESULT CSV
# ============================================================

result_df = pd.DataFrame(
    {
        "Record":
            np.arange(
                1,
                len(df) + 1
            ),

        "Actual_Category":
            actual_labels,

        "Predicted_Category":
            predicted_labels,

        "Confidence_Percent":
            np.round(
                confidence_values,
                2
            ),

        "Correct":
            np.where(
                correct_values,
                "YES",
                "NO"
            )
    }
)


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(
    "report",
    exist_ok=True
)


result_df.to_csv(
    RESULT_FILE,
    index=False
)


print()
print(
    f"Detailed results saved to: "
    f"{RESULT_FILE}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=class_names
)


fig, ax = plt.subplots(
    figsize=(12, 10)
)


image = ax.imshow(
    cm
)


ax.set_xticks(
    np.arange(
        len(class_names)
    )
)

ax.set_yticks(
    np.arange(
        len(class_names)
    )
)


ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    class_names
)


ax.set_xlabel(
    "Predicted Category"
)

ax.set_ylabel(
    "Actual Category"
)

ax.set_title(
    "Non-Overlapping UNSW-NB15 Records - Confusion Matrix"
)


# Add numbers inside matrix
for i in range(
    len(class_names)
):

    for j in range(
        len(class_names)
    ):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


fig.colorbar(
    image,
    ax=ax
)

fig.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Confusion matrix saved to: "
    f"{CONFUSION_MATRIX_FILE}"
)


# ============================================================
# SHOW SAMPLE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("FIRST 20 ACTUAL VS PREDICTED")
print("=" * 70)


print(
    result_df.head(20).to_string(
        index=False
    )
)


print()
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)