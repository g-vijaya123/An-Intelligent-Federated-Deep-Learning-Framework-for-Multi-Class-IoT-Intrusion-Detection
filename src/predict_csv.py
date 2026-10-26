import os
import sys
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from attack_info import get_attack_info


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "federated_cnn_lstm_multiclass_smote.keras"
)

PREPROCESSOR_PATH = os.path.join(
    "models",
    "preprocessing_multiclass_stratified.joblib"
)

# IMPORTANT:
# Use only records that did not exactly overlap with the
# train/test partitions in our 42-feature comparison.
INPUT_CSV = os.path.join(
    "data",
    "verified_unseen_traffic.csv"
)

OUTPUT_CSV = os.path.join(
    "report",
    "verified_unseen_predictions.csv"
)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("VERIFIED UNSEEN NETWORK TRAFFIC PREDICTION")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.isfile(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocessing artifact not found: {PREPROCESSOR_PATH}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_CSV}"
    )


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

artifacts = joblib.load(PREPROCESSOR_PATH)

categorical_encoders = artifacts["categorical_encoders"]
scaler = artifacts["scaler"]
selector = artifacts["selector"]
all_feature_names = list(artifacts["all_feature_names"])
selected_feature_names = list(
    artifacts["selected_feature_names"]
)
class_names = list(artifacts["class_names"])


print()
print("Preprocessing artifacts loaded successfully.")
print(f"Original features : {len(all_feature_names)}")
print(f"Selected features : {len(selected_feature_names)}")
print(f"Classes           : {len(class_names)}")


# ============================================================
# LOAD VERIFIED UNSEEN CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

if df.empty:
    raise ValueError(
        "verified_unseen_traffic.csv contains no records."
    )


print()
print(f"Verified unseen records loaded : {len(df)}")


# ============================================================
# VALIDATE REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in all_feature_names
    if feature not in df.columns
]

if missing_features:

    print()
    print("ERROR: Missing required features:")

    for feature in missing_features:
        print(f" - {feature}")

    raise ValueError(
        "Input CSV does not contain all required features."
    )


# Keep exact feature order used during training
input_df = df[all_feature_names].copy()


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

for column_name, encoder in categorical_encoders.items():

    if column_name not in input_df.columns:
        continue

    encoded_values = []

    known_categories = {
        str(value)
        for value in encoder.classes_
    }

    for row_index, raw_value in enumerate(
        input_df[column_name]
    ):

        value = str(raw_value).strip()

        if value in known_categories:

            encoded_value = int(
                encoder.transform([value])[0]
            )

        else:

            print(
                f"WARNING: Record {row_index + 1}: "
                f"unknown value '{value}' "
                f"for '{column_name}'. Using 0."
            )

            encoded_value = 0

        encoded_values.append(
            encoded_value
        )

    # Entire column replaced at once.
    # This avoids the Pandas StringDtype assignment error.
    input_df[column_name] = np.asarray(
        encoded_values,
        dtype=np.int64
    )


# ============================================================
# CONVERT EVERYTHING TO NUMERIC
# ============================================================

input_df = input_df.apply(
    pd.to_numeric,
    errors="coerce"
)


# Check if conversion produced missing values
nan_count = int(
    input_df.isna().sum().sum()
)

if nan_count > 0:

    print()
    print(
        f"WARNING: {nan_count} invalid/missing values "
        "were converted to 0."
    )

    input_df = input_df.fillna(0)


# Ensure exact feature order again
input_df = input_df[
    all_feature_names
]


# ============================================================
# VALIDATE 42 FEATURES
# ============================================================

if input_df.shape[1] != len(all_feature_names):

    raise ValueError(
        "Feature count mismatch before scaling."
    )


# ============================================================
# APPLY SAVED STANDARD SCALER
# ============================================================

scaled_input = scaler.transform(
    input_df
)


# ============================================================
# APPLY SAVED SELECTKBEST
# ============================================================

selected_input = selector.transform(
    scaled_input
)


if selected_input.shape[1] != len(
    selected_feature_names
):

    raise ValueError(
        "Selected feature count does not match "
        "the training pipeline."
    )


# ============================================================
# RESHAPE FOR CNN-LSTM
# ============================================================

model_input = selected_input.reshape(
    selected_input.shape[0],
    selected_input.shape[1],
    1
)


print(
    f"Final model input shape          : "
    f"{model_input.shape}"
)


# ============================================================
# LOAD FEDERATED CNN-LSTM MODEL
# ============================================================

model = load_model(
    MODEL_PATH,
    compile=False
)


print(
    "Federated CNN-LSTM model loaded successfully."
)


# ============================================================
# PREDICT ALL VERIFIED RECORDS
# ============================================================

probabilities = model.predict(
    model_input,
    verbose=0
)


if probabilities.shape[1] != len(class_names):

    raise ValueError(
        "Model output class count does not match "
        "saved class names."
    )


predicted_indices = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# CREATE RESULTS
# ============================================================

results = []


for row_number, predicted_index in enumerate(
    predicted_indices,
    start=1
):

    predicted_index = int(
        predicted_index
    )

    predicted_class = str(
        class_names[predicted_index]
    )

    confidence = float(
        probabilities[
            row_number - 1,
            predicted_index
        ]
    ) * 100.0

    attack_info = get_attack_info(
        predicted_class
    )


    # --------------------------------------------------------
    # Find second-highest prediction
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        probabilities[row_number - 1]
    )[::-1]

    second_index = int(
        sorted_indices[1]
    )

    second_class = str(
        class_names[second_index]
    )

    second_confidence = float(
        probabilities[
            row_number - 1,
            second_index
        ]
    ) * 100.0


    results.append(
        {
            "Record": row_number,

            "Predicted_Category":
                predicted_class,

            "Confidence_Percent":
                round(confidence, 2),

            "Second_Prediction":
                second_class,

            "Second_Confidence_Percent":
                round(second_confidence, 2),

            "Attack_Type":
                attack_info["type"],

            "Description":
                attack_info["description"],

            "Low_Confidence":
                confidence < 60.0
        }
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)


for result in results:

    print()

    print(
        f"Record             : "
        f"{result['Record']}"
    )

    print(
        f"Predicted Category : "
        f"{result['Predicted_Category']}"
    )

    print(
        f"Confidence         : "
        f"{result['Confidence_Percent']:.2f}%"
    )

    print(
        f"Second Prediction  : "
        f"{result['Second_Prediction']} "
        f"({result['Second_Confidence_Percent']:.2f}%)"
    )

    print(
        f"Attack Type        : "
        f"{result['Attack_Type']}"
    )

    print(
        f"Description        : "
        f"{result['Description']}"
    )


    if result["Low_Confidence"]:

        print(
            "Warning            : "
            "LOW PREDICTION CONFIDENCE"
        )

    else:

        print(
            "Confidence Status  : "
            "Above warning threshold"
        )


    print("-" * 70)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

prediction_counts = {}

for result in results:

    category = result[
        "Predicted_Category"
    ]

    prediction_counts[category] = (
        prediction_counts.get(
            category,
            0
        ) + 1
    )


print()
print("=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    f"Total records : {len(results)}"
)


for category, count in sorted(
    prediction_counts.items()
):

    print(
        f"{category:<18} : {count}"
    )


low_confidence_count = sum(
    result["Low_Confidence"]
    for result in results
)


print()
print(
    f"Low-confidence predictions : "
    f"{low_confidence_count}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)


result_df = pd.DataFrame(
    results
)


result_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print()
print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)

print(
    f"Results saved to: {OUTPUT_CSV}"
)