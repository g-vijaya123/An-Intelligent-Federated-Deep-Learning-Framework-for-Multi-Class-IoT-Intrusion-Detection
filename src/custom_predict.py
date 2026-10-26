"""
Custom prediction for a new/unseen UNSW-NB15-style network traffic record.

This script:
1. Loads saved preprocessing artifacts.
2. Creates a new manual network traffic record.
3. Applies the same categorical encoding used during training.
4. Applies the saved StandardScaler.
5. Applies the saved SelectKBest selector.
6. Loads the trained federated CNN-LSTM model.
7. Predicts one of the 10 classes.
8. Displays confidence, attack information, and all class probabilities.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ============================================================
# IMPORT attack_info.py
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


# ============================================================
# NEW / CUSTOM NETWORK TRAFFIC
# ============================================================
#
# This is a manually created sample.
# It is NOT taken from the UNSW-NB15 test dataset.
#
# NOTE:
# These values are for pipeline testing only.
# For real-world testing, values should come from an actual
# network-flow extraction process.
# ============================================================

custom_traffic = {
    "dur": 0.75,
    "proto": "tcp",
    "service": "http",
    "state": "FIN",
    "rate": 25.0,
    "sttl": 64,
    "dttl": 64,
    "sload": 150000.0,
    "dload": 80000.0,
    "sinpkt": 20.0,
    "sjit": 10.0,
    "swin": 255,
    "stcpb": 100000,
    "dtcpb": 200000,
    "dwin": 255,
    "tcprtt": 0.05,
    "synack": 0.02,
    "ackdat": 0.03,
    "smean": 100,
    "dmean": 120,
    "ct_srv_src": 5,
    "ct_state_ttl": 1,
    "ct_dst_ltm": 5,
    "ct_src_dport_ltm": 3,
    "ct_dst_sport_ltm": 2,
    "ct_dst_src_ltm": 4,
    "is_ftp_login": 0,
    "ct_ftp_cmd": 0,
    "ct_flw_http_mthd": 1,
    "ct_src_ltm": 5,
    "ct_srv_dst": 5,
    "is_sm_ips_ports": 0
}


# ============================================================
# START
# ============================================================

print()
print("=" * 65)
print("CUSTOM / UNSEEN NETWORK TRAFFIC PREDICTION")
print("=" * 65)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.isfile(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocessing artifact not found: {PREPROCESSOR_PATH}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Federated model not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

artifacts = joblib.load(PREPROCESSOR_PATH)

categorical_encoders = artifacts["categorical_encoders"]
scaler = artifacts["scaler"]
selector = artifacts["selector"]
all_feature_names = artifacts["all_feature_names"]
selected_feature_names = artifacts["selected_feature_names"]
class_names = artifacts["class_names"]


print()
print("Preprocessing artifacts loaded successfully.")
print(f"Number of original features : {len(all_feature_names)}")
print(f"Selected features           : {len(selected_feature_names)}")
print(f"Number of classes           : {len(class_names)}")


# ============================================================
# CREATE COMPLETE INPUT ROW
# ============================================================
#
# The saved StandardScaler expects exactly the same 42
# features and the same feature order used during training.
#
# Any feature not manually supplied above is temporarily
# assigned 0.
# ============================================================

input_row = {}

for feature in all_feature_names:
    input_row[feature] = custom_traffic.get(feature, 0)


input_df = pd.DataFrame(
    [input_row],
    columns=all_feature_names
)


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================
#
# IMPORTANT FIX:
# Pandas may create proto/service/state as StringDtype.
# LabelEncoder produces integer values.
#
# Therefore the categorical columns are first converted to
# object dtype before replacing strings with encoded integers.
# ============================================================

for column_name, encoder in categorical_encoders.items():

    if column_name not in input_df.columns:
        continue

    # Fix for:
    # TypeError: Invalid value '...' for dtype 'str'
    input_df[column_name] = input_df[column_name].astype(object)

    original_value = str(
        input_df.at[0, column_name]
    )

    known_categories = set(
        str(value) for value in encoder.classes_
    )

    if original_value in known_categories:

        encoded_value = int(
            encoder.transform([original_value])[0]
        )

        input_df.at[0, column_name] = encoded_value

    else:

        print()
        print(
            f"WARNING: Unseen categorical value "
            f"'{original_value}' for '{column_name}'."
        )

        print(
            "Fallback encoded value 0 will be used."
        )

        input_df.at[0, column_name] = 0


# ============================================================
# CONVERT ALL FEATURES TO NUMERIC
# ============================================================

input_df = input_df.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)


# Maintain exactly the same feature order used during training.
input_df = input_df[
    all_feature_names
]


# ============================================================
# VALIDATE INPUT
# ============================================================

expected_features = len(all_feature_names)

if input_df.shape[1] != expected_features:
    raise ValueError(
        f"Feature mismatch. Expected {expected_features} "
        f"features but received {input_df.shape[1]}."
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


if selected_input.shape[1] != len(selected_feature_names):
    raise ValueError(
        "Selected feature count does not match training."
    )


# ============================================================
# RESHAPE FOR CNN-LSTM
# ============================================================

model_input = selected_input.reshape(
    (
        1,
        selected_input.shape[1],
        1
    )
)


print()
print(
    f"Final model input shape     : {model_input.shape}"
)


# ============================================================
# LOAD TRAINED FEDERATED CNN-LSTM
# ============================================================

model = load_model(
    MODEL_PATH,
    compile=False
)


print(
    "Federated CNN-LSTM model loaded successfully."
)


# ============================================================
# MODEL PREDICTION
# ============================================================

probabilities = model.predict(
    model_input,
    verbose=0
)[0]


# Safety check
if len(probabilities) != len(class_names):
    raise ValueError(
        "Model output classes do not match saved class names."
    )


predicted_index = int(
    np.argmax(probabilities)
)


predicted_class = str(
    class_names[predicted_index]
)


confidence = float(
    probabilities[predicted_index]
) * 100.0


# ============================================================
# GET HUMAN-READABLE ATTACK INFORMATION
# ============================================================

attack_information = get_attack_info(
    predicted_class
)


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

print()
print("=" * 65)
print("PREDICTION RESULT")
print("=" * 65)

print(
    f"Predicted Category : {predicted_class}"
)

print(
    f"Confidence         : {confidence:.2f}%"
)

print(
    f"Attack Type        : "
    f"{attack_information['type']}"
)

print(
    f"Description        : "
    f"{attack_information['description']}"
)


# ============================================================
# DISPLAY ALL 10 CLASS PROBABILITIES
# ============================================================

print()
print("=" * 65)
print("ALL 10 CLASS PROBABILITIES")
print("=" * 65)


results = []

for class_name, probability in zip(
    class_names,
    probabilities
):

    probability_percent = float(
        probability
    ) * 100.0

    results.append(
        (
            str(class_name),
            probability_percent
        )
    )


# Sort highest probability first
results.sort(
    key=lambda item: item[1],
    reverse=True
)


for class_name, probability in results:

    print(
        f"{class_name:<18} : "
        f"{probability:6.2f}%"
    )


print("=" * 65)


# ============================================================
# CONFIDENCE WARNING
# ============================================================
#
# This is NOT a true unknown-attack detector.
#
# It only indicates that the model has low confidence among
# the 10 known classes.
# ============================================================

WARNING_THRESHOLD = 60.0


if confidence < WARNING_THRESHOLD:

    print()
    print("WARNING: LOW PREDICTION CONFIDENCE")

    print(
        "The model is uncertain about this traffic."
    )

    print(
        "The traffic may require additional investigation."
    )

else:

    print()
    print(
        "Prediction confidence is above "
        "the warning threshold."
    )


# ============================================================
# DISPLAY SELECTED FEATURES
# ============================================================

print()
print("=" * 65)
print("32 FEATURES USED BY THE MODEL")
print("=" * 65)


for index, feature in enumerate(
    selected_feature_names,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 65)
print("PREDICTION COMPLETED")
print("=" * 65)