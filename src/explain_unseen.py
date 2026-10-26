import os
import joblib
import numpy as np
import pandas as pd
import shap

from tensorflow.keras.models import load_model


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "data/verified_unseen_with_labels.csv"

MODEL_FILE = (
    "models/"
    "federated_cnn_lstm_multiclass_smote.keras"
)

ARTIFACT_FILE = (
    "models/"
    "preprocessing_multiclass_stratified.joblib"
)


# Record number shown in evaluation output
RECORD_NUMBER = 49

# pandas starts indexing from 0
ROW_INDEX = RECORD_NUMBER - 1


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SHAP EXPLANATION FOR NON-OVERLAPPING RECORD")
print("=" * 70)

print(f"Record number : {RECORD_NUMBER}")


# ============================================================
# LOAD ARTIFACTS
# ============================================================

artifacts = joblib.load(
    ARTIFACT_FILE
)

feature_names = list(
    artifacts["all_feature_names"]
)

selected_feature_names = list(
    artifacts["selected_feature_names"]
)

class_names = list(
    artifacts["class_names"]
)

categorical_encoders = artifacts[
    "categorical_encoders"
]

scaler = artifacts["scaler"]

selector = artifacts["selector"]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

if ROW_INDEX >= len(df):
    raise ValueError(
        f"Record {RECORD_NUMBER} does not exist."
    )


actual_category = str(
    df.iloc[ROW_INDEX]["Actual_Category"]
)


print(f"Actual category : {actual_category}")


# ============================================================
# PREPARE ALL RECORDS
# ============================================================

input_df = df[
    feature_names
].copy()


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

for column_name, encoder in categorical_encoders.items():

    if column_name not in input_df.columns:
        continue

    known_categories = {
        str(value): index
        for index, value in enumerate(
            encoder.classes_
        )
    }

    encoded_values = []

    for value in input_df[column_name]:

        value = str(value).strip()

        if value in known_categories:

            encoded_values.append(
                known_categories[value]
            )

        else:

            encoded_values.append(0)

    input_df[column_name] = np.asarray(
        encoded_values,
        dtype=np.int64
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

input_df = input_df.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)


# ============================================================
# SAME PREPROCESSING USED DURING TRAINING
# ============================================================

scaled_data = scaler.transform(
    input_df
)

selected_data = selector.transform(
    scaled_data
)

model_data = selected_data.reshape(
    selected_data.shape[0],
    selected_data.shape[1],
    1
)


print(
    f"Model input shape : {model_data.shape}"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(
    MODEL_FILE,
    compile=False
)

print("Federated CNN-LSTM model loaded.")


# ============================================================
# PREDICT RECORD 47
# ============================================================

sample = model_data[
    ROW_INDEX:ROW_INDEX + 1
]

prediction = model.predict(
    sample,
    verbose=0
)[0]


predicted_index = int(
    np.argmax(prediction)
)

predicted_category = str(
    class_names[predicted_index]
)

confidence = float(
    prediction[predicted_index] * 100
)


print()
print("=" * 70)
print("PREDICTION")
print("=" * 70)

print(
    f"Actual class    : {actual_category}"
)

print(
    f"Predicted class : {predicted_category}"
)

print(
    f"Confidence      : {confidence:.2f}%"
)


# ============================================================
# SHOW ALL CLASS PROBABILITIES
# ============================================================

print()
print("=" * 70)
print("CLASS PROBABILITIES")
print("=" * 70)

probability_order = np.argsort(
    prediction
)[::-1]

for index in probability_order:

    print(
        f"{class_names[index]:<18} "
        f"{prediction[index] * 100:6.2f}%"
    )


# ============================================================
# BACKGROUND DATA FOR SHAP
# ============================================================
# Use a subset of the verified records as background.
# Exclude the record being explained.

background_indices = [
    i
    for i in range(len(model_data))
    if i != ROW_INDEX
]

# Keep SHAP reasonably fast
background_indices = background_indices[:50]

background = model_data[
    background_indices
]


print()
print(
    f"SHAP background records : "
    f"{len(background)}"
)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.GradientExplainer(
    model,
    background
)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

shap_values = explainer.shap_values(
    sample
)


# ============================================================
# NORMALIZE SHAP OUTPUT FORMAT
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP format:
    # one array for each output class
    predicted_shap = np.asarray(
        shap_values[predicted_index]
    )[0]

else:

    shap_array = np.asarray(
        shap_values
    )

    # Possible shape:
    # (samples, features, 1, classes)
    if shap_array.ndim == 4:

        predicted_shap = (
            shap_array[
                0,
                :,
                0,
                predicted_index
            ]
        )

    # Possible shape:
    # (samples, features, classes)
    elif shap_array.ndim == 3:

        if shap_array.shape[-1] == len(class_names):

            predicted_shap = (
                shap_array[
                    0,
                    :,
                    predicted_index
                ]
            )

        else:

            predicted_shap = (
                shap_array[0, :, 0]
            )

    elif shap_array.ndim == 2:

        predicted_shap = shap_array[0]

    else:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_array.shape}"
        )


predicted_shap = np.asarray(
    predicted_shap
).reshape(-1)


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

if len(predicted_shap) != len(
    selected_feature_names
):

    raise ValueError(
        "SHAP feature count does not match "
        "selected feature count. "
        f"SHAP={len(predicted_shap)}, "
        f"Features={len(selected_feature_names)}"
    )


# ============================================================
# CREATE FEATURE EXPLANATION TABLE
# ============================================================

selected_values = selected_data[
    ROW_INDEX
]

explanation_df = pd.DataFrame(
    {
        "Feature":
            selected_feature_names,

        "Processed_Value":
            selected_values,

        "SHAP_Value":
            predicted_shap,

        "Absolute_SHAP":
            np.abs(
                predicted_shap
            )
    }
)


explanation_df = explanation_df.sort_values(
    "Absolute_SHAP",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print()
print("=" * 70)
print(
    f"TOP FEATURES INFLUENCING "
    f"PREDICTION: {predicted_category}"
)
print("=" * 70)


top_features = explanation_df.head(10)


for _, row in top_features.iterrows():

    direction = (
        "supports prediction"
        if row["SHAP_Value"] > 0
        else "opposes prediction"
    )

    print(
        f"{row['Feature']:<22} "
        f"SHAP={row['SHAP_Value']:+.6f}  "
        f"{direction}"
    )


# ============================================================
# SAVE SHAP RESULTS
# ============================================================

os.makedirs(
    "report",
    exist_ok=True
)


output_file = (
    f"report/"
    f"record_{RECORD_NUMBER}_shap_explanation.csv"
)


explanation_df.to_csv(
    output_file,
    index=False
)


print()
print("=" * 70)
print("SHAP EXPLANATION COMPLETED")
print("=" * 70)

print(
    f"Results saved to : {output_file}"
)