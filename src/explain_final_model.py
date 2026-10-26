import os
import joblib
import numpy as np
import tensorflow as tf
import shap

# ============================================================
# PATHS
# ============================================================

MODEL_FILE = os.path.join(
    "models",
    "federated_cnn_bigru_mha_full.keras"
)

ARTIFACT_FILE = os.path.join(
    "models",
    "preprocessing_full_features.joblib"
)

DATA_DIR = os.path.join(
    "data",
    "processed_full"
)

# Change this number later to explain another test record
SAMPLE_INDEX = 100


# ============================================================
# LOAD MODEL + PREPROCESSING ARTIFACT
# ============================================================

print("\n" + "=" * 70)
print("FINAL CNN-BiGRU-MHA — SHAP EXPLANATION")
print("=" * 70)

print("\nLoading final V2 model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

artifact = joblib.load(
    ARTIFACT_FILE
)

class_names = np.array(
    artifact["class_names"]
)

# Our final model uses all 43 model features.
# Try the artifact's saved feature-name keys.
if "feature_names" in artifact:
    feature_names = list(
        artifact["feature_names"]
    )

elif "all_feature_names" in artifact:
    feature_names = list(
        artifact["all_feature_names"]
    )

else:
    feature_names = [
        "sport",
        "dsport",
        "proto",
        "state",
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "service",
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "swin",
        "dwin",
        "stcpb",
        "dtcpb",
        "smean",
        "dmean",
        "trans_depth",
        "response_body_len",
        "sjit",
        "djit",
        "sinpkt",
        "dinpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "is_sm_ips_ports",
        "ct_state_ttl",
        "ct_flw_http_mthd",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_srv_src",
        "ct_srv_dst",
        "ct_dst_ltm",
        "ct_src_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm"
    ]


print("Model loaded successfully.")
print("Classes:", len(class_names))
print("Features:", len(feature_names))

assert len(feature_names) == 43


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test.npy"
    ),
    mmap_mode="r"
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test.npy"
    )
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# SELECT RECORD
# ============================================================

if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(X_test):
    raise ValueError(
        f"SAMPLE_INDEX must be between 0 and "
        f"{len(X_test) - 1}"
    )

sample = np.asarray(
    X_test[
        SAMPLE_INDEX:SAMPLE_INDEX + 1
    ],
    dtype=np.float32
)

sample_model = np.expand_dims(
    sample,
    axis=-1
)

actual_index = int(
    y_test[SAMPLE_INDEX]
)

actual_class = class_names[
    actual_index
]


# ============================================================
# MODEL PREDICTION
# ============================================================

probabilities = model.predict(
    sample_model,
    verbose=0
)[0]

predicted_index = int(
    np.argmax(probabilities)
)

predicted_class = class_names[
    predicted_index
]

confidence = float(
    probabilities[predicted_index]
)


print("\n" + "=" * 70)
print("PREDICTION")
print("=" * 70)

print(
    "\nTest Record:",
    SAMPLE_INDEX
)

print(
    "Actual Class   :",
    actual_class
)

print(
    "Predicted Class:",
    predicted_class
)

print(
    f"Confidence     : "
    f"{confidence * 100:.2f}%"
)

if actual_index == predicted_index:
    print("Verification   : CORRECT")
else:
    print("Verification   : INCORRECT")


# ============================================================
# SHOW CLASS PROBABILITIES
# ============================================================

print("\nClass Probabilities:")

probability_order = np.argsort(
    probabilities
)[::-1]

for idx in probability_order:

    print(
        f"{class_names[idx]:20s}: "
        f"{probabilities[idx] * 100:8.4f}%"
    )


# ============================================================
# BACKGROUND DATA FOR SHAP
# ============================================================

print("\nPreparing SHAP background...")

rng = np.random.default_rng(42)

BACKGROUND_SIZE = 64

background_indices = rng.choice(
    len(X_test),
    size=BACKGROUND_SIZE,
    replace=False
)

background = np.asarray(
    X_test[background_indices],
    dtype=np.float32
)

background_model = np.expand_dims(
    background,
    axis=-1
)

print(
    "Background shape:",
    background_model.shape
)


# ============================================================
# SHAP GRADIENT EXPLAINER
# ============================================================

print("\nRunning SHAP GradientExplainer...")
print("This can take some time on CPU.")

explainer = shap.GradientExplainer(
    model,
    background_model
)

shap_values = explainer.shap_values(
    sample_model
)


# ============================================================
# NORMALIZE SHAP OUTPUT FORMAT
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP format:
    # list[class] -> (samples, features, channels)

    predicted_shap = np.asarray(
        shap_values[predicted_index]
    )[0]

else:

    shap_array = np.asarray(
        shap_values
    )

    print(
        "Raw SHAP output shape:",
        shap_array.shape
    )

    # Expected newer format may be:
    # (samples, features, channels, classes)

    if shap_array.ndim == 4:

        predicted_shap = shap_array[
            0,
            :,
            :,
            predicted_index
        ]

    # Possible:
    # (samples, features, classes)
    elif shap_array.ndim == 3:

        if (
            shap_array.shape[-1]
            == len(class_names)
        ):
            predicted_shap = shap_array[
                0,
                :,
                predicted_index
            ]
        else:
            predicted_shap = shap_array[0]

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


if len(predicted_shap) != 43:
    raise ValueError(
        "Expected 43 SHAP feature values, "
        f"but received {len(predicted_shap)}."
    )


# ============================================================
# TOP IMPORTANT FEATURES
# ============================================================

importance_order = np.argsort(
    np.abs(predicted_shap)
)[::-1]

TOP_K = 10

print("\n" + "=" * 70)
print(
    f"WHY THE MODEL PREDICTED: "
    f"{predicted_class}"
)
print("=" * 70)

print(
    "\nTop feature contributions "
    "(SHAP values):\n"
)

for rank, feature_idx in enumerate(
    importance_order[:TOP_K],
    start=1
):

    shap_value = float(
        predicted_shap[feature_idx]
    )

    feature_value = float(
        sample[0, feature_idx]
    )

    if shap_value > 0:
        direction = (
            f"supports {predicted_class}"
        )
    else:
        direction = (
            f"opposes {predicted_class}"
        )

    print(
        f"{rank:2d}. "
        f"{feature_names[feature_idx]:22s} "
        f"SHAP={shap_value:+.6f}  "
        f"ProcessedValue={feature_value:+.4f}  "
        f"→ {direction}"
    )


# ============================================================
# SIMPLE HUMAN-READABLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPLANATION SUMMARY")
print("=" * 70)

top_features = [
    feature_names[i]
    for i in importance_order[:5]
]

print(
    f"\nThe federated CNN-BiGRU-MHA model "
    f"predicted this traffic record as "
    f"'{predicted_class}' with "
    f"{confidence * 100:.2f}% confidence."
)

print(
    "\nThe five most influential model "
    "features for this prediction were:"
)

for i, name in enumerate(
    top_features,
    start=1
):
    print(
        f"{i}. {name}"
    )

print(
    "\nImportant: SHAP explains which features "
    "influenced the model prediction. "
    "It does NOT prove that those features "
    "caused the network attack."
)

print("\nSHAP explanation completed.")