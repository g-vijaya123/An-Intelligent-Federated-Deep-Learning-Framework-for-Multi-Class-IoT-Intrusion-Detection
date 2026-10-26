from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import shap

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "fl_3datasets_20percent"

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "federated_20percent_cnn_bigru_mha_corrected.keras"
)

SCALER_FILE = DATA_DIR / "global_train_scaler.npz"

REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = REPORT_DIR / "final_3dataset_shap_explanation.csv"

CLIENTS = [
    "UNSW",
    "CICIoT2023",
    "EdgeIIoTset",
]

FEATURES = [
    "duration",
    "packet_rate",
    "src_packets",
    "dst_packets",
    "src_bytes",
    "dst_bytes",
    "packet_size_mean",
    "packet_size_min",
    "packet_size_max",
    "packet_size_std",
    "ttl",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "iat",
]

CLASSES = [
    "Normal",
    "DDoS",
    "DoS",
    "Reconnaissance",
    "Backdoor",
    "Injection",
    "BruteForce",
    "MITM",
    "Malware",
    "Exploits",
    "Fuzzers",
    "Generic",
    "OtherAttack",
]

STANDARDIZED_CLIP = 20.0

# Small background keeps SHAP practical on CPU.
BACKGROUND_SIZE = 50

# Number of test records to search from each dataset.
SEARCH_RECORDS_PER_CLIENT = 1000

# Explain one useful correctly predicted attack record.
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HELPER
# ============================================================

def heading(text):
    print()
    print("=" * 75)
    print(text)
    print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

heading("FINAL 3-DATASET SHAP EXPLAINABILITY")

print(f"Project root : {PROJECT_ROOT}")
print(f"Model        : {MODEL_FILE}")
print(f"Scaler       : {SCALER_FILE}")

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Final model not found:\n{MODEL_FILE}"
    )

if not SCALER_FILE.exists():
    raise FileNotFoundError(
        f"Global scaler not found:\n{SCALER_FILE}"
    )

for client in CLIENTS:
    test_file = DATA_DIR / client / "test_raw.csv"

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{test_file}"
        )

print("\nAll required files found.")


# ============================================================
# LOAD SCALER
# ============================================================

heading("LOADING GLOBAL TRAIN-ONLY SCALER")

scaler = np.load(SCALER_FILE, allow_pickle=True)

mean = scaler["mean"].astype(np.float32)
std = scaler["std"].astype(np.float32)

if len(mean) != len(FEATURES):
    raise ValueError(
        f"Scaler feature count mismatch. "
        f"Expected {len(FEATURES)}, got {len(mean)}"
    )

if len(std) != len(FEATURES):
    raise ValueError(
        f"Scaler std count mismatch. "
        f"Expected {len(FEATURES)}, got {len(std)}"
    )

std = np.where(std < 1e-12, 1.0, std).astype(np.float32)

print("Scaler loaded successfully.")
print(f"Number of features: {len(FEATURES)}")


# ============================================================
# LOAD FINAL MODEL
# ============================================================

heading("LOADING FINAL GLOBAL MODEL")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

print("Final global model loaded successfully.")
print(f"Model input shape : {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

if model.output_shape[-1] != len(CLASSES):
    raise ValueError(
        f"Model class count mismatch. "
        f"Expected {len(CLASSES)}, "
        f"got {model.output_shape[-1]}"
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_features(df):
    x = df[FEATURES].copy()

    for column in FEATURES:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce"
        )

    x = x.replace([np.inf, -np.inf], np.nan)

    if x.isna().any().any():
        bad_columns = x.columns[x.isna().any()].tolist()

        raise ValueError(
            f"NaN/Infinity found in columns: {bad_columns}"
        )

    x = x.to_numpy(dtype=np.float32)

    x = (x - mean) / std

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=STANDARDIZED_CLIP,
        neginf=-STANDARDIZED_CLIP,
    )

    x = np.clip(
        x,
        -STANDARDIZED_CLIP,
        STANDARDIZED_CLIP
    )

    return x.reshape(
        len(x),
        len(FEATURES),
        1
    ).astype(np.float32)


# ============================================================
# BUILD SHAP BACKGROUND
# ============================================================

heading("BUILDING SHAP BACKGROUND")

background_parts = []

per_client_background = max(
    1,
    BACKGROUND_SIZE // len(CLIENTS)
)

for client in CLIENTS:

    train_file = (
        DATA_DIR
        / client
        / "train_raw.csv"
    )

    df = pd.read_csv(
        train_file,
        nrows=5000,
        usecols=FEATURES
    )

    sample_size = min(
        per_client_background,
        len(df)
    )

    sampled = df.sample(
        n=sample_size,
        random_state=SEED
    )

    x_part = preprocess_features(sampled)

    background_parts.append(x_part)

    print(
        f"{client}: "
        f"{sample_size} background records"
    )

background = np.concatenate(
    background_parts,
    axis=0
)

print(
    f"\nFinal SHAP background shape: "
    f"{background.shape}"
)


# ============================================================
# FIND A CORRECTLY PREDICTED ATTACK
# ============================================================

heading("SEARCHING FOR A CORRECTLY PREDICTED ATTACK")

selected = None

for client in CLIENTS:

    test_file = (
        DATA_DIR
        / client
        / "test_raw.csv"
    )

    df = pd.read_csv(
        test_file,
        nrows=SEARCH_RECORDS_PER_CLIENT
    )

    if "label" not in df.columns:
        print(
            f"{client}: label column not found. Skipping."
        )
        continue

    x = preprocess_features(df)

    probabilities = model.predict(
        x,
        batch_size=512,
        verbose=0
    )

    pred_ids = np.argmax(
        probabilities,
        axis=1
    )

    pred_labels = [
        CLASSES[i]
        for i in pred_ids
    ]

    actual_labels = (
        df["label"]
        .astype(str)
        .tolist()
    )

    for i, (
        actual,
        predicted
    ) in enumerate(
        zip(
            actual_labels,
            pred_labels
        )
    ):

        if (
            actual != "Normal"
            and actual == predicted
        ):

            selected = {
                "client": client,
                "row": df.iloc[i],
                "x": x[i:i + 1],
                "actual": actual,
                "predicted": predicted,
                "predicted_id": int(
                    pred_ids[i]
                ),
                "confidence": float(
                    probabilities[
                        i,
                        pred_ids[i]
                    ]
                ),
                "probabilities":
                    probabilities[i],
            }

            break

    if selected is not None:
        break


if selected is None:
    raise RuntimeError(
        "No correctly predicted attack was found "
        "in the searched records. "
        "Increase SEARCH_RECORDS_PER_CLIENT."
    )


print(
    f"Dataset          : "
    f"{selected['client']}"
)

print(
    f"Actual attack    : "
    f"{selected['actual']}"
)

print(
    f"Predicted attack : "
    f"{selected['predicted']}"
)

print(
    f"Confidence       : "
    f"{selected['confidence'] * 100:.2f}%"
)


# ============================================================
# SHOW TOP CLASS PROBABILITIES
# ============================================================

heading("MODEL CLASS PROBABILITIES")

probabilities = selected["probabilities"]

order = np.argsort(
    probabilities
)[::-1]

for class_id in order[:5]:

    print(
        f"{CLASSES[class_id]:18s}: "
        f"{probabilities[class_id] * 100:7.3f}%"
    )


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

heading("CREATING SHAP EXPLAINER")

print(
    "This may take some time on CPU..."
)

explainer = shap.GradientExplainer(
    model,
    background
)

print("SHAP GradientExplainer created.")


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

heading("CALCULATING SHAP VALUES")

x_explain = selected["x"]

shap_values = explainer.shap_values(
    x_explain
)

print(
    "Raw SHAP output type:",
    type(shap_values)
)


# ============================================================
# NORMALIZE SHAP OUTPUT FORMAT
# ============================================================

predicted_id = selected["predicted_id"]

if isinstance(shap_values, list):

    # Older SHAP format:
    # list[class] -> (samples, features, channels)

    class_shap = np.asarray(
        shap_values[predicted_id]
    )

    class_shap = np.squeeze(
        class_shap[0]
    )

else:

    shap_array = np.asarray(
        shap_values
    )

    print(
        "Raw SHAP array shape:",
        shap_array.shape
    )

    # Common newer format:
    # (samples, features, channels, classes)

    if (
        shap_array.ndim == 4
        and shap_array.shape[-1]
        == len(CLASSES)
    ):

        class_shap = shap_array[
            0,
            :,
            0,
            predicted_id
        ]

    # Possible format:
    # (samples, features, classes)

    elif (
        shap_array.ndim == 3
        and shap_array.shape[-1]
        == len(CLASSES)
    ):

        class_shap = shap_array[
            0,
            :,
            predicted_id
        ]

    # Single-output style

    elif shap_array.ndim == 3:

        class_shap = np.squeeze(
            shap_array[0]
        )

    elif shap_array.ndim == 2:

        class_shap = shap_array[0]

    else:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_array.shape}"
        )


class_shap = np.asarray(
    class_shap,
    dtype=np.float64
).reshape(-1)

if len(class_shap) != len(FEATURES):

    raise ValueError(
        f"Expected {len(FEATURES)} SHAP values, "
        f"got {len(class_shap)}"
    )


# ============================================================
# CREATE EXPLANATION TABLE
# ============================================================

heading(
    f"WHY THE MODEL PREDICTED "
    f"{selected['predicted'].upper()}"
)

raw_values = []

for feature in FEATURES:

    raw_values.append(
        float(
            pd.to_numeric(
                selected["row"][feature],
                errors="coerce"
            )
        )
    )

explanation = pd.DataFrame({
    "Feature": FEATURES,
    "Raw_Value": raw_values,
    "SHAP_Value": class_shap,
})

explanation[
    "Absolute_SHAP"
] = np.abs(
    explanation["SHAP_Value"]
)

explanation[
    "Influence"
] = np.where(
    explanation["SHAP_Value"] >= 0,
    "Supports predicted class",
    "Opposes predicted class"
)

explanation = explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

print(
    "\nTop 10 influential features:\n"
)

for _, row in explanation.head(10).iterrows():

    direction = (
        "supports"
        if row["SHAP_Value"] >= 0
        else "opposes"
    )

    print(
        f"{row['Feature']:20s} "
        f"Raw={row['Raw_Value']:12.5f} | "
        f"SHAP={row['SHAP_Value']:+.6f} | "
        f"{direction} "
        f"{selected['predicted']}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

explanation.insert(
    0,
    "Dataset",
    selected["client"]
)

explanation.insert(
    1,
    "Actual_Attack",
    selected["actual"]
)

explanation.insert(
    2,
    "Predicted_Attack",
    selected["predicted"]
)

explanation.insert(
    3,
    "Confidence",
    selected["confidence"]
)

explanation.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

heading("SHAP EXPLANATION COMPLETE")

print(
    f"Dataset          : "
    f"{selected['client']}"
)

print(
    f"Actual attack    : "
    f"{selected['actual']}"
)

print(
    f"Predicted attack : "
    f"{selected['predicted']}"
)

print(
    f"Confidence       : "
    f"{selected['confidence'] * 100:.2f}%"
)

print(
    "\nIMPORTANT:"
)

print(
    "SHAP explains which input features influenced "
    "the model prediction."
)

print(
    "SHAP values should NOT be interpreted as "
    "the real-world causal reason for the attack."
)

print(
    f"\nSHAP report saved:\n{OUTPUT_CSV}"
)

print(
    "\nSTATUS: PASS"
)