import os
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


DATA_DIR = "data"
MODEL_DIR = "models"

RAW_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "service", "sload", "dload", "spkts", "dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean",
    "trans_depth", "response_body_len", "sjit", "djit",
    "stime", "ltime", "sinpkt", "dinpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label"
]

# No SelectKBest.
# Exclude identifiers, labels and absolute capture timestamps.
EXCLUDED_COLUMNS = [
    "srcip",
    "dstip",
    "stime",
    "ltime",
    "attack_cat",
    "label",
]

FEATURE_COLUMNS = [
    c for c in RAW_COLUMNS
    if c not in EXCLUDED_COLUMNS
]

CATEGORICAL_COLUMNS = [
    "proto",
    "state",
    "service",
]

NUMERIC_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if c not in CATEGORICAL_COLUMNS
]


def clean_attack_category(attack, label):

    try:
        label = int(float(label))
    except:
        label = None

    if label == 0:
        return "Normal"

    value = str(attack).strip()
    value = value.replace('"', "").replace("'", "").strip()

    if value.lower() in ["", "nan", "none", "normal"]:
        return "Normal"

    if value.lower() == "backdoors":
        return "Backdoor"

    return value


def load_full_dataset():

    frames = []

    print("\nLoading all four raw files...")

    for filename in RAW_FILES:

        path = os.path.join(DATA_DIR, filename)

        print("Loading:", filename)

        df = pd.read_csv(
            path,
            header=None,
            names=RAW_COLUMNS,
            low_memory=False
        )

        frames.append(df)

        print(f"  Records: {len(df):,}")

    print("\nCombining datasets...")

    df = pd.concat(
        frames,
        ignore_index=True
    )

    del frames
    gc.collect()

    print(f"Total records: {len(df):,}")

    return df


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("FULL-FEATURE UNSW-NB15 PREPROCESSING")
    print("=" * 70)

    print("\nNumber of model features:", len(FEATURE_COLUMNS))

    print("\nFeatures:")

    for i, feature in enumerate(FEATURE_COLUMNS, 1):
        print(f"{i:02d}. {feature}")

    df = load_full_dataset()

    # --------------------------------------------------
    # Create clean multiclass target
    # --------------------------------------------------

    print("\nCreating multiclass attack labels...")

    df["target"] = [
        clean_attack_category(a, l)
        for a, l in zip(
            df["attack_cat"],
            df["label"]
        )
    ]

    print("\nClass distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Keep only model data
    # --------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].copy()

    del df
    gc.collect()

    # --------------------------------------------------
    # Clean categorical values
    # --------------------------------------------------

    print("\nCleaning categorical features...")

    for column in CATEGORICAL_COLUMNS:

        X[column] = (
            X[column]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # --------------------------------------------------
    # Clean numeric features
    # --------------------------------------------------

    print("Cleaning numeric features...")

    for column in NUMERIC_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        X[column] = (
            X[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    # --------------------------------------------------
    # Split BEFORE fitting encoders/scaler
    # --------------------------------------------------

    print("\nCreating stratified train/test split...")

    X_train, X_test, y_train_text, y_test_text = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    del X, y
    gc.collect()

    print(
        "Training records:",
        f"{len(X_train):,}"
    )

    print(
        "Testing records:",
        f"{len(X_test):,}"
    )

    # --------------------------------------------------
    # Categorical encoder
    # --------------------------------------------------

    print("\nFitting categorical encoder...")

    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train[CATEGORICAL_COLUMNS] = (
        categorical_encoder.fit_transform(
            X_train[CATEGORICAL_COLUMNS]
        )
    )

    X_test[CATEGORICAL_COLUMNS] = (
        categorical_encoder.transform(
            X_test[CATEGORICAL_COLUMNS]
        )
    )

    # --------------------------------------------------
    # Convert all to float32
    # --------------------------------------------------

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # --------------------------------------------------
    # StandardScaler
    # --------------------------------------------------

    print("Fitting StandardScaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_test_scaled = scaler.transform(
        X_test
    ).astype(np.float32)

    del X_train, X_test
    gc.collect()

    # --------------------------------------------------
    # Encode target classes
    # --------------------------------------------------

    print("Encoding attack classes...")

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        y_train_text
    )

    y_test = label_encoder.transform(
        y_test_text
    )

    class_names = list(
        label_encoder.classes_
    )

    print("\nDetected classes:")

    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    # --------------------------------------------------
    # Save preprocessing artifacts
    # --------------------------------------------------

    artifact = {
        "feature_names": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_encoder": categorical_encoder,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "num_features": len(FEATURE_COLUMNS),
        "num_classes": len(class_names),

        # Explicitly record this design choice
        "feature_selection": None,
        "select_k_best": False,

        "excluded_columns": EXCLUDED_COLUMNS,
    }

    artifact_path = os.path.join(
        MODEL_DIR,
        "preprocessing_full_features.joblib"
    )

    joblib.dump(
        artifact,
        artifact_path
    )

    print(
        "\nSaved preprocessing artifact:",
        artifact_path
    )

    # --------------------------------------------------
    # Save processed arrays
    # --------------------------------------------------

    os.makedirs(
        os.path.join(DATA_DIR, "processed_full"),
        exist_ok=True
    )

    print("\nSaving processed arrays...")

    np.save(
        os.path.join(
            DATA_DIR,
            "processed_full",
            "X_train.npy"
        ),
        X_train_scaled
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "processed_full",
            "X_test.npy"
        ),
        X_test_scaled
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "processed_full",
            "y_train.npy"
        ),
        y_train.astype(np.int16)
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "processed_full",
            "y_test.npy"
        ),
        y_test.astype(np.int16)
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        "X_train:",
        X_train_scaled.shape
    )

    print(
        "X_test:",
        X_test_scaled.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    print(
        "Number of features:",
        len(FEATURE_COLUMNS)
    )

    print(
        "Number of classes:",
        len(class_names)
    )

    print("\nNO SelectKBest used.")
    print("NO 32-feature restriction used.")

    print(
        "\nArtifact:",
        artifact_path
    )

    print(
        "Processed data:",
        os.path.join(
            DATA_DIR,
            "processed_full"
        )
    )


if __name__ == "__main__":
    main()