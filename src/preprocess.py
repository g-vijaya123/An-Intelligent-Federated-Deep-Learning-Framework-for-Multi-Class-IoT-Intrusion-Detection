"""
Preprocessing pipeline for the UNSW-NB15 IDS project.

Steps:
    1. Load train and test CSV files.
    2. Create binary or multiclass targets.
    3. Encode categorical columns.
    4. Optionally remove leakage features.
    5. Convert all features to numeric values.
    6. Standardize features using StandardScaler.
    7. Select best K features using SelectKBest.
    8. Save preprocessing objects for future unseen/custom data.
    9. Reshape features for CNN-LSTM.
    10. One-hot encode target labels.
"""

import os
import joblib

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.utils import to_categorical


# ============================================================
# DATASET FILES
# ============================================================

# The two files have swapped names in this dataset copy.
TRAIN_FILE = "UNSW_NB15_testing-set.csv"
TEST_FILE = "UNSW_NB15_training-set.csv"


# ============================================================
# COLUMN CONFIGURATION
# ============================================================

# Categorical/text columns
CATEGORICAL_COLS = [
    "proto",
    "service",
    "state"
]


# Features that can strongly leak label information.
LEAKAGE_COLS = [
    "sttl",
    "dttl",
    "ct_state_ttl",
    "swin",
    "dwin",
    "state"
]


# Columns that must never be used as model input.
NON_FEATURE_COLS = [
    "id",
    "label",
    "attack_cat"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_raw(data_dir="data"):
    """
    Load raw UNSW-NB15 train and test datasets.
    """

    train_path = os.path.join(data_dir, TRAIN_FILE)
    test_path = os.path.join(data_dir, TEST_FILE)

    missing = [
        path
        for path in (train_path, test_path)
        if not os.path.isfile(path)
    ]

    if missing:
        raise FileNotFoundError(
            "UNSW-NB15 CSV files are missing.\n\n"
            "Place both files inside the data folder:\n"
            f"  - {TRAIN_FILE}\n"
            f"  - {TEST_FILE}\n\n"
            "Missing path(s):\n  - "
            + "\n  - ".join(missing)
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


# ============================================================
# BUILD TARGETS
# ============================================================

def _build_targets(train, test, mode):
    """
    Build integer target labels.

    binary:
        Normal / Attack

    multiclass:
        Individual UNSW-NB15 attack categories
    """

    if mode == "binary":

        y_train = train["label"].to_numpy()
        y_test = test["label"].to_numpy()

        class_names = [
            "Normal",
            "Attack"
        ]

        return y_train, y_test, class_names


    if mode == "multiclass":

        target_encoder = LabelEncoder()

        combined = pd.concat(
            [
                train["attack_cat"],
                test["attack_cat"]
            ],
            axis=0
        ).astype(str)

        target_encoder.fit(combined)

        y_train = target_encoder.transform(
            train["attack_cat"].astype(str)
        )

        y_test = target_encoder.transform(
            test["attack_cat"].astype(str)
        )

        class_names = list(target_encoder.classes_)

        return y_train, y_test, class_names


    raise ValueError(
        "mode must be 'binary' or 'multiclass'"
    )


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess(
    data_dir="data",
    mode="binary",
    k=32,
    drop_leakage=False,
    split="official",
    test_size=0.2,
    random_state=42,
    verbose=True
):
    """
    Convert raw UNSW-NB15 data into model-ready arrays.

    Returns:
        X_train
        y_train
        X_test
        y_test
        class_names
        feature_names
    """

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    train, test = load_raw(data_dir)


    # --------------------------------------------------------
    # 2. CREATE TRAIN / TEST SPLIT
    # --------------------------------------------------------

    if split == "stratified":

        full = pd.concat(
            [train, test],
            axis=0,
            ignore_index=True
        )

        if mode == "binary":
            strat_col = full["label"]
        else:
            strat_col = full["attack_cat"]

        train, test = train_test_split(
            full,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_col
        )

    elif split != "official":

        raise ValueError(
            "split must be 'official' or 'stratified'"
        )


    # --------------------------------------------------------
    # 3. BUILD TARGET LABELS
    # --------------------------------------------------------

    y_train_int, y_test_int, class_names = _build_targets(
        train,
        test,
        mode
    )


    # --------------------------------------------------------
    # 4. REMOVE TARGET / ID COLUMNS
    # --------------------------------------------------------

    drop_cols = [
        col
        for col in NON_FEATURE_COLS
        if col in train.columns
    ]

    X_train = train.drop(
        columns=drop_cols
    ).copy()

    X_test = test.drop(
        columns=drop_cols
    ).copy()


    # --------------------------------------------------------
    # 5. OPTIONAL LEAKAGE REMOVAL
    # --------------------------------------------------------

    if drop_leakage:

        leak = [
            col
            for col in LEAKAGE_COLS
            if col in X_train.columns
        ]

        X_train = X_train.drop(
            columns=leak
        )

        X_test = X_test.drop(
            columns=leak
        )


    # --------------------------------------------------------
    # 6. SAVE RAW FEATURE ORDER
    # --------------------------------------------------------

    raw_feature_names = X_train.columns.tolist()


    # --------------------------------------------------------
    # 7. ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------------

    categorical_encoders = {}

    for col in CATEGORICAL_COLS:

        if col in X_train.columns:

            encoder = LabelEncoder()

            combined = pd.concat(
                [
                    X_train[col],
                    X_test[col]
                ],
                axis=0
            ).astype(str)

            encoder.fit(combined)

            X_train[col] = encoder.transform(
                X_train[col].astype(str)
            )

            X_test[col] = encoder.transform(
                X_test[col].astype(str)
            )

            # Save encoder for future unseen/custom input
            categorical_encoders[col] = encoder


    # --------------------------------------------------------
    # 8. CONVERT EVERYTHING TO NUMERIC
    # --------------------------------------------------------

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)


    all_feature_names = X_train.columns.tolist()


    # --------------------------------------------------------
    # 9. STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # 10. FEATURE SELECTION
    # --------------------------------------------------------

    k = min(
        k,
        X_train.shape[1]
    )

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train = selector.fit_transform(
        X_train,
        y_train_int
    )

    X_test = selector.transform(
        X_test
    )


    selected_mask = selector.get_support()

    feature_names = [
        name
        for name, keep in zip(
            all_feature_names,
            selected_mask
        )
        if keep
    ]


    # --------------------------------------------------------
    # 11. SAVE PREPROCESSING ARTIFACTS
    # --------------------------------------------------------
    #
    # These objects are required when completely new network
    # traffic is supplied later.
    #
    # IMPORTANT:
    # New data must NOT fit a new scaler or selector.
    # It must use exactly these fitted objects.
    # --------------------------------------------------------

    if mode == "multiclass" and split == "stratified":

        os.makedirs(
            "models",
            exist_ok=True
        )

        preprocessing_artifacts = {

            # Encoders for proto/service/state
            "categorical_encoders":
                categorical_encoders,

            # StandardScaler fitted on training data
            "scaler":
                scaler,

            # SelectKBest fitted on training data
            "selector":
                selector,

            # Feature order BEFORE feature selection
            "all_feature_names":
                all_feature_names,

            # Original raw feature order
            "raw_feature_names":
                raw_feature_names,

            # Final selected 32 features
            "selected_feature_names":
                feature_names,

            # Output classes
            "class_names":
                class_names,

            # Configuration information
            "drop_leakage":
                drop_leakage,

            "k":
                k,

            "mode":
                mode,

            "split":
                split
        }


        artifact_path = os.path.join(
            "models",
            "preprocessing_multiclass_stratified.joblib"
        )


        joblib.dump(
            preprocessing_artifacts,
            artifact_path
        )


        if verbose:
            print(
                "  preprocessing artifacts saved :",
                artifact_path
            )


    # --------------------------------------------------------
    # 12. RESHAPE FOR CNN / LSTM
    # --------------------------------------------------------

    X_train = X_train.reshape(
        (
            X_train.shape[0],
            X_train.shape[1],
            1
        )
    )

    X_test = X_test.reshape(
        (
            X_test.shape[0],
            X_test.shape[1],
            1
        )
    )


    # --------------------------------------------------------
    # 13. ONE-HOT ENCODE TARGETS
    # --------------------------------------------------------

    n_classes = len(
        class_names
    )

    y_train = to_categorical(
        y_train_int,
        num_classes=n_classes
    )

    y_test = to_categorical(
        y_test_int,
        num_classes=n_classes
    )


    # --------------------------------------------------------
    # 14. DISPLAY INFORMATION
    # --------------------------------------------------------

    if verbose:

        print()
        print("=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

        print(
            f"Mode                 : {mode}"
        )

        print(
            f"Split                : {split}"
        )

        print(
            f"Drop leakage         : {drop_leakage}"
        )

        print(
            f"Selected features    : {len(feature_names)}"
        )

        print(
            f"Number of classes    : {n_classes}"
        )

        print(
            f"Classes              : {class_names}"
        )

        print(
            f"X_train shape        : {X_train.shape}"
        )

        print(
            f"X_test shape         : {X_test.shape}"
        )

        print(
            f"y_train shape        : {y_train.shape}"
        )

        print(
            f"y_test shape         : {y_test.shape}"
        )

        print()
        print("Selected feature names:")

        for index, feature in enumerate(
            feature_names,
            start=1
        ):
            print(
                f"  {index:02d}. {feature}"
            )

        print("=" * 60)


    # --------------------------------------------------------
    # 15. RETURN DATA
    # --------------------------------------------------------

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        feature_names
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=== BINARY PREPROCESSING ===")

    preprocess(
        mode="binary"
    )


    print()
    print("=== MULTICLASS PREPROCESSING ===")

    preprocess(
        mode="multiclass"
    )