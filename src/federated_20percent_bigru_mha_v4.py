# ============================================================

# federated_20percent_bigru_mha.py

#

# FINAL CORRECTED 3-DATASET FEDERATED TRAINING

#

# Datasets / Clients:

#   1. UNSW-NB15

#   2. CICIoT2023

#   3. Edge-IIoTset

#

# Pipeline:

#   20% prepared data

#       -> shared train-only scaler

#       -> automatic random-access cache

#       -> class-aware tempered sampling

#       -> CNN + BiGRU + Multi-Head Attention

#       -> frequent equal-client federated averaging

#       -> ONE final global model

#       -> test each client ONCE at the end

#

# IMPORTANT:

#   - Does NOT recreate the 20% dataset.

#   - Does NOT refit the scaler.

#   - Does NOT use test data during training.

#   - Does NOT select a "best round" using test data.

#   - Keeps all 16 harmonized features.

#   - Keeps all 13 common output classes.

# ============================================================



from pathlib import Path

import gc

import json

import math

import os

import random

import time



import numpy as np

import pandas as pd

import tensorflow as tf



from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix,

    f1_score,

    precision_score,

    recall_score,

)





# ============================================================

# 1. PATHS

# ============================================================



PROJECT_ROOT = Path(__file__).resolve().parents[1]



DATA_DIR = PROJECT_ROOT / "data" / "fl_3datasets_20percent"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "report"



MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)



SCALER_FILE = DATA_DIR / "global_train_scaler.npz"



FINAL_MODEL_FILE = (

    MODEL_DIR / "federated_20percent_cnn_bigru_mha_v4.keras"

)



ROUND_METRICS_FILE = (

    REPORT_DIR / "federated_20percent_v4_training_metrics.csv"

)



FINAL_REPORT_FILE = (

    REPORT_DIR / "federated_20percent_v4_final_report.txt"

)



CONFUSION_MATRIX_FILE = (

    REPORT_DIR / "federated_20percent_v4_confusion_matrix.csv"

)



PER_CLASS_METRICS_FILE = (

    REPORT_DIR / "federated_20percent_v4_per_class_metrics.csv"

)



CACHE_DIR = DATA_DIR / "random_access_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)





# ============================================================

# 2. REPRODUCIBILITY

# ============================================================



SEED = 42



os.environ["PYTHONHASHSEED"] = str(SEED)



random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)





# ============================================================

# 3. CLIENTS

# ============================================================



CLIENTS = [

    "UNSW",

    "CICIoT2023",

    "EdgeIIoTset",

]





# ============================================================

# 4. FEATURES

# ============================================================



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



NUM_FEATURES = len(FEATURES)





# ============================================================

# 5. COMMON ATTACK CLASSES

# ============================================================



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



NUM_CLASSES = len(CLASSES)



CLASS_TO_ID = {

    class_name: index

    for index, class_name in enumerate(CLASSES)

}





# ============================================================

# 6. TRAINING CONFIGURATION

# ============================================================



# Frequent synchronization:

# Smaller local training blocks reduce client drift.

NUM_ROUNDS = 10



# Each client performs only this many mini-batch updates

# before sending its weights back to the server.

LOCAL_STEPS_PER_ROUND = 100



BATCH_SIZE = 1024



LEARNING_RATE = 5e-5



CSV_CHUNK_SIZE = 200_000



EVAL_BATCH_SIZE = 4096



STANDARDIZED_CLIP = 20.0



# Tempered class sampling.

#

# alpha = 1.0 -> follows original class imbalance

# alpha = 0.0 -> uniform classes

# 0.65 gives minority attacks more opportunity without

# making extremely rare classes dominate training.

SAMPLING_ALPHA = 0.30

# V2 experiment: uniform sampling across classes naturally present at each client.
# This tests whether stronger minority-class exposure improves macro-F1.
# Held-out test data remains unused until final evaluation.



# Equal-client aggregation:

# UNSW, CIC and Edge each receive equal aggregation weight.

CLIENT_AGGREGATION_WEIGHT = 1.0 / len(CLIENTS)

# V4 validation protocol.
# Existing train_raw/cache is split class-wise into 90% effective training
# and 10% validation. Existing test_raw.csv remains untouched until the end.
VALIDATION_FRACTION = 0.10
SPLIT_SEED = 42
MAX_VALIDATION_PER_CLASS = 10_000






# ============================================================

# 7. BASIC CHECKS

# ============================================================



print("\n" + "=" * 75)

print("V4 CLEAN VALIDATION-GUIDED 3-DATASET FEDERATED IDS")

print("=" * 75)



print(f"\nProject root : {PROJECT_ROOT}")

print(f"Data folder  : {DATA_DIR}")

print(f"Clients      : {CLIENTS}")

print(f"Features     : {NUM_FEATURES}")

print(f"Classes      : {NUM_CLASSES}")

print(f"Rounds       : {NUM_ROUNDS}")

print(f"Local steps  : {LOCAL_STEPS_PER_ROUND}")

print(f"Batch size   : {BATCH_SIZE}")

print(f"Learning rate: {LEARNING_RATE}")



if not DATA_DIR.exists():

    raise FileNotFoundError(

        f"\n20% data folder not found:\n{DATA_DIR}"

    )



if not SCALER_FILE.exists():

    raise FileNotFoundError(

        f"\nGlobal TRAIN-ONLY scaler not found:\n{SCALER_FILE}"

    )



for client in CLIENTS:

    train_file = DATA_DIR / client / "train_raw.csv"

    test_file = DATA_DIR / client / "test_raw.csv"



    if not train_file.exists():

        raise FileNotFoundError(

            f"Training file missing:\n{train_file}"

        )



    if not test_file.exists():

        raise FileNotFoundError(

            f"Test file missing:\n{test_file}"

        )





# ============================================================

# 8. LOAD SHARED TRAIN-ONLY SCALER

# ============================================================



print("\n" + "=" * 75)

print("LOADING GLOBAL TRAIN-ONLY SCALER")

print("=" * 75)



scaler_data = np.load(

    SCALER_FILE,

    allow_pickle=True,

)



SCALER_MEAN = np.asarray(

    scaler_data["mean"],

    dtype=np.float32,

)



SCALER_STD = np.asarray(

    scaler_data["std"],

    dtype=np.float32,

)



if len(SCALER_MEAN) != NUM_FEATURES:

    raise ValueError(

        "Scaler feature count does not match model feature count."

    )



if len(SCALER_STD) != NUM_FEATURES:

    raise ValueError(

        "Scaler std count does not match model feature count."

    )



SCALER_STD = np.where(

    np.abs(SCALER_STD) < 1e-12,

    1.0,

    SCALER_STD,

).astype(np.float32)



print("Scaler loaded successfully.")

print("Scaler was already fitted using TRAINING DATA ONLY.")





# ============================================================

# 9. HELPER: COUNT CSV ROWS

# ============================================================



def count_csv_rows(csv_file):

    """

    Counts data rows without loading the complete CSV into RAM.

    Header is excluded.

    """



    count = 0



    with open(

        csv_file,

        "r",

        encoding="utf-8",

        errors="ignore",

    ) as file:

        # Skip header

        next(file)



        for _ in file:

            count += 1



    return count





# ============================================================

# 10. BUILD RANDOM-ACCESS TRAINING CACHE

# ============================================================



def cache_paths(client):

    client_cache = CACHE_DIR / client

    client_cache.mkdir(

        parents=True,

        exist_ok=True,

    )



    return {

        "x": client_cache / "X_train.npy",

        "y": client_cache / "y_train.npy",

        "meta": client_cache / "cache_metadata.json",

    }





def build_client_cache(client):

    """

    Converts train_raw.csv to disk-backed NumPy arrays.



    Why?



    The old CSV generator restarted from the beginning each

    federated round. CICIoT2023 therefore did not receive fair

    random access to its entire training partition.



    This cache allows sampling from ANY training record.

    """



    paths = cache_paths(client)



    if (

        paths["x"].exists()

        and paths["y"].exists()

        and paths["meta"].exists()

    ):

        try:

            with open(

                paths["meta"],

                "r",

                encoding="utf-8",

            ) as file:

                metadata = json.load(file)



            if (

                metadata.get("num_features") == NUM_FEATURES

                and metadata.get("classes") == CLASSES

                and metadata.get("completed") is True

            ):

                print(

                    f"\n{client}: existing random-access "

                    f"cache found. Reusing it."

                )



                return metadata



        except Exception:

            pass



    print("\n" + "-" * 75)

    print(f"BUILDING RANDOM-ACCESS CACHE: {client}")

    print("-" * 75)



    train_file = (

        DATA_DIR

        / client

        / "train_raw.csv"

    )



    print("Counting training rows...")



    total_rows = count_csv_rows(train_file)



    print(

        f"{client} training records: "

        f"{total_rows:,}"

    )



    # Disk-backed arrays.

    x_memmap = np.lib.format.open_memmap(

        paths["x"],

        mode="w+",

        dtype=np.float32,

        shape=(total_rows, NUM_FEATURES),

    )



    y_memmap = np.lib.format.open_memmap(

        paths["y"],

        mode="w+",

        dtype=np.int16,

        shape=(total_rows,),

    )



    position = 0

    class_counts = np.zeros(

        NUM_CLASSES,

        dtype=np.int64,

    )



    required_columns = FEATURES + [

        "label",

        "label_id",

    ]



    for chunk_number, chunk in enumerate(

        pd.read_csv(

            train_file,

            usecols=lambda c: c in required_columns,

            chunksize=CSV_CHUNK_SIZE,

        ),

        start=1,

    ):

        # -------------------------------

        # Features

        # -------------------------------



        x = (

            chunk[FEATURES]

            .apply(

                pd.to_numeric,

                errors="coerce",

            )

            .to_numpy(dtype=np.float32)

        )



        if np.isnan(x).any():

            raise ValueError(

                f"{client}: NaN detected while "

                f"building cache."

            )



        if np.isinf(x).any():

            raise ValueError(

                f"{client}: Infinity detected while "

                f"building cache."

            )



        # -------------------------------

        # Labels

        # -------------------------------



        if "label_id" in chunk.columns:

            y = pd.to_numeric(

                chunk["label_id"],

                errors="coerce",

            ).to_numpy()



            if np.isnan(y).any():

                raise ValueError(

                    f"{client}: invalid label_id found."

                )



            y = y.astype(np.int16)



        else:

            y = (

                chunk["label"]

                .map(CLASS_TO_ID)

                .to_numpy()

            )



            if pd.isna(y).any():

                raise ValueError(

                    f"{client}: unknown label found."

                )



            y = y.astype(np.int16)



        if (

            np.any(y < 0)

            or np.any(y >= NUM_CLASSES)

        ):

            raise ValueError(

                f"{client}: label ID outside expected "

                f"0..{NUM_CLASSES - 1}."

            )



        n = len(chunk)



        x_memmap[

            position:position + n

        ] = x



        y_memmap[

            position:position + n

        ] = y



        class_counts += np.bincount(

            y,

            minlength=NUM_CLASSES,

        )



        position += n



        print(

            f"\r{client}: cached "

            f"{position:,}/{total_rows:,}",

            end="",

            flush=True,

        )



    print()



    if position != total_rows:

        raise RuntimeError(

            f"{client}: cache row mismatch. "

            f"Expected {total_rows:,}, "

            f"wrote {position:,}."

        )



    x_memmap.flush()

    y_memmap.flush()



    del x_memmap

    del y_memmap



    metadata = {

        "client": client,

        "records": int(total_rows),

        "num_features": NUM_FEATURES,

        "features": FEATURES,

        "classes": CLASSES,

        "class_counts": {

            CLASSES[i]: int(class_counts[i])

            for i in range(NUM_CLASSES)

        },

        "completed": True,

    }



    with open(

        paths["meta"],

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            metadata,

            file,

            indent=2,

        )



    print(f"{client}: cache completed.")



    print("\nClass distribution:")



    for class_index, class_name in enumerate(CLASSES):

        print(

            f"  {class_name:<18} "

            f"{class_counts[class_index]:>10,}"

        )



    gc.collect()



    return metadata





# ============================================================

# 11. BUILD / VERIFY ALL CLIENT CACHES

# ============================================================



print("\n" + "=" * 75)

print("PREPARING RANDOM-ACCESS TRAINING DATA")

print("=" * 75)



CLIENT_METADATA = {}



for client in CLIENTS:

    CLIENT_METADATA[client] = (

        build_client_cache(client)

    )





# ============================================================

# 12. LOAD CACHE AS MEMORY-MAPPED ARRAYS

# ============================================================



TRAIN_X = {}

TRAIN_Y = {}

CLASS_INDICES = {}

CLASS_SAMPLING_PROBS = {}
VAL_INDICES = {}
CLIENT_PRESENT_CLASSES = {}






def prepare_client_sampling(client):
    """
    Create deterministic class-wise 90/10 train/validation pools.

    Only the 90% effective-training pool is used by the local sampler.
    Validation is used for round/model selection.
    test_raw.csv is not touched here.
    """

    paths = cache_paths(client)

    x = np.load(paths["x"], mmap_mode="r")
    y = np.load(paths["y"], mmap_mode="r")

    TRAIN_X[client] = x
    TRAIN_Y[client] = y

    train_class_indices = {}
    validation_parts = []
    available_classes = []
    train_counts = []

    print("\n" + "-" * 75)
    print(f"PREPARING TRAIN/VALIDATION POOLS: {client}")
    print("-" * 75)

    for class_id, class_name in enumerate(CLASSES):
        indices = np.flatnonzero(
            np.asarray(y) == class_id
        ).astype(np.int64)

        if len(indices) == 0:
            continue

        # Deterministic class-specific shuffle.
        rng = np.random.default_rng(
            SPLIT_SEED + class_id * 1009 + CLIENTS.index(client) * 100_003
        )
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        n_val = max(1, int(round(len(shuffled) * VALIDATION_FRACTION)))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0

        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]

        # Fixed capped validation subset for efficient per-round model selection.
        if len(val_idx) > MAX_VALIDATION_PER_CLASS:
            val_idx = np.sort(
                rng.choice(
                    val_idx,
                    size=MAX_VALIDATION_PER_CLASS,
                    replace=False,
                )
            )
        else:
            val_idx = np.sort(val_idx)

        train_idx = np.sort(train_idx)

        train_class_indices[class_id] = train_idx
        validation_parts.append(val_idx)
        available_classes.append(class_id)
        train_counts.append(len(train_idx))

        print(
            f"{class_name:<18}: "
            f"train={len(train_idx):>9,} | "
            f"validation={len(val_idx):>7,}"
        )

    if not available_classes:
        raise RuntimeError(f"{client}: no classes found.")

    counts = np.asarray(train_counts, dtype=np.float64)
    tempered = np.power(counts, SAMPLING_ALPHA)
    probabilities = tempered / tempered.sum()

    CLASS_INDICES[client] = train_class_indices
    CLIENT_PRESENT_CLASSES[client] = np.asarray(
        available_classes, dtype=np.int32
    )

    if validation_parts:
        VAL_INDICES[client] = np.sort(
            np.concatenate(validation_parts).astype(np.int64)
        )
    else:
        VAL_INDICES[client] = np.empty(0, dtype=np.int64)

    CLASS_SAMPLING_PROBS[client] = {
        "classes": np.asarray(available_classes, dtype=np.int16),
        "probabilities": probabilities,
    }

    print("\nV4 training sampling probabilities:")
    for class_id, probability in zip(available_classes, probabilities):
        print(
            f"  {CLASSES[class_id]:<18}: "
            f"{probability * 100:6.2f}%"
        )

    print(
        f"Validation subset used for model selection: "
        f"{len(VAL_INDICES[client]):,} records"
    )



for client in CLIENTS:

    prepare_client_sampling(client)

# ============================================================
# V4: RE-FIT SHARED SCALER ON EFFECTIVE TRAINING ROWS ONLY
# ============================================================

# V3 loaded the earlier train_raw-only scaler above only for compatibility.
# V4 replaces its statistics here BEFORE any model training/validation.
# CLASS_INDICES contains only the 90% effective-training rows created by
# prepare_client_sampling(). Validation indices and test_raw.csv are excluded.

V4_SCALER_FILE = DATA_DIR / "v4_effective_train_scaler.npz"

def fit_v4_effective_train_scaler():
    global SCALER_MEAN, SCALER_STD

    print("\n" + "=" * 75)
    print("FITTING V4 EFFECTIVE-TRAIN-ONLY SCALER")
    print("=" * 75)

    n_total = 0
    sum_total = np.zeros(NUM_FEATURES, dtype=np.float64)
    sumsq_total = np.zeros(NUM_FEATURES, dtype=np.float64)

    for client in CLIENTS:
        n_client = 0
        sum_client = np.zeros(NUM_FEATURES, dtype=np.float64)
        sumsq_client = np.zeros(NUM_FEATURES, dtype=np.float64)

        for class_id, indices in CLASS_INDICES[client].items():
            for start in range(0, len(indices), CSV_CHUNK_SIZE):
                idx = indices[start:start + CSV_CHUNK_SIZE]
                x = np.asarray(TRAIN_X[client][idx], dtype=np.float64)

                if not np.isfinite(x).all():
                    raise ValueError(
                        f"{client}: non-finite feature found during V4 scaler fit."
                    )

                n_client += len(x)
                sum_client += x.sum(axis=0, dtype=np.float64)
                sumsq_client += np.square(x).sum(axis=0, dtype=np.float64)

        if n_client == 0:
            raise RuntimeError(f"{client}: no effective-training rows.")

        n_total += n_client
        sum_total += sum_client
        sumsq_total += sumsq_client
        print(f"{client:<12}: effective-training rows = {n_client:,}")

    mean = sum_total / n_total
    var = (sumsq_total / n_total) - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std = np.where(np.abs(std) < 1e-12, 1.0, std)

    SCALER_MEAN = mean.astype(np.float32)
    SCALER_STD = std.astype(np.float32)

    np.savez(
        V4_SCALER_FILE,
        mean=SCALER_MEAN,
        std=SCALER_STD,
        features=np.asarray(FEATURES),
        total_training_records=np.asarray(n_total, dtype=np.int64),
        validation_fraction=np.asarray(VALIDATION_FRACTION, dtype=np.float64),
        split_seed=np.asarray(SPLIT_SEED, dtype=np.int64),
    )

    print(f"Total scaler-fit rows : {n_total:,}")
    print(f"V4 scaler saved       : {V4_SCALER_FILE}")
    print("Validation rows in scaler fit: 0")
    print("Test rows in scaler fit      : 0")

fit_v4_effective_train_scaler()






# ============================================================

# 13. PREPROCESS BATCH

# ============================================================



def preprocess_features(x):

    x = np.asarray(

        x,

        dtype=np.float32,

    )



    x = (

        x - SCALER_MEAN

    ) / SCALER_STD



    x = np.nan_to_num(

        x,

        nan=0.0,

        posinf=STANDARDIZED_CLIP,

        neginf=-STANDARDIZED_CLIP,

    )



    x = np.clip(

        x,

        -STANDARDIZED_CLIP,

        STANDARDIZED_CLIP,

    )



    # CNN / GRU sequence representation:

    # 16 semantic features -> 16 time-like positions.

    x = x.reshape(

        -1,

        NUM_FEATURES,

        1,

    )



    return x.astype(

        np.float32,

        copy=False,

    )





# ============================================================

# 14. CLASS-AWARE RANDOM BATCH SAMPLER

# ============================================================



def sample_training_batch(

    client,

    batch_size,

    rng,

):

    """

    Randomly samples across the ENTIRE client training set.



    This fixes the old problem where CIC training repeatedly

    started at the beginning of its large CSV.

    """



    sampling = (

        CLASS_SAMPLING_PROBS[client]

    )



    available_classes = (

        sampling["classes"]

    )



    probabilities = (

        sampling["probabilities"]

    )



    sampled_classes = rng.choice(

        available_classes,

        size=batch_size,

        replace=True,

        p=probabilities,

    )



    sampled_indices = np.empty(

        batch_size,

        dtype=np.int64,

    )



    # Select random records from each sampled class.

    for class_id in np.unique(

        sampled_classes

    ):

        positions = np.flatnonzero(

            sampled_classes == class_id

        )



        pool = (

            CLASS_INDICES[client][

                int(class_id)

            ]

        )



        selected = rng.choice(

            pool,

            size=len(positions),

            replace=True,

        )



        sampled_indices[

            positions

        ] = selected



    # h5py-like sorted indexing is not required here because

    # NumPy memmap supports arbitrary integer indexing.

    x_batch = np.asarray(

        TRAIN_X[client][sampled_indices],

        dtype=np.float32,

    )



    y_batch = np.asarray(

        TRAIN_Y[client][sampled_indices],

        dtype=np.int32,

    )



    x_batch = preprocess_features(

        x_batch

    )



    return x_batch, y_batch





# ============================================================

# 15. MODEL

# ============================================================



def build_model():

    """

    CNN + BiGRU + Multi-Head Attention.



    BatchNormalization has intentionally been replaced by

    LayerNormalization because BatchNorm moving statistics

    can become incompatible across strongly non-IID clients.

    """



    inputs = tf.keras.Input(

        shape=(NUM_FEATURES, 1),

        name="network_features",

    )



    # --------------------------------------------------------

    # CNN feature extraction

    # --------------------------------------------------------



    x = tf.keras.layers.Conv1D(

        filters=64,

        kernel_size=3,

        padding="same",

        activation="relu",

        name="conv1",

    )(inputs)



    x = tf.keras.layers.LayerNormalization(

        name="layer_norm_1",

    )(x)



    x = tf.keras.layers.Conv1D(

        filters=128,

        kernel_size=3,

        padding="same",

        activation="relu",

        name="conv2",

    )(x)



    x = tf.keras.layers.LayerNormalization(

        name="layer_norm_2",

    )(x)



    # --------------------------------------------------------

    # Bidirectional GRU

    # --------------------------------------------------------



    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.GRU(

            64,

            return_sequences=True,

            dropout=0.10,

        ),

        name="bigru_1",

    )(x)



    # --------------------------------------------------------

    # Multi-Head Attention

    # --------------------------------------------------------



    attention_output = (

        tf.keras.layers.MultiHeadAttention(

            num_heads=4,

            key_dim=32,

            dropout=0.10,

            name="multi_head_attention",

        )(

            query=x,

            value=x,

            key=x,

        )

    )



    x = tf.keras.layers.Add(

        name="attention_residual",

    )([

        x,

        attention_output,

    ])



    x = tf.keras.layers.LayerNormalization(

        name="attention_layer_norm",

    )(x)



    # --------------------------------------------------------

    # Second BiGRU

    # --------------------------------------------------------



    x = tf.keras.layers.Bidirectional(

        tf.keras.layers.GRU(

            32,

            return_sequences=False,

            dropout=0.10,

        ),

        name="bigru_2",

    )(x)



    # --------------------------------------------------------

    # Classification head

    # --------------------------------------------------------



    x = tf.keras.layers.Dense(

        128,

        activation="relu",

        name="dense_128",

    )(x)



    x = tf.keras.layers.Dropout(

        0.30,

        name="dropout_1",

    )(x)



    x = tf.keras.layers.Dense(

        64,

        activation="relu",

        name="dense_64",

    )(x)



    x = tf.keras.layers.Dropout(

        0.20,

        name="dropout_2",

    )(x)



    outputs = tf.keras.layers.Dense(

        NUM_CLASSES,

        activation="softmax",

        name="attack_prediction",

    )(x)



    model = tf.keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="Federated_CNN_BiGRU_MHA",

    )



    optimizer = tf.keras.optimizers.Adam(

        learning_rate=LEARNING_RATE,

        clipnorm=1.0,

    )



    model.compile(

        optimizer=optimizer,

        loss=tf.keras.losses.SparseCategoricalCrossentropy(),

        metrics=[

            tf.keras.metrics.SparseCategoricalAccuracy(

                name="accuracy"

            )

        ],

    )



    return model





# ============================================================

# 16. INITIAL GLOBAL MODEL

# ============================================================



print("\n" + "=" * 75)

print("BUILDING GLOBAL CNN-BiGRU-MHA MODEL")

print("=" * 75)



global_model = build_model()



global_model.summary()



print(

    f"\nTotal parameters: "

    f"{global_model.count_params():,}"

)





# ============================================================

# 17. FEDERATED WEIGHT AVERAGING

# ============================================================



def equal_client_average(

    client_weight_sets,

):

    """

    Equal-client parameter averaging.



    Each dataset/client contributes equally.



    This is deliberately reported as:

        Equal-client federated averaging / FedAvg-style



    It is NOT sample-count-weighted canonical FedAvg.

    """



    if len(client_weight_sets) != len(CLIENTS):

        raise ValueError(

            "Expected one weight set per client."

        )



    averaged_weights = []



    for layer_weights in zip(

        *client_weight_sets

    ):

        stacked = np.stack(

            layer_weights,

            axis=0,

        )



        averaged = np.mean(

            stacked,

            axis=0,

        )



        averaged_weights.append(

            averaged

        )



    return averaged_weights





# ============================================================

# 18. LOCAL TRAINING

# ============================================================



def train_client(

    client,

    global_weights,

    round_number,

):

    """

    Train one client for a small number of random mini-batch

    updates, then return its model parameters.

    """



    tf.keras.backend.clear_session()



    local_model = build_model()



    local_model.set_weights(

        global_weights

    )



    # Different deterministic RNG stream for every

    # client and round.

    client_number = (

        CLIENTS.index(client) + 1

    )



    rng = np.random.default_rng(

        SEED

        + round_number * 10_000

        + client_number * 1_000

    )



    loss_metric = tf.keras.metrics.Mean()



    accuracy_metric = (

        tf.keras.metrics.SparseCategoricalAccuracy()

    )



    start_time = time.time()



    print(

        f"\nTraining {client} "

        f"({LOCAL_STEPS_PER_ROUND} local steps)"

    )



    for step in range(

        1,

        LOCAL_STEPS_PER_ROUND + 1,

    ):

        x_batch, y_batch = (

            sample_training_batch(

                client=client,

                batch_size=BATCH_SIZE,

                rng=rng,

            )

        )



        result = local_model.train_on_batch(

            x_batch,

            y_batch,

            return_dict=True,

        )



        loss_value = float(

            result["loss"]

        )



        # Keras train_on_batch accuracy is cumulative

        # depending on metric state. For readable progress,

        # independently update our own metric.

        predictions = local_model(

            x_batch,

            training=False,

        )



        loss_metric.update_state(

            loss_value

        )



        accuracy_metric.update_state(

            y_batch,

            predictions,

        )



        if (

            step == 1

            or step % 20 == 0

            or step == LOCAL_STEPS_PER_ROUND

        ):

            elapsed = (

                time.time() - start_time

            )



            print(

                f"\r  Step "

                f"{step:>4}/{LOCAL_STEPS_PER_ROUND}"

                f" | Loss "

                f"{loss_metric.result().numpy():.4f}"

                f" | Acc "

                f"{accuracy_metric.result().numpy() * 100:.2f}%"

                f" | {elapsed:.1f}s",

                end="",

                flush=True,

            )



    print()



    local_weights = (

        local_model.get_weights()

    )



    local_loss = float(

        loss_metric.result().numpy()

    )



    local_accuracy = float(

        accuracy_metric.result().numpy()

    )



    del local_model



    gc.collect()



    return (

        local_weights,

        local_loss,

        local_accuracy,

    )





# ============================================================
# 18B. VALIDATION EVALUATION
# ============================================================

def evaluate_client_validation(client, model):
    """
    Evaluate the fixed validation subset carved only from train data.
    Returns macro-F1 over classes naturally present at this client.
    """
    idx = VAL_INDICES[client]

    if len(idx) == 0:
        raise RuntimeError(f"{client}: empty validation subset.")

    y_true_parts = []
    y_pred_parts = []

    for start in range(0, len(idx), EVAL_BATCH_SIZE):
        batch_idx = idx[start:start + EVAL_BATCH_SIZE]

        x = np.asarray(
            TRAIN_X[client][batch_idx],
            dtype=np.float32,
        )
        y_true = np.asarray(
            TRAIN_Y[client][batch_idx],
            dtype=np.int32,
        )

        x = preprocess_features(x)

        probabilities = model.predict(
            x,
            batch_size=EVAL_BATCH_SIZE,
            verbose=0,
        )

        y_pred = np.argmax(
            probabilities,
            axis=1,
        ).astype(np.int32)

        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    present = CLIENT_PRESENT_CLASSES[client]

    macro_f1_present = f1_score(
        y_true,
        y_pred,
        labels=present,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)

    return {
        "records": len(y_true),
        "accuracy": float(accuracy),
        "macro_f1_present": float(macro_f1_present),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================

# 19. FEDERATED TRAINING

# ============================================================



print("\n" + "=" * 75)

print("STARTING V4 CLEAN VALIDATION-GUIDED FEDERATED TRAINING")

print("=" * 75)



print(

    "\nIMPORTANT:"

    "\n  Test data will NOT be evaluated during training."

    "\n  Validation is evaluated after each round; test remains untouched.\n  Final test evaluation happens once after the best validation round is selected."

)



training_history = []

best_validation_score = -1.0
best_round = None
best_global_weights = None




for round_number in range(

    1,

    NUM_ROUNDS + 1,

):

    print("\n\n" + "=" * 75)

    print(

        f"FEDERATED ROUND "

        f"{round_number}/{NUM_ROUNDS}"

    )

    print("=" * 75)



    round_start = time.time()



    global_weights = (

        global_model.get_weights()

    )



    client_weight_sets = []



    round_record = {

        "round": round_number,

    }



    for client in CLIENTS:

        (

            local_weights,

            local_loss,

            local_accuracy,

        ) = train_client(

            client=client,

            global_weights=global_weights,

            round_number=round_number,

        )



        client_weight_sets.append(

            local_weights

        )



        round_record[

            f"{client}_local_loss"

        ] = local_loss



        round_record[

            f"{client}_local_accuracy"

        ] = local_accuracy



        print(

            f"{client} local result:"

            f" Loss={local_loss:.6f},"

            f" Accuracy="

            f"{local_accuracy * 100:.2f}%"

        )



    # --------------------------------------------------------

    # Server aggregation

    # --------------------------------------------------------



    new_global_weights = (

        equal_client_average(

            client_weight_sets

        )

    )



    global_model.set_weights(

        new_global_weights

    )

    # --------------------------------------------------------
    # Validation-based model selection (NO test data)
    # --------------------------------------------------------
    validation_scores = []

    print("\nValidation after aggregation:")

    for client in CLIENTS:
        val_result = evaluate_client_validation(
            client,
            global_model,
        )

        validation_scores.append(
            val_result["macro_f1_present"]
        )

        round_record[
            f"{client}_val_accuracy"
        ] = val_result["accuracy"]

        round_record[
            f"{client}_val_macro_f1"
        ] = val_result["macro_f1_present"]

        print(
            f"  {client:<12} | "
            f"records={val_result['records']:,} | "
            f"accuracy={val_result['accuracy'] * 100:.2f}% | "
            f"macro-F1={val_result['macro_f1_present'] * 100:.2f}%"
        )

    # Equal-client validation objective so CIC's much larger dataset
    # does not dominate model selection.
    mean_client_val_macro_f1 = float(
        np.mean(validation_scores)
    )

    round_record[
        "mean_client_val_macro_f1"
    ] = mean_client_val_macro_f1

    print(
        f"  Mean client validation Macro-F1: "
        f"{mean_client_val_macro_f1 * 100:.2f}%"
    )

    if mean_client_val_macro_f1 > best_validation_score:
        best_validation_score = mean_client_val_macro_f1
        best_round = round_number
        best_global_weights = [
            np.array(w, copy=True)
            for w in global_model.get_weights()
        ]
        print(
            f"  New best validation model: round {best_round}"
        )

    elapsed = (

        time.time() - round_start

    )



    round_record[

        "round_seconds"

    ] = elapsed



    training_history.append(

        round_record

    )



    pd.DataFrame(

        training_history

    ).to_csv(

        ROUND_METRICS_FILE,

        index=False,

    )



    # Save round checkpoint.

    round_model_file = (

        MODEL_DIR

        / (

            "federated_20percent_v4_"

            f"round_{round_number}.keras"

        )

    )



    global_model.save(

        round_model_file

    )



    print(

        f"\nRound {round_number} aggregated."

    )



    print(

        f"Round time: "

        f"{elapsed / 60:.2f} minutes"

    )



    print(

        "No test data used in this round."

    )



    del client_weight_sets

    del global_weights

    del new_global_weights



    gc.collect()





# ============================================================

# 20. SAVE FINAL GLOBAL MODEL

# ============================================================



print("\n" + "=" * 75)

print("RESTORING BEST VALIDATION MODEL")

print("=" * 75)

if best_global_weights is None:
    raise RuntimeError("No validation-selected model was recorded.")

global_model.set_weights(best_global_weights)

print(
    f"Selected round: {best_round} | "
    f"Mean client validation Macro-F1: "
    f"{best_validation_score * 100:.2f}%"
)

print("\n" + "=" * 75)

print("SAVING FINAL GLOBAL MODEL")

print("=" * 75)



global_model.save(

    FINAL_MODEL_FILE

)



print(

    f"Final model saved:\n"

    f"{FINAL_MODEL_FILE}"

)





# ============================================================

# 21. FINAL TEST EVALUATION

# ============================================================



def evaluate_client_test(

    client,

    model,

):

    """

    Evaluate the client's complete held-out test partition.



    Test data is used here for the first and only evaluation

    stage of this training script.

    """



    test_file = (

        DATA_DIR

        / client

        / "test_raw.csv"

    )



    total_records = 0

    correct_records = 0



    y_true_all = []

    y_pred_all = []



    print("\n" + "-" * 75)

    print(f"FINAL TEST: {client}")

    print("-" * 75)



    for chunk_number, chunk in enumerate(

        pd.read_csv(

            test_file,

            chunksize=CSV_CHUNK_SIZE,

        ),

        start=1,

    ):

        x = (

            chunk[FEATURES]

            .apply(

                pd.to_numeric,

                errors="coerce",

            )

            .to_numpy(dtype=np.float32)

        )



        if "label_id" in chunk.columns:

            y_true = pd.to_numeric(

                chunk["label_id"],

                errors="coerce",

            ).to_numpy()



            if np.isnan(y_true).any():

                raise ValueError(

                    f"{client}: invalid test label_id."

                )



            y_true = y_true.astype(

                np.int32

            )



        else:

            y_true = (

                chunk["label"]

                .map(CLASS_TO_ID)

                .to_numpy()

            )



            if pd.isna(y_true).any():

                raise ValueError(

                    f"{client}: unknown test label."

                )



            y_true = y_true.astype(

                np.int32

            )



        x = preprocess_features(x)



        probabilities = model.predict(

            x,

            batch_size=EVAL_BATCH_SIZE,

            verbose=0,

        )



        y_pred = np.argmax(

            probabilities,

            axis=1,

        ).astype(np.int32)



        total_records += len(y_true)



        correct_records += int(

            np.sum(

                y_true == y_pred

            )

        )



        y_true_all.append(

            y_true

        )



        y_pred_all.append(

            y_pred

        )



        print(

            f"\r{client}: tested "

            f"{total_records:,} records",

            end="",

            flush=True,

        )



    print()



    y_true_all = np.concatenate(

        y_true_all

    )



    y_pred_all = np.concatenate(

        y_pred_all

    )



    accuracy = accuracy_score(

        y_true_all,

        y_pred_all,

    )



    weighted_precision = precision_score(

        y_true_all,

        y_pred_all,

        average="weighted",

        zero_division=0,

    )



    weighted_recall = recall_score(

        y_true_all,

        y_pred_all,

        average="weighted",

        zero_division=0,

    )



    weighted_f1 = f1_score(

        y_true_all,

        y_pred_all,

        average="weighted",

        zero_division=0,

    )



    macro_precision = precision_score(

        y_true_all,

        y_pred_all,

        average="macro",

        zero_division=0,

    )



    macro_recall = recall_score(

        y_true_all,

        y_pred_all,

        average="macro",

        zero_division=0,

    )



    macro_f1 = f1_score(

        y_true_all,

        y_pred_all,

        average="macro",

        zero_division=0,

    )



    print(

        f"\n{client} FINAL:"

        f"\n  Records            : {total_records:,}"

        f"\n  Accuracy           : {accuracy * 100:.2f}%"

        f"\n  Weighted Precision : {weighted_precision * 100:.2f}%"

        f"\n  Weighted Recall    : {weighted_recall * 100:.2f}%"

        f"\n  Weighted F1        : {weighted_f1 * 100:.2f}%"

        f"\n  Macro Precision    : {macro_precision * 100:.2f}%"

        f"\n  Macro Recall       : {macro_recall * 100:.2f}%"

        f"\n  Macro F1           : {macro_f1 * 100:.2f}%"

    )



    metrics = {

        "client": client,

        "records": total_records,

        "accuracy": accuracy,

        "weighted_precision": weighted_precision,

        "weighted_recall": weighted_recall,

        "weighted_f1": weighted_f1,

        "macro_precision": macro_precision,

        "macro_recall": macro_recall,

        "macro_f1": macro_f1,

    }



    return (

        metrics,

        y_true_all,

        y_pred_all,

    )





# ============================================================

# 22. RUN FINAL TEST ON ALL THREE DATASETS

# ============================================================



print("\n\n" + "=" * 75)

print("FINAL HELD-OUT TEST EVALUATION")

print("=" * 75)



print(

    "\nTraining is complete."

    "\nThe test partitions are now being evaluated."

)



client_final_metrics = []



all_y_true = []

all_y_pred = []



for client in CLIENTS:

    (

        metrics,

        y_true,

        y_pred,

    ) = evaluate_client_test(

        client,

        global_model,

    )



    client_final_metrics.append(

        metrics

    )



    all_y_true.append(

        y_true

    )



    all_y_pred.append(

        y_pred

    )





# ============================================================

# 23. POOLED GLOBAL METRICS

# ============================================================



all_y_true = np.concatenate(

    all_y_true

)



all_y_pred = np.concatenate(

    all_y_pred

)



pooled_accuracy = accuracy_score(

    all_y_true,

    all_y_pred,

)



pooled_weighted_precision = precision_score(

    all_y_true,

    all_y_pred,

    average="weighted",

    zero_division=0,

)



pooled_weighted_recall = recall_score(

    all_y_true,

    all_y_pred,

    average="weighted",

    zero_division=0,

)



pooled_weighted_f1 = f1_score(

    all_y_true,

    all_y_pred,

    average="weighted",

    zero_division=0,

)



pooled_macro_precision = precision_score(

    all_y_true,

    all_y_pred,

    average="macro",

    zero_division=0,

)



pooled_macro_recall = recall_score(

    all_y_true,

    all_y_pred,

    average="macro",

    zero_division=0,

)



pooled_macro_f1 = f1_score(

    all_y_true,

    all_y_pred,

    average="macro",

    zero_division=0,

)



client_macro_f1_average = np.mean(

    [

        item["macro_f1"]

        for item in client_final_metrics

    ]

)





# ============================================================

# 24. CONFUSION MATRIX

# ============================================================



cm = confusion_matrix(

    all_y_true,

    all_y_pred,

    labels=np.arange(NUM_CLASSES),

)



cm_df = pd.DataFrame(

    cm,

    index=[

        f"Actual_{name}"

        for name in CLASSES

    ],

    columns=[

        f"Predicted_{name}"

        for name in CLASSES

    ],

)



cm_df.to_csv(

    CONFUSION_MATRIX_FILE

)





# ============================================================

# 25. PER-CLASS METRICS

# ============================================================



classification_dict = classification_report(

    all_y_true,

    all_y_pred,

    labels=np.arange(NUM_CLASSES),

    target_names=CLASSES,

    zero_division=0,

    output_dict=True,

)



per_class_rows = []



for class_name in CLASSES:

    values = classification_dict[

        class_name

    ]



    per_class_rows.append(

        {

            "class": class_name,

            "precision": values["precision"],

            "recall": values["recall"],

            "f1_score": values["f1-score"],

            "support": int(values["support"]),

        }

    )



pd.DataFrame(

    per_class_rows

).to_csv(

    PER_CLASS_METRICS_FILE,

    index=False,

)





# ============================================================

# 26. FINAL TEXT REPORT

# ============================================================



report_lines = []



report_lines.append(

    "=" * 75

)



report_lines.append(

    "V4 CLEAN VALIDATION-GUIDED 3-DATASET FEDERATED IDS - FINAL REPORT"

)



report_lines.append(

    "=" * 75

)



report_lines.append("")



report_lines.append(

    "Training configuration:"

)



report_lines.append(

    f"Clients: {', '.join(CLIENTS)}"

)



report_lines.append(

    f"Features: {NUM_FEATURES}"

)



report_lines.append(

    f"Classes: {NUM_CLASSES}"

)



report_lines.append(

    f"Federated rounds: {NUM_ROUNDS}"

)



report_lines.append(

    f"Local steps per client per round: "

    f"{LOCAL_STEPS_PER_ROUND}"

)



report_lines.append(

    f"Batch size: {BATCH_SIZE}"

)



report_lines.append(

    f"Learning rate: {LEARNING_RATE}"

)



report_lines.append(

    f"Tempered sampling alpha: "

    f"{SAMPLING_ALPHA}"

)



report_lines.append(

    "Aggregation: equal-client federated averaging"

)



report_lines.append(

    "Normalization: V4 shared scaler fitted only on effective 90% training rows; "
    "validation and test excluded"

)



report_lines.append(

    "Test data used during training: NO"

)



report_lines.append("")



report_lines.append(

    "=" * 75

)



report_lines.append(

    "PER-DATASET FINAL TEST RESULTS"

)



report_lines.append(

    "=" * 75

)



for item in client_final_metrics:

    report_lines.append("")



    report_lines.append(

        f"{item['client']}:"

    )



    report_lines.append(

        f"  Records: "

        f"{item['records']:,}"

    )



    report_lines.append(

        f"  Accuracy: "

        f"{item['accuracy'] * 100:.2f}%"

    )



    report_lines.append(

        f"  Weighted F1: "

        f"{item['weighted_f1'] * 100:.2f}%"

    )



    report_lines.append(

        f"  Macro Precision: "

        f"{item['macro_precision'] * 100:.2f}%"

    )



    report_lines.append(

        f"  Macro Recall: "

        f"{item['macro_recall'] * 100:.2f}%"

    )



    report_lines.append(

        f"  Macro F1: "

        f"{item['macro_f1'] * 100:.2f}%"

    )



report_lines.append("")



report_lines.append(

    "=" * 75

)



report_lines.append(

    "POOLED FINAL TEST RESULTS"

)



report_lines.append(

    "=" * 75

)



report_lines.append(

    f"Total test records: "

    f"{len(all_y_true):,}"

)



report_lines.append(

    f"Pooled Accuracy: "

    f"{pooled_accuracy * 100:.2f}%"

)



report_lines.append(

    f"Pooled Weighted Precision: "

    f"{pooled_weighted_precision * 100:.2f}%"

)



report_lines.append(

    f"Pooled Weighted Recall: "

    f"{pooled_weighted_recall * 100:.2f}%"

)



report_lines.append(

    f"Pooled Weighted F1: "

    f"{pooled_weighted_f1 * 100:.2f}%"

)



report_lines.append(

    f"Pooled Macro Precision: "

    f"{pooled_macro_precision * 100:.2f}%"

)



report_lines.append(

    f"Pooled Macro Recall: "

    f"{pooled_macro_recall * 100:.2f}%"

)



report_lines.append(

    f"Pooled Macro F1: "

    f"{pooled_macro_f1 * 100:.2f}%"

)



report_lines.append(

    f"Average Client Macro F1: "

    f"{client_macro_f1_average * 100:.2f}%"

)



report_lines.append("")



report_lines.append(

    "=" * 75

)



report_lines.append(

    "PER-CLASS RESULTS"

)



report_lines.append(

    "=" * 75

)



for row in per_class_rows:

    report_lines.append(

        f"{row['class']:<18} "

        f"Precision={row['precision'] * 100:6.2f}% "

        f"Recall={row['recall'] * 100:6.2f}% "

        f"F1={row['f1_score'] * 100:6.2f}% "

        f"Support={row['support']:,}"

    )



report_lines.append("")



report_lines.append(

    "Scientific note:"

)



report_lines.append(

    "The three source datasets have different original "

    "feature schemas and attack distributions. They were "

    "mapped into a shared 16-feature representation and "

    "13-class taxonomy. Missing semantic features in a "

    "source dataset may be represented by zeros, so "

    "cross-domain distribution shift remains a limitation."

)



report_lines.append("")



report_lines.append(

    "The aggregation used here gives equal influence to "

    "the three dataset clients. Therefore it should be "

    "described as equal-client federated averaging / "

    "FedAvg-style aggregation rather than canonical "

    "sample-count-weighted FedAvg."

)



with open(

    FINAL_REPORT_FILE,

    "w",

    encoding="utf-8",

) as file:

    file.write(

        "\n".join(report_lines)

    )





# ============================================================

# 27. FINAL CONSOLE OUTPUT

# ============================================================



print("\n\n" + "=" * 75)

print("FINAL RESULTS")

print("=" * 75)



for item in client_final_metrics:

    print(

        f"\n{item['client']}"

        f"\n  Accuracy     : "

        f"{item['accuracy'] * 100:.2f}%"

        f"\n  Weighted F1  : "

        f"{item['weighted_f1'] * 100:.2f}%"

        f"\n  Macro Recall : "

        f"{item['macro_recall'] * 100:.2f}%"

        f"\n  Macro F1     : "

        f"{item['macro_f1'] * 100:.2f}%"

    )



print("\n" + "-" * 75)



print(

    f"POOLED TEST RECORDS : "

    f"{len(all_y_true):,}"

)



print(

    f"POOLED ACCURACY     : "

    f"{pooled_accuracy * 100:.2f}%"

)



print(

    f"POOLED WEIGHTED F1  : "

    f"{pooled_weighted_f1 * 100:.2f}%"

)



print(

    f"POOLED MACRO RECALL : "

    f"{pooled_macro_recall * 100:.2f}%"

)



print(

    f"POOLED MACRO F1     : "

    f"{pooled_macro_f1 * 100:.2f}%"

)



print(

    f"CLIENT MACRO-F1 AVG : "

    f"{client_macro_f1_average * 100:.2f}%"

)



print("\n" + "=" * 75)

print("OUTPUT FILES")

print("=" * 75)



print(

    f"\nFinal model:\n{FINAL_MODEL_FILE}"

)



print(

    f"\nTraining metrics:\n{ROUND_METRICS_FILE}"

)



print(

    f"\nFinal report:\n{FINAL_REPORT_FILE}"

)



print(

    f"\nConfusion matrix:\n{CONFUSION_MATRIX_FILE}"

)



print(

    f"\nPer-class metrics:\n{PER_CLASS_METRICS_FILE}"

)



print("\n" + "=" * 75)

print("STATUS: PASS")

print("=" * 75)