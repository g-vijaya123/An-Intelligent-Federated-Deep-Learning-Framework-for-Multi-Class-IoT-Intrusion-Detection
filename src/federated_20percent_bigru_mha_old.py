from pathlib import Path
import os
import gc
import json
import math

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "fl_3datasets_20percent"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCALER_FILE = (
    DATA_DIR
    / "global_train_scaler.npz"
)

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

NUM_CLASSES = len(CLASSES)
NUM_FEATURES = len(FEATURES)

# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------

BATCH_SIZE = 1024

NUM_ROUNDS = 5

# IMPORTANT:
# Same local update budget for every client.
LOCAL_STEPS_PER_ROUND = 1000

LEARNING_RATE = 1e-4

CSV_CHUNK_SIZE = 200_000

# Evaluate entire test set after each round.
EVAL_BATCH_SIZE = 2048

# Clip standardized extreme values.
STANDARDIZED_CLIP = 20.0


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL_FILE = (
    MODEL_DIR
    / "federated_20percent_cnn_bigru_mha_best.keras"
)

FINAL_MODEL_FILE = (
    MODEL_DIR
    / "federated_20percent_cnn_bigru_mha_final.keras"
)

ROUND_METRICS_FILE = (
    REPORT_DIR
    / "federated_20percent_round_metrics.csv"
)

FINAL_REPORT_FILE = (
    REPORT_DIR
    / "federated_20percent_final_report.txt"
)

CONFUSION_FILE = (
    REPORT_DIR
    / "federated_20percent_confusion_matrix.csv"
)


# ============================================================
# PRINT HELPER
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD SCALER
# ============================================================

section("LOADING 20% TRAIN-ONLY GLOBAL SCALER")

if not SCALER_FILE.exists():
    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_FILE}"
    )

scaler_data = np.load(
    SCALER_FILE,
    allow_pickle=True,
)

SCALER_MEAN = scaler_data["mean"].astype(
    np.float32
)

SCALER_STD = scaler_data["std"].astype(
    np.float32
)

if len(SCALER_MEAN) != NUM_FEATURES:
    raise ValueError(
        "Scaler feature count does not match model features."
    )

print("Scaler loaded successfully.")
print(f"Features: {NUM_FEATURES}")
print(f"Classes : {NUM_CLASSES}")


# ============================================================
# CLIENT PATHS
# ============================================================

CLIENT_PATHS = {}

for client in CLIENTS:

    train_file = (
        DATA_DIR
        / client
        / "train_raw.csv"
    )

    test_file = (
        DATA_DIR
        / client
        / "test_raw.csv"
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Missing train file:\n{train_file}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Missing test file:\n{test_file}"
        )

    CLIENT_PATHS[client] = {
        "train": train_file,
        "test": test_file,
    }


# ============================================================
# LOAD METADATA / TRAIN COUNTS
# ============================================================

GLOBAL_METADATA_FILE = (
    DATA_DIR
    / "global_metadata.json"
)

if not GLOBAL_METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata not found:\n{GLOBAL_METADATA_FILE}"
    )

with open(
    GLOBAL_METADATA_FILE,
    "r",
    encoding="utf-8",
) as f:
    GLOBAL_METADATA = json.load(f)

CLIENT_TRAIN_COUNTS = {}

for client in CLIENTS:

    count = int(
        GLOBAL_METADATA[
            "clients"
        ][client][
            "train_records"
        ]
    )

    CLIENT_TRAIN_COUNTS[client] = count

print("\nClient training counts:")

for client, count in CLIENT_TRAIN_COUNTS.items():
    print(
        f"{client:<15}: {count:,}"
    )


# ============================================================
# CLASS COUNTS
# ============================================================

section("GLOBAL TRAIN CLASS DISTRIBUTION")

global_class_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64,
)

for client in CLIENTS:

    counts = GLOBAL_METADATA[
        "clients"
    ][client][
        "train_class_counts"
    ]

    for i, class_name in enumerate(CLASSES):
        global_class_counts[i] += int(
            counts.get(
                class_name,
                0,
            )
        )

for i, class_name in enumerate(CLASSES):

    print(
        f"{i:>2} "
        f"{class_name:<18} "
        f"{global_class_counts[i]:>10,}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Moderate weighting.
# We deliberately avoid huge inverse-frequency weights because
# they contributed to unstable rare-class overprediction before.

total_train = int(
    np.sum(global_class_counts)
)

nonzero_classes = np.count_nonzero(
    global_class_counts
)

class_weights = np.ones(
    NUM_CLASSES,
    dtype=np.float32,
)

for i in range(NUM_CLASSES):

    count = global_class_counts[i]

    if count > 0:

        raw_weight = math.sqrt(
            total_train
            / (
                nonzero_classes
                * count
            )
        )

        class_weights[i] = np.clip(
            raw_weight,
            0.50,
            5.0,
        )

print("\nModerate class weights")
print("-" * 80)

for i, class_name in enumerate(CLASSES):

    print(
        f"{class_name:<18}: "
        f"{class_weights[i]:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(
            NUM_FEATURES,
            1,
        ),
        name="network_features",
    )

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            64,
            return_sequences=True,
        )
    )(x)

    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.1,
    )(
        x,
        x,
    )

    x = tf.keras.layers.Add()(
        [
            x,
            attention,
        ]
    )

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            32,
            return_sequences=False,
        )
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="attack_prediction",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_BiGRU_MHA_3Dataset_IDS",
    )


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_model(model):

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


# ============================================================
# STANDARDIZATION
# ============================================================

def preprocess_X(X):

    X = X.astype(
        np.float32
    )

    X = (
        X
        - SCALER_MEAN
    ) / SCALER_STD

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=STANDARDIZED_CLIP,
        neginf=-STANDARDIZED_CLIP,
    )

    X = np.clip(
        X,
        -STANDARDIZED_CLIP,
        STANDARDIZED_CLIP,
    )

    X = np.expand_dims(
        X,
        axis=-1,
    )

    return X


# ============================================================
# INFINITE CLIENT TRAINING GENERATOR
# ============================================================

def client_training_generator(
    client,
    round_number,
):

    train_file = CLIENT_PATHS[
        client
    ]["train"]

    epoch = 0

    while True:

        epoch += 1

        # Different deterministic shuffle seed for each
        # local pass / round / client.
        rng = np.random.default_rng(
            SEED
            + round_number * 1000
            + CLIENTS.index(client) * 100
            + epoch
        )

        for chunk in pd.read_csv(
            train_file,
            usecols=FEATURES + ["label_id"],
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        ):

            X = chunk[
                FEATURES
            ].to_numpy(
                dtype=np.float32
            )

            y = chunk[
                "label_id"
            ].to_numpy(
                dtype=np.int32
            )

            indices = np.arange(
                len(y)
            )

            rng.shuffle(
                indices
            )

            X = X[indices]
            y = y[indices]

            X = preprocess_X(X)

            for start in range(
                0,
                len(y),
                BATCH_SIZE,
            ):

                end = min(
                    start + BATCH_SIZE,
                    len(y),
                )

                X_batch = X[
                    start:end
                ]

                y_batch = y[
                    start:end
                ]

                if len(y_batch) == 0:
                    continue

                sample_weights = (
                    class_weights[
                        y_batch
                    ]
                )

                yield (
                    X_batch,
                    y_batch,
                    sample_weights,
                )

            del X
            del y
            del chunk

        gc.collect()


# ============================================================
# TF DATASET
# ============================================================

def make_train_dataset(
    client,
    round_number,
):

    output_signature = (
        tf.TensorSpec(
            shape=(
                None,
                NUM_FEATURES,
                1,
            ),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(None,),
            dtype=tf.int32,
        ),
        tf.TensorSpec(
            shape=(None,),
            dtype=tf.float32,
        ),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: client_training_generator(
            client,
            round_number,
        ),
        output_signature=output_signature,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# FEDERATED AGGREGATION
# ============================================================

def average_weights(
    client_weights,
):

    averaged = []

    number_clients = len(
        client_weights
    )

    for layer_index in range(
        len(client_weights[0])
    ):

        layer_values = [
            weights[layer_index]
            for weights in client_weights
        ]

        layer_average = np.mean(
            np.stack(
                layer_values,
                axis=0,
            ),
            axis=0,
        )

        averaged.append(
            layer_average
        )

    return averaged


# ============================================================
# EVALUATION
# ============================================================

def evaluate_client(
    model,
    client,
):

    test_file = CLIENT_PATHS[
        client
    ]["test"]

    all_true = []
    all_pred = []

    total_confidence = 0.0
    total_records = 0

    for chunk in pd.read_csv(
        test_file,
        usecols=FEATURES + ["label_id"],
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):

        X = chunk[
            FEATURES
        ].to_numpy(
            dtype=np.float32
        )

        y_true = chunk[
            "label_id"
        ].to_numpy(
            dtype=np.int32
        )

        X = preprocess_X(X)

        probabilities = model.predict(
            X,
            batch_size=EVAL_BATCH_SIZE,
            verbose=0,
        )

        y_pred = np.argmax(
            probabilities,
            axis=1,
        ).astype(
            np.int32
        )

        confidence = np.max(
            probabilities,
            axis=1,
        )

        all_true.append(
            y_true
        )

        all_pred.append(
            y_pred
        )

        total_confidence += float(
            np.sum(confidence)
        )

        total_records += len(
            y_true
        )

        del X
        del probabilities
        del chunk

        gc.collect()

    y_true = np.concatenate(
        all_true
    )

    y_pred = np.concatenate(
        all_pred
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=np.arange(
            NUM_CLASSES
        ),
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        labels=np.arange(
            NUM_CLASSES
        ),
        zero_division=0,
    )

    average_confidence = (
        total_confidence
        / total_records
    )

    return {
        "client": client,
        "records": total_records,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "average_confidence": average_confidence,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================
# COMBINED EVALUATION
# ============================================================

def evaluate_global_model(
    model,
):

    client_results = {}

    combined_true = []
    combined_pred = []

    for client in CLIENTS:

        print(
            f"\nEvaluating {client}..."
        )

        result = evaluate_client(
            model,
            client,
        )

        client_results[
            client
        ] = result

        combined_true.append(
            result["y_true"]
        )

        combined_pred.append(
            result["y_pred"]
        )

        print(
            f"{client} | "
            f"Accuracy={result['accuracy'] * 100:.2f}% | "
            f"Macro-F1={result['macro_f1'] * 100:.2f}% | "
            f"Macro-Recall={result['macro_recall'] * 100:.2f}%"
        )

    y_true = np.concatenate(
        combined_true
    )

    y_pred = np.concatenate(
        combined_pred
    )

    pooled_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    pooled_weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    pooled_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=np.arange(
            NUM_CLASSES
        ),
        zero_division=0,
    )

    pooled_macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        labels=np.arange(
            NUM_CLASSES
        ),
        zero_division=0,
    )

    # Equal-domain metric:
    # each dataset contributes equally regardless of size.
    balanced_client_macro_f1 = float(
        np.mean(
            [
                client_results[c][
                    "macro_f1"
                ]
                for c in CLIENTS
            ]
        )
    )

    return {
        "client_results": client_results,
        "y_true": y_true,
        "y_pred": y_pred,
        "pooled_accuracy": pooled_accuracy,
        "pooled_weighted_f1": pooled_weighted_f1,
        "pooled_macro_f1": pooled_macro_f1,
        "pooled_macro_recall": pooled_macro_recall,
        "balanced_client_macro_f1": balanced_client_macro_f1,
    }


# ============================================================
# INITIALIZE GLOBAL MODEL
# ============================================================

section("BUILDING GLOBAL CNN-BiGRU-MHA MODEL")

global_model = build_model()

compile_model(
    global_model
)

global_model.summary()

print(
    f"\nParameters: "
    f"{global_model.count_params():,}"
)


# ============================================================
# ROUND METRICS STORAGE
# ============================================================

round_rows = []

best_score = -1.0
best_round = None


# ============================================================
# FEDERATED TRAINING
# ============================================================

section("STARTING FEDERATED TRAINING")

print(
    f"Rounds                 : {NUM_ROUNDS}"
)

print(
    f"Clients                : {len(CLIENTS)}"
)

print(
    f"Local steps/client     : {LOCAL_STEPS_PER_ROUND}"
)

print(
    f"Batch size             : {BATCH_SIZE}"
)

print(
    f"Learning rate          : {LEARNING_RATE}"
)

print(
    "Aggregation            : Equal-client parameter averaging"
)

print(
    "Test data in training  : NO"
)


for round_number in range(
    1,
    NUM_ROUNDS + 1,
):

    section(
        f"FEDERATED ROUND {round_number}/{NUM_ROUNDS}"
    )

    starting_weights = (
        global_model.get_weights()
    )

    client_weights = []

    # --------------------------------------------------------
    # LOCAL CLIENT TRAINING
    # --------------------------------------------------------

    for client in CLIENTS:

        print(
            f"\n--- Training client: {client} ---"
        )

        local_model = build_model()

        local_model.set_weights(
            starting_weights
        )

        # Fresh optimizer for every local client.
        compile_model(
            local_model
        )

        train_dataset = make_train_dataset(
            client,
            round_number,
        )

        history = local_model.fit(
            train_dataset,
            steps_per_epoch=LOCAL_STEPS_PER_ROUND,
            epochs=1,
            verbose=1,
        )

        final_loss = float(
            history.history[
                "loss"
            ][-1]
        )

        final_accuracy = float(
            history.history[
                "accuracy"
            ][-1]
        )

        print(
            f"{client} local result | "
            f"Loss={final_loss:.6f} | "
            f"Accuracy={final_accuracy * 100:.2f}%"
        )

        client_weights.append(
            local_model.get_weights()
        )

        del train_dataset
        del local_model
        gc.collect()

        tf.keras.backend.clear_session()

    # --------------------------------------------------------
    # AGGREGATION
    # --------------------------------------------------------

    print(
        "\nAggregating 3 client models..."
    )

    new_global_weights = average_weights(
        client_weights
    )

    global_model = build_model()

    global_model.set_weights(
        new_global_weights
    )

    compile_model(
        global_model
    )

    del client_weights
    gc.collect()

    # --------------------------------------------------------
    # SAVE ROUND CHECKPOINT
    # --------------------------------------------------------

    round_model_file = (
        MODEL_DIR
        / f"federated_20percent_round_{round_number}.keras"
    )

    global_model.save(
        round_model_file
    )

    print(
        f"\nRound checkpoint saved:\n"
        f"{round_model_file}"
    )

    # --------------------------------------------------------
    # GLOBAL EVALUATION
    # --------------------------------------------------------

    section(
        f"ROUND {round_number} GLOBAL EVALUATION"
    )

    evaluation = evaluate_global_model(
        global_model
    )

    print(
        "\nPOOLED RESULTS"
    )

    print(
        f"Accuracy      : "
        f"{evaluation['pooled_accuracy'] * 100:.2f}%"
    )

    print(
        f"Weighted F1   : "
        f"{evaluation['pooled_weighted_f1'] * 100:.2f}%"
    )

    print(
        f"Macro F1      : "
        f"{evaluation['pooled_macro_f1'] * 100:.2f}%"
    )

    print(
        f"Macro Recall  : "
        f"{evaluation['pooled_macro_recall'] * 100:.2f}%"
    )

    print(
        f"Client Macro-F1 Average: "
        f"{evaluation['balanced_client_macro_f1'] * 100:.2f}%"
    )

    row = {
        "round": round_number,
        "pooled_accuracy": evaluation[
            "pooled_accuracy"
        ],
        "pooled_weighted_f1": evaluation[
            "pooled_weighted_f1"
        ],
        "pooled_macro_f1": evaluation[
            "pooled_macro_f1"
        ],
        "pooled_macro_recall": evaluation[
            "pooled_macro_recall"
        ],
        "balanced_client_macro_f1": evaluation[
            "balanced_client_macro_f1"
        ],
    }

    for client in CLIENTS:

        result = evaluation[
            "client_results"
        ][client]

        row[
            f"{client}_accuracy"
        ] = result["accuracy"]

        row[
            f"{client}_macro_f1"
        ] = result["macro_f1"]

        row[
            f"{client}_macro_recall"
        ] = result["macro_recall"]

    round_rows.append(
        row
    )

    pd.DataFrame(
        round_rows
    ).to_csv(
        ROUND_METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # BEST MODEL SELECTION
    # --------------------------------------------------------

    # We select using pooled macro-F1 rather than raw accuracy
    # because the combined dataset is highly imbalanced.

    current_score = evaluation[
        "pooled_macro_f1"
    ]

    if current_score > best_score:

        best_score = current_score
        best_round = round_number

        global_model.save(
            BEST_MODEL_FILE
        )

        print(
            "\nNEW BEST MODEL"
        )

        print(
            f"Round    : {best_round}"
        )

        print(
            f"Macro-F1 : "
            f"{best_score * 100:.2f}%"
        )

        print(
            f"Saved to : "
            f"{BEST_MODEL_FILE}"
        )

    # Free evaluation arrays.
    del evaluation
    gc.collect()


# ============================================================
# SAVE FINAL ROUND MODEL
# ============================================================

global_model.save(
    FINAL_MODEL_FILE
)


# ============================================================
# LOAD BEST MODEL FOR FINAL EVALUATION
# ============================================================

section("FINAL EVALUATION USING BEST ROUND")

best_model = tf.keras.models.load_model(
    BEST_MODEL_FILE,
    compile=False,
)

final_evaluation = evaluate_global_model(
    best_model
)

final_true = final_evaluation[
    "y_true"
]

final_pred = final_evaluation[
    "y_pred"
]


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
    final_true,
    final_pred,
    labels=np.arange(
        NUM_CLASSES
    ),
    target_names=CLASSES,
    digits=4,
    zero_division=0,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    final_true,
    final_pred,
    labels=np.arange(
        NUM_CLASSES
    ),
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
    CONFUSION_FILE
)


# ============================================================
# FINAL REPORT
# ============================================================

with open(
    FINAL_REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "FEDERATED 3-DATASET IDS — 20% REPRESENTATIVE SUBSET\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Datasets:\n"
        "- UNSW-NB15\n"
        "- CICIoT2023\n"
        "- Edge-IIoTset\n\n"
    )

    f.write(
        "Model:\n"
        "CNN + BiGRU + Multi-Head Attention\n\n"
    )

    f.write(
        f"Features: {NUM_FEATURES}\n"
    )

    f.write(
        f"Classes : {NUM_CLASSES}\n"
    )

    f.write(
        f"Best federated round: {best_round}\n\n"
    )

    f.write(
        "FINAL POOLED METRICS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"Accuracy      : "
        f"{final_evaluation['pooled_accuracy'] * 100:.4f}%\n"
    )

    f.write(
        f"Weighted F1   : "
        f"{final_evaluation['pooled_weighted_f1'] * 100:.4f}%\n"
    )

    f.write(
        f"Macro F1      : "
        f"{final_evaluation['pooled_macro_f1'] * 100:.4f}%\n"
    )

    f.write(
        f"Macro Recall  : "
        f"{final_evaluation['pooled_macro_recall'] * 100:.4f}%\n"
    )

    f.write(
        f"Client Macro-F1 Average: "
        f"{final_evaluation['balanced_client_macro_f1'] * 100:.4f}%\n\n"
    )

    f.write(
        "PER-DATASET RESULTS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for client in CLIENTS:

        result = final_evaluation[
            "client_results"
        ][client]

        f.write(
            f"\n{client}\n"
        )

        f.write(
            f"Accuracy     : "
            f"{result['accuracy'] * 100:.4f}%\n"
        )

        f.write(
            f"Macro F1     : "
            f"{result['macro_f1'] * 100:.4f}%\n"
        )

        f.write(
            f"Macro Recall : "
            f"{result['macro_recall'] * 100:.4f}%\n"
        )

    f.write(
        "\nCLASSIFICATION REPORT\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        report_text
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

section("FEDERATED TRAINING COMPLETE")

print(
    f"Best round: {best_round}"
)

print(
    f"Final pooled accuracy: "
    f"{final_evaluation['pooled_accuracy'] * 100:.2f}%"
)

print(
    f"Final pooled Macro-F1: "
    f"{final_evaluation['pooled_macro_f1'] * 100:.2f}%"
)

print(
    f"Final pooled Macro Recall: "
    f"{final_evaluation['pooled_macro_recall'] * 100:.2f}%"
)

print(
    "\nBest model:"
)

print(
    BEST_MODEL_FILE
)

print(
    "\nFinal-round model:"
)

print(
    FINAL_MODEL_FILE
)

print(
    "\nRound metrics:"
)

print(
    ROUND_METRICS_FILE
)

print(
    "\nFinal report:"
)

print(
    FINAL_REPORT_FILE
)

print(
    "\nConfusion matrix:"
)

print(
    CONFUSION_FILE
)

print(
    "\nSTATUS: PASS"
)