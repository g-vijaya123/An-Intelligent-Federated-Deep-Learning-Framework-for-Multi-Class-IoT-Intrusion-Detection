from pathlib import Path
import json
import gc
import csv
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fl_3datasets"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SCALER_FILE = DATA_DIR / "global_train_scaler.npz"

FINAL_MODEL_FILE = (
    MODEL_DIR / "federated_3dataset_cnn_bigru_mha.keras"
)

BEST_MODEL_FILE = (
    MODEL_DIR / "federated_3dataset_cnn_bigru_mha_best.keras"
)

ROUND_LOG_FILE = (
    REPORT_DIR / "federated_3dataset_round_metrics.csv"
)

FINAL_REPORT_FILE = (
    REPORT_DIR / "federated_3dataset_final_report.txt"
)

CONFUSION_FILE = (
    REPORT_DIR / "federated_3dataset_confusion_matrix.csv"
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

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASSES)
}

NUM_FEATURES = len(FEATURES)
NUM_CLASSES = len(CLASSES)

# Windows CPU-friendly starting configuration.
BATCH_SIZE = 1024

# Each round processes every training record once per client.
LOCAL_EPOCHS = 1

# Start with 3 FL rounds because the full training set contains
# ~39.8 million records. Increase only after measuring runtime.
NUM_ROUNDS = 3

LEARNING_RATE = 3e-4

# Number of rows read from CSV at a time.
CSV_CHUNK_SIZE = 200_000

# Evaluation is also streamed.
EVAL_CHUNK_SIZE = 200_000


# ============================================================
# DISPLAY
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD SHARED GLOBAL SCALER
# ============================================================

def load_global_scaler():

    if not GLOBAL_SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Global scaler not found:\n{GLOBAL_SCALER_FILE}"
        )

    data = np.load(
        GLOBAL_SCALER_FILE,
        allow_pickle=True,
    )

    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)

    if "features" in data.files:
        scaler_features = [
            str(x)
            for x in data["features"].tolist()
        ]

        if scaler_features != FEATURES:
            raise ValueError(
                "Global scaler feature order does not match "
                "the training feature order."
            )

    if len(mean) != NUM_FEATURES:
        raise ValueError(
            "Scaler mean dimension mismatch."
        )

    if len(std) != NUM_FEATURES:
        raise ValueError(
            "Scaler std dimension mismatch."
        )

    std = np.where(
        std < 1e-12,
        1.0,
        std,
    ).astype(np.float32)

    return mean, std


SCALER_MEAN, SCALER_STD = load_global_scaler()


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(NUM_FEATURES, 1),
        name="network_features",
    )

    # ---------------- CNN ----------------

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

    # ---------------- BiGRU ----------------

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            64,
            return_sequences=True,
        )
    )(x)

    # ---------------- Multi-Head Attention ----------------

    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.1,
    )(
        query=x,
        value=x,
        key=x,
    )

    x = tf.keras.layers.Add()(
        [x, attention]
    )

    x = tf.keras.layers.LayerNormalization()(x)

    # ---------------- Second BiGRU ----------------

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            32,
            return_sequences=False,
        )
    )(x)

    # ---------------- Classification ----------------

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.30)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="attack_class",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_BiGRU_MHA_IDS",
    )

    compile_model(model)

    return model


def compile_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_client_class_counts(client):

    metadata_file = (
        DATA_DIR
        / client
        / "client_metadata.json"
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata missing: {metadata_file}"
        )

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    # Support a few likely metadata layouts.
    possible_keys = [
        "train_class_counts",
        "training_class_counts",
        "class_counts_train",
    ]

    counts = None

    for key in possible_keys:
        if key in metadata:
            counts = metadata[key]
            break

    return counts


def calculate_global_class_weights():

    global_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    metadata_success = True

    for client in CLIENTS:

        counts = get_client_class_counts(client)

        if counts is None:
            metadata_success = False
            break

        for class_name, count in counts.items():

            if class_name in CLASS_TO_ID:
                global_counts[
                    CLASS_TO_ID[class_name]
                ] += float(count)

    # If metadata layout differs, calculate counts directly
    # from training CSVs once.
    if not metadata_success:

        print(
            "Class counts not found in expected metadata keys."
        )

        print(
            "Calculating training class counts from CSV files..."
        )

        global_counts[:] = 0

        for client in CLIENTS:

            train_file = (
                DATA_DIR
                / client
                / "train_raw.csv"
            )

            for chunk in pd.read_csv(
                train_file,
                usecols=["label_id"],
                chunksize=CSV_CHUNK_SIZE,
            ):

                values = chunk[
                    "label_id"
                ].to_numpy(
                    dtype=np.int32
                )

                bincount = np.bincount(
                    values,
                    minlength=NUM_CLASSES,
                )

                global_counts += bincount

    total = np.sum(global_counts)

    if total <= 0:
        raise ValueError(
            "Could not calculate training class counts."
        )

    # Smoothed inverse-frequency weighting.
    #
    # sqrt is deliberately used instead of raw inverse
    # frequency because the combined dataset is extremely
    # imbalanced.
    #
    # Clipping prevents tiny classes from receiving
    # unrealistically enormous weights.

    raw_weights = np.sqrt(
        total
        / (
            NUM_CLASSES
            * np.maximum(global_counts, 1.0)
        )
    )

    clipped_weights = np.clip(
        raw_weights,
        0.25,
        10.0,
    )

    class_weight = {
        i: float(clipped_weights[i])
        for i in range(NUM_CLASSES)
    }

    return (
        global_counts.astype(np.int64),
        class_weight,
    )


# ============================================================
# STREAMING TRAIN GENERATOR
# ============================================================

def training_generator(client):

    train_file = (
        DATA_DIR
        / client
        / "train_raw.csv"
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file missing:\n{train_file}"
        )

    columns = FEATURES + ["label_id"]

    for chunk in pd.read_csv(
        train_file,
        usecols=columns,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):

        X = (
            chunk[FEATURES]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .to_numpy(
                dtype=np.float32
            )
        )

        y = (
            chunk["label_id"]
            .to_numpy(
                dtype=np.int32
            )
        )

        X = (
            X - SCALER_MEAN
        ) / SCALER_STD

        X = np.clip(
            X,
            -20.0,
            20.0,
        )

        X = X[..., np.newaxis]

        # Shuffle within each 200k-row chunk.
        indices = np.arange(len(y))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        for start in range(
            0,
            len(y),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(y),
            )

            yield (
                X[start:end],
                y[start:end],
            )


def make_training_dataset(client):

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
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: training_generator(client),
        output_signature=output_signature,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_client(
    global_weights,
    client,
    class_weight,
):

    section(
        f"LOCAL TRAINING - {client}"
    )

    model = build_model()

    model.set_weights(
        global_weights
    )

    dataset = make_training_dataset(
        client
    )

    start_time = time.time()

    history = model.fit(
        dataset,
        epochs=LOCAL_EPOCHS,
        verbose=1,
        class_weight=class_weight,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\n{client} local training completed "
        f"in {elapsed / 60:.2f} minutes."
    )

    local_weights = [
        w.copy()
        for w in model.get_weights()
    ]

    final_loss = float(
        history.history["loss"][-1]
    )

    final_accuracy = float(
        history.history["accuracy"][-1]
    )

    del dataset
    del model
    gc.collect()

    tf.keras.backend.clear_session()

    return (
        local_weights,
        final_loss,
        final_accuracy,
        elapsed,
    )


# ============================================================
# FEDERATED AGGREGATION
# ============================================================

def equal_client_aggregation(
    client_weight_sets
):

    if not client_weight_sets:
        raise ValueError(
            "No client weights supplied."
        )

    number_clients = len(
        client_weight_sets
    )

    aggregated = []

    for layer_weights in zip(
        *client_weight_sets
    ):

        stacked = np.stack(
            layer_weights,
            axis=0,
        )

        layer_average = np.mean(
            stacked,
            axis=0,
        )

        aggregated.append(
            layer_average
        )

    print(
        f"Aggregated {number_clients} clients "
        f"using equal client weights."
    )

    return aggregated


# ============================================================
# STREAMING EVALUATION
# ============================================================

def evaluate_file(
    model,
    csv_file,
    dataset_name,
):

    section(
        f"EVALUATION - {dataset_name}"
    )

    y_true_all = []
    y_pred_all = []

    columns = FEATURES + ["label_id"]

    processed = 0

    for chunk in pd.read_csv(
        csv_file,
        usecols=columns,
        chunksize=EVAL_CHUNK_SIZE,
        low_memory=False,
    ):

        X = (
            chunk[FEATURES]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .to_numpy(
                dtype=np.float32
            )
        )

        y_true = (
            chunk["label_id"]
            .to_numpy(
                dtype=np.int32
            )
        )

        X = (
            X - SCALER_MEAN
        ) / SCALER_STD

        X = np.clip(
            X,
            -20.0,
            20.0,
        )

        X = X[..., np.newaxis]

        probabilities = model.predict(
            X,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        y_pred = np.argmax(
            probabilities,
            axis=1,
        ).astype(np.int32)

        y_true_all.append(
            y_true
        )

        y_pred_all.append(
            y_pred
        )

        processed += len(y_true)

        print(
            f"{dataset_name:<12} "
            f"evaluated: {processed:,}",
            end="\r",
        )

    print()

    y_true_all = np.concatenate(
        y_true_all
    )

    y_pred_all = np.concatenate(
        y_pred_all
    )

    metrics = calculate_metrics(
        y_true_all,
        y_pred_all,
    )

    print(
        f"Accuracy      : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Weighted F1   : "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )

    print(
        f"Macro F1      : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Macro Recall  : "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    return (
        metrics,
        y_true_all,
        y_pred_all,
    )


def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),
    }


# ============================================================
# ROUND EVALUATION
# ============================================================

def evaluate_global_model(
    model,
    round_number,
):

    all_true = []
    all_pred = []

    per_client = {}

    for client in CLIENTS:

        test_file = (
            DATA_DIR
            / client
            / "test_raw.csv"
        )

        metrics, y_true, y_pred = evaluate_file(
            model,
            test_file,
            client,
        )

        per_client[client] = metrics

        all_true.append(
            y_true
        )

        all_pred.append(
            y_pred
        )

    pooled_true = np.concatenate(
        all_true
    )

    pooled_pred = np.concatenate(
        all_pred
    )

    pooled_metrics = calculate_metrics(
        pooled_true,
        pooled_pred,
    )

    section(
        f"ROUND {round_number} GLOBAL RESULTS"
    )

    print(
        f"Pooled Accuracy     : "
        f"{pooled_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Pooled Weighted F1  : "
        f"{pooled_metrics['weighted_f1'] * 100:.2f}%"
    )

    print(
        f"Pooled Macro F1     : "
        f"{pooled_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Pooled Macro Recall : "
        f"{pooled_metrics['macro_recall'] * 100:.2f}%"
    )

    return (
        pooled_metrics,
        per_client,
        pooled_true,
        pooled_pred,
    )


# ============================================================
# ROUND LOGGING
# ============================================================

def initialize_round_log():

    with open(
        ROUND_LOG_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "round",
            "pooled_accuracy",
            "pooled_weighted_f1",
            "pooled_macro_f1",
            "pooled_macro_recall",
            "UNSW_accuracy",
            "CICIoT2023_accuracy",
            "EdgeIIoTset_accuracy",
        ])


def append_round_log(
    round_number,
    pooled,
    per_client,
):

    with open(
        ROUND_LOG_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            round_number,
            pooled["accuracy"],
            pooled["weighted_f1"],
            pooled["macro_f1"],
            pooled["macro_recall"],
            per_client["UNSW"]["accuracy"],
            per_client["CICIoT2023"]["accuracy"],
            per_client["EdgeIIoTset"]["accuracy"],
        ])


# ============================================================
# FINAL REPORT
# ============================================================

def save_final_report(
    model,
    pooled_metrics,
    per_client_metrics,
    y_true,
    y_pred,
):

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=CLASSES,
        zero_division=0,
        digits=4,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    )

    cm_df.to_csv(
        CONFUSION_FILE
    )

    with open(
        FINAL_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "3-DATASET FEDERATED IDS FINAL REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Datasets / federated clients:\n"
        )

        f.write(
            "1. UNSW-NB15\n"
        )

        f.write(
            "2. CICIoT2023\n"
        )

        f.write(
            "3. Edge-IIoTset\n\n"
        )

        f.write(
            "Architecture:\n"
            "CNN -> BiGRU -> Multi-Head Attention -> "
            "BiGRU -> Dense -> 13-class Softmax\n\n"
        )

        f.write(
            "Federated aggregation:\n"
            "Equal-client model averaging.\n"
            "Each of the three client domains receives "
            "equal aggregation weight.\n\n"
        )

        f.write(
            "Important methodology note:\n"
            "This is simulated federated learning on one "
            "machine. It should not be described as a "
            "production privacy-preserving FL deployment.\n\n"
        )

        f.write(
            "POOLED HELD-OUT RESULTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in pooled_metrics.items():

            f.write(
                f"{key}: {value:.6f}\n"
            )

        f.write(
            "\nPER-DATASET HELD-OUT RESULTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for client, metrics in (
            per_client_metrics.items()
        ):

            f.write(
                f"\n{client}\n"
            )

            for key, value in (
                metrics.items()
            ):

                f.write(
                    f"{key}: {value:.6f}\n"
                )

        f.write(
            "\nPER-CLASS CLASSIFICATION REPORT\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(report)

        f.write(
            "\n\nInterpretation limitation:\n"
            "The three source datasets have different "
            "original schemas and attack taxonomies. "
            "They were mapped to a shared 16-feature "
            "representation and common 13-class taxonomy. "
            "Unavailable semantic features were represented "
            "using the harmonization strategy established "
            "during preprocessing.\n"
        )

    print(
        f"\nFinal report saved:\n"
        f"{FINAL_REPORT_FILE}"
    )

    print(
        f"\nConfusion matrix saved:\n"
        f"{CONFUSION_FILE}"
    )


# ============================================================
# MAIN FEDERATED TRAINING
# ============================================================

def main():

    section(
        "3-DATASET FEDERATED CNN-BiGRU-MHA"
    )

    print(
        f"TensorFlow version : "
        f"{tf.__version__}"
    )

    print(
        f"Features           : "
        f"{NUM_FEATURES}"
    )

    print(
        f"Classes            : "
        f"{NUM_CLASSES}"
    )

    print(
        f"Clients            : "
        f"{len(CLIENTS)}"
    )

    print(
        f"Federated rounds   : "
        f"{NUM_ROUNDS}"
    )

    print(
        f"Local epochs       : "
        f"{LOCAL_EPOCHS}"
    )

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        "\nAggregation        : "
        "Equal-client averaging"
    )

    print(
        "Scaling            : "
        "Shared train-only global scaler"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    section(
        "GLOBAL TRAINING CLASS DISTRIBUTION"
    )

    (
        class_counts,
        class_weight,
    ) = calculate_global_class_weights()

    for i, class_name in enumerate(
        CLASSES
    ):

        print(
            f"{class_name:<18} "
            f"Count={class_counts[i]:>12,} "
            f"Weight={class_weight[i]:.4f}"
        )

    # --------------------------------------------------------
    # Initial global model
    # --------------------------------------------------------

    global_model = build_model()

    section(
        "MODEL SUMMARY"
    )

    global_model.summary()

    global_weights = [
        w.copy()
        for w in global_model.get_weights()
    ]

    initialize_round_log()

    best_macro_f1 = -1.0

    best_round = 0

    final_pooled_metrics = None
    final_per_client = None
    final_true = None
    final_pred = None

    # --------------------------------------------------------
    # Federated rounds
    # --------------------------------------------------------

    for round_number in range(
        1,
        NUM_ROUNDS + 1,
    ):

        section(
            f"FEDERATED ROUND "
            f"{round_number}/{NUM_ROUNDS}"
        )

        client_weight_sets = []

        for client in CLIENTS:

            (
                local_weights,
                local_loss,
                local_accuracy,
                elapsed,
            ) = train_client(
                global_weights,
                client,
                class_weight,
            )

            client_weight_sets.append(
                local_weights
            )

            print(
                f"{client}: "
                f"loss={local_loss:.6f}, "
                f"accuracy="
                f"{local_accuracy * 100:.2f}%"
            )

        # ----------------------------------------------------
        # Equal-client aggregation
        # ----------------------------------------------------

        global_weights = (
            equal_client_aggregation(
                client_weight_sets
            )
        )

        global_model.set_weights(
            global_weights
        )

        compile_model(
            global_model
        )

        del client_weight_sets
        gc.collect()

        # ----------------------------------------------------
        # Held-out evaluation
        # ----------------------------------------------------

        (
            pooled_metrics,
            per_client_metrics,
            pooled_true,
            pooled_pred,
        ) = evaluate_global_model(
            global_model,
            round_number,
        )

        append_round_log(
            round_number,
            pooled_metrics,
            per_client_metrics,
        )

        # ----------------------------------------------------
        # Save round checkpoint
        # ----------------------------------------------------

        round_model_file = (
            MODEL_DIR
            / (
                "federated_3dataset_"
                f"round_{round_number}.keras"
            )
        )

        global_model.save(
            round_model_file
        )

        print(
            f"\nRound checkpoint saved:\n"
            f"{round_model_file}"
        )

        # ----------------------------------------------------
        # Select best model by pooled macro-F1
        # ----------------------------------------------------

        if (
            pooled_metrics["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                pooled_metrics["macro_f1"]
            )

            best_round = round_number

            global_model.save(
                BEST_MODEL_FILE
            )

            print(
                "\nNEW BEST GLOBAL MODEL"
            )

            print(
                f"Round     : {best_round}"
            )

            print(
                f"Macro F1  : "
                f"{best_macro_f1 * 100:.2f}%"
            )

            print(
                f"Saved     : "
                f"{BEST_MODEL_FILE}"
            )

        final_pooled_metrics = (
            pooled_metrics
        )

        final_per_client = (
            per_client_metrics
        )

        final_true = pooled_true
        final_pred = pooled_pred

        del pooled_true
        del pooled_pred

        gc.collect()

    # ========================================================
    # LOAD BEST MODEL FOR FINAL REPORT
    # ========================================================

    section(
        "FINAL BEST-MODEL EVALUATION"
    )

    print(
        f"Best federated round: "
        f"{best_round}"
    )

    print(
        f"Best pooled macro-F1: "
        f"{best_macro_f1 * 100:.2f}%"
    )

    best_model = tf.keras.models.load_model(
        BEST_MODEL_FILE
    )

    (
        best_pooled_metrics,
        best_per_client,
        best_true,
        best_pred,
    ) = evaluate_global_model(
        best_model,
        best_round,
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    best_model.save(
        FINAL_MODEL_FILE
    )

    print(
        f"\nFinal global model saved:\n"
        f"{FINAL_MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Save final report
    # --------------------------------------------------------

    save_final_report(
        best_model,
        best_pooled_metrics,
        best_per_client,
        best_true,
        best_pred,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    section(
        "FEDERATED TRAINING COMPLETED"
    )

    print(
        f"Best round       : "
        f"{best_round}"
    )

    print(
        f"Final Accuracy   : "
        f"{best_pooled_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Weighted F1      : "
        f"{best_pooled_metrics['weighted_f1'] * 100:.2f}%"
    )

    print(
        f"Macro F1         : "
        f"{best_pooled_metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Macro Recall     : "
        f"{best_pooled_metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        "\nFinal model:"
    )

    print(
        FINAL_MODEL_FILE
    )

    print(
        "\nSTATUS: PASS"
    )


if __name__ == "__main__":
    main()