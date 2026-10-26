import os
import gc
import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Bidirectional,
    GRU,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CLIENTS = 5
FEDERATED_ROUNDS = 5
LOCAL_EPOCHS = 1

BATCH_SIZE = 512
LEARNING_RATE = 0.0005

DATA_DIR = os.path.join("data", "processed_full")
MODEL_DIR = "models"
REPORT_DIR = "report"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "federated_cnn_bigru_mha_full.keras"
)

ARTIFACT_FILE = os.path.join(
    MODEL_DIR,
    "preprocessing_full_features.joblib"
)

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# LOAD FULL DATASET
# ============================================================

print("\n" + "=" * 70)
print("FEDERATED CNN-BiGRU-MULTIHEAD ATTENTION IDS")
print("=" * 70)

print("\nLoading FULL training dataset...")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy"),
    mmap_mode="r"
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy"),
    mmap_mode="r"
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

artifact = joblib.load(ARTIFACT_FILE)

class_names = artifact["class_names"]

NUM_FEATURES = X_train.shape[1]
NUM_CLASSES = len(class_names)

print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)
print("Features:", NUM_FEATURES)
print("Classes :", NUM_CLASSES)

assert len(X_train) == 2032037
assert NUM_FEATURES == 43

print("\n✓ ALL 2,032,037 training records will be used.")
print("✓ 43 features.")
print("✓ No SelectKBest.")
print("✓ No 1M balanced dataset.")


# ============================================================
# SMOOTHED CLASS WEIGHTS
# ============================================================
#
# Previous balanced weights were extremely large:
# Worms ~1462, Shellcode ~168 etc.
#
# Here:
# weight = sqrt(total / (classes * class_count))
#
# Square-root smoothing reduces extreme weighting while still
# giving minority classes more importance.
#
# We additionally clip weights to avoid unstable gradients.
# ============================================================

print("\nCalculating SMOOTHED class weights...")

counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

total_samples = len(y_train)

class_weights = {}

for cls in range(NUM_CLASSES):

    raw_weight = (
        total_samples
        /
        (NUM_CLASSES * counts[cls])
    )

    smooth_weight = np.sqrt(raw_weight)

    # Keep training numerically reasonable.
    smooth_weight = np.clip(
        smooth_weight,
        0.25,
        20.0
    )

    class_weights[cls] = float(
        smooth_weight
    )

    print(
        f"{class_names[cls]:20s} "
        f"Count={counts[cls]:8,d} "
        f"Weight={smooth_weight:.4f}"
    )


# ============================================================
# CNN + BiGRU + MULTI-HEAD ATTENTION
# ============================================================

def build_model():

    inputs = Input(
        shape=(NUM_FEATURES, 1),
        name="network_features"
    )

    # --------------------------------------------------------
    # CNN FEATURE EXTRACTION
    # --------------------------------------------------------

    x = Conv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling1D(
        pool_size=2
    )(x)

    x = Dropout(0.20)(x)

    x = Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(
        pool_size=2
    )(x)

    x = Dropout(0.25)(x)


    # --------------------------------------------------------
    # BIDIRECTIONAL GRU
    # --------------------------------------------------------

    x = Bidirectional(
        GRU(
            64,
            return_sequences=True
        )
    )(x)

    x = Dropout(0.25)(x)


    # --------------------------------------------------------
    # MULTI-HEAD SELF ATTENTION
    # --------------------------------------------------------

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.10,
        name="multihead_attention"
    )(
        query=x,
        value=x,
        key=x
    )

    # Residual connection
    x = Add()([
        x,
        attention
    ])

    x = LayerNormalization()(x)


    # --------------------------------------------------------
    # GLOBAL REPRESENTATION
    # --------------------------------------------------------

    x = GlobalAveragePooling1D()(x)


    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(0.30)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dropout(0.20)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax",
        name="attack_prediction"
    )(x)


    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# GLOBAL MODEL
# ============================================================

print("\nBuilding CNN-BiGRU-MultiHeadAttention model...")

global_model = build_model()

global_model.summary()


# ============================================================
# STRATIFIED FEDERATED CLIENT SPLIT
# ============================================================

print("\nCreating 5 federated clients...")

rng = np.random.default_rng(SEED)

client_indices = [
    []
    for _ in range(NUM_CLIENTS)
]

for cls in range(NUM_CLASSES):

    indices = np.where(
        y_train == cls
    )[0]

    rng.shuffle(indices)

    class_splits = np.array_split(
        indices,
        NUM_CLIENTS
    )

    for client_id in range(NUM_CLIENTS):

        client_indices[
            client_id
        ].extend(
            class_splits[
                client_id
            ].tolist()
        )


for client_id in range(NUM_CLIENTS):

    client_indices[client_id] = np.array(
        client_indices[client_id],
        dtype=np.int64
    )

    rng.shuffle(
        client_indices[client_id]
    )

    print(
        f"Client {client_id + 1}: "
        f"{len(client_indices[client_id]):,}"
    )


# ============================================================
# WEIGHTED FEDAVG
# ============================================================

def weighted_fedavg(
    model_weights,
    client_sizes
):

    total = float(
        sum(client_sizes)
    )

    averaged = []

    for layer_weights in zip(
        *model_weights
    ):

        layer_average = np.zeros_like(
            layer_weights[0],
            dtype=np.float32
        )

        for weights, size in zip(
            layer_weights,
            client_sizes
        ):

            layer_average += (
                weights.astype(np.float32)
                * (size / total)
            )

        averaged.append(
            layer_average
        )

    return averaged


# ============================================================
# FEDERATED TRAINING
# ============================================================

round_accuracies = []

print("\n" + "=" * 70)
print("STARTING FEDERATED TRAINING")
print("=" * 70)


for round_number in range(
    1,
    FEDERATED_ROUNDS + 1
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"FEDERATED ROUND "
        f"{round_number}/{FEDERATED_ROUNDS}"
    )

    print(
        "=" * 70
    )

    global_weights = (
        global_model.get_weights()
    )

    local_weights = []
    local_sizes = []


    # ========================================================
    # LOCAL CLIENT TRAINING
    # ========================================================

    for client_id in range(
        NUM_CLIENTS
    ):

        print(
            f"\n--- Client "
            f"{client_id + 1}/{NUM_CLIENTS} ---"
        )

        indices = client_indices[
            client_id
        ]

        print(
            f"Records: "
            f"{len(indices):,}"
        )

        X_client = np.asarray(
            X_train[indices],
            dtype=np.float32
        )

        y_client = np.asarray(
            y_train[indices],
            dtype=np.int16
        )

        X_client = np.expand_dims(
            X_client,
            axis=-1
        )

        local_model = build_model()

        local_model.set_weights(
            global_weights
        )

        local_model.fit(
            X_client,
            y_client,

            epochs=LOCAL_EPOCHS,

            batch_size=BATCH_SIZE,

            class_weight=class_weights,

            shuffle=True,

            verbose=1
        )

        local_weights.append(
            local_model.get_weights()
        )

        local_sizes.append(
            len(y_client)
        )

        del X_client
        del y_client
        del local_model

        gc.collect()

        tf.keras.backend.clear_session()


    # ========================================================
    # FEDERATED AGGREGATION
    # ========================================================

    print(
        "\nApplying weighted FedAvg..."
    )

    averaged_weights = (
        weighted_fedavg(
            local_weights,
            local_sizes
        )
    )

    global_model.set_weights(
        averaged_weights
    )

    del local_weights
    del averaged_weights

    gc.collect()


    # ========================================================
    # ROUND TEST
    # ========================================================

    print(
        "\nEvaluating global model..."
    )

    X_test_round = np.expand_dims(
        np.asarray(
            X_test,
            dtype=np.float32
        ),
        axis=-1
    )

    test_loss, test_accuracy = (
        global_model.evaluate(
            X_test_round,
            y_test,
            batch_size=1024,
            verbose=1
        )
    )

    round_accuracies.append(
        test_accuracy
    )

    print(
        f"\nRound {round_number} "
        f"Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    del X_test_round

    gc.collect()


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving final model..."
)

global_model.save(
    MODEL_FILE
)

print(
    "Saved:",
    MODEL_FILE
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

prediction_batch_size = 50000

prediction_parts = []


for start in range(
    0,
    len(X_test),
    prediction_batch_size
):

    end = min(
        start + prediction_batch_size,
        len(X_test)
    )

    X_batch = np.asarray(
        X_test[start:end],
        dtype=np.float32
    )

    X_batch = np.expand_dims(
        X_batch,
        axis=-1
    )

    probabilities = (
        global_model.predict(
            X_batch,
            batch_size=1024,
            verbose=0
        )
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    prediction_parts.append(
        predictions
    )

    print(
        f"Predicted "
        f"{end:,}/{len(X_test):,}"
    )

    del X_batch
    del probabilities

    gc.collect()


y_pred = np.concatenate(
    prediction_parts
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

wp, wr, wf1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)

mp, mr, mf1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
)


print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Accuracy           : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision : "
    f"{wp * 100:.2f}%"
)

print(
    f"Weighted Recall    : "
    f"{wr * 100:.2f}%"
)

print(
    f"Weighted F1        : "
    f"{wf1 * 100:.2f}%"
)

print(
    f"Macro Precision    : "
    f"{mp * 100:.2f}%"
)

print(
    f"Macro Recall       : "
    f"{mr * 100:.2f}%"
)

print(
    f"Macro F1           : "
    f"{mf1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,

    labels=np.arange(
        NUM_CLASSES
    ),

    target_names=class_names,

    digits=4,

    zero_division=0
)

print(
    "\nCLASSIFICATION REPORT\n"
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,

    labels=np.arange(
        NUM_CLASSES
    )
)

cm_file = os.path.join(
    REPORT_DIR,
    "cnn_bigru_mha_confusion_matrix.csv"
)

np.savetxt(
    cm_file,
    cm,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = os.path.join(
    REPORT_DIR,
    "cnn_bigru_mha_full_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Federated CNN-BiGRU-"
        "MultiHeadAttention IDS\n\n"
    )

    f.write(
        f"Training Records: "
        f"{len(y_train)}\n"
    )

    f.write(
        f"Testing Records: "
        f"{len(y_test)}\n"
    )

    f.write(
        f"Features: "
        f"{NUM_FEATURES}\n"
    )

    f.write(
        f"Classes: "
        f"{NUM_CLASSES}\n\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{wp:.6f}\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{wr:.6f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{wf1:.6f}\n"
    )

    f.write(
        f"Macro Precision: "
        f"{mp:.6f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{mr:.6f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{mf1:.6f}\n\n"
    )

    f.write(
        "Round Accuracies\n"
    )

    for i, value in enumerate(
        round_accuracies,
        1
    ):

        f.write(
            f"Round {i}: "
            f"{value:.6f}\n"
        )

    f.write(
        "\nClassification Report\n"
    )

    f.write(report)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nModel:",
    MODEL_FILE
)

print(
    "Report:",
    report_file
)

print(
    "Confusion Matrix:",
    cm_file
)

print(
    "\nFull training records used:",
    f"{len(y_train):,}"
)

print(
    "Test records:",
    f"{len(y_test):,}"
)

print(
    "Features:",
    NUM_FEATURES
)

print(
    "Classes:",
    NUM_CLASSES
)