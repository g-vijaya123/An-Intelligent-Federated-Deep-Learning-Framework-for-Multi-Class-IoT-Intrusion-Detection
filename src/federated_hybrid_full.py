import os
import gc
import joblib
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
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
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
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
    "federated_cnn_bilstm_attention_full.keras"
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
# GPU INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FEDERATED CNN-BiLSTM-ATTENTION IDS")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:", gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
        except:
            pass
else:
    print("\nGPU not detected.")
    print("TensorFlow will use CPU.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading FULL original training data...")

X_train = np.load(
    os.path.join(
        DATA_DIR,
        "X_train.npy"
    ),
    mmap_mode="r"
)

y_train = np.load(
    os.path.join(
        DATA_DIR,
        "y_train.npy"
    )
)

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

artifact = joblib.load(
    ARTIFACT_FILE
)

class_names = artifact["class_names"]

NUM_CLASSES = len(class_names)
NUM_FEATURES = X_train.shape[1]

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)

print("Features      :", NUM_FEATURES)
print("Classes       :", NUM_CLASSES)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# VERIFY THAT WE ARE NOT USING THE 1M BALANCED DATASET
# ============================================================

assert len(X_train) == 2032037, (
    "Unexpected training size. "
    "Expected the full 2,032,037-record training set."
)

assert NUM_FEATURES == 43, (
    "Expected 43 full features."
)

print("\n✓ Full original training set confirmed.")
print("✓ 2,032,037 training records.")
print("✓ 43 features.")
print("✓ No SelectKBest.")
print("✓ Balanced 1M dataset NOT used.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

unique_classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight
    in zip(unique_classes, weights)
}

print("\nClass weights:")

for cls in sorted(class_weights):

    print(
        f"{class_names[cls]:20s}: "
        f"{class_weights[cls]:.4f}"
    )


# ============================================================
# HYBRID MODEL
# CNN + BiLSTM + ATTENTION
# ============================================================

def build_hybrid_model():

    inputs = Input(
        shape=(NUM_FEATURES, 1),
        name="network_features"
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    x = Conv1D(
        filters=64,
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
        filters=128,
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
    # BiLSTM
    # --------------------------------------------------------

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(x)

    x = Dropout(0.30)(x)

    # --------------------------------------------------------
    # SELF ATTENTION
    # --------------------------------------------------------

    attention_output = Attention(
        name="self_attention"
    )([x, x])

    # Residual combination
    x = x + attention_output

    x = GlobalAveragePooling1D()(x)

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
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
# CREATE GLOBAL MODEL
# ============================================================

print("\nBuilding CNN-BiLSTM-Attention model...")

global_model = build_hybrid_model()

global_model.summary()


# ============================================================
# CREATE STRATIFIED CLIENTS
# ============================================================

print("\n" + "=" * 70)
print("CREATING 5 SIMULATED FEDERATED CLIENTS")
print("=" * 70)

rng = np.random.default_rng(SEED)

client_indices = [
    []
    for _ in range(NUM_CLIENTS)
]

# Split each class across all clients.
# This keeps every client reasonably representative.

for cls in unique_classes:

    indices = np.where(
        y_train == cls
    )[0]

    rng.shuffle(indices)

    splits = np.array_split(
        indices,
        NUM_CLIENTS
    )

    for client_id in range(NUM_CLIENTS):

        client_indices[
            client_id
        ].extend(
            splits[client_id].tolist()
        )


# Shuffle each client's indexes

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
        f"{len(client_indices[client_id]):,} records"
    )


# ============================================================
# FEDAVG
# ============================================================

def federated_average(
    local_weights,
    local_sizes
):

    total_samples = float(
        sum(local_sizes)
    )

    averaged_weights = []

    for weights_tuple in zip(
        *local_weights
    ):

        weighted_sum = np.zeros_like(
            weights_tuple[0],
            dtype=np.float32
        )

        for weights, size in zip(
            weights_tuple,
            local_sizes
        ):

            weighted_sum += (
                weights.astype(np.float32)
                * (size / total_samples)
            )

        averaged_weights.append(
            weighted_sum
        )

    return averaged_weights


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
        f"{'=' * 70}"
    )

    global_weights = (
        global_model.get_weights()
    )

    local_weights = []
    local_sizes = []

    # --------------------------------------------------------
    # Train each client
    # --------------------------------------------------------

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
            "Records:",
            f"{len(indices):,}"
        )

        # Materialize one client only.
        # Helps avoid unnecessary full-data copies.

        X_client = np.asarray(
            X_train[indices],
            dtype=np.float32
        )

        y_client = np.asarray(
            y_train[indices],
            dtype=np.int16
        )

        # CNN/BiLSTM expects:
        # (samples, features, channels)

        X_client = np.expand_dims(
            X_client,
            axis=-1
        )

        local_model = (
            build_hybrid_model()
        )

        local_model.set_weights(
            global_weights
        )

        local_model.fit(
            X_client,
            y_client,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            verbose=1,
            shuffle=True
        )

        local_weights.append(
            local_model.get_weights()
        )

        local_sizes.append(
            len(y_client)
        )

        # Free memory before next client.

        del X_client
        del y_client
        del local_model

        gc.collect()

        tf.keras.backend.clear_session()

    # --------------------------------------------------------
    # FedAvg
    # --------------------------------------------------------

    print("\nApplying FedAvg...")

    averaged_weights = federated_average(
        local_weights,
        local_sizes
    )

    global_model.set_weights(
        averaged_weights
    )

    del local_weights
    del averaged_weights

    gc.collect()

    # --------------------------------------------------------
    # Evaluate after this federated round
    # --------------------------------------------------------

    print(
        "\nEvaluating global model..."
    )

    # Evaluate in chunks through Keras.
    X_test_3d = np.expand_dims(
        np.asarray(
            X_test,
            dtype=np.float32
        ),
        axis=-1
    )

    test_loss, test_accuracy = (
        global_model.evaluate(
            X_test_3d,
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
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    del X_test_3d
    gc.collect()


# ============================================================
# SAVE FINAL GLOBAL MODEL
# ============================================================

print("\nSaving final global model...")

global_model.save(
    MODEL_FILE
)

print(
    "Model saved:",
    MODEL_FILE
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print(
    "\nPredicting all untouched test records..."
)

# Predict directly in manageable chunks.

prediction_batch_size = 50000

predicted_parts = []

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

    probabilities = global_model.predict(
        X_batch,
        batch_size=1024,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    predicted_parts.append(
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
    predicted_parts
)

del predicted_parts


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
)


print("\n" + "=" * 70)
print("FINAL METRICS")
print("=" * 70)

print(
    f"Accuracy           : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision : "
    f"{weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall    : "
    f"{weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1        : "
    f"{weighted_f1 * 100:.2f}%"
)

print(
    f"Macro Precision    : "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall       : "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1           : "
    f"{macro_f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(
    "\nCLASSIFICATION REPORT\n"
)

print(report)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = os.path.join(
    REPORT_DIR,
    "hybrid_full_classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Federated CNN-BiLSTM-Attention\n"
    )

    file.write(
        "Full UNSW-NB15 Dataset\n\n"
    )

    file.write(
        f"Training Records: "
        f"{len(y_train)}\n"
    )

    file.write(
        f"Testing Records: "
        f"{len(y_test)}\n"
    )

    file.write(
        f"Features: "
        f"{NUM_FEATURES}\n"
    )

    file.write(
        f"Classes: "
        f"{NUM_CLASSES}\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )

    file.write(
        f"Weighted Precision: "
        f"{weighted_precision:.6f}\n"
    )

    file.write(
        f"Weighted Recall: "
        f"{weighted_recall:.6f}\n"
    )

    file.write(
        f"Weighted F1: "
        f"{weighted_f1:.6f}\n"
    )

    file.write(
        f"Macro Precision: "
        f"{macro_precision:.6f}\n"
    )

    file.write(
        f"Macro Recall: "
        f"{macro_recall:.6f}\n"
    )

    file.write(
        f"Macro F1: "
        f"{macro_f1:.6f}\n\n"
    )

    file.write(
        "Federated Round Accuracies\n"
    )

    for i, value in enumerate(
        round_accuracies,
        1
    ):

        file.write(
            f"Round {i}: "
            f"{value:.6f}\n"
        )

    file.write(
        "\nClassification Report\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

np.savetxt(
    os.path.join(
        REPORT_DIR,
        "hybrid_full_confusion_matrix.csv"
    ),
    cm,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nFinal model:"
)

print(
    MODEL_FILE
)

print(
    "\nClassification report:"
)

print(
    report_file
)

print(
    "\nConfusion matrix:"
)

print(
    os.path.join(
        REPORT_DIR,
        "hybrid_full_confusion_matrix.csv"
    )
)

print(
    "\nFull original training records used:",
    f"{len(y_train):,}"
)

print(
    "Untouched test records:",
    f"{len(y_test):,}"
)

print(
    "Input features:",
    NUM_FEATURES
)

print(
    "Output classes:",
    NUM_CLASSES
)