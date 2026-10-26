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
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Bidirectional, GRU, MultiHeadAttention,
    LayerNormalization, Add, GlobalAveragePooling1D,
    Dense, Dropout
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIG
# ============================================================

SEED = 42

NUM_CLIENTS = 5
FEDERATED_ROUNDS = 5
LOCAL_EPOCHS = 1

BATCH_SIZE = 512
LEARNING_RATE = 0.0003

DATA_DIR = os.path.join("data", "processed_full")
MODEL_DIR = "models"
REPORT_DIR = "report"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "federated_cnn_bigru_mha_focal.keras"
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
# LOAD FULL DATA
# ============================================================

print("\n" + "=" * 70)
print("FEDERATED CNN-BiGRU-MHA + FOCAL LOSS")
print("=" * 70)

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

print("\n✓ Full 2,032,037 training records")
print("✓ 43 features")
print("✓ 10 dataset classes")
print("✓ No undersampling")
print("✓ No oversampling")
print("✓ No 1M balanced dataset")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

print("\nTraining class distribution:")

for i in range(NUM_CLASSES):
    print(
        f"{i:2d} {class_names[i]:20s}: "
        f"{counts[i]:,}"
    )


# ============================================================
# FOCAL ALPHA
# ============================================================
#
# Focal loss already down-weights easy examples.
#
# We therefore use MODERATE class-specific alpha values,
# rather than the previous extreme class weights.
#
# alpha is derived from sqrt inverse frequency and normalized.
# Finally clip it to avoid huge Worms weight.
# ============================================================

frequency = counts / len(y_train)

alpha = 1.0 / np.sqrt(
    frequency + 1e-12
)

# Normalize so average alpha = 1
alpha = alpha / np.mean(alpha)

# Prevent extremely tiny / huge alpha
alpha = np.clip(
    alpha,
    0.25,
    5.0
)

alpha = alpha.astype(
    np.float32
)

print("\nFocal alpha values:")

for i in range(NUM_CLASSES):
    print(
        f"{class_names[i]:20s}: "
        f"{alpha[i]:.4f}"
    )


# ============================================================
# SPARSE MULTICLASS FOCAL LOSS
# ============================================================

GAMMA = 2.0

alpha_tensor = tf.constant(
    alpha,
    dtype=tf.float32
)


def sparse_focal_loss(
    y_true,
    y_pred
):
    """
    Multiclass focal loss for sparse integer labels.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    y_true = tf.cast(
        tf.reshape(y_true, [-1]),
        tf.int32
    )

    y_pred = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0 - 1e-7
    )

    indices = tf.stack(
        [
            tf.range(
                tf.shape(y_true)[0]
            ),
            y_true
        ],
        axis=1
    )

    p_t = tf.gather_nd(
        y_pred,
        indices
    )

    alpha_t = tf.gather(
        alpha_tensor,
        y_true
    )

    focal_factor = tf.pow(
        1.0 - p_t,
        GAMMA
    )

    loss = (
        -alpha_t
        * focal_factor
        * tf.math.log(p_t)
    )

    return tf.reduce_mean(loss)


# ============================================================
# MODEL
# CNN + BiGRU + MULTI-HEAD ATTENTION
# ============================================================

def build_model():

    inputs = Input(
        shape=(NUM_FEATURES, 1),
        name="network_features"
    )

    # CNN BLOCK 1
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


    # CNN BLOCK 2
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


    # BiGRU
    x = Bidirectional(
        GRU(
            64,
            return_sequences=True
        )
    )(x)

    x = Dropout(0.25)(x)


    # MULTI-HEAD SELF ATTENTION
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


    # RESIDUAL CONNECTION
    x = Add()([
        x,
        attention
    ])

    x = LayerNormalization()(x)


    # GLOBAL FEATURE VECTOR
    x = GlobalAveragePooling1D()(x)


    # CLASSIFICATION HEAD
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
        loss=sparse_focal_loss,
        metrics=["accuracy"]
    )

    return model


# ============================================================
# GLOBAL MODEL
# ============================================================

print("\nBuilding model...")

global_model = build_model()

global_model.summary()


# ============================================================
# CREATE STRATIFIED FEDERATED CLIENTS
# ============================================================

print("\nCreating 5 stratified simulated clients...")

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

    splits = np.array_split(
        indices,
        NUM_CLIENTS
    )

    for client_id in range(NUM_CLIENTS):

        client_indices[
            client_id
        ].extend(
            splits[
                client_id
            ].tolist()
        )


for client_id in range(NUM_CLIENTS):

    client_indices[
        client_id
    ] = np.array(
        client_indices[
            client_id
        ],
        dtype=np.int64
    )

    rng.shuffle(
        client_indices[
            client_id
        ]
    )

    print(
        f"Client {client_id + 1}: "
        f"{len(client_indices[client_id]):,}"
    )


# ============================================================
# WEIGHTED FEDAVG
# ============================================================

def weighted_fedavg(
    local_weights,
    local_sizes
):

    total_samples = float(
        sum(local_sizes)
    )

    averaged_weights = []

    for layer_tuple in zip(
        *local_weights
    ):

        layer_average = np.zeros_like(
            layer_tuple[0],
            dtype=np.float32
        )

        for weights, size in zip(
            layer_tuple,
            local_sizes
        ):

            layer_average += (
                weights.astype(
                    np.float32
                )
                * (
                    size
                    / total_samples
                )
            )

        averaged_weights.append(
            layer_average
        )

    return averaged_weights


# ============================================================
# FEDERATED TRAINING
# ============================================================

round_accuracies = []

print("\n" + "=" * 70)
print("STARTING FEDERATED FOCAL-LOSS TRAINING")
print("=" * 70)


for round_number in range(
    1,
    FEDERATED_ROUNDS + 1
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"FEDERATED ROUND "
        f"{round_number}/"
        f"{FEDERATED_ROUNDS}"
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
    # EACH CLIENT
    # ========================================================

    for client_id in range(
        NUM_CLIENTS
    ):

        print(
            f"\n--- Client "
            f"{client_id + 1}/"
            f"{NUM_CLIENTS} ---"
        )

        indices = client_indices[
            client_id
        ]

        print(
            "Records:",
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


        # IMPORTANT:
        # No class_weight here.
        # Focal loss already contains class alpha.

        local_model.fit(
            X_client,
            y_client,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
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
    # FEDAVG
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
    # ROUND EVALUATION
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
# SAVE FINAL MODEL
# ============================================================

print(
    "\nSaving final Focal-Loss model..."
)

global_model.save(
    MODEL_FILE
)

print(
    "Saved:",
    MODEL_FILE
)


# ============================================================
# FINAL PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

prediction_parts = []

PREDICTION_CHUNK = 50000


for start in range(
    0,
    len(X_test),
    PREDICTION_CHUNK
):

    end = min(
        start + PREDICTION_CHUNK,
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
        f"{end:,}/"
        f"{len(X_test):,}"
    )


    del X_batch
    del probabilities

    gc.collect()


y_pred = np.concatenate(
    prediction_parts
)

del prediction_parts


# ============================================================
# FINAL METRICS
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
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)


cm_file = os.path.join(
    REPORT_DIR,
    "cnn_bigru_mha_focal_confusion_matrix.csv"
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
    "cnn_bigru_mha_focal_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Federated CNN-BiGRU-MultiHeadAttention "
        "+ Focal Loss\n\n"
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
        f"{NUM_CLASSES}\n"
    )

    f.write(
        f"Focal Gamma: "
        f"{GAMMA}\n\n"
    )

    f.write(
        "Focal Alpha Values:\n"
    )

    for i in range(NUM_CLASSES):

        f.write(
            f"{class_names[i]}: "
            f"{alpha[i]:.6f}\n"
        )

    f.write("\n")

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
        "Round Accuracies:\n"
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
        "\nClassification Report\n\n"
    )

    f.write(report)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("V3 TRAINING COMPLETE")
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
    "\nTraining records:",
    f"{len(y_train):,}"
)

print(
    "Testing records:",
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