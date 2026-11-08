from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GRU,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "independent_fl_v1"
MODEL_ROOT = PROJECT_ROOT / "models" / "independent_fl_v1"
RESULT_ROOT = PROJECT_ROOT / "results" / "independent_fl_v1"

DATASET_NAMES = ["UNSW_NB15", "CICIoT2023", "EdgeIIoTset"]

DEFAULT_ROUNDS = 10
DEFAULT_LOCAL_STEPS = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_CLIENTS = 3
READ_CHUNK_SIZE = 20_000
CLASS_WEIGHT_CAP = 5.0
EARLY_STOPPING_PATIENCE = 4


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_cnn_bigru_mha(num_features: int, num_classes: int) -> Model:
    """CNN + bidirectional GRU + multi-head self-attention classifier."""
    inputs = Input(shape=(num_features, 1), name="network_features")

    x = Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn_conv_1",
    )(inputs)
    x = BatchNormalization(name="cnn_batch_norm")(x)
    x = Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn_conv_2",
    )(x)

    x = Bidirectional(
        GRU(64, return_sequences=True, dropout=0.15),
        name="bigru",
    )(x)

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.10,
        name="multi_head_attention",
    )(x, x)

    x = Add(name="attention_residual")([x, attention])
    x = LayerNormalization(name="attention_layer_norm")(x)

    feed_forward = Conv1D(
        filters=128,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="attention_feed_forward",
    )(x)
    x = Add(name="feed_forward_residual")([x, feed_forward])
    x = LayerNormalization(name="feed_forward_layer_norm")(x)

    x = GlobalAveragePooling1D(name="global_average_pooling")(x)
    x = Dense(128, activation="relu", name="dense_features")(x)
    x = Dropout(0.30, name="classifier_dropout")(x)
    outputs = Dense(
        num_classes,
        activation="softmax",
        name="dataset_specific_probabilities",
    )(x)

    return Model(inputs=inputs, outputs=outputs, name="CNN_BiGRU_MHA")


def merge_running_stats(
    count_a: int,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    count_b: int,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
):
    if count_a == 0:
        return count_b, mean_b.copy(), m2_b.copy()
    if count_b == 0:
        return count_a, mean_a.copy(), m2_a.copy()

    delta = mean_b - mean_a
    total = count_a + count_b
    mean = mean_a + delta * (count_b / total)
    m2 = (
        m2_a
        + m2_b
        + (delta * delta) * (count_a * count_b / total)
    )
    return total, mean, m2


def fit_training_only_scaler(
    dataset_dir: Path,
    features: list[str],
    num_clients: int,
    chunk_size: int = READ_CHUNK_SIZE,
):
    """Fit one scaler per dataset using only that dataset's training-client CSVs."""
    count = 0
    mean = np.zeros(len(features), dtype=np.float64)
    m2 = np.zeros(len(features), dtype=np.float64)
    client_counts = {}

    for client_id in range(1, num_clients + 1):
        path = dataset_dir / f"train_client_{client_id:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Training client file not found: {path}")

        rows_for_client = 0
        for chunk in pd.read_csv(
            path,
            usecols=features,
            chunksize=chunk_size,
            low_memory=False,
        ):
            X = chunk.apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            values = X.to_numpy(dtype=np.float64, copy=False)
            n = len(values)
            if n == 0:
                continue

            batch_mean = values.mean(axis=0)
            batch_m2 = np.square(values - batch_mean).sum(axis=0)
            count, mean, m2 = merge_running_stats(
                count, mean, m2, n, batch_mean, batch_m2
            )
            rows_for_client += n

        client_counts[str(client_id)] = int(rows_for_client)

    if count < 2:
        raise RuntimeError(f"Not enough training rows to fit scaler in {dataset_dir}")

    variance = np.maximum(m2 / count, 0.0)
    std = np.sqrt(variance)
    constant_mask = std < 1e-12
    std[constant_mask] = 1.0

    scaler_path = dataset_dir / "scaler.npz"
    np.savez(
        scaler_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        features=np.asarray(features, dtype=str),
        training_records=np.asarray([count], dtype=np.int64),
    )

    scaler_metadata = {
        "fit_scope": "training client files only",
        "training_records_used": int(count),
        "client_training_records": client_counts,
        "features": features,
        "mean": {features[i]: float(mean[i]) for i in range(len(features))},
        "std": {features[i]: float(std[i]) for i in range(len(features))},
        "constant_features": [
            features[i] for i, is_constant in enumerate(constant_mask) if is_constant
        ],
    }
    (dataset_dir / "scaler_metadata.json").write_text(
        json.dumps(scaler_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Training-only scaler saved: {scaler_path}")
    return mean.astype(np.float32), std.astype(np.float32), client_counts


def calculate_class_weights(metadata: dict, classes: list[str]) -> np.ndarray:
    """Mild square-root inverse-frequency weights, capped to reduce instability."""
    counts_dict = metadata.get("train_class_counts", {})
    counts = np.asarray([int(counts_dict.get(label, 0)) for label in classes], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise RuntimeError("No training class counts found in dataset metadata.")

    weights = np.ones(len(classes), dtype=np.float64)
    present = counts > 0
    weights[present] = np.sqrt(total / (len(classes) * counts[present]))
    weights = np.minimum(weights, CLASS_WEIGHT_CAP)

    # Normalize so the average sample weight across the training set is approximately 1.
    weighted_average = np.sum(weights * counts) / total
    if weighted_average > 0:
        weights /= weighted_average

    return weights.astype(np.float32)


def make_tf_dataset(
    csv_path: Path,
    features: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    repeat: bool,
    training: bool,
    class_weights: np.ndarray | None = None,
    seed: int = 42,
):
    """Read CSV data in chunks; avoid loading a complete dataset into RAM."""
    num_features = len(features)

    def generator():
        rng = np.random.default_rng(seed)
        for chunk in pd.read_csv(
            csv_path,
            chunksize=max(batch_size * 20, 4096),
            low_memory=False,
        ):
            X_frame = chunk[features].apply(pd.to_numeric, errors="coerce")
            X_frame = X_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X = X_frame.to_numpy(dtype=np.float32, copy=True)
            X = (X - mean) / std
            # Limit extreme standardized values to keep recurrent training stable.
            np.clip(X, -20.0, 20.0, out=X)

            y = chunk["label_id"].to_numpy(dtype=np.int32, copy=True)

            indices = np.arange(len(y))
            if training:
                rng.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                xb = X[batch_indices].reshape(-1, num_features, 1)
                yb = y[batch_indices]

                if training:
                    if class_weights is None:
                        raise ValueError("class_weights must be supplied for training.")
                    sample_weights = class_weights[yb]
                    yield xb.astype(np.float32), yb.astype(np.int32), sample_weights.astype(np.float32)
                else:
                    yield xb.astype(np.float32), yb.astype(np.int32)

    if training:
        signature = (
            tf.TensorSpec(shape=(None, num_features, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
    else:
        signature = (
            tf.TensorSpec(shape=(None, num_features, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(1)


def fedavg(local_weights: list[list[np.ndarray]], client_sizes: list[int]) -> list[np.ndarray]:
    """Sample-count-weighted FedAvg for clients within one dataset."""
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("Cannot aggregate clients with zero training records.")

    aggregated = []
    for weight_index in range(len(local_weights[0])):
        accumulator = np.zeros_like(
            local_weights[0][weight_index],
            dtype=np.float64,
        )
        for weights, size in zip(local_weights, client_sizes):
            accumulator += weights[weight_index].astype(np.float64) * (size / total)
        aggregated.append(accumulator.astype(local_weights[0][weight_index].dtype))
    return aggregated


def confusion_matrix_from_dataset(
    model: Model,
    dataset: tf.data.Dataset,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for xb, yb in dataset:
        probabilities = model(xb, training=False).numpy()
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
        actual = yb.numpy().astype(np.int64)
        encoded = actual * num_classes + predictions
        matrix += np.bincount(
            encoded,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
    return matrix


def metrics_from_confusion_matrix(matrix: np.ndarray, classes: list[str]):
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)

    precision = np.divide(
        tp, predicted, out=np.zeros_like(tp), where=predicted > 0
    )
    recall = np.divide(
        tp, support, out=np.zeros_like(tp), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    total = support.sum()
    accuracy = float(tp.sum() / total) if total else 0.0

    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    weighted_precision = float(np.average(precision, weights=support)) if total else 0.0
    weighted_recall = float(np.average(recall, weights=support)) if total else 0.0
    weighted_f1 = float(np.average(f1, weights=support)) if total else 0.0

    report_rows = []
    for i, class_name in enumerate(classes):
        report_rows.append({
            "class": class_name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        })

    report_rows.extend([
        {
            "class": "macro_avg",
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "support": int(total),
        },
        {
            "class": "weighted_avg",
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
            "support": int(total),
        },
    ])

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "test_records": int(total),
        "number_of_classes": len(classes),
    }
    return metrics, report_rows


def train_one_dataset(
    dataset_name: str,
    rounds: int,
    local_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    print("\n" + "=" * 88)
    print(f"FEDERATED TRAINING: {dataset_name}")
    print("=" * 88)

    dataset_dir = DATA_ROOT / dataset_name
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run prepare_independent_fl_data.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["feature_columns"]
    classes = metadata["class_names"]
    num_clients = int(metadata["number_of_simulated_clients"])
    num_features = len(features)
    num_classes = len(classes)

    for required in ["validation.csv", "test.csv"]:
        if not (dataset_dir / required).exists():
            raise FileNotFoundError(f"Required split missing: {dataset_dir / required}")

    set_seeds(seed)

    mean, std, observed_client_counts = fit_training_only_scaler(
        dataset_dir,
        features,
        num_clients,
    )
    class_weights = calculate_class_weights(metadata, classes)

    expected_client_counts = metadata.get("train_client_counts", {})
    client_sizes = [
        int(expected_client_counts.get(str(client_id), observed_client_counts.get(str(client_id), 0)))
        for client_id in range(1, num_clients + 1)
    ]
    if any(size <= 0 for size in client_sizes):
        raise RuntimeError(
            f"{dataset_name}: at least one simulated client has no training records: {client_sizes}"
        )

    validation_ds = make_tf_dataset(
        dataset_dir / "validation.csv",
        features,
        mean,
        std,
        batch_size,
        repeat=False,
        training=False,
        seed=seed + 100,
    )

    global_model = build_cnn_bigru_mha(num_features, num_classes)
    global_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    global_weights = global_model.get_weights()

    model_dir = MODEL_ROOT / dataset_name
    result_dir = RESULT_ROOT / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "federated_cnn_bigru_mha.keras"
    round_history = []
    best_validation_loss = float("inf")
    best_round = 0
    patience_counter = 0

    print(f"Features             : {num_features}")
    print(f"Classes              : {num_classes}")
    print(f"Simulated clients    : {num_clients}")
    print(f"Client training rows : {client_sizes}")
    print(f"Rounds/local steps   : {rounds}/{local_steps}")
    print(f"Batch size           : {batch_size}")

    for round_number in range(1, rounds + 1):
        local_weights = []
        local_losses = []

        print(f"\nRound {round_number}/{rounds}")

        for client_id in range(1, num_clients + 1):
            client_path = dataset_dir / f"train_client_{client_id:02d}.csv"
            client_seed = seed + round_number * 1000 + client_id

            local_model = build_cnn_bigru_mha(num_features, num_classes)
            local_model.set_weights(global_weights)
            local_model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            client_train_ds = make_tf_dataset(
                client_path,
                features,
                mean,
                std,
                batch_size,
                repeat=True,
                training=True,
                class_weights=class_weights,
                seed=client_seed,
            )

            history = local_model.fit(
                client_train_ds,
                epochs=1,
                steps_per_epoch=local_steps,
                verbose=0,
            )

            local_loss = float(history.history["loss"][-1])
            if not np.isfinite(local_loss):
                raise RuntimeError(
                    f"{dataset_name} client {client_id}: non-finite local loss."
                )

            local_losses.append(local_loss)
            local_weights.append(local_model.get_weights())

            print(
                f"  Client {client_id}: rows={client_sizes[client_id - 1]:,}, "
                f"local_loss={local_loss:.5f}"
            )

            del local_model, client_train_ds, history
            gc.collect()

        # Aggregation is strictly within the current dataset experiment.
        global_weights = fedavg(local_weights, client_sizes)
        global_model.set_weights(global_weights)

        validation_metrics = global_model.evaluate(
            validation_ds,
            verbose=0,
            return_dict=True,
        )
        validation_loss = float(validation_metrics["loss"])
        validation_accuracy = float(validation_metrics["accuracy"])
        average_local_loss = float(np.mean(local_losses))

        row = {
            "round": round_number,
            "average_local_loss": average_local_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        round_history.append(row)

        print(
            f"  FedAvg validation: loss={validation_loss:.5f}, "
            f"accuracy={validation_accuracy:.5f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_round = round_number
            patience_counter = 0
            global_model.save(best_model_path)
            print(f"  Saved best model at round {best_round}.")
        else:
            patience_counter += 1
            print(f"  No validation-loss improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE}).")

        pd.DataFrame(round_history).to_csv(
            result_dir / "round_history.csv",
            index=False,
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping: validation loss did not improve.")
            break

        del local_weights
        gc.collect()

    if not best_model_path.exists():
        raise RuntimeError(f"Training did not produce a model: {best_model_path}")

    best_model = tf.keras.models.load_model(best_model_path, compile=False)

    test_ds = make_tf_dataset(
        dataset_dir / "test.csv",
        features,
        mean,
        std,
        batch_size,
        repeat=False,
        training=False,
        seed=seed + 999,
    )

    confusion = confusion_matrix_from_dataset(best_model, test_ds, num_classes)
    evaluation_metrics, report_rows = metrics_from_confusion_matrix(confusion, classes)

    evaluation_metrics.update({
        "dataset": dataset_name,
        "best_round": int(best_round),
        "best_validation_loss": float(best_validation_loss),
        "model": "CNN + BiGRU + Multi-Head Attention",
        "federated_method": "sample-count-weighted FedAvg within this dataset only",
        "simulated_clients": num_clients,
        "features": num_features,
        "classes": num_classes,
        "rounds_requested": rounds,
        "local_steps_per_client_per_round": local_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "scaler_fit_on": "training split only",
    })

    (result_dir / "evaluation_metrics.json").write_text(
        json.dumps(evaluation_metrics, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(report_rows).to_csv(
        result_dir / "classification_report.csv",
        index=False,
    )
    pd.DataFrame(
        confusion,
        index=classes,
        columns=classes,
    ).to_csv(result_dir / "confusion_matrix.csv", index=True)

    experiment_config = {
        "dataset": dataset_name,
        "architecture": "CNN + BiGRU + Multi-Head Attention",
        "feature_columns": features,
        "class_names": classes,
        "num_clients": num_clients,
        "fedavg": "sample-count-weighted",
        "rounds_requested": rounds,
        "local_steps_per_client_per_round": local_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_round": int(best_round),
        "model_path": str(best_model_path),
        "scaler_path": str(dataset_dir / "scaler.npz"),
    }
    (result_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2),
        encoding="utf-8",
    )

    print("\nTEST RESULTS")
    print(json.dumps(evaluation_metrics, indent=2))
    print(f"\nModel      : {best_model_path}")
    print(f"Metrics    : {result_dir / 'evaluation_metrics.json'}")
    print(f"Confusion  : {result_dir / 'confusion_matrix.csv'}")
    print(f"Report     : {result_dir / 'classification_report.csv'}")

    del best_model, global_model, validation_ds, test_ds
    tf.keras.backend.clear_session()
    gc.collect()

    return evaluation_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train three independent CNN-BiGRU-MHA federated experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=["all"] + DATASET_NAMES,
        default="all",
        help="Train one dataset experiment or all three sequentially.",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--local-steps", type=int, default=DEFAULT_LOCAL_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rounds < 1 or args.local_steps < 1 or args.batch_size < 1:
        raise ValueError("rounds, local-steps and batch-size must be positive.")

    selected_datasets = DATASET_NAMES if args.dataset == "all" else [args.dataset]

    print("=" * 88)
    print("THREE INDEPENDENT FEDERATED LEARNING EXPERIMENTS")
    print("=" * 88)
    print("Model: CNN + BiGRU + Multi-Head Attention")
    print("Aggregation: sample-count-weighted FedAvg")
    print("Cross-dataset aggregation: DISABLED")
    print(f"Datasets: {selected_datasets}")

    all_results = {}
    for dataset_name in selected_datasets:
        all_results[dataset_name] = train_one_dataset(
            dataset_name=dataset_name,
            rounds=args.rounds,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

    summary_path = RESULT_ROOT / "all_dataset_results.json"
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("FEDERATED EXPERIMENTS COMPLETED")
    print("=" * 88)
    print(f"Combined results index: {summary_path}")
    print("Remember: each dataset has its own model, scaler, features and class probabilities.")


if __name__ == "__main__":
    main()
