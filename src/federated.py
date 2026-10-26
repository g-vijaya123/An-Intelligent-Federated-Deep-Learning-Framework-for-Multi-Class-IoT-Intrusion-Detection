"""
Federated Learning simulation for the UNSW-NB15 IDS.

Flow:
1. Preprocess UNSW-NB15 in multiclass mode.
2. Apply Controlled SMOTE ONLY to training data.
3. Divide balanced training data among simulated organizations.
4. Each organization trains a local CNN-LSTM.
5. Aggregate local weights using FedAvg.
6. Update the global model.
7. Evaluate the final global model on untouched test data.
"""

import argparse
import os
import sys

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# --------------------------------------------------
# Import project modules
# --------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from preprocess import preprocess
from model import build_cnn_lstm
from balance import apply_smote_tomek


# ==================================================
# Split training data between clients
# ==================================================

def split_clients(X, y, num_clients=5):
    """
    Randomly divide training data between
    simulated organizations.
    """

    indices = np.arange(len(X))

    # Reproducible shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    client_indices = np.array_split(
        indices,
        num_clients
    )

    clients = []

    for idx in client_indices:
        clients.append(
            (X[idx], y[idx])
        )

    return clients


# ==================================================
# Federated Averaging
# ==================================================

def fedavg(client_weights, client_sizes):
    """
    Perform sample-size weighted
    Federated Averaging (FedAvg).
    """

    total_samples = sum(client_sizes)

    global_weights = []

    for layer_weights in zip(*client_weights):

        averaged_layer = np.zeros_like(
            layer_weights[0]
        )

        for weights, size in zip(
            layer_weights,
            client_sizes,
        ):
            averaged_layer += (
                weights *
                (size / total_samples)
            )

        global_weights.append(
            averaged_layer
        )

    return global_weights


# ==================================================
# Main
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Federated CNN-LSTM Multiclass IDS "
            "with Controlled SMOTE"
        )
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=5
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=5
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--k",
        type=int,
        default=32
    )

    parser.add_argument(
        "--split",
        default="stratified"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FEDERATED MULTICLASS IDS")
    print("CNN-LSTM + CONTROLLED SMOTE + FEDAVG")
    print("=" * 60)

    # ==================================================
    # STEP 1: Preprocessing
    # ==================================================

    X_train, y_train, X_test, y_test, \
        class_names, feature_names = preprocess(

            mode="multiclass",
            k=args.k,
            split=args.split,
            verbose=True,
        )

    print("\nAttack classes:")

    for i, name in enumerate(class_names):
        print(
            i,
            "->",
            name
        )

    # ==================================================
    # STEP 2: Controlled SMOTE
    # ==================================================
    #
    # IMPORTANT:
    # Balancing is applied ONLY to X_train/y_train.
    #
    # X_test/y_test remain completely untouched.
    # ==================================================

    print("\n" + "=" * 60)
    print("APPLYING CONTROLLED SMOTE")
    print("=" * 60)

    X_train, y_train = apply_smote_tomek(
        X_train,
        y_train
    )

    print("\nBalanced federated training dataset:")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "Untouched X_test:",
        X_test.shape
    )

    print(
        "Untouched y_test:",
        y_test.shape
    )

    # Get shapes AFTER balancing
    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1]

    # ==================================================
    # STEP 3: Create simulated organizations
    # ==================================================

    clients = split_clients(
        X_train,
        y_train,
        num_clients=args.clients,
    )

    print("\n" + "=" * 60)
    print("CLIENT DISTRIBUTION")
    print("=" * 60)

    for i, (X_client, y_client) in enumerate(clients):

        print(
            f"Organization {i + 1}: "
            f"{len(X_client)} training samples"
        )

        # Show attack distribution at each organization
        labels = np.argmax(
            y_client,
            axis=1
        )

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        distribution = dict(
            zip(
                unique.tolist(),
                counts.tolist()
            )
        )

        print(
            "  Class distribution:",
            distribution
        )

    # ==================================================
    # STEP 4: Build global CNN-LSTM
    # ==================================================

    global_model = build_cnn_lstm(
        input_shape,
        n_classes,
    )

    # ==================================================
    # STEP 5: Federated training
    # ==================================================

    round_accuracies = []

    for round_number in range(args.rounds):

        print("\n" + "=" * 60)

        print(
            f"FEDERATED ROUND "
            f"{round_number + 1}/"
            f"{args.rounds}"
        )

        print("=" * 60)

        # Current global model weights
        global_weights = (
            global_model.get_weights()
        )

        client_weights = []
        client_sizes = []

        # ==============================================
        # Local client training
        # ==============================================

        for client_id, (
            X_client,
            y_client
        ) in enumerate(clients):

            print(
                f"\nTraining Organization "
                f"{client_id + 1}"
            )

            # Build local CNN-LSTM
            local_model = build_cnn_lstm(
                input_shape,
                n_classes,
            )

            # Start every organization from
            # current global model weights
            local_model.set_weights(
                global_weights
            )

            # Local training
            local_model.fit(
                X_client,
                y_client,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                verbose=1,
            )

            # Save local weights
            client_weights.append(
                local_model.get_weights()
            )

            # Save local sample count
            client_sizes.append(
                len(X_client)
            )

            # Delete local model reference
            del local_model

        # ==============================================
        # Server-side FedAvg
        # ==============================================

        print(
            "\nAggregating local models "
            "using FedAvg..."
        )

        averaged_weights = fedavg(
            client_weights,
            client_sizes,
        )

        global_model.set_weights(
            averaged_weights
        )

        # ==============================================
        # Evaluate current global model
        # ==============================================

        loss, accuracy = global_model.evaluate(
            X_test,
            y_test,
            verbose=0,
        )

        round_accuracies.append(
            accuracy
        )

        print(
            f"Round {round_number + 1} "
            f"Global Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

    # ==================================================
    # STEP 6: Final prediction
    # ==================================================

    probabilities = global_model.predict(
        X_test,
        verbose=0,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    y_true = np.argmax(
        y_test,
        axis=1,
    )

    # ==================================================
    # STEP 7: Final metrics
    # ==================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ==================================================
    # Display round progression
    # ==================================================

    print("\n" + "=" * 60)
    print("FEDERATED ROUND ACCURACY")
    print("=" * 60)

    for i, round_accuracy in enumerate(
        round_accuracies,
        start=1,
    ):
        print(
            f"Round {i}: "
            f"{round_accuracy * 100:.2f}%"
        )

    # ==================================================
    # Final results
    # ==================================================

    print("\n" + "=" * 60)
    print("FINAL BALANCED FEDERATED RESULTS")
    print("=" * 60)

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Weighted Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Weighted F1-score  : "
        f"{f1 * 100:.2f}%"
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
        f"Macro F1-score     : "
        f"{macro_f1 * 100:.2f}%"
    )

    print("\nPer-class results:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    # ==================================================
    # STEP 8: Save global model
    # ==================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # Different filename so original 79.43% model
    # is NOT overwritten.
    model_path = os.path.join(
        "models",
        "federated_cnn_lstm_multiclass_smote.keras",
    )

    global_model.save(
        model_path
    )

    print(
        "\nBalanced global model saved ->",
        model_path,
    )


if __name__ == "__main__":
    main()