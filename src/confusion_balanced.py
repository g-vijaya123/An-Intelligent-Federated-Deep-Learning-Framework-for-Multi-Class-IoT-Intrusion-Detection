import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from preprocess import preprocess


# ==================================================
# STEP 1: Preprocess test data
# ==================================================

print("=" * 60)
print("BALANCED FEDERATED MODEL - CONFUSION MATRIX")
print("=" * 60)

X_train, y_train, X_test, y_test, \
    class_names, feature_names = preprocess(
        mode="multiclass",
        k=32,
        split="stratified",
        verbose=False
    )

print("Test samples:", len(X_test))
print("Classes:", class_names)


# ==================================================
# STEP 2: Load balanced federated model
# ==================================================

model_path = os.path.join(
    "models",
    "federated_cnn_lstm_multiclass_smote.keras"
)

print("\nLoading model:", model_path)

model = load_model(
    model_path,
    compile=False
)


# ==================================================
# STEP 3: Predictions
# ==================================================

print("Generating predictions...")

probabilities = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)


# ==================================================
# STEP 4: Confusion Matrix
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==================================================
# STEP 5: Plot
# ==================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=True
)

plt.title(
    "Confusion Matrix - Federated CNN-LSTM with Controlled SMOTE"
)

plt.xlabel(
    "Predicted Attack Category"
)

plt.ylabel(
    "Actual Attack Category"
)

plt.tight_layout()


# ==================================================
# STEP 6: Save
# ==================================================

os.makedirs(
    "report",
    exist_ok=True
)

output_path = os.path.join(
    "report",
    "balanced_federated_confusion_matrix.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved ->",
    output_path
)

print("=" * 60)
print("CONFUSION MATRIX GENERATED SUCCESSFULLY")
print("=" * 60)