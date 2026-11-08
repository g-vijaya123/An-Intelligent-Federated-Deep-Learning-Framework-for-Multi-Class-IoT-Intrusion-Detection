from pathlib import Path
import re
import shutil

target = Path("src/independent_fl/train_independent_fl.py")

if not target.exists():
    raise SystemExit(f"ERROR: File not found: {target}")

source = target.read_text(encoding="utf-8-sig")

if "def evaluate_dataset_once(" not in source:
    marker = "\ndef metrics_from_confusion_matrix("
    if marker not in source:
        raise SystemExit("ERROR: Could not find insertion point.")

    helper = '''

def evaluate_dataset_once(
    model: Model,
    dataset: tf.data.Dataset,
    num_classes: int,
):
    """Calculate loss, accuracy and confusion matrix in one dataset pass."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_records = 0

    for batch_number, (xb, yb) in enumerate(dataset, start=1):
        probabilities = model(xb, training=False).numpy()
        actual = yb.numpy().astype(np.int64)
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)

        batch_losses = tf.keras.losses.sparse_categorical_crossentropy(
            actual, probabilities
        ).numpy()

        total_loss += float(np.sum(batch_losses))
        total_records += len(actual)

        encoded = actual * num_classes + predictions
        matrix += np.bincount(
            encoded,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

        if batch_number % 500 == 0:
            print(
                f"    Evaluation progress: {total_records:,} records processed",
                flush=True,
            )

    if total_records == 0:
        raise ValueError("Evaluation dataset contains no records.")

    accuracy = float(np.trace(matrix) / total_records)
    average_loss = total_loss / total_records

    return average_loss, accuracy, matrix

'''
    source = source.replace(marker, helper + marker, 1)

pattern = re.compile(
    r'(?m)^            validation_loss_metrics = global_model\.evaluate\(\n'
    r'                validation_ds, verbose=0, return_dict=True\n'
    r'            \)\n'
    r'            validation_loss = float\(validation_loss_metrics\["loss"\]\)\n'
    r'            validation_accuracy = float\(validation_loss_metrics\["accuracy"\]\)\n'
    r'            validation_confusion = confusion_matrix_from_dataset\(\n'
    r'                global_model, validation_ds, num_classes\n'
    r'            \)\n'
    r'            validation_class_metrics, _ = metrics_from_confusion_matrix\(\n'
    r'                validation_confusion, classes\n'
    r'            \)\n'
    r'            validation_macro_f1 = float\(validation_class_metrics\["macro_f1"\]\)\n'
    r'            average_local_loss = float\(np\.mean\(local_losses\)\)'
)

replacement = '''            print("  Starting global validation...", flush=True)
            validation_loss, validation_accuracy, validation_confusion = (
                evaluate_dataset_once(
                    global_model,
                    validation_ds,
                    num_classes,
                )
            )
            validation_class_metrics, _ = metrics_from_confusion_matrix(
                validation_confusion, classes
            )
            validation_macro_f1 = float(validation_class_metrics["macro_f1"])
            average_local_loss = float(np.mean(local_losses))'''

source, count = pattern.subn(replacement, source, count=1)

if count != 1:
    raise SystemExit(
        "ERROR: Validation block did not match. Original file was not modified."
    )

backup = target.with_suffix(target.suffix + ".before_one_pass_validation.bak")

if not backup.exists():
    shutil.copy2(target, backup)

target.write_text(source, encoding="utf-8")

print("SUCCESS: Validation updated to one-pass evaluation.")
print(f"Backup created: {backup}")
print(f"Updated file: {target}")
