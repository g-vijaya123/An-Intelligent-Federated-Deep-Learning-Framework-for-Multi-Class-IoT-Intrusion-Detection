from pathlib import Path

path = Path("src/independent_fl/train_independent_fl.py")
source = path.read_text(encoding="utf-8")

old1 = '''        client_caches = []
        for client_id in range(1, num_clients + 1):'''

new1 = '''        client_caches = []
        client_datasets = []
        for client_id in range(1, num_clients + 1):'''

old2 = '''                client_train_ds = make_balanced_training_dataset(
                    cache=client_caches[client_id - 1],
                    num_features=num_features,
                    batch_size=batch_size,
                    seed=seed + round_number * 1000 + client_id,
                )
                history = local_model.fit('''

new2 = '''                client_train_ds = make_balanced_training_dataset(
                    cache=client_caches[client_id - 1],
                    num_features=num_features,
                    batch_size=batch_size,
                    seed=seed + round_number * 1000 + client_id,
                )
                client_datasets.append(client_train_ds)
                history = local_model.fit('''

old3 = '''        # Release disk-backed files before TemporaryDirectory cleanup on Windows.
        for cache in client_caches:
            cache["X"].flush()
            cache["y"].flush()
            del cache["X"]
            del cache["y"]
        del client_caches
        gc.collect()'''

new3 = '''        # Release TensorFlow generators and memory-mapped files on Windows.
        client_datasets.clear()
        gc.collect()

        for cache in client_caches:
            for key in ("X", "y"):
                array = cache.get(key)
                if array is not None:
                    array.flush()
                    mmap_handle = getattr(array, "_mmap", None)
                    if mmap_handle is not None:
                        mmap_handle.close()
                    cache[key] = None

        client_caches.clear()
        gc.collect()'''

for name, old, new in [
    ("client cache declaration", old1, new1),
    ("dataset reference tracking", old2, new2),
    ("Windows memory-map cleanup", old3, new3),
]:
    count = source.count(old)
    if count != 1:
        raise SystemExit(
            f"PATCH STOPPED: Expected one occurrence of {name}, found {count}. "
            "Original file was not modified."
        )
    source = source.replace(old, new, 1)

path.write_text(source, encoding="utf-8")
print("PATCH SUCCESS: All three changes applied.")
