from pathlib import Path

p = Path("src/independent_fl/prepare_independent_fl_data.py")
s = p.read_text(encoding="utf-8")

replacements = [
    (
        "SAMPLE_FRACTION = 0.20",
        "SAMPLE_FRACTION = 0.10",
    ),
    (
        '"sample_20_percent"',
        '"sample_10_percent"',
    ),
    (
        'description="Prepare independent 20% dataset subsets for three separate FL experiments."',
        'description="Prepare independent dataset subsets for separate FL experiments."',
    ),
    (
        'print("Sampling     : 20% per class (deterministic, approximate)")',
        'print("Sampling     : 10% per class (deterministic, approximate)")',
    ),
]

for old, new in replacements:
    count = s.count(old)
    if count != 1:
        raise SystemExit(
            f"PATCH STOPPED: Expected one occurrence, found {count}: {old}"
        )
    s = s.replace(old, new, 1)

old_args = '''    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate only data/independent_fl_v1. Existing V5 data is not touched.",
    )
    args = parser.parse_args()'''

new_args = '''    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild only the selected dataset output folder.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="all",
        help="Prepare one dataset or all datasets.",
    )
    args = parser.parse_args()
    selected_datasets = (
        list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    )'''

if s.count(old_args) != 1:
    raise SystemExit("PATCH STOPPED: Could not uniquely locate argument section.")

s = s.replace(old_args, new_args, 1)

old_output = '''    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"\\nOutput already exists: {OUTPUT_ROOT}\\n"
                "To intentionally rebuild this NEW output folder, rerun with --overwrite.\\n"
                "This script never deletes data/fl_3datasets_20percent or V5 model files."
            )
        print(f"\\nRemoving only the new output folder: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for dataset_name, config in DATASETS.items():
        summaries[dataset_name] = process_dataset(
            dataset_name,
            config,
            OUTPUT_ROOT,
        )'''

new_output = '''    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for dataset_name in selected_datasets:
        dataset_output = OUTPUT_ROOT / dataset_name

        if dataset_output.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {dataset_output}. "
                    "Use --overwrite to rebuild only this dataset."
                )
            print(f"Removing only selected dataset folder: {dataset_output}")
            shutil.rmtree(dataset_output)

        summaries[dataset_name] = process_dataset(
            dataset_name,
            DATASETS[dataset_name],
            OUTPUT_ROOT,
        )'''

if s.count(old_output) != 1:
    raise SystemExit("PATCH STOPPED: Could not uniquely locate output section.")

s = s.replace(old_output, new_output, 1)

old_summary = 'summary_path = OUTPUT_ROOT / "global_summary.json"'
new_summary = 'summary_path = OUTPUT_ROOT / ("global_summary_" + args.dataset + ".json")'

if s.count(old_summary) != 1:
    raise SystemExit("PATCH STOPPED: Could not uniquely locate summary path.")

s = s.replace(old_summary, new_summary, 1)

p.write_text(s, encoding="utf-8")
print("SUCCESS: CICIoT2023-only 10% preparation patch applied.")
