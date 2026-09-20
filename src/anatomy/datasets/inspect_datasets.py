import os

# Root folder containing all anatomy datasets
DATA_ROOT = os.path.join("data", "anatomy")


def print_tree(path, prefix="", max_depth=4, current_depth=0):
    """Print folder/file structure."""

    if current_depth > max_depth:
        return

    if not os.path.exists(path):
        print(f"[NOT FOUND] {path}")
        return

    items = sorted(os.listdir(path))

    for i, item in enumerate(items):
        full_path = os.path.join(path, item)

        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")

        if os.path.isdir(full_path):
            print(f"{prefix}{connector}{item}/")
            print_tree(
                full_path,
                next_prefix,
                max_depth,
                current_depth + 1
            )
        else:
            print(f"{prefix}{connector}{item}")


def dataset_summary(dataset_path):
    """Print basic information about one dataset."""

    if not os.path.exists(dataset_path):
        print("NOT FOUND")
        return

    files = []
    folders = []

    for root, dirs, filenames in os.walk(dataset_path):
        folders.extend(dirs)
        files.extend(
            os.path.join(root, f)
            for f in filenames
        )

    print(f"Folders : {len(folders)}")
    print(f"Files   : {len(files)}")

    # File extensions
    extensions = {}

    for file in files:
        ext = os.path.splitext(file)[1].lower()

        if ext:
            extensions[ext] = extensions.get(ext, 0) + 1

    print("Extensions:")

    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count}")

    # First few files
    print("\nSample files:")

    for file in files[:10]:
        print(f"  {os.path.relpath(file, dataset_path)}")


def main():

    print("=" * 70)
    print("TRINAY - ANATOMY DATASET INSPECTOR")
    print("=" * 70)

    if not os.path.exists(DATA_ROOT):
        print(f"\nDataset folder not found:")
        print(os.path.abspath(DATA_ROOT))
        return

    datasets = sorted(
        name
        for name in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, name))
    )

    print("\nDatasets found:")
    for dataset in datasets:
        print(f"  ✓ {dataset}")

    # Print complete structure
    print("\n" + "=" * 70)
    print("FOLDER STRUCTURE")
    print("=" * 70)

    print(f"\n{DATA_ROOT}/")
    print_tree(DATA_ROOT, max_depth=4)

    # Print summaries
    print("\n" + "=" * 70)
    print("DATASET SUMMARIES")
    print("=" * 70)

    for dataset in datasets:

        dataset_path = os.path.join(DATA_ROOT, dataset)

        print("\n" + "-" * 70)
        print(f"DATASET: {dataset}")
        print("-" * 70)

        dataset_summary(dataset_path)


if __name__ == "__main__":
    main()