"""
CS3IV Kinect Object Recognition - Person B: Classifier Training

Loads labelled crops from Person A's output_frames folder, extracts HOG
and ORB features, and trains an SVM classifier. Two kernels (linear, RBF)
are compared on a held-out validation split and the better one is retrained
on the full dataset before saving.

Run from the pipeline directory:
    pdm run train-classifier output_frames/

Saves model.joblib, scaler.joblib, and classes.txt to the data directory.
Person C loads these for test set evaluation.

- Karim
"""

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import typer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from vi2026_pythonpackage.features import extract_features

app = typer.Typer()


def load_dataset(data_dir: Path):
    """
    Load paired depth and RGB crops from Person A's output_frames layout.

    data_dir: root folder with one subfolder per class
    returns: features (N, D), labels (N,), class_names, class_dirs
    """
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"No class subfolders found in {data_dir}")
        raise SystemExit(1)

    class_names = [d.name for d in class_dirs]
    all_features, all_labels = [], []

    for class_idx, class_dir in enumerate(class_dirs):
        depth_files = sorted(class_dir.glob("depth_*.png"))
        for depth_path in depth_files:
            rgb_path = class_dir / depth_path.name.replace("depth_", "rgb_")

            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            if depth_img is None:
                continue

            rgb_img = cv2.imread(str(rgb_path)) if rgb_path.exists() else None

            feat = extract_features(depth_img, rgb_img)
            all_features.append(feat)
            all_labels.append(class_idx)

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
        class_names,
        class_dirs,
    )


@app.command()
def main(
    data_dir: Path = typer.Argument(..., help="Path to labelled training data folder"),
    test_size: float = typer.Option(0.2, help="Fraction of data held out for validation"),
):
    """Train and save the best SVM classifier on the labelled crop dataset."""
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        raise SystemExit(1)

    print("Loading dataset")
    X, y, class_names, class_dirs = load_dataset(data_dir)

    # stratify keeps class proportions balanced in both splits
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print("\nDataset summary:")
    print(f"  Total samples     : {len(X)}")
    print(f"  Feature vector    : {X.shape[1]}")
    print(f"  Training samples  : {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print("\nPer-class counts:")
    for class_idx, class_dir in enumerate(class_dirs):
        n_train = int(np.sum(y_train == class_idx))
        n_val   = int(np.sum(y_val   == class_idx))
        print(f"  {class_dir.name:20s}: {n_train} train, {n_val} val")

    # fit scaler on training split only to avoid leaking val statistics
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    classifiers = {
        "SVM linear": SVC(kernel="linear", C=1.0),
        "SVM RBF"   : SVC(kernel="rbf",    C=10.0, gamma="scale"),
    }

    print("\nValidation results:")
    best_name, best_acc, best_clf = "", -1.0, SVC()

    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        predictions = clf.predict(X_val_s)

        accuracy = np.sum(predictions == y_val) / len(y_val)
        print(f"\n  {name}  overall accuracy: {accuracy:.1%}")

        # per-class accuracy
        for class_idx, class_name in enumerate(class_names):
            mask = y_val == class_idx
            if not np.any(mask):
                continue
            class_acc = np.sum(predictions[mask] == y_val[mask]) / np.sum(mask)
            print(f"    {class_name:20s}: {class_acc:.1%}")

        if accuracy > best_acc:
            best_acc, best_name, best_clf = accuracy, name, clf

    print(f"\nBest: {best_name} ({best_acc:.1%} on validation set)")

    # retrain on full dataset before saving so all labelled data is used
    X_all_s = scaler.fit_transform(X)
    best_clf.fit(X_all_s, y)

    save_dir = Path(__file__).parent.parent / "models"
    save_dir.mkdir(exist_ok=True)

    joblib.dump(best_clf, save_dir / "model.joblib")
    joblib.dump(scaler,   save_dir / "scaler.joblib")
    (save_dir / "classes.txt").write_text("\n".join(class_names))

    print(f"\nSaved model   -> {save_dir / 'model.joblib'}")
    print(f"Saved scaler  -> {save_dir / 'scaler.joblib'}")
    print(f"Saved classes -> {save_dir / 'classes.txt'}")


if __name__ == "__main__":
    sys.exit(app())
