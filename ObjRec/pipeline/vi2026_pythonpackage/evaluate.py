"""
CS3IV Kinect Object Recognition - Person C: Test Set Evaluation

Loads a labelled test dataset, runs the trained classifier, and produces:
  - Per-class accuracy table (printed to terminal via Rich)
  - Overall accuracy
  - A confusion matrix saved as a PNG

Run from the pipeline directory:
    pdm run evaluate <path-to-test-data-folder>

The test data folder must use the same layout as Person A's output_frames:
    <test_dir>/<class_name>/depth_NNNN.png  (+ rgb_NNNN.png where available)

Models are loaded from the models/ directory next to this package by default.

Ben
"""

import sys
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix

from vi2026_pythonpackage.features import extract_features

console = Console()
app = typer.Typer()


def load_test_dataset(data_dir: Path, class_names: list[str]):
    """
    Load test crops and extract features, mapping labels to the training class order.

    Parameters
    ----------
    data_dir : Path
        Root folder with one subfolder per class (same layout as training output_frames).
    class_names : list[str]
        Ordered class list from classes.txt — must match the label indices used at
        training time.  Any folder in data_dir not in this list is skipped with a warning.

    Returns
    -------
    X : np.ndarray  (N, D)  feature matrix
    y : np.ndarray  (N,)    integer labels aligned to class_names
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    X, y = [], []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label_name = class_dir.name
        if label_name not in name_to_idx:
            console.print(f"[yellow]Warning: folder '{label_name}' not in training classes — skipped[/yellow]")
            continue

        class_idx = name_to_idx[label_name]
        depth_files = sorted(class_dir.glob("depth_*.png"))

        if not depth_files:
            console.print(f"[yellow]Warning: no depth frames found in {class_dir}[/yellow]")
            continue

        for depth_path in depth_files:
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            if depth_img is None:
                continue

            rgb_path = class_dir / depth_path.name.replace("depth_", "rgb_")
            rgb_img = cv2.imread(str(rgb_path)) if rgb_path.exists() else None

            feat = extract_features(depth_img, rgb_img)
            X.append(feat)
            y.append(class_idx)

    if not X:
        console.print("[red]No test samples loaded — check the test data path.[/red]")
        raise SystemExit(1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path):
    """
    Save a colour-coded confusion matrix as a PNG using matplotlib.

    Rows = true labels, columns = predicted labels.
    Diagonal values show correct predictions; off-diagonal are misclassifications.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix — Test Set",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with the count; use white text on dark cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@app.command()
def main(
    test_dir: Path = typer.Argument(..., help="Path to labelled test data folder"),
    models_dir: Path = typer.Option(None, help="Directory containing model.joblib / scaler.joblib / classes.txt"),
    output_dir: Path = typer.Option(None, help="Where to save the confusion matrix PNG (defaults to test_dir)"),
):
    """Evaluate the trained classifier on a labelled test dataset and produce a confusion matrix."""
    if not test_dir.is_dir():
        console.print(f"[red]Error: {test_dir} is not a directory[/red]")
        raise SystemExit(1)

    # Resolve paths
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"
    if output_dir is None:
        output_dir = test_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model artefacts
    model_path   = models_dir / "model.joblib"
    scaler_path  = models_dir / "scaler.joblib"
    classes_path = models_dir / "classes.txt"

    for p in (model_path, scaler_path, classes_path):
        if not p.exists():
            console.print(f"[red]Error: required file not found: {p}[/red]")
            raise SystemExit(1)

    clf          = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    class_names  = classes_path.read_text().strip().splitlines()

    console.print(f"\n[bold]Loaded model:[/bold]  {model_path}")
    console.print(f"[bold]Classes ({len(class_names)}):[/bold]  {', '.join(class_names)}\n")

    # Load test features
    console.print(f"[bold]Loading test data from:[/bold] {test_dir}")
    X_test, y_test = load_test_dataset(test_dir, class_names)
    console.print(f"  {len(X_test)} samples loaded, feature dim = {X_test.shape[1]}\n")

    # Scale and predict
    X_test_s = scaler.transform(X_test)
    y_pred   = clf.predict(X_test_s)

    # Overall accuracy
    overall_acc = float(np.sum(y_pred == y_test)) / len(y_test)

    # Per-class accuracy using the confusion matrix diagonal
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    # Print results table
    table = Table(title=f"Per-class Accuracy   (overall: {overall_acc:.1%})")
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Total", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")

    for idx, name in enumerate(class_names):
        total   = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        acc_str = f"{correct / total:.1%}" if total > 0 else "—"
        table.add_row(name, str(total), str(correct), acc_str)

    console.print(table)
    console.print(f"\n[bold green]Overall accuracy: {overall_acc:.1%}[/bold green]  ({int(np.sum(y_pred == y_test))}/{len(y_test)} correct)\n")

    # Save confusion matrix PNG
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    console.print(f"[bold]Confusion matrix saved →[/bold] {cm_path}\n")


if __name__ == "__main__":
    sys.exit(app())
