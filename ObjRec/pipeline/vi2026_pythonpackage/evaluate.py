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
from sklearn.metrics import confusion_matrix, classification_report

from vi2026_pythonpackage.features import extract_features

console = Console()
app = typer.Typer()


def load_test_dataset(data_dir: Path, class_names: list[str]):
    """
    Load test crops and extract features, mapping labels to the training class order.
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    X, y = [],[]
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
    output_dir: Path = typer.Option(None, help="Where to save the confusion matrix PNG"),
):
    if not test_dir.is_dir():
        console.print(f"[red]Error: {test_dir} is not a directory[/red]")
        raise SystemExit(1)

    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"
    if output_dir is None:
        output_dir = test_dir

    output_dir.mkdir(parents=True, exist_ok=True)

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
    
    # --- Register the unknown class for Open-Set Recognition ---
    if "unknown" not in class_names:
        class_names.append("unknown")

    console.print(f"\n[bold]Loaded model:[/bold]  {model_path}")
    console.print(f"[bold]Classes ({len(class_names)}):[/bold]  {', '.join(class_names)}\n")

    console.print(f"[bold]Loading test data from:[/bold] {test_dir}")
    X_test, y_test = load_test_dataset(test_dir, class_names)
    console.print(f"  {len(X_test)} samples loaded, feature dim = {X_test.shape[1]}\n")

    X_test_s = scaler.transform(X_test)
    
    # --- CONFIDENCE THRESHOLDING (Open-Set Recognition) ---
    try:
        probs = clf.predict_proba(X_test_s)
        max_probs = np.max(probs, axis=1)
        best_preds = np.argmax(probs, axis=1)
        
        UNKNOWN_IDX = class_names.index("unknown")
        CONFIDENCE_THRESHOLD = 0.10 
        
        # Re-route low confidence predictions to 'unknown'
        y_pred = np.where(max_probs < CONFIDENCE_THRESHOLD, UNKNOWN_IDX, best_preds)
        console.print("[green]Confidence thresholding applied for 'unknown' objects.[/green]")
        
    except AttributeError:
        # Fallback if the SVM was not trained with probability=True
        console.print("[yellow]Warning: Model does not support probabilities. Please retrain with `probability=True` to enable 'unknown' object rejection. Falling back to standard prediction.[/yellow]")
        y_pred = clf.predict(X_test_s)

    # --- RESULTS ---
    overall_acc = float(np.sum(y_pred == y_test)) / len(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

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

    # --- ADVANCED METRICS (Worksheet 3 - Extension A) ---
    console.print("[bold]Classification Report (Precision, Recall, F1-Score):[/bold]")
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    console.print(report)

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    console.print(f"[bold]Confusion matrix saved →[/bold] {cm_path}\n")

if __name__ == "__main__":
    sys.exit(app())