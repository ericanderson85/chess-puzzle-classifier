import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from src.util.logger import get_logger
from src.util.config import Config, ModelType
from src.util.plotting import (
    _setup_plot_style,
    _save_plot,
    COLOR_CYCLE,
    bar_chart,
)
from src.model.train import get_dataloaders, gradient_descent
from src.model.puzzle_dataset import get_datasets
from src.model.puzzle_cnn import PuzzleCNN
from src.model.puzzle_mlp import PuzzleMLP
from src.model.puzzle_logistic_regression import PuzzleLogisticRegression


def evaluate_and_plot_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    logger,
):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = {}
    tpr = {}
    aucs = {}
    for i, cls in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        aucs[i] = auc(fpr[i], tpr[i])
        logger.info(f"Class {cls}: AUC = {aucs[i]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cls in enumerate(class_names):
        ax.plot(
            fpr[i],
            tpr[i],
            label=f"{cls} (AUC={aucs[i]:.2f})",
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
            linewidth=2.5,
            alpha=0.9,
        )

    _setup_plot_style(
        ax,
        title="ROC Curve",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    _save_plot(fig, "roc.png", logger)
    return aucs


def main():
    config = Config()
    logger = get_logger(__name__, config.TRAIN_LOG_PATH)
    logger.info("Starting training + evaluation pipeline")
    logger.info(f"Using device: {config.DEVICE}")

    labeled_ds, unlabeled_ds = get_datasets(config, logger)
    (
        train_loader,
        unlabeled_loader,
        val_loader,
        test_loader,
    ) = get_dataloaders(config, labeled_ds, unlabeled_ds, logger)

    if config.MODEL_TYPE == ModelType.CNN:
        model = PuzzleCNN(config).to(config.DEVICE)
    elif config.MODEL_TYPE == ModelType.MLP:
        model = PuzzleMLP(config).to(config.DEVICE)
    elif config.MODEL_TYPE == ModelType.LOGISTIC_REGRESSION:
        model = PuzzleLogisticRegression(config).to(config.DEVICE)
    else:
        raise ValueError("Unrecognized MODEL_TYPE")

    optimizer = config.OPTIMIZATION_FUNCTION(
        model.parameters(), lr=config.LEARNING_RATE
    )
    scheduler = config.SCHEDULER(optimizer) if config.SCHEDULER else None
    loss_fn = config.LOSS_FUNCTION()

    gradient_descent(
        config,
        model,
        train_loader,
        unlabeled_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        loss_fn,
        logger,
    )

    if test_loader is None or len(test_loader.dataset) == 0:
        logger.warning("No test set available; exiting.")
        return

    model.eval()
    y_true_list, y_score_list, y_pred_list = [], [], []
    with torch.no_grad():
        for boards, labels in test_loader:
            boards = boards.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(boards)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true_list.append(labels.cpu().numpy())
            y_score_list.append(probs)
            y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_score = np.vstack(y_score_list)
    y_pred = np.concatenate(y_pred_list)

    evaluate_and_plot_roc(
        y_true, y_score, config.PUZZLE_CLASSES, logger
    )

    test_acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info(f"Test Accuracy : {test_acc:.4f}")
    logger.info(f"Macro Precision: {macro_p:.4f}")
    logger.info(f"Macro Recall   : {macro_r:.4f}")
    logger.info(f"Macro F1 Score : {macro_f1:.4f}")

    metrics = np.array([test_acc, macro_p, macro_r, macro_f1])
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    bar_chart(
        x=labels,
        y=metrics,
        logger=logger,
        title="Test Metrics",
        xlabel="Metric",
        ylabel="Value",
        filename="metrics.png",
    )

    errors = np.where(y_true != y_pred)[0]
    subset_idxs = getattr(test_loader.dataset, "indices", None)
    if subset_idxs is not None:
        errors = [subset_idxs[i] for i in errors]
    sample_errors = errors[:10]
    logger.info(f"Sample error indices: {sample_errors}")
    print("Sample error indices (first 10):", sample_errors)


if __name__ == "__main__":
    main()
