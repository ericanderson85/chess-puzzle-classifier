import os
import numbers
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Union

from src.util.config import Config, LearningType, BoardRepresentation, PuzzleRepresentation
from src.util.logger import get_logger
from src.model.puzzle_cnn import PuzzleCNN
from src.model.puzzle_dataset import get_datasets
from src.model.train import get_dataloaders, gradient_descent
from src.util.plotting import bar_chart, line_plot


def tune_parameters(
    base_config: Config,
    parameter_ranges: Dict[str, List[Any]],
    logger
) -> Dict[str, List[float]]:

    results = {}

    for param_name, param_values in parameter_ranges.items():
        logger.info(f"\n--- Tuning parameter: {param_name} ---")

        param_results = {
            "param_values": [],
            "param_value_names": [],
            "train_losses": [],
            "validation_losses": []
        }

        for value in param_values:
            config = Config()

            for key, val in vars(base_config).items():
                if hasattr(config, key):
                    setattr(config, key, val)

            setattr(config, param_name, value)

            config.update_model_parameters()

            if hasattr(value, 'name'):
                value_name = value.name
            elif callable(value):
                value_name = value.__name__
            else:
                value_name = str(value)

            logger.info(f"Testing {param_name} = {value_name}")

            try:
                labeled_dataset, unlabeled_dataset = get_datasets(config, logger)
                labeled_train_loader, unlabeled_train_loader, validation_loader, test_loader = get_dataloaders(
                    config, labeled_dataset, unlabeled_dataset, logger
                )

                model = PuzzleCNN(config).to(config.DEVICE)
                optimizer = config.OPTIMIZATION_FUNCTION(
                    model.parameters(),
                    lr=config.LEARNING_RATE
                )
                loss_fn = config.LOSS_FUNCTION()

                best_train_loss, best_validation_loss = gradient_descent(
                    config,
                    model,
                    labeled_train_loader,
                    unlabeled_train_loader,
                    validation_loader,
                    test_loader,
                    optimizer,
                    loss_fn,
                    logger
                )

                param_results["param_values"].append(value)
                param_results["param_value_names"].append(value_name)
                param_results["validation_losses"].append(best_validation_loss)
                param_results["train_losses"].append(best_train_loss)

                logger.info(
                    f"Completed {param_name} = {value_name}, validation loss: {best_validation_loss:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {param_name} = {value_name}: {e}")

        results[param_name] = param_results

        df = pd.DataFrame({
            "param_value": param_results["param_value_names"],
            "validation_loss": param_results["validation_losses"],
            "train_loss": param_results["train_losses"]
        })
        df.to_csv(os.path.join(config.LOGS_DIRECTORY,
                  f"{param_name}_tuning_results.csv"), index=False)

    return results


def plot_tuning_results(results: dict[str, Dict], logger):
    for param_name, param_results in results.items():
        formatted_name = param_name.replace("_", " ").title()
        title = f"Loss Curves for {formatted_name}"
        xlabel = formatted_name
        ylabel = "Loss"
        filename = f"{param_name}.png"

        param_values = param_results["param_values"]
        param_value_names = param_results["param_value_names"]
        train_losses = param_results["train_losses"]
        validation_losses = param_results["validation_losses"]

        def is_numeric(x):
            return isinstance(x, numbers.Real) or isinstance(x, np.generic)

        if all(is_numeric(v) for v in param_values):
            line_plot(
                x=param_values,
                y=[train_losses, validation_losses],
                labels=["Training Loss", "Validation Loss"],
                logger=logger,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                filename=filename,
            )
        else:
            bar_chart(
                x=param_value_names,
                y=[np.array(train_losses), np.array(validation_losses)],
                labels=["Training Loss", "Validation Loss"],
                logger=logger,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                filename=filename,
            )


def main():
    base_config = Config()

    logger = get_logger(__name__, base_config.TUNE_LOG_PATH)

    parameter_ranges = {
        "PUZZLE_REPRESENTATION": [PuzzleRepresentation.FIRST, PuzzleRepresentation.FIRST_AND_LAST, PuzzleRepresentation.FEW],
        # "BOARD_REPRESENTATION": [BoardRepresentation.PIECE_INDEX, BoardRepresentation.PIECE_TYPES, BoardRepresentation.PIECE_TYPES_AND_COLORS],
        # "CONVOLUTION_LAYERS": [
        #     [
        #         (base_config.FLATTENED_CHANNEL_DIMENSION, 32, 3, 1, 1),
        #     ],
        #     [
        #         (base_config.FLATTENED_CHANNEL_DIMENSION, 32, 3, 1, 1),
        #         (32, 64, 3, 1, 1),
        #     ],
        #     [
        #         (base_config.FLATTENED_CHANNEL_DIMENSION, 32, 3, 1, 1),
        #         (32, 64, 3, 1, 1),
        #         (64, 128, 3, 2, 1),
        #     ]
        # ],
        # "NUM_SAMPLES": [10, 25, 50, 100, 200, 400, 800, 1600, 2400],
        # "DROPOUT": [0.0, 0.2, 0.4, 0.6],
        # "ACTIVATION_FUNCTION": [nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid],
        # "NUM_UNLABELED_SAMPLES": [500, 1000, 2000, 3000, 5000]
    }

    results = tune_parameters(base_config, parameter_ranges, logger)

    plot_tuning_results(results, logger)

    logger.info("\n--- Best Parameter Values ---")
    for param_name, param_results in results.items():
        if param_results["validation_losses"]:
            best_idx = np.argmin(param_results["validation_losses"])
            best_value = param_results["param_value_names"][best_idx]
            best_loss = param_results["validation_losses"][best_idx]
            logger.info(f"{param_name}: {best_value} (validation loss: {best_loss:.4f})")

    logger.info("Parameter tuning completed!")


if __name__ == "__main__":
    main()
