import os
import random
import numbers
import pandas as pd
import numpy as np
import torch
from torch import nn, optim

from src.model.dice_loss import DiceLoss
from src.model.puzzle_logistic_regression import PuzzleLogisticRegression
from src.model.puzzle_mlp import PuzzleMLP
from src.model.puzzle_cnn import PuzzleCNN
from src.model.puzzle_dataset import get_datasets
from src.model.train import get_dataloaders, gradient_descent
from src.util.config import BoardRepresentation, Config, LearningType, ModelType, PuzzleRepresentation
from src.util.logger import get_logger
from src.util.plotting import bar_chart, line_plot


def random_search(
    base_config: Config,
    param_spaces: dict[str, list],
    n_iter: int,
    logger
) -> Config:
    best_val_loss = float("inf")
    best_config = None
    best_model_state_dict = None
    records = []

    for i in range(1, n_iter + 1):

        sampled = {
            name: random.choice(choices)
            for name, choices in param_spaces.items()
        }

        cfg = Config()
        for k, v in vars(base_config).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        for name, val in sampled.items():
            setattr(cfg, name, val)

        logger.info(f"[Random Search {i}/{n_iter}] sampling {sampled}")

        try:

            ld, ud = get_datasets(cfg, logger)
            ltr, utr, val_dl, tst = get_dataloaders(
                cfg, ld, ud, logger
            )

            if cfg.MODEL_TYPE == ModelType.CNN:
                model = PuzzleCNN(cfg).to(cfg.DEVICE)
            elif cfg.MODEL_TYPE == ModelType.MLP:
                model = PuzzleMLP(cfg).to(cfg.DEVICE)
            elif cfg.MODEL_TYPE == ModelType.LOGISTIC_REGRESSION:
                model = PuzzleLogisticRegression(cfg).to(cfg.DEVICE)
            else:
                raise ValueError("Unknown MODEL_TYPE")

            optim_fn = cfg.OPTIMIZATION_FUNCTION
            optimizer = optim_fn(model.parameters(), lr=cfg.LEARNING_RATE)
            scheduler = cfg.SCHEDULER(optimizer) if cfg.SCHEDULER else None
            loss_fn = cfg.LOSS_FUNCTION()

            tr_loss, val_loss = gradient_descent(
                cfg, model, ltr, utr, val_dl, tst,
                optimizer, scheduler, loss_fn, logger
            )

            records.append({**sampled,
                            "train_loss": tr_loss,
                            "validation_loss": val_loss})

            logger.info(
                f"--> val_loss={val_loss:.4f} (train_loss={tr_loss:.4f})"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = cfg
                best_model_state_dict = model.state_dict()
                logger.info(f"New best validation loss: {best_val_loss:.4f}")

        except Exception as e:
            logger.error(f"Random search iteration {i} failed: {e}", exc_info=True)

    df = pd.DataFrame(records)
    results_csv_path = os.path.join(
        base_config.LOGS_DIRECTORY, "random_search_results.csv"
    )
    os.makedirs(base_config.LOGS_DIRECTORY, exist_ok=True)
    df.to_csv(results_csv_path, index=False)
    logger.info(f"Random search results saved to {results_csv_path}")

    if best_model_state_dict and best_config:
        os.makedirs(base_config.MODELS_DIRECTORY, exist_ok=True)
        model_save_path = os.path.join(
            base_config.MODELS_DIRECTORY, "model.pt"
        )
        torch.save(best_model_state_dict, model_save_path)
        logger.info(
            f"Saved best model from random search to {model_save_path}"
        )
    elif not best_config:
        logger.warning("No best configuration found from random search. No model saved.")
    else:
        logger.warning("Best configuration found, but no model state was captured. No model saved.")

    logger.info(f"Best random-search val_loss={best_val_loss:.4f}")
    return best_config


def tune_parameters(
    base_config: Config,
    parameter_ranges: dict[str, list],
    logger
) -> dict[str, dict]:
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
            cfg = Config()

            for k, v_base in vars(base_config).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v_base)

            setattr(cfg, param_name, value)

            if hasattr(value, "name"):
                vname = value.name
            elif callable(value):
                vname = value.__name__
            else:
                vname = str(value)

            logger.info(f"Testing {param_name} = {vname}")

            try:
                ld, ud = get_datasets(cfg, logger)
                ltr, utr, val_dl, tst = get_dataloaders(
                    cfg, ld, ud, logger
                )
                if cfg.MODEL_TYPE == ModelType.CNN:
                    model = PuzzleCNN(cfg).to(cfg.DEVICE)
                elif cfg.MODEL_TYPE == ModelType.MLP:
                    model = PuzzleMLP(cfg).to(cfg.DEVICE)
                elif cfg.MODEL_TYPE == ModelType.LOGISTIC_REGRESSION:
                    model = PuzzleLogisticRegression(cfg).to(cfg.DEVICE)
                else:
                    raise ValueError("Unrecognized model type")

                optimizer = cfg.OPTIMIZATION_FUNCTION(
                    model.parameters(), lr=cfg.LEARNING_RATE
                )
                scheduler = cfg.SCHEDULER(optimizer) if cfg.SCHEDULER else None
                loss_fn = cfg.LOSS_FUNCTION()

                tr_loss, val_loss = gradient_descent(
                    cfg, model, ltr, utr, val_dl, tst,
                    optimizer, scheduler, loss_fn, logger
                )

                param_results["param_values"].append(value)
                param_results["param_value_names"].append(vname)
                param_results["train_losses"].append(tr_loss)
                param_results["validation_losses"].append(val_loss)

                logger.info(
                    f"Completed {param_name} = {vname}; val_loss={val_loss:.4f}"
                )

            except Exception as e:
                logger.error(f"Error for {param_name}={vname}: {e}", exc_info=True)

        if param_results["param_values"]:
            df = pd.DataFrame({
                "param_value": param_results["param_value_names"],
                "train_loss": param_results["train_losses"],
                "validation_loss": param_results["validation_losses"],
            })
            os.makedirs(base_config.LOGS_DIRECTORY, exist_ok=True)
            csv_path = os.path.join(
                base_config.LOGS_DIRECTORY,
                f"{param_name}_tuning_results.csv"
            )
            df.to_csv(csv_path, index=False)
            logger.info(f"Tuning results for {param_name} saved to {csv_path}")
        results[param_name] = param_results

    return results


def plot_tuning_results(config: Config, results: dict[str, dict], logger):
    for param_name, param_results in results.items():
        if not param_results["param_values"]:
            logger.info(f"No results to plot for {param_name}.")
            continue

        formatted_name = param_name.replace("_", " ").title()
        title = f"Loss Curves for {formatted_name}"
        xlabel = formatted_name
        ylabel = "Loss"
        filename = os.path.join(f"{param_name}_tuning.png")

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

    os.makedirs(os.path.dirname(base_config.TUNE_LOG_PATH), exist_ok=True)
    logger = get_logger(__name__, base_config.TUNE_LOG_PATH)

    random_search_spaces = {
        "BATCH_SIZE":      [8, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64, 128],
        "LEARNING_RATE":   [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01],
        # "DROPOUT":         [0.2, 0.35, 0.5, 0.55, 0.6, 0.65, 0.7],

        # "CONVOLUTION_LAYERS": [
        #     [(32, 3, 1, 1)],
        #     [(32, 3, 1, 1), (64, 3, 1, 1)],
        #     [(8, 7, 1, 3), (16, 5, 1, 2)],
        # ],

        # "FULLY_CONNECTED_LAYERS": [
        #     [128],
        #     [128, 64],
        #     [64, 32],
        #     [32, 16],
        #     [64, 64],
        #     [256, 64],
        # ],
    }

    individual_ranges = {
        "NUM_SAMPLES": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 1250, 1500, 1750, 2000, 2386],
        "BOARD_REPRESENTATION": [BoardRepresentation.PIECE_INDEX, BoardRepresentation.PIECE_TYPES, BoardRepresentation.PIECE_TYPES_AND_COLORS],
        "PUZZLE_REPRESENTATION": [PuzzleRepresentation.FIRST, PuzzleRepresentation.FIRST_AND_LAST, PuzzleRepresentation.ALL],
        "LEARNING_RATE":   [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01],
        # "ACTIVATION_FUNCTION": [nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid],
        # "DROPOUT": [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    N_RANDOM_TRIALS = 1

    logger.info(f"Starting random search with {N_RANDOM_TRIALS} trials...")
    best_config_from_random_search = random_search(
        base_config,
        random_search_spaces,
        N_RANDOM_TRIALS,
        logger
    )

    if best_config_from_random_search is None:
        logger.error("Random search did not yield a best configuration. Aborting tuning.")
        return

    logger.info("Random search completed. Starting individual parameter tuning with best config...")

    results = tune_parameters(
        base_config,
        individual_ranges,
        logger
    )
    logger.info("Individual parameter tuning completed. Plotting results...")
    plot_tuning_results(base_config, results, logger)

    logger.info("\n--- Best Parameter Values from Individual Tuning ---")
    final_best_config_params = {}
    if best_config_from_random_search:
        logger.info("Parameters from best random search config (base for individual tuning):")
        for param_name in random_search_spaces.keys():
            if hasattr(best_config_from_random_search, param_name):
                val = getattr(best_config_from_random_search, param_name)
                if hasattr(val, "name"):
                    vname = val.name
                elif callable(val):
                    vname = val.__name__
                else:
                    vname = str(val)
                logger.info(f"  {param_name}: {vname}")
                final_best_config_params[param_name] = val

    for pname, pres in results.items():
        if pres["validation_losses"]:
            valid_losses_np = np.array(pres["validation_losses"])

            if np.all(np.isnan(valid_losses_np)) or not np.any(np.isfinite(valid_losses_np)):
                logger.warning(f"No valid validation losses for {pname}. Skipping.")
                continue

            min_loss_idx = np.nanargmin(valid_losses_np)
            vname = pres["param_value_names"][min_loss_idx]
            vloss = pres["validation_losses"][min_loss_idx]
            best_value_for_param = pres["param_values"][min_loss_idx]
            logger.info(f"Best for {pname}: {vname} (val_loss={vloss:.4f})")
            final_best_config_params[pname] = best_value_for_param
        else:
            logger.info(f"No results for {pname} in individual tuning.")

    logger.info("\n--- Final Recommended Configuration ---")

    final_reco_config = Config()

    for k, v in vars(base_config).items():
        if hasattr(final_reco_config, k):
            setattr(final_reco_config, k, v)

    if best_config_from_random_search:
        for k, v in vars(best_config_from_random_search).items():
            if hasattr(final_reco_config, k):
                setattr(final_reco_config, k, v)

    for k, v in final_best_config_params.items():
        if hasattr(final_reco_config, k):
            setattr(final_reco_config, k, v)

    for param_name in sorted(final_best_config_params.keys()):
        val = getattr(final_reco_config, param_name)
        if hasattr(val, "name"):
            vname = val.name
        elif callable(val):
            vname = val.__name__
        else:
            vname = str(val)
        logger.info(f"{param_name}: {vname}")

    logger.info("Tuning completed")


if __name__ == "__main__":
    main()
