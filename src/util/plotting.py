import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
from logging import Logger


BLACK = '#1F1F1F'
GRAY = '#676F79'
WHITE = '#D4D4D4'

LIGHT_BLUE = '#9ADAFB'
DARK_BLUE = '#569CD6'
ORANGE = '#CE9178'
PURPLE = '#C586C0'
GREEN = '#4DC9B0'
YELLOW = '#D2D2A2'


PLOT_BACKGROUND_COLOR = BLACK
TEXT_COLOR = WHITE
AXIS_COLOR = WHITE
COLOR_CYCLE = [LIGHT_BLUE, ORANGE, PURPLE, GREEN, YELLOW]
CUSTOM_FONT_PATH: str | None = '/Users/eric/Library/Fonts/Inter.ttc'
DEFAULT_FONT_SIZE = 12
TITLE_FONT_SIZE_OFFSET = 2
DEFAULT_LEGEND_LOCATION = "best"
DEFAULT_FIGURE_SIZE = (10, 6)


SCATTER_MARKER_SIZE = 80
SCATTER_ALPHA = 0.8
STEP_LINE_WIDTH = 2.5
STEP_WHERE = 'mid'
BAR_WIDTH = 0.8
BAR_ALPHA = 0.85
FILL_ALPHA = 0.4
BOX_WIDTH = 0.6
BOX_NOTCH = False
BOX_FLIER_MARKER = 'o'
BOX_FLIER_MARKER_FACE_COLOR = WHITE
BOX_FLIER_MARKER_EDGE_COLOR = GRAY
BOX_FLIER_ALPHA = 0.45
HIST_BINS: int | list[float] | str = 'auto'
HIST_ALPHA = 0.75
HIST_DENSITY = False
HIST_EDGE_COLOR = AXIS_COLOR
HIST_LINEWIDTH = 0.8
ADJUST_HIST_ALPHA_OVERLAP = True

MULTI_HIST_TYPE = 'stepfilled'


PLOTS_DIRECTORY = 'plots/'
DEFAULT_DPI = 600
DEFAULT_SCATTER_FILENAME = "scatter_plot.png"
DEFAULT_STEP_FILENAME = "step_plot.png"
DEFAULT_BAR_FILENAME = "bar_chart.png"
DEFAULT_FILL_FILENAME = "fill_between_plot.png"
DEFAULT_BOX_FILENAME = "box_plot.png"
DEFAULT_HIST_FILENAME = "histogram.png"


DEFAULT_SCATTER_TITLE = "Scatter Plot"
DEFAULT_STEP_TITLE = "Step Plot"
DEFAULT_BAR_TITLE = "Bar Chart"
DEFAULT_FILL_TITLE = "Fill Between Plot"
DEFAULT_BOX_TITLE = "Box Plot"
DEFAULT_HIST_TITLE = "Histogram"
DEFAULT_X_LABEL = "X"
DEFAULT_Y_LABEL = "Y"
DEFAULT_BAR_X_LABEL = "Categories"
DEFAULT_BAR_Y_LABEL = "Values"
DEFAULT_BOX_X_LABEL = "Categories"
DEFAULT_BOX_Y_LABEL = "Values"
DEFAULT_HIST_X_LABEL = "Value"
DEFAULT_HIST_Y_LABEL = "Frequency"


sns.set(style="darkgrid", rc={"axes.facecolor": PLOT_BACKGROUND_COLOR,
                              "figure.facecolor": PLOT_BACKGROUND_COLOR})
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', COLOR_CYCLE)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.color'] = GRAY
plt.rcParams['font.size'] = DEFAULT_FONT_SIZE
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0


def _setup_plot_style(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    ax.figure.patch.set_facecolor(PLOT_BACKGROUND_COLOR)
    ax.set_facecolor(PLOT_BACKGROUND_COLOR)

    if CUSTOM_FONT_PATH and os.path.exists(CUSTOM_FONT_PATH):
        font_prop = FontProperties(fname=CUSTOM_FONT_PATH, size=DEFAULT_FONT_SIZE)
        title_font = FontProperties(
            fname=CUSTOM_FONT_PATH, size=DEFAULT_FONT_SIZE + TITLE_FONT_SIZE_OFFSET
        )
    else:
        font_prop = FontProperties(size=DEFAULT_FONT_SIZE)
        title_font = FontProperties(size=DEFAULT_FONT_SIZE + TITLE_FONT_SIZE_OFFSET)

    ax.set_title(title, color=TEXT_COLOR, fontproperties=title_font, fontweight='bold')
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontproperties=font_prop, fontweight='semibold')
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontproperties=font_prop, fontweight='semibold')

    ax.tick_params(colors=AXIS_COLOR, which='both')
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_color(TEXT_COLOR)

    ax.grid(True, alpha=0.15, color=GRAY, linestyle='-', linewidth=0.8)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        legend_obj = ax.legend(
            handles,
            legend_labels,
            loc=DEFAULT_LEGEND_LOCATION,
            facecolor=PLOT_BACKGROUND_COLOR,
            edgecolor=AXIS_COLOR,
            framealpha=0.8
        )
        for text in legend_obj.get_texts():
            text.set_color(TEXT_COLOR)
            text.set_fontproperties(font_prop)


def _save_plot(
    fig: plt.Figure,
    default_filename: str,
    logger: Logger,
) -> None:
    if not os.path.exists(PLOTS_DIRECTORY):
        try:
            os.makedirs(PLOTS_DIRECTORY)
            logger.info(f"Created directory: {PLOTS_DIRECTORY}")
        except OSError as e:
            logger.error(f"Error creating directory {PLOTS_DIRECTORY}: {e}")
            return

    filepath = os.path.join(PLOTS_DIRECTORY, default_filename)
    try:
        fig.savefig(
            filepath,
            dpi=DEFAULT_DPI,
            bbox_inches='tight',
            facecolor=PLOT_BACKGROUND_COLOR,
            edgecolor='none',
            transparent=False
        )
        logger.info(f"Saved plot to {filepath}")
    except Exception as e:
        logger.error(f"Error saving plot to {filepath}: {e}")


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray | list[np.ndarray],
    logger: Logger,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str] | None = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    y_data = [y] if isinstance(y, np.ndarray) and y.ndim == 1 else y
    num_series = len(y_data)

    for i, yi in enumerate(y_data):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        label = (
            labels[i]
            if labels and i < len(labels)
            else (f"Series {i+1}" if num_series > 1 else None)
        )

        sns.scatterplot(
            x=x,
            y=yi,
            s=SCATTER_MARKER_SIZE,
            alpha=SCATTER_ALPHA,
            label=label,
            color=color,
            edgecolor=AXIS_COLOR,
            linewidth=0.5,
            ax=ax
        )

    _setup_plot_style(
        ax,
        title=title or DEFAULT_SCATTER_TITLE,
        xlabel=xlabel or DEFAULT_X_LABEL,
        ylabel=ylabel or DEFAULT_Y_LABEL,
    )
    _save_plot(fig, DEFAULT_SCATTER_FILENAME, logger)
    return fig


def step_plot(
    x: np.ndarray,
    y: np.ndarray | list[np.ndarray],
    logger: Logger,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str] | None = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    y_data = [y] if isinstance(y, np.ndarray) and y.ndim == 1 else y
    num_series = len(y_data)

    for i, yi in enumerate(y_data):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        label = (
            labels[i]
            if labels and i < len(labels)
            else (f"Series {i+1}" if num_series > 1 else None)
        )

        ax.step(
            x,
            yi,
            where=STEP_WHERE,
            linewidth=STEP_LINE_WIDTH,
            label=label,
            color=color,
            alpha=0.9
        )

    _setup_plot_style(
        ax,
        title=title or DEFAULT_STEP_TITLE,
        xlabel=xlabel or DEFAULT_X_LABEL,
        ylabel=ylabel or DEFAULT_Y_LABEL,
    )
    _save_plot(fig, DEFAULT_STEP_FILENAME, logger)
    return fig


def bar_chart(
    x: np.ndarray | list[str],
    y: np.ndarray | list[np.ndarray],
    logger: Logger,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str] | None = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    is_single_series = isinstance(y, np.ndarray) and y.ndim == 1
    y_data = [y] if is_single_series else y
    num_series = len(y_data)
    num_categories = len(x)
    x_indices = np.arange(num_categories)

    if is_single_series:
        label = labels[0] if labels and len(labels) > 0 else None

        sns.barplot(
            x=x,
            y=y_data[0],
            width=BAR_WIDTH,
            alpha=BAR_ALPHA,
            color=COLOR_CYCLE[0],
            label=label,
            ax=ax,
            edgecolor=AXIS_COLOR,
            linewidth=0.8,
        )
    else:
        group_width = BAR_WIDTH / num_series
        offsets = np.linspace(
            -(BAR_WIDTH / 2) + (group_width / 2),
            (BAR_WIDTH / 2) - (group_width / 2),
            num_series,
        )
        for i, (yi, offset) in enumerate(zip(y_data, offsets)):
            color_idx = i % len(COLOR_CYCLE)
            label = (
                labels[i] if labels and i < len(labels) else f"Series {i+1}"
            )

            ax.bar(
                x_indices + offset,
                yi,
                width=group_width,
                alpha=BAR_ALPHA,
                color=COLOR_CYCLE[color_idx],
                label=label,
                edgecolor=AXIS_COLOR,
                linewidth=0.8,
            )
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x)

    _setup_plot_style(
        ax,
        title=title or DEFAULT_BAR_TITLE,
        xlabel=xlabel or DEFAULT_BAR_X_LABEL,
        ylabel=ylabel or DEFAULT_BAR_Y_LABEL,
    )
    _save_plot(fig, DEFAULT_BAR_FILENAME, logger)
    return fig


def fill_between(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    logger: Logger,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str | None = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    fill_color = COLOR_CYCLE[0]

    ax.fill_between(x, y1, y2, alpha=FILL_ALPHA, color=fill_color, label=label)

    ax.plot(x, y1, color=fill_color, alpha=BAR_ALPHA, linewidth=2.0)
    ax.plot(x, y2, color=fill_color, alpha=BAR_ALPHA, linewidth=2.0)

    _setup_plot_style(
        ax,
        title=title or DEFAULT_FILL_TITLE,
        xlabel=xlabel or DEFAULT_X_LABEL,
        ylabel=ylabel or DEFAULT_Y_LABEL,
    )
    _save_plot(fig, DEFAULT_FILL_FILENAME, logger)
    return fig


def box_plot(
    data: list[np.ndarray],
    logger: Logger,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str] | None = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    box_data = []
    group_labels = []
    for i, group_data in enumerate(data):
        group_label = labels[i] if labels and i < len(labels) else f"Group {i+1}"
        group_labels.append(group_label)
        for value in group_data:
            box_data.append({'Group': group_label, 'Value': value})

    df = pd.DataFrame(box_data)

    sns.boxplot(
        x='Group',
        y='Value',
        data=df,
        ax=ax,
        width=BOX_WIDTH,
        palette=dict(zip(group_labels, COLOR_CYCLE[:len(data)])),
        hue='Group',
        legend=False,
        notch=BOX_NOTCH,
        flierprops={
            'marker': BOX_FLIER_MARKER,
            'markerfacecolor': BOX_FLIER_MARKER_FACE_COLOR,
            'markeredgecolor': BOX_FLIER_MARKER_EDGE_COLOR,
            'alpha': BOX_FLIER_ALPHA,
            'markersize': 6
        },
        linewidth=1.5
    )

    _setup_plot_style(
        ax,
        title=title or DEFAULT_BOX_TITLE,
        xlabel=xlabel or DEFAULT_BOX_X_LABEL,
        ylabel=ylabel or DEFAULT_BOX_Y_LABEL,
    )
    _save_plot(fig, DEFAULT_BOX_FILENAME, logger)
    return fig


def histogram(
    data: np.ndarray | list[np.ndarray],
    logger: Logger,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str] | None = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    data_sets = [data] if isinstance(data, np.ndarray) and data.ndim == 1 else data
    num_series = len(data_sets)

    if num_series == 1:
        label = labels[0] if labels and len(labels) > 0 else None
        sns.histplot(
            data_sets[0],
            color=COLOR_CYCLE[0],
            alpha=HIST_ALPHA,
            label=label,
            bins=HIST_BINS,
            kde=False,
            edgecolor=HIST_EDGE_COLOR,
            linewidth=HIST_LINEWIDTH,
            ax=ax
        )
    else:

        current_alpha = HIST_ALPHA
        if ADJUST_HIST_ALPHA_OVERLAP and num_series > 1:
            current_alpha = max(0.3, HIST_ALPHA / (num_series * 0.8))

        if MULTI_HIST_TYPE == 'stepfilled' or MULTI_HIST_TYPE == 'step':

            hist_kwargs = {
                'bins': HIST_BINS,
                'density': HIST_DENSITY,
                'edgecolor': HIST_EDGE_COLOR,
                'linewidth': HIST_LINEWIDTH,
                'alpha': current_alpha,
                'histtype': MULTI_HIST_TYPE
            }

            for i, d in enumerate(data_sets):
                color_idx = i % len(COLOR_CYCLE)
                label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                ax.hist(
                    d,
                    color=COLOR_CYCLE[color_idx],
                    label=label,
                    **hist_kwargs
                )
        else:

            for i, d in enumerate(data_sets):
                color_idx = i % len(COLOR_CYCLE)
                label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                sns.histplot(
                    d,
                    color=COLOR_CYCLE[color_idx],
                    alpha=current_alpha,
                    label=label,
                    bins=HIST_BINS,
                    kde=False,
                    edgecolor=HIST_EDGE_COLOR,
                    linewidth=HIST_LINEWIDTH,
                    ax=ax,
                    element="bars"
                )

    _setup_plot_style(
        ax,
        title=title or DEFAULT_HIST_TITLE,
        xlabel=xlabel or DEFAULT_HIST_X_LABEL,
        ylabel=ylabel or DEFAULT_HIST_Y_LABEL,
    )
    _save_plot(fig, DEFAULT_HIST_FILENAME, logger)
    return fig
