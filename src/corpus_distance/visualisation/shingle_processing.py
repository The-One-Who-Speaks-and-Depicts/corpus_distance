"""
Visualises the results of shingle processing. Contains the following functions:

* plot_shingle_diversity_barchart: produces a stacked barchart that shows, how much
unique n-shingles there are in the texts of the dataset for each n in the given range.
"""
from logging import getLogger, NullHandler

from numpy import zeros, issubdtype, number, std, arange, argmin
from matplotlib import rc
from matplotlib.pyplot import subplots, close, Figure, figure, plot, annotate
from collections.abc import Iterable

logger = getLogger(__name__)
logger.addHandler(NullHandler())

def plot_shingle_diversity_barchart(
        quantities_dict: dict,
        bottom_limit: int = 1,
        upper_limit: int = 16,
        width: float = 0.5
        ) -> Figure:
    """
    Takes a dictionary with lect names as keys and quantities of n-shingles
    (with n from bottom_limit to upper_limit) in the texts of these lect, and builds
    a stacked bar chart with the width being set by the width parameter.

    Parameters:
        quantities_dict (dict): a dictionary with strings as keys (names of entities,
        in this case, lects), and lists of integers as values.
        bottom_limit (int): the value that denotes the first n, by which the dataset tokens
        were split.
        uppper_limit (int): the value that denotes the value next to the last n,
        by which the dataset tokens were split.
        width (float): the width of the bar.
    
    Returns:
        Figure: a bar chart that on its X-axis has different n-s,
        into the sequences of length of which the texts got split, on its Y-axis has the overall
        number of n-shingles for a specific n, the colors denoting the lects (or any entities).
    """
    logger.debug("Input params: %s", locals())  
    if not isinstance(quantities_dict, dict) or len(quantities_dict.keys()) == 0 or not all(
        isinstance(key, str) for key in quantities_dict.keys()
    ) or not all(
        isinstance(val, Iterable) and all(
            issubdtype(type(num), number) for num in val
        ) and len(val) == len(
            list(quantities_dict.values())[0]
            ) for val in quantities_dict.values()
    ):
        raise ValueError((f"quantities_dict should be a dict, received {quantities_dict}"))
    if not isinstance(bottom_limit, int) or bottom_limit < 1:
        raise ValueError(f"bottom_limit should be a positive integer, received {bottom_limit}")
    if not isinstance(upper_limit, int) or upper_limit < 1 or upper_limit < bottom_limit:
        raise ValueError("upper_limit should be a positive integer"
                         f"more than bottom_limit({bottom_limit}), received {upper_limit}")
    if not issubdtype(type(width), number):
        raise ValueError(f"width should be a number, received {width}")
    if width <= 0:
        # this is not a problem for building a bar chart, the bars are just going to be
        # one pixel in length; still, I think it is necessary to give a warning to a user
        logger.warning(
            "Width (%s) is negative or neutral, reconsider setting the parameter", width
            )
    fig, ax = subplots()
    range_of_ns = range(bottom_limit, upper_limit)
    current_layer_bottom = zeros(upper_limit - bottom_limit)
    for lect in quantities_dict.keys():
        ax.bar(range_of_ns, quantities_dict[lect], width, label=lect, bottom=current_layer_bottom)
        current_layer_bottom += quantities_dict[lect]        

    ax.set_title("Quantity of n-shingles (axis Y) depending on n (X) in the dataset")
    ax.legend(loc="best")

    close()

    return fig

def plot_shingle_variance(
    quantities_dict: dict,
    bottom_limit: int = 1,
    upper_limit: int = 16) -> Figure:
    logger.debug("Input params: %s", locals())  
    if not isinstance(quantities_dict, dict) or len(quantities_dict.keys()) == 0 or not all(
        isinstance(key, str) for key in quantities_dict.keys()
    ) or not all(
        isinstance(val, Iterable) and all(
            issubdtype(type(num), number) for num in val
        ) and len(val) == len(
            list(quantities_dict.values())[0]
            ) for val in quantities_dict.values()
    ):
        raise ValueError(f"quantities_dict should be a dict, received {quantities_dict}")
    if not isinstance(bottom_limit, int) or bottom_limit < 1:
        raise ValueError(f"bottom_limit should be a positive integer, received {bottom_limit}")
    if not isinstance(upper_limit, int) or upper_limit < 1 or upper_limit < bottom_limit:
        raise ValueError("upper_limit should be a positive integer"
                         f"more than bottom_limit({bottom_limit}), received {upper_limit}")
    font = {'family':'DejaVu Sans', 'weight':'normal', 'size':20}
    rc('font', **font)
    fig, ax = subplots(figsize=(40, 15), dpi = 300)
    variances = []
    for i in range(0, upper_limit - bottom_limit):
        n_quantity = []
        for lect in quantities_dict.keys():
            n_quantity.append(quantities_dict[lect][i])
        variances.append(std(n_quantity))
    ax.plot(range(bottom_limit, upper_limit), variances, color="#009E73", linestyle='-', linewidth=2, marker='o')
    # Defining a palette of shades of gray (do not worry there is not 50 of them)
    GREY10 = "#1a1a1a"
    GREY30 = "#4d4d4d"
    GREY40 = "#666666"

    fig.text(
        0.015, 0.93,
        "How equal are n-shingle splits in the dataset?",
        color="black", fontname="Georgia",
        fontsize=36, weight="bold"
    )

    fig.text(
        0.015, 0.90, # (x,y) coordinates
        "Scoring MSD (mean square deviation) between unique n-shingles in lects " \
        "of the dataset",
        color=GREY30, fontsize=24,
    )
    
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.spines['bottom'].set_color(GREY40)
    ax.tick_params(axis='x', colors=GREY40)
    ax.xaxis.label.set_color(GREY40)

    # X-Axis: Position ticks center, except for first and last
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('center')

    ax.xaxis.majorTicks[0].label1.set_horizontalalignment('left')
    ax.xaxis.majorTicks[6].label1.set_horizontalalignment('right')

    HLINES = arange(0, max(variances) + max(variances)/len(variances), max(variances)/len(variances))

    # Y-axis: Add vertical gridlines every 2 billion tons
    for h in HLINES:
        ax.axhline(h, color=GREY30, lw=1, zorder=0, linestyle="--")

    # Y-Axis: Remove y ticks
    ax.yaxis.set_tick_params(width=0)

    ax.set_yticks([y for y in HLINES])

    # Y-axis: Change y axis labels to have Unit#
    ax.set_yticklabels(
        [f"{f"{y:.0f}"}" if y!=0 else "0" for y in HLINES], 
        fontsize=24,
        weight=750,
        color=GREY10
    )

    y_end_value = variances[-1]
    y_offset = 0.5
    annotation_x = upper_limit - 0.8
    annotation_y = y_end_value + y_offset
    ax.text(
        annotation_x, annotation_y + y_offset, "Lesser = more equal amount of n-shingles", 
        color=GREY10, fontsize=24, weight="650", va="center"
    )

    min_y = min(variances)
    min_x = range(bottom_limit, upper_limit)[argmin(variances)]
    plot(min_x, min_y, marker="$𖣠$", markersize=38, color="#D55E00", label="Maximum")

    
    close()

    return fig