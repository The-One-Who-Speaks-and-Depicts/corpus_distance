"""
Visualises the results of shingle processing. Contains the following functions:

* plot_shingle_diversity_barchart: produces a stacked barchart that shows, how much
unique n-shingles there are in the texts of the dataset for each n in the given range.
"""
from logging import getLogger, NullHandler

from numpy import zeros, issubdtype, number
from matplotlib.pyplot import subplots, close, Figure
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
        raise ValueError("quantities_dict should be a dict")
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
    for key in quantities_dict.keys():
        ax.bar(range_of_ns, quantities_dict[key], width, label=key, bottom=current_layer_bottom)
        current_layer_bottom += quantities_dict[key]        

    ax.set_title("Quantity of n-shingles (axis Y) depending on n (X) in the dataset")
    ax.legend(loc="best")

    close()

    return fig