"""
Visualises the results of shingle processing
"""
from logging import getLogger, NullHandler

from numpy import zeros
from matplotlib.pyplot import subplots, close

logger = getLogger(__name__)
logger.addHandler(NullHandler())

def plot_shingle_diversity_barchart(
        quantities_dict: dict,
        bottom_limit: int = 1,
        upper_limit: int = 16,
        width: float = 0.5
        ):
    """
    Takes 
    """
    logger.debug("Input params: %s", locals())  

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