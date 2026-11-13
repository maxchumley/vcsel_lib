"""Runnable examples. Importing these modules never starts training."""


def running_cells():
    """Detect IPython/Jupyter so cell execution never parses kernel arguments."""
    from IPython import get_ipython
    return get_ipython() is not None
