import numpy as np
import matplotlib.pyplot as plt
import warnings

def anomolyDist(anomolyData):
    """
    Graphs distribution of anomoly zone branch length
    Input: List of tuple/list of (x,y) branch lengths in coalescent units
    Ouput: Graph of distribution of anomoly zone data (x,y) branch lengths
    """
    #graph anomolyData

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["figure.figsize"] = 10, 6

    f = lambda x: -0.8 + (10496 / ((1 + (x / 8.423 * (10 ** -17)) ** 0.33)))

    x = np.arange(100)
    y = f(x)

    # use this context manager to make "xkcd-style" plots!
    with plt.xkcd():
        # we might be missing some xkcd fonts, so we'll ignore those
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.plot(x, (y / y.max() + 1) * 2)
        plt.title("Expected GPS vs. Semester Progress")
        plt.ylabel("E[GPA]")
        plt.xlabel("Days from Semester Start")
        plt.show()