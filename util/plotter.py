import visdom
import numpy as np

#Connect to Visdom
vis = visdom.Visdom(env='main')

def line(name,X,Y,window='Main',xlabel="Epoch"):
    """
    Plots onto a visdom plot
    """
    vis.line(
        X=np.array(X),
        Y=np.array(Y),
        win=window,
        name=name,
        update='append',
        opts = dict(title=window, xlabel=xlabel, showlegend=True)
    )

def heatmap(name, X, window="Main",xlabel="X-Axis",ylabel="Y-Axis"):
    """
    Plots a visdom heatmap
    """
    vis.heatmap(
        X=np.array(X),
        win=window,
        opts = dict(columnnames=xlabel, rownames=ylabel)
    )

def scatter(name, X, Y, window="Main", xlabel="X-Axis", ylabel="Y-Axis"):
    """
    Plots a visdom scatter plot
    """
    vis.scatter(
        X=np.array(X),
        Y=np.array(Y),
        win=window,
        name=name,
        update='append',
        opts = dict(title=window, xlabel=xlabel, showlegend=True)
    )