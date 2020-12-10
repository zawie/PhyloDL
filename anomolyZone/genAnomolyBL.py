
"""
Checks whether the input branch lengths are in the anomoly zone
"""
def isAnomolyBL(x, y, z=0):
    #x branch length bounds
    if (x > 0.27 or x <= 0):
        return False
    #y branch length bounds in terms of x
    y_bound = -0.8 + (10496 / ((1 + (x / 8.423 * (10 ** -17)) ** 0.33)))
    if (y > y_bound):
        return False

    #no bounds on z branch length
    return True
